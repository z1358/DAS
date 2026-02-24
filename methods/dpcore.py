import torch
import torch.nn as nn
import torch.jit

from methods.base import TTAMethod
from utils.registry import ADAPTATION_REGISTRY

import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer, Mlp
import math
from functools import reduce
from operator import mul
import logging
logger = logging.getLogger(__name__)

class PromptViT(nn.Module):
    '''
    Vision Transformer with added prompts at the input layer
    '''
    def __init__(self,
                vit:VisionTransformer,
                num_prompts = 1):
        super().__init__()
        self.vit = vit
        self.num_prompts = num_prompts
        self.prompt_dim = vit.embed_dim

        if num_prompts > 0:
            self.prompts = nn.Parameter(torch.zeros(1, num_prompts, self.prompt_dim)) # [1 8 768]
            # initialization adopted from vpt, https://arxiv.org/abs/2203.12119
            val = math.sqrt(6. / float(3 * reduce(mul, vit.patch_embed.patch_size, 1) + self.prompt_dim)) # noqa
            nn.init.uniform_(self.prompts.data, -val, val) # xavier_uniform initialization
    
    def reset(self):
        val = math.sqrt(6. / float(3 * reduce(mul, self.vit.patch_embed.patch_size, 1) + self.prompt_dim)) # noqa
        nn.init.uniform_(self.prompts.data, -val, val) # xavier_uniform initialization

    def prompt_injection(self, x):
        if self.num_prompts > 0:
            x = torch.cat((
                x[:,:1,:],
                self.prompts.expand(x.shape[0],-1,-1),
                x[:,1:,:]
            ), dim=1)
        return x
    
    def _collect_layers_features(self, x):
        # collecting features for each layer
        cls_features = []
        for i in range(len(self.vit.blocks)):
            x = self.vit.blocks[i](x)
            if i < len(self.vit.blocks) - 1:
                cls_features.append(self.vit.blocks[i+1].norm1(x[:, 0]))
            else:
                cls_features.append(self.vit.norm(x[:, 0]))
        cls_features = torch.cat(cls_features, dim=1)
        return cls_features

    def forward_features(self, x):
        '''
        Forwarding a batch of samples with prompts' embeddings inserted
        We added only the highlighted line of code based on `timm` library
        '''
        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)
        # inject prompts
        x = self.prompt_injection(x)
        # !!end
        x = self.vit.norm_pre(x)
        x = self.vit.blocks(x)
        x = self.vit.norm(x)
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.vit.forward_head(x)
        return x
    
    def layers_cls_features(self, x):
        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)
        x = self.vit.norm_pre(x)
        return self._collect_layers_features(x)
    
    def layers_cls_features_with_prompts(self, x):
        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)
        # inject prompts
        x = self.prompt_injection(x)
        # !!end
        x = self.vit.norm_pre(x)
        return self._collect_layers_features(x)
    
    def forward_raw_features(self, x):
        '''
        Forwarding a batch of samples without prompts' embeddings inserted
        We added only the highlighted line of code based on `timm` library
        '''
        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)

        # !!end
        x = self.vit.norm_pre(x)
        x = self.vit.blocks(x)
        x = self.vit.norm(x)
        return x
    

@ADAPTATION_REGISTRY.register()
class DPCore(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        self.lamda = 1.0
        self.temp_tau = cfg.DPCore.TEMP_TAU
        self.ema_alpha = cfg.DPCore.EMA_ALPHA
        self.thr_rho = cfg.DPCore.THR_RHO
        self.E_ID = 1
        self.E_OOD = cfg.DPCore.E_OOD

        self.coreset = []

        self.models = [self.model]
        self.model_states, self.optimizer_state = self.copy_model_and_optimizer()

        self.obtain_src_stat(cfg.DATA_DIR, num_samples=cfg.DPCore.src_num_samples)

    def _update_coreset(self, weights, batch_mean, batch_std):
        """Update overall test statistics"""
        updated_prompts = self.model.prompts.clone().detach().cpu()
        for p_idx in range(len(self.coreset)):
            self.coreset[p_idx][0] += self.ema_alpha * weights[p_idx] * (batch_mean - self.coreset[p_idx][0])
            self.coreset[p_idx][1] += self.ema_alpha * weights[p_idx] * torch.clamp(batch_std - self.coreset[p_idx][1], min=0.0)
            self.coreset[p_idx][2] += self.ema_alpha * weights[p_idx] * (updated_prompts - self.coreset[p_idx][2])  

    @torch.no_grad()
    def _eval_coreset(self, x):
        """Evaluate the coreset on a batch of samples."""
        
        loss, batch_mean, batch_std = forward_and_get_loss(x, self.model, self.lamda, self.train_info, with_prompt=False)
        is_ID = False
        weights = None
        weighted_prompts = None
        if self.coreset:
            weights = calculate_weights(self.coreset, batch_mean, batch_std, self.lamda, self.temp_tau)
            weighted_prompts = torch.stack([w * p[2] for w, p in zip(weights, self.coreset)], dim=0).sum(dim=0)
            assert weighted_prompts.shape == self.model.prompts.shape, f'{weighted_prompts.shape} != {self.model.prompts.shape}'
            self.model.prompts = torch.nn.Parameter(weighted_prompts.cuda())
            self.model.prompts.requires_grad_(False)
            
            loss_new, _, _ = forward_and_get_loss(x, self.model, self.lamda, self.train_info, with_prompt=True)
            if loss_new < loss * self.thr_rho:
                self.model.prompts.requires_grad_(True)
                self.optimizer = torch.optim.AdamW([self.model.prompts], lr=1e-1)
                is_ID = True
        else:
            loss_new = loss
            
        return is_ID, batch_mean, batch_std, weighted_prompts, weights, loss, loss_new

    def obtain_src_stat(self, data_path, num_samples=300):
        num = 0
        features = []
        import timm
        from torchvision.datasets import ImageNet, STL10
        net = timm.create_model('vit_base_patch16_224.augreg_in1k', pretrained=True)
        data_config = timm.data.resolve_model_data_config(net)
        src_transforms = timm.data.create_transform(**data_config, is_training=False)
        src_dataset = ImageNet(root=f"{data_path}/imagenet2012", split= 'train', transform=src_transforms)
        src_loader = torch.utils.data.DataLoader(src_dataset, batch_size=64, shuffle=True)
        
        with torch.no_grad():
            for idx, dl in enumerate(src_loader):
                images = dl[0].cuda()
                feature = self.model.forward_raw_features(images)
                
                output = self.model(images)
                ent = softmax_entropy(output)
                selected_indices = torch.where(ent < math.log(self.num_classes)/2-1)[0]
                feature = feature[selected_indices]
                
                features.append(feature[:, 0])
                num += feature.shape[0]
                if num >= num_samples:
                    break

            features = torch.cat(features, dim=0)
            features = features[:num_samples, :]
            print(f'Source Statistics computed with {features.shape[0]} examples actually used {idx} batches.')
            self.train_info = torch.std_mean(features, dim=0)
        del features

    def loss_calculation(self, x):
        imgs_test = x[0]
        # imgs_test = self.img_norm(imgs_test) if hasattr(self, 'img_norm') else imgs_test # both source and target images are normalized 

        is_ID, batch_mean, batch_std, weighted_prompts, weights, loss_raw, loss_new = self._eval_coreset(imgs_test)
        if is_ID:
            for _ in range(self.E_ID):
                self.model.prompts = torch.nn.Parameter(weighted_prompts.cuda())
                optimizer = torch.optim.AdamW([self.model.prompts], lr=1e-1)
                outputs, loss, batch_mean, batch_std = prompt_forward_and_adapt(imgs_test, self.model, optimizer, self.lamda, self.train_info)
            self._update_coreset(weights, batch_mean, batch_std)
            
        else:
            load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_states[0], self.optimizer_state)
            self.model.prompts.requires_grad_(True)
            self.optimizer = torch.optim.AdamW([self.model.prompts], lr=1e-1)
            
            for _ in range(self.E_OOD):
                outputs, loss, _, _ = prompt_forward_and_adapt(imgs_test, self.model, self.optimizer, self.lamda, self.train_info)

            self.coreset.append([batch_mean, batch_std, self.model.prompts.clone().detach().cpu()])
            logger.info(f'New coreset added, current size: {len(self.coreset)}, loss_raw: {loss_raw.item():.3f}, loss_new: {loss_new.item():.3f}')

        return outputs, loss_raw, loss_new, loss
    
    @torch.enable_grad()
    def forward_and_adapt(self, x):
        if self.mixed_precision and self.device == "cuda":
            with torch.cuda.amp.autocast():
                outputs = self.loss_calculation(x)
        else:
            outputs = self.loss_calculation(x)
        return outputs[0]
    
    def configure_model(self):
        """Configure model."""
        # 学cmae，将norm与模型分割开
        self.img_norm = self.model[0]
        self.model = self.model[1]

        self.model = PromptViT(self.model, self.cfg.DPCore.PROMPT_NUM)
        self.model.to(self.device)
        self.img_norm.to(self.device)
        self.model.train()

    def collect_params_vit(self):
        return [self.model.prompts], 'prompts'

@torch.enable_grad()
def prompt_forward_and_adapt(x, model: PromptViT, optimizer, lamda, train_info):
    """Forward and adapt model on batch of data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    features = model.forward_features(x)
    cls_features = features[:, 0]
    batch_std, batch_mean = torch.std_mean(cls_features, dim=0)
    # std_mse, mean_mse = criterion_mse(batch_std, train_info[0].cuda()), criterion_mse(batch_mean, train_info[1].cuda())

    std_loss = torch.norm(batch_std - train_info[0].cuda(), p=2)
    mean_loss = torch.norm(batch_mean - train_info[1].cuda(), p=2)
    loss = lamda * std_loss + mean_loss
    
    # output = model.vit.head(cls_features)
    output = model.vit.forward_head(features)
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return output, loss, batch_mean, batch_std   

def forward_and_get_loss(images, model:PromptViT, lamda, train_info, with_prompt=False):
    if with_prompt:
        cls_features = model.forward_features(images)[:, 0]
    else:
        cls_features = model.forward_raw_features(images)[:, 0]
    

    """discrepancy loss"""
    batch_std, batch_mean = torch.std_mean(cls_features, dim=0)
    # std_mse, mean_mse = criterion_mse(batch_std, train_info[0].cuda()), criterion_mse(batch_mean, train_info[1].cuda())
    std_loss = torch.norm(batch_std - train_info[0].cuda(), p=2)
    mean_loss = torch.norm(batch_mean - train_info[1].cuda(), p=2)
    
    loss = lamda * std_loss + mean_loss
    # output = model.vit.forward_head(raw_features)

    return loss, batch_mean, batch_std

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x/ temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x

def calculate_weights(coreset, batch_mean, batch_std, lamda, temp_tau):
    mean_tensor = torch.stack([p[0] for p in coreset])
    std_tensor = torch.stack([p[1] for p in coreset])
    assert mean_tensor.shape[1] == 768 and mean_tensor.shape[0] == len(coreset)
    
    mean_match = torch.norm(batch_mean - mean_tensor, p=2, dim=1)
    std_match = torch.norm(batch_std - std_tensor, p=2, dim=1)
    
    match_loss = mean_match + lamda *  std_match
    weights = torch.nn.functional.softmax(-match_loss/temp_tau, dim=0)
    return weights.detach().cpu()