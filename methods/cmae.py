"""
Builds upon: https://github.com/qinenergy/cotta
Corresponding paper: https://arxiv.org/abs/2203.13591
"""
import logging
import torch
import torch.nn as nn
import torch.jit
from methods.base import TTAMethod
from augmentations.transforms_cotta import get_tta_transforms
from utils.registry import ADAPTATION_REGISTRY
from utils.operators import HOGLayerC
from typing import Tuple
from torch import Tensor

logger = logging.getLogger(__name__)


class ImageNormalizer(nn.Module):
    def __init__(self, mean: Tuple[float, float, float],
        std: Tuple[float, float, float]) -> None:
        super(ImageNormalizer, self).__init__()

        self.register_buffer('mean', torch.as_tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.as_tensor(std).view(1, 3, 1, 1))

    def forward(self, input: Tensor) -> Tensor:
        return (input - self.mean) / self.std


@ADAPTATION_REGISTRY.register()
class CMAE(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        # self.mt = cfg.CMAE.mt_alpha # in face no use
        self.rst = cfg.CMAE.rst_m
        self.block_size = cfg.CMAE.block_size # 16
        self.mask_ratio = cfg.CMAE.mask_ratio # 0.5

        self.softmax_entropy = softmax_entropy_cifar if "cifar" in self.dataset_name else softmax_entropy_imagenet
        self.transform = get_tta_transforms(self.img_size)

        vit_mu = (0.5, 0.5, 0.5)
        vit_sigma = (0.5, 0.5, 0.5)

        self.ctta_img_norm = ImageNormalizer(
            vit_mu,
            vit_sigma
        ).to(self.device)

        head_dim = 768
        if self.cfg.CMAE.use_hog:
            nbins = 9
            cell_sz = 8
            self.hogs = HOGLayerC(
                    nbins=nbins,
                    pool=cell_sz
                )
            # hogs = nn.DataParallel(hogs) # make parallel
            self.hogs.to(self.device)

            # hog_projection
            num_class = int(nbins*3*(16/cell_sz)*(16/cell_sz)) 
            self.projections = nn.Linear(head_dim, num_class, bias=True)
            if isinstance(self.projections, nn.Linear):
                nn.init.trunc_normal_(self.projections.weight, std=0.02)
                if isinstance(self.projections, nn.Linear) and self.projections.bias is not None:
                    nn.init.constant_(self.projections.bias, 0)
            # projections = nn.DataParallel(projections) # make parallel
            self.hogs.to(self.device)
            self.projections.to(self.device)
        else:
            self.hogs = None
            self.projections = None

        # mask_token
        mask_token_dim = (1, 1, head_dim)
        self.mask_token = nn.Parameter(torch.zeros(*mask_token_dim), requires_grad=True)
        # mask_token = nn.DataParallel(mask_token) # make parallel
        self.mask_token.to(self.device)
        self.random_mask = self.cfg.CMAE.random_mask

        self.hog_ratio = cfg.CMAE.hog_ratio
        self.mse_func = nn.MSELoss(reduction="mean")

        self.params, param_names = self.collect_params_cmae()
        logger.info(f"Trainable parameters: {param_names}")
        logger.info(f"Number of trainable parameters: {len(self.params)}")
        self.optimizer = self.setup_optimizer() if len(self.params) > 0 else None
        self.num_trainable_params, self.num_total_params = self.get_number_trainable_params()
        
        # Setup EMA and anchor/source model
        self.model_ema = self.copy_model(self.model)
        for param in self.model_ema.parameters():
            param.detach_()    

        # note: if the self.model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.models = [self.model, self.model_ema]
        self.model_states, self.optimizer_state = self.copy_model_and_optimizer()

    def _get_hog_label_2d(self, input_frames, output_mask, block_size): # output_mask True代表需要mask
        # input_frames, B C H W
        feat_size = input_frames.shape[-1] // block_size # num of the window ; ori = [1, H//patch_size, W//patch_size]
        tmp_hog = self.hogs(input_frames).flatten(1, 2)
        unfold_size = tmp_hog.shape[-1] // feat_size
        tmp_hog = (
            tmp_hog.permute(0, 2, 3, 1)
            .unfold(1, unfold_size, unfold_size)
            .unfold(2, unfold_size, unfold_size)
            .flatten(1, 2)
            .flatten(2)
        )
        tmp_hog = tmp_hog[output_mask]
        return tmp_hog

    def loss_calculation(self, imgs_test):
        x = imgs_test[0]
        # 如果没有这一步，效果会差多少呢？ 49.6 -> 61.0 
        x = self.ctta_img_norm(x)

        if self.random_mask:
            # random mask
            batch_size = x.size(0)
            num_tokens = (self.img_size[0] // self.block_size) * (self.img_size[1] // self.block_size)
            top_k = int(num_tokens * self.mask_ratio)
            random_numbers = torch.rand(batch_size, num_tokens, device=x.device)
            sorted_indices = torch.argsort(random_numbers, dim=1)
            masked_indices = sorted_indices[:, :top_k]
            mask_chosed = torch.zeros((batch_size, num_tokens), device=x.device)
            mask_chosed.scatter_(1, masked_indices, 1)
        else:
            n_outputs = []
            for i in range(10):
                _, outputs_  = self.model_ema(self.transform(x), return_norm=True)
                outputs_ = outputs_[:,1:,:].mean(dim=2) # cls token not need [bs, 196, 768]
                n_outputs.append(outputs_)            

            stacked_outputs_ = torch.stack(n_outputs, dim=0)
            variance = torch.var(stacked_outputs_, dim=0)
            sorted_data, sorted_indices = torch.sort(variance, dim=1, descending=True)
            top_k = int(sorted_indices.shape[1] * self.mask_ratio)
            masked_indices = sorted_indices[:, :top_k]
            mask_chosed = torch.zeros_like(sorted_data)
            mask_chosed.scatter_(1, masked_indices, 1) 

        outputs, outputs_hog = self.model(x, self.mask_token, mask_chosed, return_norm=True)

        if self.hogs is not None:
            output_mask = mask_chosed.to(bool) #mask_chosed True代表需要mask
            hog_preds = self.projections(outputs_hog[:,1:,:]) # cls token not need [bs, 196, 108]
            hog_preds = hog_preds[output_mask] 
            hog_labels = self._get_hog_label_2d(x, output_mask, block_size=self.block_size)
            hog_loss = self.mse_func(hog_preds, hog_labels) 

        outputs_temp = self.model_ema(x)

        loss_ori = (self.softmax_entropy(outputs, outputs_temp)).mean(0) 
        if self.hogs is not None:
            loss = loss_ori + self.hog_ratio*hog_loss
            # print("loss_ori:",loss_ori, "hog_loss:", hog_loss)
        else:
            loss = loss_ori
            # print("loss_ori:",loss)

        return outputs_temp, loss

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        if self.mixed_precision and self.device == "cuda":
            with torch.cuda.amp.autocast():
                outputs_ema, loss = self.loss_calculation(x)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        else:
            outputs_ema, loss = self.loss_calculation(x)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()    

        if True:
            for nm, m  in self.model.named_modules():
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        mask = (torch.rand(p.shape)<self.rst).float().cuda() 
                        with torch.no_grad():
                            p.data = self.model_states[0][f"{nm}.{npp}"] * mask + p * (1.-mask)
        # one model setting
        for temp_param, param in zip(self.model_ema.parameters(), self.model.parameters()):
            temp_param.data[:] = param[:].data[:]

        return outputs_ema


    def configure_model(self):
        """Configure model."""
        self.model.train()
        self.model.requires_grad_(False)
        # enable all trainable
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            else:
                m.requires_grad_(True)

    
    def collect_params_vit(self):
        return None, None

    def collect_params_cmae(self):
        params = []
        names = []
        for nm, p in self.model.named_parameters():
            if True: #isinstance(m, nn.BatchNorm2d): collect all 
                if p.requires_grad:
                    params.append(p)
                    names.append(nm)
        if self.projections is not None:
                for np, p in self.projections.named_parameters():
                        params.append(p)
                        names.append(f"{np}")
        if self.mask_token is not None:
            params.append(self.mask_token)
            names.append("mask_token")
        return params, names

        # params = []
        # names = []
        # for nm, m in self.model.named_modules():
        #     if True: #isinstance(m, nn.BatchNorm2d): collect all 
        #         for np, p in m.named_parameters():
        #             if np in ['weight', 'bias'] and p.requires_grad:
        #                 params.append(p)
        #                 names.append(f"{nm}.{np}")
        # if self.projections is not None:
        #         for np, p in self.projections.named_parameters():
        #                 params.append(p)
        #                 names.append(f"{np}")
        # if self.mask_token is not None:
        #     params.append(self.mask_token)
        #     names.append("mask_token")
        # return params, names
        
@torch.jit.script
def softmax_entropy_cifar(x, x_ema) -> torch.Tensor:
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)


@torch.jit.script
def softmax_entropy_imagenet(x, x_ema) -> torch.Tensor:
    return -0.5*(x_ema.softmax(1) * x.log_softmax(1)).sum(1)-0.5*(x.softmax(1) * x_ema.log_softmax(1)).sum(1)
