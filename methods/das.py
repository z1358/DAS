import logging

import os
import tqdm
import torch
import torch.nn as nn
import torch.jit
import torch.nn.functional as F
import wandb
from methods.base import TTAMethod
from augmentations.transforms_cotta import get_tta_transforms, FourierStyleTransferFDA
from datasets.data_loading import get_source_loader
from utils.registry import ADAPTATION_REGISTRY
from utils.losses import SymmetricCrossEntropy
from utils.misc import ema_update_model
import numpy as np
import torch
import random
from collections import defaultdict
from typing import List, Set

logger = logging.getLogger(__name__)
    
@ADAPTATION_REGISTRY.register()
class DAS(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        batch_size_src = cfg.TEST.BATCH_SIZE if cfg.TEST.BATCH_SIZE > 1 else cfg.TEST.WINDOW_LENGTH
        if cfg.SOURCE.USE_SYN:
            DATA_DIR_ROOT = os.path.join(cfg.DATA_DIR, "Syn_datasets")
        else:
            DATA_DIR_ROOT = cfg.DATA_DIR
        
        syn_dataset, self.src_loader = get_source_loader(dataset_name=cfg.CORRUPTION.DATASET,
                                               adaptation=cfg.MODEL.ADAPTATION,
                                               preprocess=model.model_preprocess,
                                               data_root_dir=DATA_DIR_ROOT,
                                               batch_size=batch_size_src,
                                               ckpt_path=cfg.MODEL.CKPT_PATH,
                                               num_samples=cfg.SOURCE.NUM_SAMPLES,
                                               percentage=cfg.SOURCE.PERCENTAGE,
                                               workers=min(cfg.SOURCE.NUM_WORKERS, os.cpu_count()),
                                               use_synthetic=cfg.SOURCE.USE_SYN,
                                               model_arch=cfg.MODEL.ARCH)
        self.src_loader_iter = iter(self.src_loader)
        self.contrast_mode = cfg.CONTRAST.MODE
        self.temperature = cfg.CONTRAST.TEMPERATURE
        self.base_temperature = cfg.CONTRAST.BASE_TEMPERATURE
        self.projection_dim = cfg.CONTRAST.PROJECTION_DIM
        self.lambda_ce_src = cfg.STYLE.LAMBDA_CE_SRC
        self.lambda_ce_trg = cfg.STYLE.LAMBDA_CE_TRG
        self.lambda_cont = cfg.STYLE.LAMBDA_CONT
        self.m_teacher_momentum = cfg.M_TEACHER.MOMENTUM
        # arguments neeeded for warm up
        self.warmup_steps = cfg.STYLE.NUM_SAMPLES_WARM_UP // batch_size_src
        self.final_lr = cfg.OPTIM.LR
        arch_name = cfg.MODEL.ARCH
        ckpt_path = cfg.MODEL.CKPT_PATH
        self.use_fst = cfg.STYLE.USE_FourierST

        self.num_classes = num_classes
        self.confusion_matrix = torch.zeros((num_classes, num_classes), device=self.device)
        self.adaptive_sampler = AdaptiveSampler(syn_dataset)

        # Style transfer
        self.st_layers = cfg.STYLE.ST_LAYERS
        inject_pos = [False] * 15
        if len(self.st_layers) > 0:
            for layer in self.st_layers:
                if layer >= 0:
                    inject_pos[layer] = True
        self.pos = inject_pos
        print(f"Injecting style transfer in layers {self.st_layers} at positions {self.pos}")

        self.tta_transform = get_tta_transforms(self.img_size)
        self.fourier_src_style_transfer = FourierStyleTransferFDA(
            low_freq_ratio=self.cfg.STYLE.FourierST_RATIO, alpha=self.cfg.STYLE.FourierST_ALPHA
        )
        
        # setup loss functions
        self.symmetric_cross_entropy = SymmetricCrossEntropy()

        # Setup EMA model
        self.model_ema = self.copy_model(self.model)
        for param in self.model_ema.parameters():
            param.detach_()

        self.num_batches = 0

        self.lambda_cont = cfg.STYLE.LAMBDA_CONT
        self.projection_dim = cfg.CONTRAST.PROJECTION_DIM
        # self.contrast_criterion = SupConLoss(0.07, base_temperature=1.0)
        num_channels = 768
        if self.cfg.CORRUPTION.DATASET == "cifar100_c" and "VIT" not in self.cfg.MODEL.ARCH:
            num_channels = 1024
        elif self.cfg.CORRUPTION.DATASET == "cifar10_c" and "VIT" not in self.cfg.MODEL.ARCH:
            num_channels = 640
        elif self.cfg.CORRUPTION.DATASET == "imagenet_c" and "VIT" not in self.cfg.MODEL.ARCH:
            num_channels = 2048
        # self.projector = nn.Sequential(nn.Linear(num_channels, self.projection_dim), nn.ReLU(),
        #                                 nn.Linear(self.projection_dim, self.projection_dim)).to(self.device)
        # self.optimizer.add_param_group({'params': self.projector.parameters(), 'lr': self.optimizer.param_groups[0]["lr"]})

        self.projector = nn.Identity()

    def update_confusion_matrix(self, preds, labels, momentum=0.5):
        # preds, labels: 1D tensor, shape [batch_size]
        # self.confusion_matrix = torch.zeros((self.num_classes, self.num_classes), device=self.device)
        cur_matrix = torch.zeros_like(self.confusion_matrix)
        for t, p in zip(labels.view(-1), preds.view(-1)):
            cur_matrix[t.long(), p.long()] += 1
        self.confusion_matrix = momentum * self.confusion_matrix + (1 - momentum) * cur_matrix
    
    def get_sampling_weights(self):
        # 统计每个类别的“混淆度”
        confusion = self.confusion_matrix + self.confusion_matrix.t()  # 对称化
        confusion.fill_diagonal_(0)  # 不考虑自身
        class_confusion = confusion.sum(1)  # 每个类别与其他类别的混淆总和
        weights = class_confusion + 1e-6  # 防止全为0
        weights = weights / weights.sum()  # 归一化为概率
        return weights.cpu().numpy()

    def sample_syn_batch(self):
        # 按混淆权重采样合成域batch
        weights = self.get_sampling_weights()
        top2_indices = np.argsort(weights)[-2:][::-1].tolist()
        images, labels = self.adaptive_sampler.sample(
            target_class_indices=top2_indices,
            num_samples=10
        )
        return images.to(self.device), labels.to(self.device)
    
    # Integrated from: https://github.com/HobbitLong/SupContrast/blob/master/losses.py
    def contrastive_loss(self, features, labels=None, mask=None):
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            mask = mask.float().to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        contrast_feature = self.projector(contrast_feature)
        contrast_feature = F.normalize(contrast_feature, p=2, dim=1)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # 1. 计算每个锚点对应的正样本对的 log_prob 之和
        sum_of_log_probs_for_positives = (mask * log_prob).sum(1)
        # 2. 计算每个锚点对应的正样本对的数量 (此时 mask 已经排除了自对比 mask[i,i]=0)
        num_positive_pairs = mask.sum(1)

        # 3. 识别那些至少有一个正样本对的锚点
        #    (即 num_positive_pairs > 0 的锚点)
        valid_anchors_mask = num_positive_pairs > 0

        # 4. 过滤，只保留有效锚点的数据
        sum_of_log_probs_for_positives_filtered = sum_of_log_probs_for_positives[valid_anchors_mask]
        num_positive_pairs_filtered = num_positive_pairs[valid_anchors_mask]

        # 5. 如果没有任何锚点有正样本对（例如，所有样本标签都唯一），则损失为0
        if num_positive_pairs_filtered.numel() == 0:
            # 返回一个与输入特征设备相同、梯度跟踪状态一致的0值张量
            return torch.tensor(0.0, device=features.device, 
                                requires_grad=features.requires_grad)

        # 6. 计算有效锚点的平均正样本对数似然
        mean_log_prob_pos_for_valid_anchors = sum_of_log_probs_for_positives_filtered / num_positive_pairs_filtered
        # 修改结束

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos_for_valid_anchors

        # 我们这里对有效锚点的损失取平均。
        if loss.numel() > 0: # 确保 loss 不是空张量 (虽然上面的 numel()==0 检查应该已经处理了)
            loss = loss.mean()
        else:
            # 如果经过筛选后 loss 为空 (理论上不应发生，因为 num_positive_pairs_filtered.numel()==0 已处理)
            return torch.tensor(0.0, device=features.device,
                                requires_grad=features.requires_grad)
            
        return loss

        # loss = loss.view(anchor_count, batch_size).mean()
        # return loss
    
    def loss_calculation(self, x):
        imgs_test = x[0] # 此时传进来的就是[0,1]之间的tensor
        self.num_batches += 1

        # forward original test data
        features_test, cur_styles = self.model.forward_features_rst(imgs_test, return_style=True)
        outputs_test = self.model.forward_fc(features_test)

        # forward augmented test data
        features_aug_test = self.model.forward_features_rst(self.tta_transform(imgs_test))
        outputs_aug_test = self.model.forward_fc(features_aug_test)

        # forward original test data through the ema model
        outputs_ema = self.model_ema(imgs_test)

        # energy_ema = outputs_ema.logsumexp(1).mean()
        # energy_test = outputs_test.logsumexp(1).mean()
        # energy_aug_test = outputs_aug_test.logsumexp(1).mean()

        loss_self_training = (0.5 * self.symmetric_cross_entropy(outputs_test, outputs_ema) + 0.5 * self.symmetric_cross_entropy(outputs_aug_test, outputs_ema)).mean(0)
        
        log_data = {
            "losses/loss_self_training": loss_self_training.item(),
        }
        loss = self.lambda_ce_trg * loss_self_training #  + self.lambda_cont * loss_contrastive

        if self.lambda_ce_src > 0:
            # sample source batch
            try:
                batch = next(self.src_loader_iter)
            except StopIteration:
                self.src_loader_iter = iter(self.src_loader)
                batch = next(self.src_loader_iter)

            imgs_src, labels_src = batch[0].to(self.device), batch[1].to(self.device)
            # imgs_adap, labels_adap = self.sample_syn_batch()

            # imgs_src = torch.cat([init_imgs_src[:40], imgs_adap], dim=0)
            # labels_src = torch.cat([init_labels_src[:40], labels_adap], dim=0)
            # print(labels_src)
            # train on labeled source data
            if self.use_fst:
                with torch.no_grad(): # Style transfer is usually not backpropagated through the style image
                    stylized_imgs_src = self.fourier_src_style_transfer(imgs_src, imgs_test)

                # features_src = self.model.forward_features_wst(imgs_src, cur_styles, inject_pos=self.pos)
                features_src = self.model.forward_features_wst(stylized_imgs_src, cur_styles, inject_pos=self.pos)
            else:
                features_src = self.model.forward_features_wst(imgs_src, cur_styles, inject_pos=self.pos)
                
            outputs_src = self.model.forward_fc(features_src)
            loss_ce_src = F.cross_entropy(outputs_src, labels_src.long())
            loss += self.lambda_ce_src * loss_ce_src
            # energy_src = outputs_src.logsumexp(1).mean()
            # 将源域损失添加到日志字典
            log_data["losses/loss_ce_src"] = loss_ce_src.item()

            with torch.no_grad():
                src_preds = outputs_src.argmax(1)
                self.update_confusion_matrix(src_preds, labels_src.long()) 

        if self.lambda_cont > 0.0:
            supcon_features = torch.cat([features_src, features_test, features_aug_test], dim=0)
            supcon_features.unsqueeze_(1)
            labels_cur = (outputs_test + outputs_ema).argmax(1)
            supcon_labels = torch.cat([labels_src.long(), labels_cur, labels_cur], dim=0)
            loss_contrastive = self.contrastive_loss(features=supcon_features, labels=supcon_labels)
            loss += self.lambda_cont * loss_contrastive
        
        # lambda_energy = 0.1
        # if lambda_energy > 0.0:
        #     log_data["energy/energy_ema"] = energy_ema.item()
        #     log_data["energy/energy_test"] = energy_test.item()
        #     log_data["energy/energy_aug_test"] = energy_aug_test.item()
        #     log_data["energy/energy_src"] = energy_src.item()

        #     # energy loss
        #     loss_energy = (energy_test - energy_ema).abs() + (energy_aug_test - energy_ema).abs()
        #     loss += lambda_energy * loss_energy

        #     log_data["losses/loss_energy"] = loss_energy.item()

        # print(log_data)
        # --- 执行 W&B 日志记录 (合并调用并指定 step) ---
        if self.cfg.USE_WANDB:
            wandb.log(log_data, step=self.num_batches)

        # create and return the ensemble prediction
        outputs = {"ensemble": outputs_test + outputs_ema, "outputs_test": outputs_test, "outputs_ema": outputs_ema}
        return outputs, loss

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        if self.mixed_precision and self.device == "cuda":
            with torch.cuda.amp.autocast():
                outputs, loss = self.loss_calculation(x)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        else:
            outputs, loss = self.loss_calculation(x)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.model_ema = ema_update_model(
            model_to_update=self.model_ema,
            model_to_merge=self.model,
            momentum=self.m_teacher_momentum,
            device=self.device,
            update_all=True
        )
                    
        return outputs

    def configure_model(self):
        """Configure model"""
        # model.train()
        self.model.eval()  # eval mode to avoid stochastic depth in swin. test-time normalization is still applied
        # disable grad, to (re-)enable only what we update
        self.model.requires_grad_(False)
        # enable all trainable
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.BatchNorm1d):
                m.train()   # always forcing train mode in bn1d will cause problems for single sample tta
                m.requires_grad_(True)
            else:
                m.requires_grad_(True)

class AdaptiveSampler:
    def __init__(self, dataset):
        print("initing AdaptiveSampler")
        self.dataset = dataset
        self.class_to_indices_map = defaultdict(list)
        for idx, (_, target) in enumerate(dataset.samples):
            self.class_to_indices_map[target].append(idx)

    def sample(self, target_class_indices: List[int], num_samples: int):

        candidate_pool = []
        for class_idx in target_class_indices:
            candidate_pool.extend(self.class_to_indices_map[class_idx])
            
        if not candidate_pool:
            return torch.empty(0), torch.empty(0)
        sampled_indices = random.choices(candidate_pool, k=num_samples)

        batch_data = [self.dataset[i] for i in sampled_indices]
        images, labels = torch.utils.data.default_collate(batch_data)
        
        return images, labels