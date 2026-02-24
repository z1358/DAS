import torch
import logging
import math

import torch.nn.functional as F
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
from timm.models.helpers import build_model_with_cfg, resolve_pretrained_cfg
from .wide_resnet import get_style, style_inject
from robustbench.model_zoo.architectures.utils_architectures import ImageNormalizer

_logger = logging.getLogger(__name__)


IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)

def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb

def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
        out_dict[k] = v
    return out_dict


def style_vit_base_patch16_224(variant, pretrained=True):
    kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')
    
    pretrained_cfg = resolve_pretrained_cfg(variant)
    default_num_classes = pretrained_cfg.num_classes
    num_classes = kwargs.get('num_classes', default_num_classes)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        repr_size = None

    model = build_model_with_cfg(
        style_vit, variant, pretrained,
        pretrained_cfg=pretrained_cfg,
        # representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        # pretrained_custom_load='npz' in pretrained_cfg.url,
        **kwargs)
    # model.ctta_img_norm = ImageNormalizer((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    model.ctta_img_norm = ImageNormalizer(IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD)
    return model

def style_vit_base_patch16_384(variant, pretrained=False):
    kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')
    
    pretrained_cfg = resolve_pretrained_cfg(variant)
    default_num_classes = pretrained_cfg.num_classes
    num_classes = kwargs.get('num_classes', default_num_classes)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        repr_size = None

    model = build_model_with_cfg(
        style_vit, variant, pretrained,
        pretrained_cfg=pretrained_cfg,
        # representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        # pretrained_custom_load='npz' in pretrained_cfg.url,
        **kwargs)
    # model.ctta_img_norm = ImageNormalizer((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    model.ctta_img_norm = ImageNormalizer(IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD)
    return model


class style_vit(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ctta_img_norm = None

    def forward_features_rst(self, x, return_style=False, need_norm=True):
        """提取特征并可选择性返回风格
        Args:
            x (torch.Tensor): 输入图像张量, shape [B, 3, H, W]
            return_style (bool): 是否返回风格向量
        """
        if self.ctta_img_norm is not None and need_norm:
            x = self.ctta_img_norm(x)

        style_list = []
        x = self.patch_embed(x)
        B, N, C = x.shape
        
        if return_style:
            # 转换为形状[B, C, h, w]以计算风格
            patch_h = patch_w = int(math.sqrt(N))
            tokens = x.permute(0, 2, 1).reshape(B, C, patch_h, patch_w)
            style_list.append(get_style(tokens))
            
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        x = self.pos_drop(x + self.pos_embed)

        # 依次通过每个transformer block并提取风格
        for block in self.blocks:
            x = block(x)
            if return_style and len(style_list) < len(self.blocks) / 2:
                # 去掉cls token后重新排列计算风格 
                tokens = x[:, 1:, :]  # [B, N, C]
                tokens = tokens.permute(0, 2, 1).reshape(B, C, patch_h, patch_w)
                style_list.append(get_style(tokens))

        x = self.norm(x)
        # x_norm = x
        if return_style:
            return x[:, 0], style_list
        else:
            return x[:, 0]

    def forward_features_wst(self, x: torch.Tensor, styles=None, inject_pos=None, need_norm=True):
        """带风格注入的特征提取
        Args:
            x (torch.Tensor): 输入图像张量, shape [B, 3, H, W]  
            styles (list): 风格向量列表
            inject_pos (list): 注入位置列表,长度等于blocks数量+1
        """
        if self.ctta_img_norm is not None and need_norm:
            x = self.ctta_img_norm(x)
        
        if inject_pos is None:
            # 默认在所有位置注入
            inject_pos = [True] * (len(self.blocks) + 1)
            
        x = self.patch_embed(x)
        B, N, C = x.shape
        patch_h = patch_w = int(math.sqrt(N))

        # patch embedding 后注入风格
        if inject_pos[0]:
            tokens = x.permute(0, 2, 1).reshape(B, C, patch_h, patch_w)
            tokens = style_inject(tokens, styles[0])
            x = tokens.reshape(B, C, -1).permute(0, 2, 1)

        # 添加class token    
        cls_token = self.cls_token.expand(B, -1, -1)

        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        # 依次通过blocks,在指定位置注入风格
        for i, block in enumerate(self.blocks):
            x = block(x)
            if inject_pos[i+1]:
                tokens = x[:, 1:, :]  # 去掉cls token
                tokens = tokens.permute(0, 2, 1).reshape(B, C, patch_h, patch_w)
                tokens = style_inject(tokens, styles[i+1])
                tokens = tokens.reshape(B, C, -1).permute(0, 2, 1)
                x = torch.cat([x[:, :1, :], tokens], dim=1)  # 重新加入cls token

        x = self.norm(x)
        return x[:, 0]


    def forward(self, x, return_style=False, styles=None, inject_pos=None):
        """前向传播
        Args:
            x (torch.Tensor): 输入图像张量, shape [B, 3, H, W]
            return_style (bool): 是否返回风格向量
            styles (list): 用于注入的风格向量列表
            inject_pos (list): 风格注入位置列表
        """
        if self.ctta_img_norm is not None:
            x = self.ctta_img_norm(x)

        if return_style:
            x, style_list = self.forward_features_rst(x, return_style=True, need_norm=False)
            x = self.head(x)
            return x, style_list
        elif styles is not None:
            x = self.forward_features_wst(x, styles, inject_pos, need_norm=False)
            x = self.head(x)
            return x
        else:
            x = self.forward_features_rst(x, need_norm=False)
            x = self.head(x)
            return x
        
    def forward_fc(self, x):
        return self.head(x)