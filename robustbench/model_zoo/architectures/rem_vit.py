import torch
import logging

import torch.nn.functional as F
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
from timm.models.helpers import build_model_with_cfg, resolve_pretrained_cfg
from robustbench.model_zoo.architectures.utils_architectures import ImageNormalizer
from timm.models.layers import Mlp, DropPath
from robustbench.model_zoo.architectures.utils_architectures import checkpoint_filter_fn

_logger = logging.getLogger(__name__)


IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)


def rem_vit_base_patch16_224(variant, pretrained=True):
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
        rem_vit, variant, pretrained,
        pretrained_cfg=pretrained_cfg,
        # representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        # pretrained_custom_load='npz' in pretrained_cfg.url,
        **kwargs)
    model.ctta_img_norm = ImageNormalizer(IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD)
    return model


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn
    
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List
class REM_Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, return_attn=False):
        y, attn = self.attn(self.norm1(x))
        x = x + self.drop_path1(y)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        if return_attn:
            return x, attn
        return x
    
class rem_vit(VisionTransformer):
    def __init__(self, block_fn=REM_Block, **kwargs):
        super().__init__(block_fn=block_fn, **kwargs)

    def forward_features(self, x, len_keep=None, return_attn=False, need_norm=True):
        if self.ctta_img_norm is not None and need_norm:
            x = self.ctta_img_norm(x)
    
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed

        if len_keep is not None:
            B, _, D = x.shape  # batch, length, dim
            cls_save = x[:, 0, :].unsqueeze(dim=1)
            x = x[:, 1:, :]
            x = torch.gather(x, dim=1, index=len_keep.unsqueeze(-1).repeat(1, 1, D))
            x = torch.cat((cls_save, x), dim=1)

        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                x, attn = blk(x, return_attn=True)

        x = self.norm(x)
        if return_attn:
            return x, attn
        else:
            return x

    def forward(self, x, len_keep=None, return_attn=False):
        if self.ctta_img_norm is not None:
            x = self.ctta_img_norm(x)
    
        if return_attn is True:
            feat, attn = self.forward_features(x, len_keep, True, need_norm=False)
            x = self.head(feat[:,0])
            return x, attn
        else:
            feat = self.forward_features(x, len_keep, False, need_norm=False)
            x = self.head(feat[:,0])
            return x
