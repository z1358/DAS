import timm
import torch
import torch.nn as nn
from timm.models.convnext import ConvNeXt
from timm.models.helpers import build_model_with_cfg

from .wide_resnet import get_style, style_inject
from robustbench.model_zoo.architectures.utils_architectures import ImageNormalizer

def style_convnext_base(variant, pretrained=True, **kwargs):
    """
    构建一个StyleConvNeXt Base模型
    """
    model_args = dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
    model = build_model_with_cfg(
        StyleConvNeXt,
        variant,
        pretrained=pretrained,
        **dict(model_args, **kwargs)
    )

    # 自动配置归一化参数
    data_config = timm.data.resolve_data_config({}, model=model)
    mean = data_config['mean']
    std = data_config['std']
    model.ctta_img_norm = ImageNormalizer(mean, std)
    
    return model

class StyleConvNeXt(ConvNeXt):
    """
    ConvNeXt with style extraction and injection capabilities.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ctta_img_norm = None

    def forward_features_rst(self, x, return_style=False, need_norm=True):
        if self.ctta_img_norm is not None and need_norm:
            x = self.ctta_img_norm(x)

        style_list = []
        
        x = self.stem(x)

        # Stem 后的风格
        if return_style:
            style_list.append(get_style(x))

        # 2. Stages
        for stage in self.stages:
            x = stage(x)
            
            # 每个Stage之后的风格
            if return_style:
                style_list.append(get_style(x))

        x = self.norm_pre(x)

        if return_style:
            return x, style_list
        else:
            return x

    def forward_features_wst(self, x, styles, inject_pos=None, need_norm=True):
        if self.ctta_img_norm is not None and need_norm:
            x = self.ctta_img_norm(x)

        if inject_pos is None:
            inject_pos = [True] * (1 + len(self.stages))
        style_idx = 0
        
        x = self.stem(x)

        # Stem 后注入风格
        if inject_pos[style_idx]:
            x = style_inject(x, styles[style_idx])

        style_idx += 1

        for stage in self.stages:
            x = stage(x)
            # 每个Stage之后的风格
            if inject_pos[style_idx]:
                x = style_inject(x, styles[style_idx])
            style_idx += 1

        x = self.norm_pre(x)
        return x

    def forward(self, x, return_style=False, styles=None, inject_pos=None):
        if self.ctta_img_norm is not None:
            x = self.ctta_img_norm(x)

        if return_style:
            x_features, style_list = self.forward_features_rst(x, return_style=True, need_norm=False)
            x_head = self.forward_head(x_features)
            return x_head, style_list
        elif styles is not None:
            x_features = self.forward_features_wst(x, styles, inject_pos, need_norm=False)
            x_head = self.forward_head(x_features)
            return x_head
        else:
            return super().forward(x)
        
    def forward_fc(self, x):
        return self.forward_head(x)