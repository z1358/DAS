import timm
from timm.models.swin_transformer import SwinTransformer, checkpoint_filter_fn
from timm.models.helpers import build_model_with_cfg

from .wide_resnet import get_style, style_inject
from robustbench.model_zoo.architectures.utils_architectures import ImageNormalizer

def style_swin_base_patch4_window7_224(variant, pretrained=True, **kwargs):
    """
    构建一个StyleSwinTransformer Base模型
    """
    model_kwargs = dict(
        patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32),
        **kwargs
    )
    model = build_model_with_cfg(
        StyleSwinTransformer, 
        variant,
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_strict=False,
        **model_kwargs
    )

    data_config = timm.data.resolve_data_config({}, model=model)
    mean = data_config['mean']
    std = data_config['std']

    model.ctta_img_norm = ImageNormalizer(mean, std)
    return model

def style_swin_tiny_patch4_window7_224(variant, pretrained=True, **kwargs):
    """
    构建一个StyleSwinTransformer Base模型
    """
    model_kwargs = dict(
        patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
        **kwargs
    )
    model = build_model_with_cfg(
        StyleSwinTransformer, 
        variant,
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_strict=False,
        **model_kwargs
    )

    data_config = timm.data.resolve_data_config({}, model=model)
    mean = data_config['mean']
    std = data_config['std']

    model.ctta_img_norm = ImageNormalizer(mean, std)
    return model

class StyleSwinTransformer(SwinTransformer):
    """
    Swin Transformer with style extraction and injection capabilities.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 可以在这里添加自定义属性，例如 ImageNormalizer
        self.ctta_img_norm = None

    def forward_features_rst(self, x, return_style=False, need_norm=True):
        """
        前向传播并提取每一层的风格。
        Swin的forward_features逻辑比较复杂，我们直接重写它。
        """
        if self.ctta_img_norm is not None and need_norm:
            x = self.ctta_img_norm(x)

        style_list = []
        
        x = self.patch_embed(x) # 50 56 56 128
        
        # Patch Embedding 后的风格
        if return_style:
            tokens = x.permute(0, 3, 1, 2).contiguous()
            style_list.append(get_style(tokens))

        for layer in self.layers:
            x = layer(x)
            
            # 每个Stage之后的风格
            if return_style:
                tokens = x.permute(0, 3, 1, 2).contiguous()
                style_list.append(get_style(tokens))

        x = self.norm(x)
        if return_style:
            return x, style_list
        else:
            return x

    def forward_features_wst(self, x, styles, inject_pos=None, need_norm=True):
        """
        带风格注入的前向传播。
        """
        if self.ctta_img_norm is not None and need_norm:
            x = self.ctta_img_norm(x)

        if inject_pos is None:
            inject_pos = [True] * (1 + len(self.layers))
        style_idx = 0
        
        x = self.patch_embed(x)
        
        # 1. Patch Embedding 后注入风格
        if inject_pos[style_idx]:
            tokens = x.permute(0, 3, 1, 2).contiguous()
            tokens = style_inject(tokens, styles[style_idx])
            x = tokens.permute(0, 2, 3, 1).contiguous()
        style_idx += 1
            
        for layer in self.layers:
            x = layer(x)
            # 每个Stage之后的风格
            if inject_pos[style_idx]:
                tokens = x.permute(0, 3, 1, 2).contiguous()
                tokens = style_inject(tokens, styles[style_idx])
                x = tokens.permute(0, 2, 3, 1).contiguous()
            style_idx += 1

        x = self.norm(x)
        return x

    def forward(self, x, return_style=False, styles=None, inject_pos=None):
        """
        总的前向传播路由。
        """
        if self.ctta_img_norm is not None:
            x = self.ctta_img_norm(x)
        if return_style:
            x_features, style_list = self.forward_features_rst(x, return_style=True, need_norm=False)
            # 在Swin中，分类头作用于norm之后的[B, L, C]的平均值
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