from collections import OrderedDict

from torchvision import models as pt_models

from robustbench.model_zoo.enums import ThreatModel
from robustbench.model_zoo.architectures.utils_architectures import normalize_model
from robustbench.model_zoo.cmae_vit import cmae_vit_base_patch16_224 as create_model_mae
from robustbench.model_zoo.architectures.style_resnet import create_resnet50_with_style
from robustbench.model_zoo.architectures.style_vit import style_vit_base_patch16_224 as create_model_vit_style
from robustbench.model_zoo.architectures.style_swin import style_swin_base_patch4_window7_224 as create_model_swin_style
from robustbench.model_zoo.architectures.style_swin import style_swin_tiny_patch4_window7_224 as create_model_swin_style_tiny
from robustbench.model_zoo.architectures.rem_vit import rem_vit_base_patch16_224 as create_model_vit_rem
from robustbench.model_zoo.architectures.style_convnext import style_convnext_base as create_model_convnext_style

mu = (0.485, 0.456, 0.406)
sigma = (0.229, 0.224, 0.225)

vit_mu = (0.5, 0.5, 0.5)
vit_sigma = (0.5, 0.5, 0.5)

linf = OrderedDict(
    [
        ('Wong2020Fast', {  # requires resolution 288 x 288
            'model': lambda: normalize_model(pt_models.resnet50(), mu, sigma),
            'gdrive_id': '1deM2ZNS5tf3S_-eRURJi-IlvUL8WJQ_w',
            'preprocessing': 'Crop288'
        }),
        ('Engstrom2019Robustness', {
            'model': lambda: normalize_model(pt_models.resnet50(), mu, sigma),
            'gdrive_id': '1T2Fvi1eCJTeAOEzrH_4TAIwO8HTOYVyn',
            'preprocessing': 'Res256Crop224',
        }),
        ('Salman2020Do_R50', {
            'model': lambda: normalize_model(pt_models.resnet50(), mu, sigma),
            'gdrive_id': '1TmT5oGa1UvVjM3d-XeSj_XmKqBNRUg8r',
            'preprocessing': 'Res256Crop224'
        }),
        ('Salman2020Do_R18', {
            'model': lambda: normalize_model(pt_models.resnet18(), mu, sigma),
            'gdrive_id': '1OThCOQCOxY6lAgxZxgiK3YuZDD7PPfPx',
            'preprocessing': 'Res256Crop224'
        }),
        ('Salman2020Do_50_2', {
            'model': lambda: normalize_model(pt_models.wide_resnet50_2(), mu, sigma),
            'gdrive_id': '1OT7xaQYljrTr3vGbM37xK9SPoPJvbSKB',
            'preprocessing': 'Res256Crop224'
        }),
        ('Standard_R50', {
            'model': lambda: normalize_model(pt_models.resnet50(pretrained=True), mu, sigma),
            'gdrive_id': '',
            'preprocessing': 'Res256Crop224'
        }),
        ('Style_R50', {
            # 'model': lambda: normalize_model(create_resnet50_with_style(), mu, sigma),
            'model': lambda: create_resnet50_with_style(),
            'gdrive_id': '',
            'preprocessing': 'Res256Crop224'
        }),
        ('VIT_B_224_Style', {
            'model': lambda: create_model_vit_style(),
            'gdrive_id': '',
            'preprocessing': 'Res256Crop224'
        }),
        ('Swin_B_224_Style', {
            'model': lambda: create_model_swin_style(),
            'gdrive_id': '',
            'preprocessing': 'Res256Crop224'
        }),
        ('Swin_T_224_Style', {
            'model': lambda: create_model_swin_style_tiny(),
            'gdrive_id': '',
            'preprocessing': 'Res256Crop224'
        }),
        ('convnext_b_224_Style', {
            'model': lambda: create_model_convnext_style(),
            'gdrive_id': '',
            'preprocessing': 'Res256Crop224'
        }),
    ])

common_corruptions = OrderedDict(
    [
        ('Geirhos2018_SIN', {
            'model': lambda: normalize_model(pt_models.resnet50(), mu, sigma),
            'gdrive_id': '1hLgeY_rQIaOT4R-t_KyOqPNkczfaedgs',
            'preprocessing': 'Res256Crop224'
        }),
        ('Geirhos2018_SIN_IN', {
            'model': lambda: normalize_model(pt_models.resnet50(), mu, sigma),
            'gdrive_id': '139pWopDnNERObZeLsXUysRcLg6N1iZHK',
            'preprocessing': 'Res256Crop224'
        }),
        ('Geirhos2018_SIN_IN_IN', {
            'model': lambda: normalize_model(pt_models.resnet50(), mu, sigma),
            'gdrive_id': '1xOvyuxpOZ8I5CZOi0EGYG_R6tu3ZaJdO',
            'preprocessing': 'Res256Crop224'
        }),
        ('Hendrycks2020Many', {
            'model': lambda: normalize_model(pt_models.resnet50(), mu, sigma),
            'gdrive_id': '1kylueoLtYtxkpVzoOA1B6tqdbRl2xt9X',
            'preprocessing': 'Res256Crop224'
        }),
        ('Hendrycks2020AugMix', {
            'model': lambda: normalize_model(pt_models.resnet50(), mu, sigma),
            'gdrive_id': '1xRMj1GlO93tLoCMm0e5wEvZwqhIjxhoJ',
            'preprocessing': 'Res256Crop224'
        }),
        ('Salman2020Do_50_2_Linf', {
            'model': lambda: normalize_model(pt_models.wide_resnet50_2(), mu, sigma),
            'gdrive_id': '1OT7xaQYljrTr3vGbM37xK9SPoPJvbSKB',
            'preprocessing': 'Res256Crop224'
        }),
        ('Standard_R50', {
            'model': lambda: normalize_model(pt_models.resnet50(pretrained=True), mu, sigma),
            'gdrive_id': '',
            'preprocessing': 'Res256Crop224'
        }),
        ('VIT_B_224_MAE', {
            'model': lambda: create_model_mae("vit_base_patch16_224.augreg_in1k", pretrained=True),
            # 'model': lambda: normalize_model(create_model_mae("vit_base_patch16_224.augreg_in1k", pretrained=True), vit_mu, vit_sigma),
            'gdrive_id': '',
        }),

        ('Style_R50', {
            # 'model': lambda: normalize_model(create_resnet50_with_style(), mu, sigma),
            'model': lambda: create_resnet50_with_style(),
            'gdrive_id': '',
            'preprocessing': 'Res256Crop224'
        }),
        ('VIT_B_224_Style', {
            'model': lambda: create_model_vit_style("vit_base_patch16_224.augreg_in1k", pretrained=True), # vit_base_patch16_224.augreg_in21k_ft_in1k
            'gdrive_id': '',
            'preprocessing': 'Res256Crop224'
        }),
        ('VIT_B_224_REM', {
            'model': lambda: create_model_vit_rem("vit_base_patch16_224.augreg_in1k", pretrained=True),
            'gdrive_id': '',
            'preprocessing': 'Res256Crop224'
        }),
        ('Swin_B_224_Style', {
            'model': lambda: create_model_swin_style("swin_base_patch4_window7_224.ms_in1k", pretrained=True),
            'gdrive_id': '',
            'preprocessing': 'Res256Crop224'
        }),
        ('Swin_T_224_Style', {
            'model': lambda: create_model_swin_style_tiny("swin_tiny_patch4_window7_224.ms_in1k", pretrained=True),
            'gdrive_id': '',
            'preprocessing': 'Res256Crop224'
        }),
        ('convnext_b_224_Style', {
            'model': lambda: create_model_convnext_style("convnext_base.fb_in1k", pretrained=True),
            'gdrive_id': '',
            'preprocessing': 'Res256Crop224'
        }),
    ])

imagenet_models = OrderedDict([(ThreatModel.Linf, linf),
                               (ThreatModel.corruptions, common_corruptions)])


