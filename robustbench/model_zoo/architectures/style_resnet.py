import torch.nn as nn
import torch
from torchvision.models.resnet import ResNet
from .wide_resnet import get_style, style_inject
from torchvision import models as pt_models
from robustbench.model_zoo.architectures.utils_architectures import ImageNormalizer

class StyleResNet(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # mu = (0.485, 0.456, 0.406)
        # sigma = (0.229, 0.224, 0.225)
        self.ctta_img_norm = None
    
    def forward(self, x, return_style=False):
        x = self.ctta_img_norm(x)
        style_list = []
        
        # Conv1
        x = self.conv1(x)
        if return_style:
            style_list.append(get_style(x.detach()))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if return_style:
            style_list.append(get_style(x.detach()))
        
        # Layer1
        for i, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            x = layer(x)
            if return_style:
                style_list.append(get_style(x.detach()))
            
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        if return_style:
            return x, style_list
        return x
    
    def forward_features(self, x, return_style=False):
        x = self.ctta_img_norm(x)
        style_list = []
        
        # Conv1
        x = self.conv1(x)
        if return_style:
            style_list.append(get_style(x.detach()))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if return_style:
            style_list.append(get_style(x.detach()))
        
        # Layer1
        for i, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            x = layer(x)
            if return_style:
                style_list.append(get_style(x.detach()))
            
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        if return_style:
            return x, style_list
        return x
        
    
    def forward_features_wst(self, x, styles, inject_pos=[False, False, False, False, False, False]):
        x = self.ctta_img_norm(x)
        x = self.conv1(x)
        if inject_pos[0]:
            x = style_inject(x, styles[0])
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if inject_pos[1]:
            x = style_inject(x, styles[1])
        
        x = self.layer1(x)
        if inject_pos[2]:
            x = style_inject(x, styles[2])
            
        x = self.layer2(x)
        if inject_pos[3]:
            x = style_inject(x, styles[3])
            
        x = self.layer3(x)
        if inject_pos[4]:
            x = style_inject(x, styles[4])
            
        x = self.layer4(x)
        if inject_pos[5]:
            x = style_inject(x, styles[5])
            
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
    def forward_fc(self, x):
        return self.fc(x)
    
def create_resnet50_with_style():
    model = StyleResNet(
        block=pt_models.resnet.Bottleneck,
        layers=[3, 4, 6, 3]  # ResNet50的标准层配置
    )
    # 加载预训练权重
    state_dict = pt_models.resnet50(pretrained=True).state_dict()
    model.load_state_dict(state_dict, strict=True)
    model.ctta_img_norm = ImageNormalizer((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    return model