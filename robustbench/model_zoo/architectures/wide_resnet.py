import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def style_inject(feat, style_pair, style_index=None, eps=1e-5):
    style_var, style_mu = style_pair
    style_var = style_var.detach()
    style_mu = style_mu.detach()

    var = torch.var(feat.flatten(2), dim=2, unbiased=False, keepdim=True)
    mu = torch.mean(feat.flatten(2), dim=2, keepdim=True)

    var = var.unsqueeze(2)
    mu = mu.unsqueeze(2)

    if style_index is None:
        if feat.shape[0] > style_var.shape[0]:
            random_index1 = torch.randperm(feat.shape[0]) % style_var.shape[0]
            random_index2 = torch.randperm(feat.shape[0]) % style_mu.shape[0]
        else:
            random_index1 = torch.randperm(style_var.shape[0])[:feat.shape[0]]
            random_index2 = torch.randperm(style_mu.shape[0])[:feat.shape[0]]
        style_var = style_var[random_index1, :]
        style_mu = style_mu[random_index2, :]

    else:   
        style_var = style_var[style_index, :]
        style_mu = style_mu[style_index, :]

    stylized_feat = ((feat - mu) / (var + eps).sqrt()) * (style_var + eps).sqrt() + style_mu

    return stylized_feat

def get_style(feat):
    var = torch.var(feat.flatten(2), dim=2, unbiased=False, keepdim=True)
    mu = torch.mean(feat.flatten(2), dim=2, keepdim=True)
    var = var.unsqueeze(2)
    mu = mu.unsqueeze(2)

    return var, mu

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    """ Based on code from https://github.com/yaodongyu/TRADES """
    def __init__(self, depth=28, num_classes=10, widen_factor=10, sub_block1=False, dropRate=0.0, bias_last=True):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        if sub_block1:
            # 1st sub-block
            self.sub_block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes, bias=bias_last)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear) and not m.bias is None:
                m.bias.data.zero_()

    def forward(self, x, return_style=False):
        style_list = []
        out = self.conv1(x)
        if return_style:
            style_list.append(get_style(out.detach()))
        out = self.block1(out)
        if return_style:
            style_list.append(get_style(out.detach()))
        out = self.block2(out)
        if return_style:
            style_list.append(get_style(out.detach()))
        out = self.block3(out)
        if return_style:
            style_list.append(get_style(out.detach()))
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        if return_style:
            return self.fc(out), style_list
        return self.fc(out)

    def forward_features_wst(self, x, styles, inject_pos=[False, False, False, False]):
        if styles is None:
            raise NotImplemented
        
        out = self.conv1(x)
        if inject_pos[0]:
            out = style_inject(out, styles[0])
        out = self.block1(out)
        if inject_pos[1]:
            out = style_inject(out, styles[1])
        out = self.block2(out)
        if inject_pos[2]:
            out = style_inject(out, styles[2])
        out = self.block3(out)
        if inject_pos[3]:
            out = style_inject(out, styles[3])
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return out
    
    def forward_features(self, x, return_style=False):
        style_list = []
        out = self.conv1(x)
        if return_style:
            style_list.append(get_style(out.detach()))
        out = self.block1(out)
        if return_style:
            style_list.append(get_style(out.detach()))
        out = self.block2(out)
        if return_style:
            style_list.append(get_style(out.detach()))
        out = self.block3(out)
        if return_style:
            style_list.append(get_style(out.detach()))
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        if return_style:
            return out, style_list
        return out

    def forward_fc(self, x):
        return self.fc(x)