import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.pyt_utils import load_model

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d, last_relu=True):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.last_relu = last_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.last_relu:
            out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, multi_grid=1, norm_layer=nn.BatchNorm2d, last_relu=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation*multi_grid, dilation=dilation*multi_grid, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride
        self.last_relu = last_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual      
        if self.last_relu:
            out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, norm_layer=nn.BatchNorm2d, dilated=True, multi_grid=False, os=8, relu_l3=True, relu_l4=True, **kwargs):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.deep_channels = 2048
        self.dsn_channels = 1024
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        grid = (1, 2, 4) if multi_grid else (1, 1, 1)
        if dilated:
            if os == 8:
                self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2, norm_layer=norm_layer, last_relu=relu_l3)
                self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, multi_grid=grid, norm_layer=norm_layer, last_relu=relu_l4)
            else:
                self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer, last_relu=relu_l3)
                self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2, multi_grid=grid, norm_layer=norm_layer, last_relu=relu_l4)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer, last_relu=relu_l3)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer, last_relu=relu_l4)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1, norm_layer=nn.BatchNorm2d, last_relu=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index%len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample, multi_grid=generate_multi_grid(0, multi_grid), norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            use_relu = True if i != blocks - 1 else last_relu
            layers.append(block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid), norm_layer=norm_layer, last_relu=use_relu))

        return nn.Sequential(*layers)

    def base_forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        # return [x1, x2, x3, x4]
        return x4

    def forward_base_in(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        return x

class ResNetv2(nn.Module):
    def __init__(self, block, layers, norm_layer=nn.BatchNorm2d, dilated=True, multi_grid=False, os=8, relu_l3=True, relu_l4=True, **kwargs):
        self.inplanes = 128
        super(ResNetv2, self).__init__()
        self.deep_channels = 2048
        self.dsn_channels = 1024
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = norm_layer(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = norm_layer(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        grid = (1, 2, 4) if multi_grid else (1, 1, 1)
        if dilated:
            if os == 8:
                self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2, norm_layer=norm_layer, last_relu=relu_l3)
                self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, multi_grid=grid, norm_layer=norm_layer, last_relu=relu_l4)
            else:
                self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer, last_relu=relu_l3)
                self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2, multi_grid=grid, norm_layer=norm_layer, last_relu=relu_l4)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer, last_relu=relu_l3)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer, last_relu=relu_l4)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1, norm_layer=nn.BatchNorm2d, last_relu=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index%len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample, multi_grid=generate_multi_grid(0, multi_grid), norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            use_relu = True if i != blocks - 1 else last_relu
            layers.append(block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid), norm_layer=norm_layer, last_relu=use_relu))

        return nn.Sequential(*layers)

    def base_forward(self, x, return_list=False):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        if return_list:
            x2 = self.layer2(x)
            x3 = self.layer3(x2)
            x4 = self.layer4(x3)
            return [x4, x3, x2, x]
        else:
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            return x

    def forward_base_in(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        return x
