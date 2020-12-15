import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Callable

EMBEDDING_VEC_SIZE = 8
DNA_BASES = 64


def conv(
    in_planes: int,
    out_planes: int,
    kernel_size,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
) -> nn.Conv1d:
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv1d:
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv1d:
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


class Miner(nn.Module):
    def __init__(self, bs):
        super(Miner, self).__init__()
        self.freq = FreqModel(bs)
        self.pat = PatternModel(bs)
        self.fc = nn.Linear(2200, 1)

    def forward(self, x: Tensor) -> Tensor:
        a = self.freq(x)
        b = self.pat(x)
        res = torch.cat((a, b), dim=1)
        x = self.fc(res)
        return x


class FreqModel(nn.Module):
    def __init__(self, bs):
        super(FreqModel, self).__init__()
        self.bs = bs
        self.relu = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.embed = nn.Embedding(DNA_BASES, EMBEDDING_VEC_SIZE)
        self.conv = conv(5, 1000, 8, stride=1)
        self.pool = nn.AvgPool1d(293)
        self.drop1 = nn.Dropout(0.1)
        self.fc = nn.Linear(1000, 1000)
        self.drop2 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(1000, 1)  # Change later

    def forward(self, x: Tensor) -> Tensor:
        x = F.one_hot(x, num_classes=5)
        x = x.float()
        # x = self.embed(x)
        # print(x.shape)
        # x = x.view(self.bs, EMBEDDING_VEC_SIZE, -1)
        x = x.view(self.bs, 5, -1)
        x = self.conv(x)
        # print(x.shape)
        x = self.relu(x)
        # print(x.shape)
        x = self.pool(x)
        # print(x.shape)
        x = self.drop1(x)
        x = x.view(self.bs, 1000)
        # print(x.shape)
        x = self.fc(x)
        # print(x.shape)
        x = self.relu2(x)
        # print(x.shape)
        x = self.drop2(x)
        # print(x.shape)
        x = self.sigmoid(x)
        return x


class PatternModel(nn.Module):
    def __init__(self, bs):
        super(PatternModel, self).__init__()
        self.bs = bs
        self.relu = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.embed = nn.Embedding(DNA_BASES, EMBEDDING_VEC_SIZE)
        self.conv = conv(5, 1200, 11, stride=1)
        self.pool = nn.MaxPool1d(290)
        self.drop1 = nn.Dropout(0.1)
        self.fc = nn.Linear(1200, 1200)
        self.drop2 = nn.Dropout(0.1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.one_hot(x, num_classes=5)
        x = x.float()
        # x = self.embed(x)
        # print(x.shape)
        x = x.view(self.bs, 5, -1)
        x = self.conv(x)
        # print(x.shape)
        x = self.relu(x)
        # print(x.shape)
        x = self.pool(x)
        # print(x.shape)
        x = self.drop1(x)
        x = x.view(self.bs, 1200)
        # print(x.shape)
        x = self.fc(x)
        # print(x.shape)
        x = self.relu2(x)
        # print(x.shape)
        x = self.drop2(x)
        # print(x.shape)
        x = self.sigmoid(x)
        return x


class CNNTest(nn.Module):
    def __init__(self, bs):
        super(CNNTest, self).__init__()
        self.embed = nn.Embedding(DNA_BASES, EMBEDDING_VEC_SIZE)
        self.first = nn.Conv1d(EMBEDDING_VEC_SIZE, 64, 3)
        self.bb = BasicBlock(64, 64)
        self.pool = nn.MaxPool1d(5)
        self.drop = nn.Dropout(0.1)
        self.fc = nn.Linear(1216, 1)
        self.bs = bs

    def forward(self, x: Tensor) -> Tensor:
        x = self.embed(x)
        x = x.view(self.bs, EMBEDDING_VEC_SIZE, -1)
        x = self.first(x)
        x = self.bb(x)
        x = self.pool(x)
        x = x.view(self.bs, -1)
        x = self.drop(x)
        x = self.fc(x)
        return x


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self.conv1 = conv3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


# class Bottleneck(nn.Module):
#     # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
#     # while original implementation places the stride at the first 1x1 convolution(self.conv1)
#     # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
#     # This variant is also known as ResNet V1.5 and improves accuracy according to
#     # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

#     expansion: int = 4

#     def __init__(
#         self,
#         inplanes: int,
#         planes: int,
#         stride: int = 1,
#         downsample: Optional[nn.Module] = None,
#         groups: int = 1,
#         base_width: int = 64,
#         dilation: int = 1,
#         norm_layer: Optional[Callable[..., nn.Module]] = None,
#     ) -> None:
#         super(Bottleneck, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm1d
#         width = int(planes * (base_width / 64.0)) * groups
#         # Both self.conv2 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv1(inplanes, width)
#         self.bn1 = norm_layer(width)
#         self.conv2 = conv3(width, width, stride, groups, dilation)
#         self.bn2 = norm_layer(width)
#         self.conv3 = conv1(width, planes * self.expansion)
#         self.bn3 = norm_layer(planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x: Tensor) -> Tensor:
#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out += identity
#         out = self.relu(out)

#         return out

