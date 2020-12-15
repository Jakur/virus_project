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


class ResNet(nn.Module):
    def __init__(self, bs):
        super(ResNet, self).__init__()
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

