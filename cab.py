#!/usr/bin/env python
# coding=utf-8
"""
author: yisun
date : 2020-12-10
des :  channel attention block
ref :  channel attention block  CVPR 2017. Squeeze-and-Excitation Networks
       residual channel attention block   ECCV 2018. Image Super-Resolution Using Very Deep Residual Channel Attention Networks

       only (residual) channel attention block, not network
"""
import torch
import torch.nn as nn
from math import sqrt
from torchsummary import summary

## Channel Attention Block
class CABlock(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(CABlock, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, in_channels // ratio, kernel_size=1, padding=0, bias=True),
        self.relu = nn.ReLU(inplace=True),
        self.conv2 = nn.Conv2d(in_channels // ratio, in_channels, kernel_size=1, padding=0, bias=True),
        self.sig = nn.Sigmoid(),

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.sig(y)
        out = x * y.expand_as(x)
        return out

## Residual Channel Attention Block
class RCABlock(nn.Module):
    def __init__(
            self, n_feat, kernel_size, reduction):
        super(RCABlock, self).__init__()
        # conv-relu-conv-ca
        self.conv1 = nn.Conv2d(n_feat, n_feat, kernel_size)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_feat, n_feat, kernel_size)
        self.ca = CABlock(n_feat, reduction)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        res = self.ca(out)
        out = res + x
        return out