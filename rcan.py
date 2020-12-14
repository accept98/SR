#!/usr/bin/env python
# coding=utf-8
"""
author: yisun
date : 2020-12-10
des : RCAN
ref: ECCV 2018. Image Super-Resolution Using Very Deep Residual Channel Attention Networks
     https://github.com/yulunzhang/RCAN
     https://github.com/thstkdgus35/EDSR-PyTorch
"""
import torch
import torch.nn as nn
from math import sqrt
from torchsummary import summary
import math

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # feature squeeze
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature excitate
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        # feature scale
        out = x * y
        return out

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RCAB, self).__init__()
        # conv-relu-conv-ca
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        out = res + x
        return out

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        # RCAB(20)-conv
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction,
                bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)
        ]
        modules_body.append(conv(n_feat, n_feat, kernel_size)) # RG末尾的conv
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        out = res + x
        return out

## default conv
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

## upsample layer
class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

## Residual Channel Attention Network (RCAN)
class RCAN(nn.Module):
    def __init__(self, args, conv=default_conv):
        super(RCAN, self).__init__()

        n_resgroups = args['n_resgroups']
        n_resblocks = args['n_resblocks']
        n_feats = args['n_feats']
        kernel_size = 3
        reduction = args['reduction']
        scale = args['scale']
        act = nn.ReLU(True)

        # head
        modules_head = [conv(args['n_colors'], n_feats, kernel_size)]

        # body
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=args['res_scale'], n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        # tail
        modules_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args['n_colors'], kernel_size)]


        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
        	    torch.nn.init.xavier_uniform_(m.weight, gain=1)
        	    if m.bias is not None:
        		    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
        	    torch.nn.init.xavier_uniform_(m.weight, gain=1)
        	    if m.bias is not None:
        		    m.bias.data.zero_()

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        y = res + x
        x = self.tail(y)
        return x


if __name__ == '__main__':
    cfg = {
        'n_resgroups' :10,
        'n_resblocks' :20,
        'n_feats' :64, # conv out channel
        'reduction' :16,
        'scale' :3,
        'n_colors':3,  # conv in channel
        'res_scale':1 # reference edsr, in order to reduce channel
    }
    rcan = RCAN(args=cfg)
    summary(rcan, input_size=(3,41,41), batch_size=-1)
    # print(rcan)
    pass






