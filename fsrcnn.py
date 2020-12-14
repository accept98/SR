#!/usr/bin/env python
# coding=utf-8
"""
author: yisun
date : 2020-12-14 
des : FSRCNN
ref: ECCV 2016. Accelerating the Super-Resolution Convolutional Neural Network.
    https://github.com/yjn870/FSRCNN-pytorch
"""
import torch
import torch.nn as nn
from torchsummary import summary
from math import sqrt

def conv_prelu(in_channels,out_channels,kernel_size):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                  kernel_size=kernel_size, stride=1,
                  padding=kernel_size//2, bias=True),
        nn.PReLU(out_channels)
    )

class Net(nn.Module):
    def __init__(self, scale_factor, n_feats=1, d=56, s=12, m=4):
        super(Net, self).__init__()

        # head
        self.head = conv_prelu(n_feats, d, 5)

        body_layers = []

        # shrinking
        body_layers.append(conv_prelu(d, s, 1))

        # non-linear mapping
        for _ in range(m):
            body_layers.append(conv_prelu(s, s, 3))

        # expanding
        body_layers.append(conv_prelu(s, d, 1))

        # body
        self.body = nn.Sequential(*body_layers)
        # tail = deconv
        self.tail = nn.ConvTranspose2d(
            d, n_feats, kernel_size=9, stride=scale_factor, padding=9//2, output_padding=scale_factor-1
        )

        self._initialize_weights()

    def forward(self, x):
        y = self.head(x)
        y = self.body(y)
        y = self.tail(y)
        return y


    def _initialize_weights(self):
        for m in self.head:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        for m in self.body:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        nn.init.normal_(self.tail.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.tail.bias.data)

if __name__ == '__main__':
    fscrnn = Net(3)
    print(fscrnn)
    summary(fscrnn, input_size=(1, 33, 33), batch_size=-1)


