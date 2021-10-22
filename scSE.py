#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/10/22 19:02
# @File    : scSE.py
# @Software: PyCharm


'''
《Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks》发表于MICCAI 2018，
是一种更轻量化的SE-Net变体，在SE的基础上提出cSE、sSE、scSE这三个变种。
cSE和sSE分别是根据通道和空间的重要性来校准采样。scSE则是同时进行两种不同采样校准，得到一个更优异的结果。
'''


import torch.nn as nn

class SCSEModule(nn.Module):
    def __init__(self, ch, re=16):
        super().__init__()
        self.cSE = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(ch,ch//re,1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(ch//re,ch,1),
                                 nn.Sigmoid())
        self.sSE = nn.Sequential(nn.Conv2d(ch,ch,1),
                                 nn.Sigmoid())
    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)