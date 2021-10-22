#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/10/22 18:56
# @File    : SPA-Net.py
# @Software: PyCharm

'''
《Spatial Pyramid Attention Network for Enhanced Image Recognition》 发表于ICME 2020 并获得了最佳学生论文。
考虑到 SE-Net 这种利用 GAP 去建模全局上下文的方式会导致空间信息的损失，SPA-Net另辟蹊径，
利用多个自适应平均池化(Adaptive Averatge Pooling, APP) 组成的空间金字塔结构来建模局部和全局的上下文语义信息，
使得空间语义信息被更加充分的利用到。


'''

import torch
import torch.nn as nn


class CPSPPSELayer(nn.Module):
    def __init__(self,in_channel, channel, reduction=16):
        super(CPSPPSELayer, self).__init__()
        if in_channel != channel:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channel, channel, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True)
            )
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(2)
        self.avg_pool4 = nn.AdaptiveAvgPool2d(4)
        self.fc = nn.Sequential(
            nn.Linear(channel*21, channel*21 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel*21 // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x) if hasattr(self, 'conv1') else x
        b, c, _, _ = x.size()
        y1 = self.avg_pool1(x).view(b, c)  # like resize() in numpy
        y2 = self.avg_pool2(x).view(b, 4 * c)
        y3 = self.avg_pool4(x).view(b, 16 * c)
        y = torch.cat((y1, y2, y3), 1)
        y = self.fc(y)
        b, out_channel = y.size()
        y = y.view(b, out_channel, 1, 1)
        return y