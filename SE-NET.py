#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/10/22 18:53
# @File    : SE-NET.py
# @Software: PyCharm

'''
《Squeeze-and-Excitation Networks》发表于CVPR 2018，是CV领域将注意力机制应用到通道维度的代表作，
后续大量基于通道域的工作均是基于此进行润(魔)色(改)。
SE-Net是ImageNet 2017大规模图像分类任务的冠军，结构简单且效果显著，
可以通过特征重标定的方式来自适应地调整通道之间的特征响应。

Squeeze 利用全局平均池化(Global Average Pooling, GAP) 操作来提取全局感受野，将所有特征通道都抽象为一个点；
Excitation 利用两层的多层感知机(Multi-Layer Perceptron, MLP) 网络来进行非线性的特征变换，显示地构建特征图之间的相关性；
Transform 利用Sigmoid激活函数实现特征重标定，强化重要特征图，弱化非重要特征图。
'''

import torch
import torch.nn as nn

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)