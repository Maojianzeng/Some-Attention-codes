#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/10/22 18:59
# @File    : ECA-Net.py
# @Software: PyCharm

'''
《ECANet：Efficient Channel Attention for Deep Convolutional Neural Networks》发表于CVPR 2020，
是对SE-Net中特征变换部分进行了改进。SE-Net的通道信息交互方式是通过全连接实现的，
在降维和升维的过程中会损害一部分的特征表达。ECA-Net则进一步地利用一维卷积来实现通道间的信息交互，
相对于全连接实现的全局通道信息交互所带来的计算开销，ECA-Net提出了一种基于自适应选择卷积核大小的方法，
以实现局部交互，从而显著地降低模型复杂度且保持性能。

'''

import torch.nn as nn


class ECALayer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)