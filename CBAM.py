#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/10/22 19:00
# @File    : CBAM.py
# @Software: PyCharm


'''
《CBAM: Convolutional Block Attention Module》发表于CVPR 2018，在原有通道注意力的基础上，
衔接了一个空间注意力模块(Spatial Attention Modul, SAM)。SAM是基于通道进行全局平均池化以及全局最大池化操作，
产生两个代表不同信息的特征图，合并后再通过一个感受野较大的7×7卷积进行特征融合，
最后再通过Sigmoid操作来生成权重图叠加回原始的输入特征图，从而使得目标区域得以增强。
总的来说，对于空间注意力来说，由于将每个通道中的特征都做同等处理，忽略了通道间的信息交互；
而通道注意力则是将一个通道内的信息直接进行全局处理，容易忽略空间内的信息交互。

作者最终通过实验验证先通道后空间的方式比先空间后通道或者通道空间并行的方式效果更佳。
'''

# https://github.com/Jongchan/attention-module/blob/master/MODELS/cbam.py

import torch
import torch.nn as nn


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out