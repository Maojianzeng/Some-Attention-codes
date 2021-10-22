#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/10/22 19:03
# @Author  : MJZ
# @File    : A2-Nets.py
# @Software: PyCharm

'''
《A2-Nets: Double Attention Networks》发表于NIPS 2018，提出了一种双重注意力网络。
该网络首先使用二阶的注意力池化(Second-order Attention Pooling, SAP) 用于将整幅图的所有关键特征归纳到一个集合当中，
然后再利用另一种注意力机制将这些特征分别应用到图像中的每个区域。
'''


import torch.nn as nn

class DoubleAtten(nn.Module):
    """
    A2-Nets: Double Attention Networks. NIPS 2018
    """
    def __init__(self,in_c):
        super(DoubleAtten,self).__init__()
        self.in_c = in_c
        """Convolve the same input feature map to produce three feature maps with the same scale, i.e., A, B, V (as shown in paper).
        """
        self.convA = nn.Conv2d(in_c,in_c,kernel_size=1)
        self.convB = nn.Conv2d(in_c,in_c,kernel_size=1)
        self.convV = nn.Conv2d(in_c,in_c,kernel_size=1)

    def forward(self,input):

        feature_maps = self.convA(input)
        atten_map = self.convB(input)
        b, _, h, w = feature_maps.shape

        feature_maps = feature_maps.view(b, 1, self.in_c, h*w) # reshape A
        atten_map = atten_map.view(b, self.in_c, 1, h*w)       # reshape B to generate attention map
        global_descriptors = torch.mean((feature_maps * F.softmax(atten_map, dim=-1)),dim=-1) # Multiply the feature map and the attention weight map to generate a global feature descriptor

        v = self.convV(input)
        atten_vectors = F.softmax(v.view(b, self.in_c, h*w), dim=-1) # 生成 attention_vectors
        out = torch.bmm(atten_vectors.permute(0,2,1), global_descriptors).permute(0,2,1)

        return out.view(b, _, h, w)