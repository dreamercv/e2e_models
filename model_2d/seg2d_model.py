#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   seg2d_model.py
@Time    :   2026/2/24 17:28
@Author  :   Binge.van
@Email   :   1367955240@qq.com
@description   :
'''
import os
import torch
from  torch import  nn
# from loss_seg import SegLoss

class ModelSeg2D(nn.Module):
    def __init__(self,out_channels,seg_class_num=3,pos_weight=20.):
        super().__init__()
        self.up_sample = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, 2, 2),
            nn.Conv2d(out_channels,out_channels,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels, 2, 2),
            nn.Conv2d(out_channels,out_channels,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
        )
        self.seg_2d = nn.Sequential(
            nn.Conv2d(out_channels,seg_class_num,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(seg_class_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(seg_class_num,seg_class_num,kernel_size=3, stride=1, padding=1)
        )
                # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # 最后一层（输出层）的bias初始化为较低的值
        # 使得sigmoid后约为0.05，有助于减少背景误检
        nn.init.constant_(self.seg_2d[-1].bias, -2.94)  # sigmoid(-2.94) ≈ 0.05


    def forward(self,x):
        seg_out = self.seg_2d(self.up_sample(x))
        return seg_out


if __name__ == '__main__':
    x = torch.zeros((2,256,32,96))
    model = ModelSeg2D(256,3)
    out = model(x)
    print(out.shape)
