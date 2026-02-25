#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   det2d_model.py
@Time    :   2026/2/24 17:26
@Author  :   Binge.van
@Email   :   1367955240@qq.com
@description   :
'''
import os
from torch import nn
import torch

class ModelDet2D(nn.Module):
    """CenterNet检测头"""
    
    def __init__(self, out_channels=512, det_class_num=80):
        super().__init__()
        
        # Heatmap分支
        self.heatmap = nn.Sequential(
            nn.Conv2d(out_channels, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, det_class_num, 1)
        )
        
        # Offset分支
        self.offset = nn.Sequential(
            nn.Conv2d(out_channels, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 1)
        )
        
        # Size分支
        self.size = nn.Sequential(
            nn.Conv2d(out_channels, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 1)
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
        
        # Heatmap最后一层初始化为更低的值，使得sigmoid后约为0.05
        # 更低的初始值有助于减少背景误检
        nn.init.constant_(self.heatmap[-1].bias, -2.94)  # sigmoid(-2.94) ≈ 0.05
    
    def forward(self, x):
        heatmap = self.heatmap(x)
        offset = self.offset(x)
        size = self.size(x)
        
        # Heatmap应用sigmoid
        heatmap = torch.sigmoid(heatmap)
        
        return heatmap, offset, size


if __name__ == '__main__':
    pass
