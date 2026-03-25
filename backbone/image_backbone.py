#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   image_backbone.py
@Time    :   2026/2/24 16:56
@Author  :   Binge.van
@Email   :   1367955240@qq.com
@description   :
'''
import os
import torch
from  torch import nn
from torchvision.models import resnet18, resnet34




class ImageBackboneResNetFPN(nn.Module):
    """ResNet + FPN，将 C1~C4 上采样到 32x96 并融合."""

    def __init__(self, backbone: str = "resnet18", out_channels: int = 256, pretrained: bool = True,pretrain_path:str=None,device="cuda"):
        super().__init__()
        if backbone == "resnet18":
            resnet = resnet18(pretrained=False)
            c1_channels, c2_channels, c3_channels, c4_channels = 64, 128, 256, 512
        elif backbone == "resnet34":
            resnet = resnet34(pretrained=False)
            c1_channels, c2_channels, c3_channels, c4_channels = 64, 128, 256, 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
            
        if pretrained and os.path.exists(pretrain_path):
            checkpoint = torch.load(pretrain_path, map_location=device)
            resnet.load_state_dict(checkpoint, strict=True)

        # ResNet stem + stages
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # -> 32x96
        self.layer2 = resnet.layer2  # -> 16x48
        self.layer3 = resnet.layer3  # -> 8x24
        # self.layer4 = resnet.layer4  # -> 4x12

        # FPN 1x1 lateral conv，把通道统一到 out_channels
        self.lateral1 = nn.Conv2d(c1_channels, out_channels, kernel_size=1)
        self.lateral2 = nn.Conv2d(c2_channels, out_channels, kernel_size=1)
        self.lateral3 = nn.Conv2d(c3_channels, out_channels, kernel_size=1)

        # 用转置卷积上采样到 32x96（替代插值）
        # p2: 16x48 -> 32x96 (2x)
        self.upsample_p2 = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels,out_channels,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)

        )
        #  nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
        # p3: 8x24 -> 32x96 (4x)
        self.upsample_p3 = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels,out_channels,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels,out_channels,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 融合后的 3x3 平滑卷积
        self.smooth = nn.Conv2d(out_channels*3, out_channels, kernel_size=3, padding=1)

    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B', 3, 128, 384) -> (B', out_channels, 32, 96)."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c1 = self.layer1(x)  # (B', 64, 32, 96)
        c2 = self.layer2(c1)  # (B', 128, 16, 48)
        c3 = self.layer3(c2)  # (B', 256, 8, 24)
        # c4 = self.layer4(c3)  # (B', 512, 4, 12)

        # 统一通道数
        p1 = self.lateral1(c1)  # (B', out, 32, 96)
        p2 = self.lateral2(c2)  # (B', out, 16, 48)
        p3 = self.lateral3(c3)  # (B', out, 8, 24)

        # 用转置卷积上采样到 32x96
        p2 = self.upsample_p2(p2)  # -> (B', out, 32, 96)
        p3 = self.upsample_p3(p3)  # -> (B', out, 32, 96)
        # p4 = self.upsample_p4(p4)  # -> (B', out, 32, 96)

        # 融合（简单相加）
        fused = torch.cat([p1, p2, p3], dim=1) #p1 + p2 + p3 + p4
        fused = self.smooth(fused)  # (B', out_channels, 32, 96)
        return fused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, S, N, 3, 128, 384)
        返回: (B*S*N, out_channels, 32, 96)，和原 ImageBackBone 输出尺寸一致。
        """
        # B, S, N, C, H, W = x.shape
        # x = x.view(B * S * N, C, H, W)
        feat = self.forward_single(x)
        return feat  # (B*S*N, out_channels, 32, 96)

class ImageBackBone(nn.Module):
    def __init__(self,out_channels=256):
        super().__init__()
        self.backbone =  nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, out_channels, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2)

        )

    def forward(self,x):
        backbone = self.backbone(x)
        return  backbone


if __name__ == '__main__':
    x = torch.randn(1, 6,2,  3, 128, 384)
    model = ImageBackboneResNetFPN()
    out = model(x)
    print(out.shape)