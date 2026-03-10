# -*- encoding: utf-8 -*-
'''
@File         :model_test.py
@Date         :2026/03/10 10:29:08
@Author       :Binge.Van
@E-mail       :afb5szh@bosch.com
@Version      :V1.0.0
@Description  :

'''


import os,sys

import numpy as np
import torch
from torch import nn

from model.models import Model
from config.config import configs



def build_dummy_metas(
    batch_size: int,
    num_temporal_frames: int,
    num_anchor: int,
    num_det_classes: int,
    num_map_classes: int,
    num_pts_per_gt_vec: int = 20,
) -> dict:
    """
    构造假的 GT，测试检测 + 跟踪 + MapTR 的 loss 和匈牙利匹配是否能正常跑。
    """
    B, T, N = batch_size, num_temporal_frames, num_anchor

    # 检测 GT（3D boxes）
    gt_bboxes_3d = []
    gt_labels_3d = []
    for b in range(B):
        n_det = 5
        gt_bboxes_3d.append(torch.randn(n_det, 11,device="cuda"))  # 11-dim box 编码
        gt_labels_3d.append(
            torch.randint(low=0, high=num_det_classes, size=(n_det,), dtype=torch.long,device="cuda")
        )

    # MapTR GT（2D 线 + 多点）
    gt_map_bboxes = []
    gt_map_labels = []
    gt_map_pts = []
    for b in range(B):
        n_map = 4
        # xyxy in pc_range [-15,15] x [-30,30]
        x1y1 = torch.rand(n_map, 2)
        x2y2 = x1y1 + 0.5 * torch.rand(n_map, 2)
        bboxes = torch.cat([x1y1, x2y2], dim=-1)
        gt_map_bboxes.append(bboxes.cuda())
        gt_map_labels.append(
            torch.randint(low=0, high=num_map_classes, size=(n_map,), dtype=torch.long,device="cuda")
        )
        pts = torch.rand(n_map, num_pts_per_gt_vec, 2).cuda()
        gt_map_pts.append(pts)

    # 跟踪 GT：gt_track_match (B, T, N)
    gt_track_match = torch.full((B, T, N), -1, dtype=torch.long)
    for b in range(B):
        # 第一帧全部 ignore
        # 后续帧：简单地 1:1 匹配到上一帧同 index
        for t in range(1, T):
            gt_track_match[b, t] = torch.arange(N, dtype=torch.long)

    metas = {
        "gt_bboxes_3d": gt_bboxes_3d,
        "gt_labels_3d": gt_labels_3d,
        "gt_map_bboxes": gt_map_bboxes,
        "gt_map_labels": gt_map_labels,
        "gt_map_pts": gt_map_pts,
        "gt_track_match": gt_track_match,
    }
    return metas

def build_dummy_inputs(
    num_mode: int = 2,
    batch_size: int = 2,
    num_temporal_frames: int = 5,
    num_cams: int = 8,
):
    """构造一组假的相机输入和几何变换，用于快速前向测试。"""
    B, T, Cams,M = batch_size, num_temporal_frames, num_cams,num_mode
    H_img, W_img = 128, 384

    x = torch.randn(B, M, T, Cams, 3, H_img, W_img).cuda()

    # 几何相关全部用单位阵 / 零向量占位
    rots = torch.eye(3).view(1,1, 1, 1, 3, 3).expand(B, M, T, Cams, 3, 3).cuda()
    trans = torch.zeros(B, M, T, Cams, 1, 3).cuda()
    intrins = torch.eye(3).view(1, 1, 1, 1, 3, 3).expand(B, M, T, Cams, 3, 3).cuda()
    distorts = torch.zeros(B, M, T, Cams, 1, 8).cuda()
    post_rot3 = torch.eye(3).view(1, 1, 1, 1, 3, 3).expand(B, M, T, Cams, 3, 3).cuda()
    post_tran3 = torch.zeros(B, M, T, Cams, 1, 3).cuda()
    theta_mats = torch.zeros(B, M, T, 2, 3).cuda()
    T_ego_his2curs = torch.eye(4).view(1, 1, 1, 4, 4).expand(B, M, T, 4, 4).cuda()

    return (
        x,
        rots,
        trans,
        intrins,
        distorts,
        post_rot3,
        post_tran3,
        theta_mats,
        T_ego_his2curs,
    )



if __name__ == '__main__':
    model = Model(configs).cuda()

    input = build_dummy_inputs(2,configs["batch_size"],configs["seq_len"],configs["num_cams"])
    x, rots, trans, intrins, distorts, post_rot, pos_tran, theta_mats, T_ego_his2curs = input

    num_anchor = 900
    num_det_classes = 10
    num_map_classes = 3
    metas = build_dummy_metas(
        batch_size=configs["batch_size"],
        num_temporal_frames=configs["seq_len"],
        num_anchor=num_anchor,
        num_det_classes=num_det_classes,
        num_map_classes=num_map_classes,
        num_pts_per_gt_vec=20,
    )

    out_put = model(x, rots, trans, intrins, distorts, post_rot, pos_tran, theta_mats, T_ego_his2curs,metas)
    