#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   build_model.py
@Time    :   2026/3/9 21:48
@Author  :   Binge.van
@Email   :   1367955240@qq.com
@description   :
'''
import os
import numpy as np
import torch
from torch import nn
from sparse4d.instance_bank import InstanceBank
from sparse4d.detection3d_blocks import (
    SparseBox3DEncoder,
    SparseBox3DRefinementModule,
    SparseBox3DKeyPointsGenerator
)
from sparse4d.bev_aggregation import BEVFeatureAggregation
from sparse4d.head import Sparse4DHead, MHAWrapper, SimpleFFN
from sparse4d.dn_sampler import DenoisingSampler

from sparse4d.decoder import SparseBox3DDecoder
from sparse4d.losses import FocalLoss, SparseBox3DLoss


def build_det3D_head(
        num_anchor: int = 900,
        embed_dims: int = 256,
        num_decoder: int = 6,
        num_single_frame_decoder: int = 5,
        num_classes: int = 10,
        bev_bounds=([-80.0, 120.0, 1.0], [-40.0, 40.0, 1.0]),
        anchor_init: np.ndarray = None,
        decouple_attn: bool = True,
        num_heads: int = 8,
        dropout: float = 0.1,
        feedforward_dims: int = 1024,
        use_dn: bool = True,
        num_dn_groups: int = 10,
        dn_noise_scale: float = 0.5,
        max_dn_gt: int = 32,
        add_neg_dn: bool = True,
        reg_weights: list = None,
        use_decoder: bool = False,
        decoder_num_output: int = 300,
        decoder_score_threshold: float = None,
):
    """从 model.py 拷贝的 Sparse4D BEV head 构建函数，用于测试。"""
    if anchor_init is None:
        anchor_init = np.zeros((num_anchor, 11), dtype=np.float32)
    if reg_weights is None:
        reg_weights = [2.0] * 3 + [0.5] * 3 + [0.0] * 5

    instance_bank = InstanceBank(
        num_anchor=num_anchor,
        embed_dims=embed_dims,
        anchor=anchor_init,
        anchor_handler=None,
        num_temp_instances=0,
        default_time_interval=0.5,
        confidence_decay=0.6,
        feat_grad=False,
    )
    anchor_encoder = SparseBox3DEncoder(
        embed_dims=[128, 32, 32, 64] if decouple_attn else 256,
        vel_dims=3,
        mode="cat" if decouple_attn else "add",
        output_fc=not decouple_attn,
        in_loops=1,
        out_loops=4 if decouple_attn else 2,
        out_dim = embed_dims,
    )
    gnn_dim = embed_dims * 2 if decouple_attn else embed_dims
    graph_model = MHAWrapper(gnn_dim, num_heads, dropout=dropout, batch_first=True)
    norm_layer = nn.LayerNorm(embed_dims)
    ffn = SimpleFFN(embed_dims, feedforward_dims, dropout=dropout)
    kps = SparseBox3DKeyPointsGenerator(embed_dims=embed_dims, num_learnable_pts=0, fix_scale=((0.0, 0.0, 0.0),))
    bev_agg = BEVFeatureAggregation(
        embed_dims=embed_dims,
        bev_bounds=bev_bounds,
        kps_generator=kps,
        proj_drop=dropout,
        residual_mode="add",
    )
    refine_layer = SparseBox3DRefinementModule(
        embed_dims=embed_dims,
        output_dim=11,
        num_cls=num_classes,
        refine_yaw=True,
        with_cls_branch=True,
        with_quality_estimation=True,
    )

    operation_order = (
            ["gnn", "norm", "deformable", "ffn", "norm", "refine"] * num_single_frame_decoder
            + ["gnn", "norm", "deformable", "ffn", "norm", "refine"] * (num_decoder - num_single_frame_decoder)
    )
    operation_order = operation_order[2:]

    denoise = None
    loss_cls = None
    loss_reg = None
    if use_dn:
        denoise = DenoisingSampler(
            num_dn_groups=num_dn_groups,
            dn_noise_scale=dn_noise_scale,
            max_dn_gt=max_dn_gt,
            add_neg_dn=add_neg_dn,
            reg_weights=reg_weights,
        )
    sampler = DenoisingSampler(
            num_dn_groups=num_dn_groups,
            dn_noise_scale=dn_noise_scale,
            max_dn_gt=max_dn_gt,
            add_neg_dn=add_neg_dn,
            reg_weights=reg_weights,
        )
    loss_cls = FocalLoss(alpha=0.25, gamma=2.0, ignore_index=-1)
    loss_reg = SparseBox3DLoss(reg_weights=reg_weights, loss_centerness=True, loss_yawness=True)

    decoder = None
    if use_decoder:
        decoder = SparseBox3DDecoder(
            num_output=decoder_num_output,
            score_threshold=decoder_score_threshold,
            sorted=True,
        )

    head = Sparse4DHead(
        instance_bank=instance_bank,
        anchor_encoder=anchor_encoder,
        graph_model=graph_model,
        norm_layer=norm_layer,
        ffn=ffn,
        deformable_model=bev_agg,
        refine_layer=refine_layer,
        operation_order=operation_order,
        num_single_frame_decoder=num_single_frame_decoder,
        decouple_attn=decouple_attn,
        denoise=denoise,
        sampler=sampler,
        decoder=decoder,
        loss_cls=loss_cls,
        loss_reg=loss_reg,
        reg_weights=reg_weights,
        # 与 dataset 真值命名保持一致
        gt_cls_key="gt_labels_det3D",
        gt_reg_key="gt_bboxes_det3D",
        gt_cls_key_mask="gt_labels_det3D_mask",
        gt_reg_key_mask="gt_bboxes_det3D_mask",
        cls_threshold_to_reg=0.05,
    )
    head.init_weights()
    return head


if __name__ == '__main__':
    pass