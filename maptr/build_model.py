#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   build_model.py
@Time    :   2026/3/9 22:32
@Author  :   Binge.van
@Email   :   1367955240@qq.com
@description   :
'''
import os
from maptr.map_head import MapHead


def build_map_head(
        bev_feat_dim: int = 256,
        embed_dims: int = 256,
        num_vec: int = 50,
        num_pts_per_vec: int = 20,
        num_pts_per_gt_vec: int = 20,
        num_classes: int = 3,
        grid_conf=None,
        bev_bounds=None,
        decoder_num_layers: int = 2,
        decoder_num_heads: int = 4,
        decoder_im2col_step: int = 192,
        decoder_feedforward_channels: int = 512,
        decoder_ffn_dropout: float = 0.1,
        bbox_coder_max_num: int = 50,
        voxel_size=None,
        **kwargs,
) -> MapHead:
    point_cloud_range = [
        grid_conf["xbound"][0], grid_conf["ybound"][0], grid_conf["zbound"][0],
        grid_conf["xbound"][1], grid_conf["ybound"][1], grid_conf["zbound"][1],
    ]
    xmin, xmax = grid_conf["xbound"][0], grid_conf["xbound"][1]
    ymin, ymax = grid_conf["ybound"][0], grid_conf["ybound"][1]
    bbox_coder_post_center_range = [xmin, ymin, xmin, ymin, xmax, ymax, xmax, ymax]
    bev_h, bev_w = (grid_conf["xbound"][1] - grid_conf["xbound"][0] / grid_conf["xbound"][2],
                    grid_conf["ybound"][1] - grid_conf["ybound"][0] / grid_conf["ybound"][2])
    """按 maptr_nano_r18_110e.py 中 pts_bbox_head 的 decoder/bbox_coder/positional_encoding 配置构建 MapHead（use_maptr_decoder=True）。"""
    return MapHead(
        bev_feat_dim=bev_feat_dim,
        embed_dims=embed_dims,
        num_vec=num_vec,
        num_pts_per_vec=num_pts_per_vec,
        num_pts_per_gt_vec=num_pts_per_gt_vec,
        num_classes=num_classes,
        num_decoder_layers=decoder_num_layers,
        num_heads=decoder_num_heads,
        dropout=decoder_ffn_dropout,
        feedforward_dims=decoder_feedforward_channels,
        pc_range=point_cloud_range,
        bev_bounds=bev_bounds,
        row_num_embed=bev_h,
        col_num_embed=bev_w,
        use_maptr_decoder=True,
        maptr_decoder_num_layers=decoder_num_layers,
        maptr_num_heads=decoder_num_heads,
        maptr_im2col_step=decoder_im2col_step,
        maptr_feedforward_channels=decoder_feedforward_channels,
        maptr_num_levels=1,
        maptr_num_points=4,
        post_center_range=bbox_coder_post_center_range,
        bbox_coder_max_num=bbox_coder_max_num,
        **kwargs,
    )


if __name__ == '__main__':
    pass
