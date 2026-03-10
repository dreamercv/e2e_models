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
        num_decoder_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        feedforward_dims: int = 512,
        grid_conf=None,
        bev_bounds=None,

        with_box_refine= True,
        use_instance_pts = True,
        dir_interval= 1,
        cls_weight= 1.0,
        reg_weight= 1.0,
        pts_weight= 1.0,
        loss_pts_src_weight= 1.0,
        loss_pts_dst_weight= 1.0,
        loss_dir_weight= 0.005,
        aux_loss_weight= 0.5,

        use_maptr_decoder= False,


        maptr_im2col_step= 192,
        maptr_num_levels=1,
        maptr_num_points=4,
        bbox_coder_max_num = 50,


) -> MapHead:
    point_cloud_range = [
        grid_conf["xbound"][0], grid_conf["ybound"][0], grid_conf["zbound"][0],
        grid_conf["xbound"][1], grid_conf["ybound"][1], grid_conf["zbound"][1],
    ]
    xmin, xmax = grid_conf["xbound"][0], grid_conf["xbound"][1]
    ymin, ymax = grid_conf["ybound"][0], grid_conf["ybound"][1]
    bbox_coder_post_center_range = [xmin, ymin, xmin, ymin, xmax, ymax, xmax, ymax]
    bev_h  = int((grid_conf["xbound"][1] - grid_conf["xbound"][0]) // grid_conf["xbound"][2])
    bev_w  = int((grid_conf["ybound"][1] - grid_conf["ybound"][0]) // grid_conf["ybound"][2])
    """
    按 maptr_nano_r18_110e.py 中 pts_bbox_head 的 decoder/bbox_coder/positional_encoding 配置构建 MapHead（use_maptr_decoder=True）。
    """
    head =  MapHead(
        bev_feat_dim=bev_feat_dim,
        embed_dims=embed_dims,
        num_vec=num_vec,
        num_pts_per_vec=num_pts_per_vec,
        num_pts_per_gt_vec=num_pts_per_gt_vec,
        num_classes=num_classes,
        num_decoder_layers=num_decoder_layers,
        num_heads=num_heads,
        dropout=dropout,
        feedforward_dims=feedforward_dims,
        pc_range=point_cloud_range,
        bev_bounds=bev_bounds,

        with_box_refine=with_box_refine,
        use_instance_pts = use_instance_pts,
        dir_interval= dir_interval,
        cls_weight= cls_weight,
        reg_weight=reg_weight,
        pts_weight= pts_weight,
        loss_pts_src_weight= loss_pts_src_weight,
        loss_pts_dst_weight= loss_pts_dst_weight,
        loss_dir_weight= loss_dir_weight,
        aux_loss_weight= aux_loss_weight,


        row_num_embed=bev_h,
        col_num_embed=bev_w,

        use_maptr_decoder=use_maptr_decoder,
        maptr_decoder_num_layers=num_decoder_layers,
        maptr_num_heads=num_heads,
        maptr_im2col_step=maptr_im2col_step,
        maptr_feedforward_channels=feedforward_dims,
        maptr_num_levels=maptr_num_levels,
        maptr_num_points=maptr_num_points,
        post_center_range=bbox_coder_post_center_range,
        bbox_coder_max_num=bbox_coder_max_num,
    )

    return head

    


if __name__ == '__main__':
    pass
