#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   models.py
@Time    :   2026/3/9 21:34
@Author  :   Binge.van
@Email   :   1367955240@qq.com
@description   :
'''
import os
import torch
import torch.nn as nn
from torch.utils.collect_env import run_and_return_first_line

from backbone.image_backbone import ImageBackBone
from model_2d.det2d_model import ModelDet2D
from model_2d.seg2d_model import ModelSeg2D
from backbone.bev_backbone import BEVBackbone

from sparse4d.build_model import build_det3D_head
from sparse4d.track_head import TrackHead

from maptr.build_model import build_map_head

from einops import rearrange


class Model(nn.Module):
    def __init__(self,
                 config,
                 ):
        super().__init__()
        img_outchannels = config["img_outchannels"]

        self.grid_conf = config["grid_conf"]
        self.input_size = config["final_dim"]
        self.seq_len = config["seq_len"]

        self.image_backbone = ImageBackBone(img_outchannels)
        self.det2d_head = ModelDet2D(out_channels=img_outchannels, det_class_num=config["det_2d_num"])
        self.map2d_head = ModelSeg2D(out_channels=img_outchannels, seg_class_num=config["map_2d_num"])
        self.bev_backbone = BEVBackbone(channels=img_outchannels,
                                        grid_conf=self.grid_conf,
                                        input_size=self.input_size,
                                        num_temporal=self.seq_len)

        config["det_3d_head"]["embed_dims"] = img_outchannels
        config["det_3d_head"]["bev_bounds"] = (self.grid_conf["xbound"], self.grid_conf["ybound"])
        self.det3d_head = build_det3D_head(**config["det_3d_head"])

        config["track_head"]["feat_dim"] = config["det_3d_head"]["embed_dims"]
        self.track_head = TrackHead(**config["track_head"])

        config["map_3d_head"]["bev_feat_dim"] = img_outchannels
        self.map3d_head = build_map_head(bev_feat_dim=256,
                                         embed_dims=256,
                                         num_vec=50,
                                         num_pts_per_vec=20,
                                         num_pts_per_gt_vec=20,
                                         num_classes=3,
                                         decoder_num_layers=2,
                                         decoder_num_heads=4,
                                         decoder_im2col_step=192,
                                         decoder_feedforward_channels=512,
                                         bbox_coder_max_num=50, )

        self.config = config

    def forward_img_backbone(self,x):
        return self.image_backbone(x)

    def forward_det2d(self,x):
        return self.det2d_head(x)

    def forward_map2d(self,x):
        return self.map2d_head(x)

    def forward_bev_backbone(self,x, rots, trans, intrins, distorts, post_rot, pos_tran, theta_mats):
        bev_feat = self.bev_backbone(x, rots, trans, intrins, distorts, post_rot, pos_tran, theta_mats)
        return bev_feat

    def forward(self, x, rots, trans, intrins, distorts, post_rot, pos_tran, theta_mats, T_ego_his2curs=None):
        b, m, t, n, c, h, w = x.shape
        x = rearrange(x, 'b,m,t,n,c,h,w -> (b,m,t,n),c,h,w')
        rots = rearrange(rots, 'b,m,t,n,h,w -> (b,m),t,n,h,w')
        trans = rearrange(trans, 'b,m,t,n,h,w -> (b,m),t,n,h,w')
        intrins = rearrange(intrins, 'b,m,t,n,h,w -> (b,m),t,n,h,w')
        distorts = rearrange(distorts, 'b,m,t,n,h,w -> (b,m),t,n,h,w')
        post_rot = rearrange(post_rot, 'b,m,t,n,h,w -> (b,m),t,n,h,w')
        pos_tran = rearrange(pos_tran, 'b,m,t,n,h,w -> (b,m),t,n,h,w')
        theta_mats = rearrange(theta_mats, 'b,m,t,h,w -> b,m,t,h,w')
        T_ego_his2curs = torch.eye(4).view(1, 1, 1, 4, 4).expand(b, m, t, 4, 4)

        x = self.forward_img_backbone(x)

        det2d_out = self.det_head(x)
        map2d_out = self.forward_map2d(x)

        bev_feat = self.forward_bev_backbone(x, rots, trans, intrins, distorts, post_rot, pos_tran, theta_mats)








if __name__ == '__main__':
    pass
