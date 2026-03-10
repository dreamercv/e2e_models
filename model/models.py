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
                                        num_temporal=self.seq_len,
                                        num_cams = config["num_cams"] )

        config["det_3d_head"]["embed_dims"] = img_outchannels
        config["det_3d_head"]["bev_bounds"] = (self.grid_conf["xbound"], self.grid_conf["ybound"])
        self.det3d_head = build_det3D_head(**config["det_3d_head"])

        config["track_head"]["feat_dim"] = config["det_3d_head"]["embed_dims"]
        self.track_head = TrackHead(**config["track_head"])

        config["map_3d_head"]["bev_feat_dim"] = img_outchannels
        config["map_3d_head"]["grid_conf"] = self.grid_conf 
        # self.map3d_head = build_map_head(bev_feat_dim=config["map_3d_head"]["bev_feat_dim"],
        #                                  embed_dims=config["map_3d_head"]["embed_dims"],
        #                                  num_vec=config["map_3d_head"]["num_vec"],
        #                                  num_pts_per_vec=config["map_3d_head"]["num_pts_per_vec"],
        #                                  num_pts_per_gt_vec=config["map_3d_head"]["num_pts_per_gt_vec"],
        #                                  num_classes=config["map_3d_head"]["num_pts_per_vec"],
        #                                  grid_conf = self.grid_conf,
        #                                  decoder_num_layers=2,
        #                                  decoder_num_heads=4,
        #                                  decoder_im2col_step=192,
        #                                  decoder_feedforward_channels=512,
        #                                  bbox_coder_max_num=50, )

        self.map3d_head = build_map_head(**config["map_3d_head"] )

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
    
    def forward_dynamic_branch(self,bev_feat,T_ego_his2curs,metas,decoder=True,comp_loss = True):
        out,seq_features,seq_anchors = self.det3d_head(bev_feat,metas)
        if comp_loss:
            det_loss = self.det3d_head.loss(out,metas)
        else:
            det_loss = None

        if decoder:
            result = self.det3d_head.decoder.decode(
                out["classification"],
                out["prediction"],
                quality=out.get("quality"),
                instance_id=out.get("instance_id"),
            )
        else:
            result = None


        return out,det_loss,result


    def forward_static_branch(self,bev_feat,metas):
        Bt, C, H, W = bev_feat.shape
        T = self.seq_len
        B = Bt // T
        bev_for_map = bev_feat.view(B, T, C, H, W)[:, -1] # 只预测最后一帧的地图
        map_pred = self.map3d_head(bev_for_map, metas)
        map_out = {
            "map_cls_scores": map_pred["map_cls_scores"],
            "map_bbox_preds": map_pred["map_bbox_preds"],
            "map_pts_preds": map_pred["map_pts_preds"],
        }
        gt_map_bboxes = metas.get("gt_map_bboxes")
        gt_map_labels = metas.get("gt_map_labels")
        gt_map_pts = metas.get("gt_map_pts")
        if gt_map_bboxes is not None and gt_map_labels is not None and gt_map_pts is not None:
            map_losses = self.map3d_head.loss(map_pred, gt_map_bboxes, gt_map_labels, gt_map_pts)
            map_out["loss_map"] = map_losses
        map_out["map_polylines"] = self.map3d_head.decode(map_pred, score_threshold=0.5)
        return map_out


    def forward(self, x, rots, trans, intrins, distorts, post_rot, pos_tran, theta_mats, T_ego_his2curs=None,metas=None):
        b, m, t, n, c, h, w = x.shape
        x = rearrange(x, 'b m t n c h w -> (b m t n) c h w')
        rots = rearrange(rots, 'b m t n h w -> (b m) t n h w')
        trans = rearrange(trans, 'b m t n h w -> (b m) t n h w')
        intrins = rearrange(intrins, 'b m t n h w -> (b m) t n h w')
        distorts = rearrange(distorts, 'b m t n h w -> (b m) t n h w')
        post_rot = rearrange(post_rot, 'b m t n h w -> (b m) t n h w')
        pos_tran = rearrange(pos_tran, 'b m t n h w -> (b m) t n h w')
        theta_mats = rearrange(theta_mats, 'b m t h w -> (b m) t h w')
        T_ego_his2curs = torch.eye(4).view(1, 1, 4, 4).expand(b,  t, 4, 4)

        x = self.forward_img_backbone(x)

        det2d_out = self.forward_det2d(x)
        map2d_out = self.forward_map2d(x)

        bev_feat = self.forward_bev_backbone(x, rots, trans, intrins, distorts, post_rot, pos_tran, theta_mats)
        print(bev_feat)

        bev_feat = bev_feat.reshape(b, m, t,*bev_feat.shape[1:])
        dynamic_bev_feture = bev_feat[:,0].flatten(0,1)
        static_bev_feture = bev_feat[:,1].flatten(0,1)



        print(bev_feat)

        det_out,det_loss,det_result = self.forward_dynamic_branch(dynamic_bev_feture,T_ego_his2curs, metas)
        print(det_out)

        map_out = self.forward_static_branch(static_bev_feture, metas)










if __name__ == '__main__':
    pass
