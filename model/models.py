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

from backbone.image_backbone import ImageBackBone,ImageBackboneResNetFPN
from model_2d.det2d_model import ModelDet2D
from model_2d.seg2d_model import ModelSeg2D
from backbone.bev_backbone import BEVBackbone

from sparse4d.build_model import build_det3D_head
from sparse4d.track_head import TrackHead

from trajectory_dynamic.trajectory_predictor import build_dynamic_trajectory_predictor

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
        self.bn2gn_num_groups = config["bn2gn_num_groups"]

        self.image_backbone = ImageBackboneResNetFPN(
            backbone=config.get("backbone", "resnet18"),
            out_channels=img_outchannels,
            pretrain_path=config.get("backbone_path", None),
            pretrained=False if config.get("pretrain", None) is not None else True,
            device=config.get("device", "cuda"),
        )
        self.det2d_head = ModelDet2D(out_channels=img_outchannels, det_class_num=config["det_2d_num"])
        self.map2d_head = ModelSeg2D(out_channels=img_outchannels, seg_class_num=config["map_2d_num"])
        self.bev_backbone = BEVBackbone(
            channels=img_outchannels,
            grid_conf=self.grid_conf,
            input_size=self.input_size,
            num_temporal=self.seq_len,
            num_cams=config["num_cams"],
            use_cam_pos_embed=config.get("bev_use_cam_pos_embed", True),
            cam_pos_L=config.get("bev_cam_pos_L", 4),
        )

        config["det_3d_head"]["embed_dims"] = img_outchannels
        config["det_3d_head"]["bev_bounds"] = (self.grid_conf["xbound"], self.grid_conf["ybound"])
        self.det3d_head = build_det3D_head(**config["det_3d_head"])

        config["traj_dynamic"]["obj_num"] = config["det_3d_head"]["num_anchor"]
        config["traj_dynamic"]["traj_dim"] = (config["his_lens"]["obj_dynamic_traj"] + config["fur_lens"]["obj_dynamic_traj"])*2
        config["traj_dynamic"]["instance_feat_dim"] = config["det_3d_head"]["embed_dims"]
        config["traj_dynamic"]["anchor_dim"] = config["det_3d_head"]["anchor_dim"]
        self.dynamic_traj = build_dynamic_trajectory_predictor(**config["traj_dynamic"])

        config["track_head"]["feat_dim"] = config["det_3d_head"]["embed_dims"]
        self.track_head = TrackHead(**config["track_head"])

        config["map_3d_head"]["bev_feat_dim"] = img_outchannels
        config["map_3d_head"]["grid_conf"] = self.grid_conf 
        self.map3d_head = build_map_head(**config["map_3d_head"] )

        self.config = config
        self.det3d_loss_weights = config.get("det3d_loss_weights", {})

        self._replace_all_bn_with_gn(target_groups=self.bn2gn_num_groups)

    def _replace_all_bn_with_gn(self, target_groups=32):
        def _get_groups(c):
            g = min(target_groups, c)
            while c % g != 0 and g > 1:
                g -= 1
            return g
        
        def _replace(module):
            for name, child in module.named_children():
                if isinstance(child, nn.BatchNorm2d):
                    c = child.num_features
                    gn = nn.GroupNorm(_get_groups(c), c, eps=child.eps, affine=True)
                    setattr(module, name, gn)
                else:
                    _replace(child)
        
        _replace(self)

    def _get_loss_weight(self, name: str) -> float:
        if not name.startswith("det3d_"):
            return 1.0
        if "loss_cls" in name:
            return float(self.det3d_loss_weights.get("cls", 1.0))
        if "loss_box" in name:
            return float(self.det3d_loss_weights.get("box", 1.0))
        if "loss_cns" in name:
            return float(self.det3d_loss_weights.get("cns", 1.0))
        if "loss_yns" in name:
            return float(self.det3d_loss_weights.get("yns", 1.0))
        if "loss_giou" in name:
            return float(self.det3d_loss_weights.get("giou", 1.0))
        return 1.0

    def forward_img_backbone(self,x):
        return self.image_backbone(x)

    def forward_det2d(self,x):
        return self.det2d_head(x)

    def forward_map2d(self,x):
        return self.map2d_head(x)

    def forward_bev_backbone(self,x, rots, trans, intrins, distorts, post_rot, pos_tran, ego_poses,pre_bev=[],prepare_pre_bev = False):
        bev_feat,cur_bev = self.bev_backbone(x, rots, trans, intrins, distorts, post_rot, pos_tran, ego_poses,pre_bev,prepare_pre_bev = prepare_pre_bev)
        return bev_feat,cur_bev
    
    def forward_dynamic_branch(self,bev_feat,T_ego_his2curs,metas,decoder=False):
        out,instance_features,instance_anchors = self.det3d_head(bev_feat,metas)
        det_loss = self.det3d_head.loss(out,metas)
            

        if decoder:
            result = self.det3d_head.decoder.decode(
                out["classification"],
                out["prediction"],
                quality=out.get("quality"),
                instance_id=out.get("instance_id"),
            )
        else:
            result = None


        return out,det_loss,result,instance_features,instance_anchors

    def forward_dynamic_trajectory(self,instance_feature,anchor,decode_trajectory=True,data_gt=None,det_out=None):
        out = self.dynamic_traj(instance_feature,anchor,decode_trajectory=decode_trajectory,data_gt=data_gt,det_out=det_out)
        return out

    def forward_static_branch(self,bev_feat,metas,decoder=False):
        # bev_feat: (B * T, C, H, W)，静态分支希望对整个时间序列 T 帧都计算地图 loss
        Bt, C, H, W = bev_feat.shape
        T = self.seq_len
        B = Bt // T
        # 直接把每一帧当作一个样本送入 MapHead，batch 视作 B*T
        map_pred = self.map3d_head(bev_feat, metas)
        map_out = {
            "map_cls_scores": map_pred["map_cls_scores"],
            "map_bbox_preds": map_pred["map_bbox_preds"],
            "map_pts_preds": map_pred["map_pts_preds"],
        }
        # 与 dataset 输出命名保持一致：gt_labels_map3D / gt_bboxes_map3D / gt_pts_map3D
        # data 层对单个样本输出的是长度为 T 的 list，collate 后 metas 中是形如：
        #   gt_labels_map3D: [ [t0_labels, t1_labels, ...],  [...], ... ]  (长度 B，每个元素长度 T)
        # 这里需要展平成长度 B*T 的一维 list，与 bev_feat 的 batch 维对齐。
        raw_bboxes = metas.get("gt_bboxes_map3D")
        raw_labels = metas.get("gt_labels_map3D")
        raw_pts = metas.get("gt_pts_map3D")

        def _flatten_seq_list(x):
            if not isinstance(x, list) or len(x) == 0:
                return x
            if isinstance(x[0], list):
                out = []
                for seq in x:
                    out.extend(seq)
                return out
            return x

        gt_map_bboxes = _flatten_seq_list(raw_bboxes)
        gt_map_labels = _flatten_seq_list(raw_labels)
        gt_map_pts = _flatten_seq_list(raw_pts)

        if gt_map_bboxes is not None and gt_map_labels is not None and gt_map_pts is not None:
            map_losses = self.map3d_head.loss(map_pred, gt_map_bboxes, gt_map_labels, gt_map_pts)
            map_out["loss_map"] = map_losses
        if decoder:
            map_polylines = self.map3d_head.decode(map_pred, score_threshold=0.5)
        else:
            map_polylines = None
        map_out["map_polylines"] = map_polylines
        return map_out


    def forward(self, x, rots, trans, intrins, distorts, post_rot, pos_tran, ego_poses,pre_bev, T_ego_his2curs=None, metas=None,decoder=False,task_names=[],prepare_pre_bev = False):
        b, m, t, n, c, h, w = x.shape
        # 2, 1, 5, 8, 3, 128, 384
        # from dataset.dataset import denormalize_img
        # import cv2
        # import numpy as np
        # a = denormalize_img(x[0,0,0,0])
        # opencv_image = cv2.cvtColor(np.array(a), cv2.COLOR_RGB2BGR)


        x = rearrange(x, 'b m t n c h w -> (b m t n) c h w')
        rots = rearrange(rots, 'b m t n h w -> (b m) t n h w')          # torch.Size([2, 1, 5, 8, 3, 3])
        trans = rearrange(trans, 'b m t n h w -> (b m) t n h w')        # torch.Size([2, 1, 5, 8, 1, 3])
        intrins = rearrange(intrins, 'b m t n h w -> (b m) t n h w')    # torch.Size([2, 1, 5, 8, 3, 3])
        distorts = rearrange(distorts, 'b m t n h w -> (b m) t n h w')  # torch.Size([2, 1, 5, 8, 1, 8])
        post_rot = rearrange(post_rot, 'b m t n h w -> (b m) t n h w')  # torch.Size([2, 1, 5, 8, 3, 3])
        pos_tran = rearrange(pos_tran, 'b m t n h w -> (b m) t n h w')  # torch.Size([2, 1, 5, 8, 1, 3])
        ego_poses = rearrange(ego_poses, 'b m t h w -> (b m) t h w')  # torch.Size([2, 1, 5, 2, 3])
        if T_ego_his2curs is None:
            T_ego_his2curs = torch.eye(4, device=x.device).view(1, 1, 4, 4).expand(b, t, 4, 4)


        out = {}
        x = self.forward_img_backbone(x)

        det2d_out = self.forward_det2d(x)
        map2d_out = self.forward_map2d(x)
        out.update({
            "det2d_out": det2d_out,
            "map2d_out": map2d_out,
        })

        bev_feat,cur_bev = self.forward_bev_backbone(x, rots, trans, intrins, distorts, post_rot, pos_tran, ego_poses,pre_bev,prepare_pre_bev = prepare_pre_bev)
        out.update({
            "cur_bev":cur_bev,
        })
        if prepare_pre_bev:
            return out

        bev_feat = bev_feat.reshape(b, m, t,*bev_feat.shape[1:])

        losses = {}
        det_out,det_result = None,None
        map_out = None
        dyn_traj = None
        for i, task_name in enumerate(task_names) :
            sub_bev_feture = bev_feat[:,i].flatten(0,1)
            if task_name == "dynamic" or task_name == "dynamic_static":
                det_out, det_loss, det_result,instance_features,instance_anchors = self.forward_dynamic_branch(sub_bev_feture, T_ego_his2curs, metas["dynamic"],decoder=decoder)
                if isinstance(det_loss, dict):
                    for k, v in det_loss.items():
                        losses[f"det3d_{k}"] = v

                # B, N = instance_features.shape[:2]
                # traj_dim = 55 * 2
                # device = instance_features.device
                # gt_traj = torch.randn(B, N, traj_dim, device=device)
                # traj_mask = torch.ones(B, N, traj_dim, device=device)
                # instance_mask = torch.ones(B, N, device=device)
                # instance_mask[:, N//2:] = 0
                dyn_traj_loss,dyn_traj = self.forward_dynamic_trajectory(instance_features,instance_anchors,decode_trajectory=True,det_out=det_out,data_gt= metas["dynamic"])
                if isinstance(dyn_traj_loss, dict):
                    for k, v in dyn_traj_loss.items():
                        losses[f"dyntraj_{k}"] = v


            if task_name == "static" or task_name == "dynamic_static":
                map_out = self.forward_static_branch(sub_bev_feture, metas["static"],decoder=decoder)
                map_loss = map_out.get("loss_map", None)
                if isinstance(map_loss, dict):
                    for k, v in map_loss.items():
                        losses[f"map3d_{k}"] = v

        total_loss = None
        if len(losses) > 0:
            weighted_losses = []
            for k, v in losses.items():
                if not torch.is_tensor(v):
                    continue
                weighted_losses.append(v * self._get_loss_weight(k))
            total_loss = sum(weighted_losses) if len(weighted_losses) > 0 else None

        out.update({
            "total_loss": total_loss,
            "losses": losses,
            "det3d_out": det_out,
            "det3d_result": det_result,
            "map3d_out": map_out,
            "dyn_traj":dyn_traj
        })

        return out
if __name__ == '__main__':
    pass