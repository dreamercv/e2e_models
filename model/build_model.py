import numpy as np
import torch
import torch.nn as nn



from backbone.image_backbone import ImageBackBone
from model_2d.det2d_model import ModelDet2D
from model_2d.seg2d_model import ModelSeg2D
from backbone.bev_backbone import BEVBackbone

from sparse4d.instance_bank import InstanceBank
from sparse4d.detection3d_blocks import (
    SparseBox3DEncoder,
    SparseBox3DRefinementModule,
    SparseBox3DKeyPointsGenerator,
)
from sparse4d.bev_aggregation import BEVFeatureAggregation
from sparse4d.head import Sparse4DHead, MHAWrapper, SimpleFFN
from sparse4d.decoder import SparseBox3DDecoder
from sparse4d.dn_sampler import DenoisingSampler
from sparse4d.losses import FocalLoss, SparseBox3DLoss

from sparse4d.track_head import TrackHead
from sparse4d.track_head import track_affinity_loss, decode_track
from sparse4d.track_head import align_anchors_to_frame

from maptr import MapHead



def build_image_backbone(out_channels):
    return ImageBackBone(out_channels=out_channels)

def build_det2d_model(out_channels=256,det_class_num=3):
    return ModelDet2D(out_channels=out_channels,det_class_num=det_class_num)

def build_seg2d_model(out_channels=256,seg_class_num=3,pos_weight=20.):
    return ModelSeg2D(out_channels=out_channels,seg_class_num=seg_class_num,pos_weight=pos_weight)


def build_bev_backbone(channels,grid_conf,input_size=(128, 384),num_temporal=7):
    model = BEVBackbone(channels=channels,grid_conf=grid_conf, input_size=input_size,num_temporal=num_temporal)
    return model



def build_sparse4d_bev_head(
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

    sampler = None
    loss_cls = None
    loss_reg = None
    if use_dn:
        sampler = DenoisingSampler(
            num_dn_groups=num_dn_groups,
            dn_noise_scale=dn_noise_scale,
            max_dn_gt=max_dn_gt,
            add_neg_dn=add_neg_dn,
            reg_weights=reg_weights,
        )
    loss_cls = FocalLoss(alpha=0.25, gamma=2.0, ignore_index=num_classes)
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
        sampler=sampler,
        decoder=decoder,
        loss_cls=loss_cls,
        loss_reg=loss_reg,
        reg_weights=reg_weights,
        gt_cls_key="gt_labels_det3D",
        gt_reg_key="gt_bboxes_det3D",
        cls_threshold_to_reg=0.05,
    )
    head.init_weights()
    return head


def build_track_head(
    feat_dim: int = 256,
    anchor_dim: int = 11,
    num_heads: int = 8,
    dropout: float = 0.1,
    embed_dim: int = 256,
):
    """从 model.py 拷贝的跟踪 head 构建函数。"""
    return TrackHead(
        feat_dim=feat_dim,
        anchor_dim=anchor_dim,
        num_heads=num_heads,
        dropout=dropout,
        embed_dim=embed_dim,
    )




def build_map_head(
    bev_feat_dim = 256,
    embed_dims = 256,
    num_vec = 50,
    num_pts_per_vec = 20,
    num_pts_per_gt_vec = 20,
    num_classes = 3,
    pc_range= None,
    bev_bounds=None,
    bev_h = 80,
    bev_w = 40,
    decoder_num_layers = 2,
    decoder_num_heads = 4,
    decoder_im2col_step = 192,
    decoder_feedforward_channels = 512,
    decoder_ffn_dropout = 0.1,
    bbox_coder_post_center_range= None,
    bbox_coder_max_num = 50,
    voxel_size = None,
    **kwargs):
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
        pc_range=pc_range,
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



def build_sparse4d_bev_model(
    image_backbone: nn.Module,
    det2d_head:nn.Module,
    seg2d_head:nn.Module,
    bev_backbone: nn.Module,
    det_head: nn.Module,
    track_head: nn.Module = None,
    map_head: nn.Module = None,
    num_temporal_frames: int = 7,
):
    """从 model.py 拷贝的完整模型封装。"""

    class Sparse4DBEVModel(nn.Module):
        def __init__(self, image_backbone,bev_backbone,
                     det2d_head=None,seg2d_head=None,
                     det_head=None, track_head=None, map_head=None, num_temporal_frames=7):
            super().__init__()
            self.image_backbone = image_backbone
            self.bev_backbone = bev_backbone
            self.det2d_head = det2d_head
            self.seg2d_head = seg2d_head
            self.det_head = det_head
            self.track_head = track_head
            self.map_head = map_head
            self.num_temporal_frames = num_temporal_frames

        def forward(
            self,
            x,
            rots,
            trans,
            intrins,
            distorts,
            post_rot3,
            post_tran3,
            theta_mats,
            T_ego_his2curs,
            metas=None,
        ):
            det2d_out,seg2d_out = None,None
            det_out, track_out = None, None
            map_out = None

            x = self.image_backbone(x)
            if self.det_head is not None:
                det2d_out = self.det_head(x)
            if self.seg2d_head is not None:
                seg2d_out = self.seg2d_head(x)

            feature_maps = self.bev_backbone(x, rots, trans, intrins, distorts, post_rot3, post_tran3, theta_mats)

            if self.det_head is not None:
                det_out, seq_features, seq_anchors = self.det_head(feature_maps, metas)
                if self.track_head is not None and T_ego_his2curs is not None:
                    track_affinity = self.track_head(seq_features, seq_anchors, T_ego_his2curs)
                    track_out = {"track_affinity": track_affinity}
                    if self.training and metas is not None and metas.get("gt_track_match") is not None:
                        track_out["loss_track"] = track_affinity_loss(
                            track_affinity, metas["gt_track_match"], ignore_index=-1
                        )
                    if not self.training:
                        track_ids, positions = decode_track(track_affinity, seq_anchors, use_hungarian=True)
                        track_out["track_ids"] = track_ids
                        track_out["track_positions"] = positions


            if self.map_head is not None:
                Bt, C, H, W = feature_maps.shape
                T = self.num_temporal_frames
                B = Bt // T if (T > 0 and Bt % T == 0) else Bt
                if T > 0 and Bt % T == 0:
                    bev_for_map = feature_maps.view(B, T, C, H, W)[:, -1]
                else:
                    bev_for_map = feature_maps
                map_pred = self.map_head(bev_for_map, metas)
                map_out = {
                    "map_cls_scores": map_pred["map_cls_scores"],
                    "map_bbox_preds": map_pred["map_bbox_preds"],
                    "map_pts_preds": map_pred["map_pts_preds"],
                }
                if self.training and metas is not None:
                    gt_map_bboxes = metas.get("gt_map_bboxes")
                    gt_map_labels = metas.get("gt_map_labels")
                    gt_map_pts = metas.get("gt_map_pts")
                    if gt_map_bboxes is not None and gt_map_labels is not None and gt_map_pts is not None:
                        map_losses = self.map_head.loss(map_pred, gt_map_bboxes, gt_map_labels, gt_map_pts)
                        map_out["loss_map"] = map_losses
                if not self.training:
                    map_out["map_polylines"] = self.map_head.decode(map_pred, score_threshold=0.5)

            return det2d_out,seg2d_out,det_out, track_out, map_out

    return Sparse4DBEVModel(image_backbone,bev_backbone, det2d_head,seg2d_head,det_head, track_head, map_head, num_temporal_frames)