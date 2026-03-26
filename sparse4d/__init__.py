"""
Sparse4D + BEV 纯 PyTorch 实现，不依赖 mmcv/mmdet，便于调试开发。

用法:
  from sparse4d_bev_torch import build_sparse4d_bev_head
  head = build_sparse4d_bev_head(num_anchor=900, embed_dims=256, ...)
  out = head([bev_feature], metas={})  # bev_feature: (B*S, 256, H, W)
"""
from sparse4d.box3d import X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ, decode_box
from sparse4d.decoder import SparseBox3DDecoder
from sparse4d.instance_bank import InstanceBank, topk
from sparse4d.detection3d_blocks import (
    SparseBox3DEncoder,
    SparseBox3DRefinementModule,
    SparseBox3DKeyPointsGenerator,
)
from sparse4d.bev_aggregation import BEVFeatureAggregation
from sparse4d.head import Sparse4DHead, MHAWrapper, FFN
# from model import build_sparse4d_bev_head, build_sparse4d_bev_model, build_track_head
from sparse4d.track_head import TrackHead, track_affinity_loss, decode_track, align_anchors_to_frame

__all__ = [
    # "build_sparse4d_bev_head",
    # "build_sparse4d_bev_model",
    # "build_track_head",
    "TrackHead",
    "track_affinity_loss",
    "decode_track",
    "align_anchors_to_frame",
    "InstanceBank",
    "Sparse4DHead",
    "SparseBox3DDecoder",
    "decode_box",
    "BEVFeatureAggregation",
    "SparseBox3DEncoder",
    "SparseBox3DRefinementModule",
    "SparseBox3DKeyPointsGenerator",
    "MHAWrapper",
    "FFN",
    "topk",
]