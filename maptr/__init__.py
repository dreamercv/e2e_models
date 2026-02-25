"""
MapTR 相关模块：decoder、bbox_coder、deformable_attn，纯 PyTorch，不依赖 mmcv。
"""
from .deformable_attn import (
    multi_scale_deformable_attn_pytorch,
    CustomMSDeformableAttention,
)
from .decoder import (
    inverse_sigmoid,
    DetrTransformerDecoderLayer,
    MapTRDecoder,
    build_maptr_decoder,
)
from .bbox_coder import (
    MapTRNMSFreeCoder,
    build_maptr_bbox_coder,
)
from .map_head import (
    MapHead,
    build_map_head,
    build_map_head_from_maptr_config,
)

__all__ = [
    "multi_scale_deformable_attn_pytorch",
    "CustomMSDeformableAttention",
    "inverse_sigmoid",
    "DetrTransformerDecoderLayer",
    "MapTRDecoder",
    "build_maptr_decoder",
    "MapTRNMSFreeCoder",
    "build_maptr_bbox_coder",
    "MapHead",
    "build_map_head",
    "build_map_head_from_maptr_config",
]
