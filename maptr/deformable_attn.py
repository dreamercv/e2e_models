"""
纯 PyTorch 多尺度可变形注意力，不依赖 mmcv。
用于 MapTR decoder 的 cross-attn（在 BEV 上采样）。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

__all__ = [
    "multi_scale_deformable_attn_pytorch",
    "CustomMSDeformableAttention",
]


def multi_scale_deformable_attn_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """纯 PyTorch 多尺度可变形注意力（CPU/CUDA 通用）。

    Args:
        value: (bs, num_keys, num_heads, embed_dims_per_head)
        value_spatial_shapes: (num_levels, 2)，每级的 (H, W)
        sampling_locations: (bs, num_queries, num_heads, num_levels, num_points, 2)，归一化坐标 [0,1]
        attention_weights: (bs, num_queries, num_heads, num_levels, num_points)

    Returns:
        (bs, num_queries, num_heads * embed_dims_per_head)
    """
    bs, num_value, num_heads, embed_dims = value.shape
    _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape

    # [0,1] -> [-1,1] for grid_sample
    sampling_grids = 2 * sampling_locations - 1

    # split value by level
    level_sizes = (
        value_spatial_shapes[:, 0] * value_spatial_shapes[:, 1]
    )
    if level_sizes.dim() > 0:
        level_sizes = [int(s.item()) for s in level_sizes]
    else:
        level_sizes = [int(level_sizes.item())]
    value_list = value.split(level_sizes, dim=1)

    sampling_value_list = []
    for level_idx in range(value_spatial_shapes.shape[0]):
        H_ = int(value_spatial_shapes[level_idx, 0].item())
        W_ = int(value_spatial_shapes[level_idx, 1].item())
        value_l_ = value_list[level_idx]
        # (bs, H_*W_, num_heads, embed_dims) -> (bs*num_heads, embed_dims, H_, W_)
        value_l_ = value_l_.flatten(2).permute(0, 2, 1).reshape(
            bs * num_heads, embed_dims, H_, W_
        )
        # (bs, num_queries, num_heads, num_points, 2) for this level
        sampling_grid_l_ = sampling_grids[:, :, :, level_idx, :, :]
        # (bs, num_heads, num_queries, num_points, 2) -> (bs*num_heads, num_queries, num_points, 2)
        sampling_grid_l_ = sampling_grid_l_.permute(0, 2, 1, 3, 4).reshape(
            bs * num_heads, num_queries, num_points, 2
        )
        # grid_sample: input (N,C,H,W), grid (N,H_out,W_out,2) -> (N,C,H_out,W_out)
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        # (bs*num_heads, embed_dims, num_queries, num_points)
        sampling_value_list.append(sampling_value_l_)

    # (bs*num_heads, embed_dims, num_queries, num_levels*num_points)
    sampling_values = torch.cat(sampling_value_list, dim=-1)
    # (bs, num_queries, num_heads, num_levels*num_points)
    attention_weights_flat = attention_weights.reshape(
        bs, num_queries, num_heads, num_levels * num_points
    )
    # (bs, num_heads, num_queries, num_levels*num_points)
    attention_weights_flat = attention_weights_flat.permute(0, 2, 1, 3)
    # (bs*num_heads, 1, num_queries, num_levels*num_points)
    attention_weights_flat = attention_weights_flat.reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )
    # (bs*num_heads, embed_dims, num_queries, num_levels*num_points) * weight -> sum -> (bs*num_heads, embed_dims, num_queries)
    output = (sampling_values * attention_weights_flat).sum(dim=-1)
    # (bs, num_queries, num_heads*embed_dims)
    output = output.view(bs, num_heads, embed_dims, num_queries).permute(
        0, 3, 1, 2
    ).reshape(bs, num_queries, num_heads * embed_dims)
    return output


class CustomMSDeformableAttention(nn.Module):
    """MapTR 使用的多尺度可变形注意力，纯 PyTorch，不依赖 mmcv。
    Query 为 map instance，value 为 BEV 特征，按 reference_points 在 BEV 上采样。
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 1,
        num_points: int = 4,
        im2col_step: int = 64,
        dropout: float = 0.1,
        batch_first: bool = False,
    ):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_heads, "
                f"but got {embed_dims} and {num_heads}"
            )
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.im2col_step = im2col_step
        self.batch_first = batch_first
        dim_per_head = embed_dims // num_heads

        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2
        )
        self.attention_weights = nn.Linear(
            embed_dims, num_heads * num_levels * num_points
        )
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        nn.init.constant_(self.sampling_offsets.bias, 0.0)
        thetas = torch.arange(
            self.num_heads, dtype=torch.float32
        ) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], dim=-1)
        grid_init = (
            grid_init / grid_init.abs().max(dim=-1, keepdim=True)[0]
        ).view(self.num_heads, 1, 1, 2).repeat(
            1, self.num_levels, self.num_points, 1
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        self.sampling_offsets.bias.data = grid_init.view(-1)
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        identity: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        reference_points: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.Tensor] = None,
        level_start_index: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum().item() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        dim_per_head = self.embed_dims // self.num_heads
        value = value.view(bs, num_value, self.num_heads, dim_per_head)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(dim=-1)
        attention_weights = attention_weights.view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], dim=-1
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets
                / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.num_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                f"reference_points last dim must be 2 or 4, got {reference_points.shape[-1]}"
            )

        output = multi_scale_deformable_attn_pytorch(
            value,
            spatial_shapes,
            sampling_locations,
            attention_weights,
        )
        output = self.output_proj(output)
        if not self.batch_first:
            output = output.permute(1, 0, 2)
        return self.dropout(output) + identity