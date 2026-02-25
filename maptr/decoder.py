"""
MapTR Decoder：DetrTransformerDecoderLayer + MapTRDecoder，纯 PyTorch，不依赖 mmcv。
与 config 中 decoder / transformerlayers / operation_order 对应。
"""
import copy
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict, Any

from .deformable_attn import CustomMSDeformableAttention

__all__ = [
    "inverse_sigmoid",
    "DetrTransformerDecoderLayer",
    "MapTRDecoder",
    "build_maptr_decoder",
]


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(min=eps, max=1 - eps)
    return torch.log(x / (1 - x))


class FFN(nn.Module):
    """两层 MLP，用于 decoder layer 的 ffn。"""

    def __init__(
        self,
        embed_dims: int,
        feedforward_channels: int,
        ffn_dropout: float = 0.1,
        act_cfg: Optional[dict] = None,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.linear1 = nn.Linear(embed_dims, feedforward_channels)
        self.linear2 = nn.Linear(feedforward_channels, embed_dims)
        self.dropout = nn.Dropout(ffn_dropout)
        self.act = nn.ReLU(inplace=True)

    def forward(
        self, x: torch.Tensor, identity: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if identity is None:
            identity = x
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x + identity


class DetrTransformerDecoderLayer(nn.Module):
    """单层 MapTR Decoder：self_attn -> norm -> cross_attn -> norm -> ffn -> norm。
    与 config transformerlayers (DetrTransformerDecoderLayer) 及 operation_order 一致。
    输入 query 格式：(num_query, bs, embed_dims)，即 batch_first=False。
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int = 4,
        num_levels: int = 1,
        num_points: int = 4,
        im2col_step: int = 192,
        feedforward_channels: int = 512,
        ffn_dropout: float = 0.1,
        dropout: float = 0.1,
        batch_first: bool = False,
        operation_order: Tuple[str, ...] = (
            "self_attn",
            "norm",
            "cross_attn",
            "norm",
            "ffn",
            "norm",
        ),
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.operation_order = operation_order
        self.batch_first = batch_first

        self.self_attn = nn.MultiheadAttention(
            embed_dims,
            num_heads,
            dropout=dropout,
            batch_first=batch_first,
        )
        self.cross_attn = CustomMSDeformableAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            im2col_step=im2col_step,
            dropout=dropout,
            batch_first=batch_first,
        )
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
        )

        num_norms = operation_order.count("norm")
        self.norms = nn.ModuleList(
            [nn.LayerNorm(embed_dims) for _ in range(num_norms)]
        )

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        reference_points: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.Tensor] = None,
        level_start_index: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        norm_idx = 0
        identity = query

        for op in self.operation_order:
            if op == "self_attn":
                if query_pos is not None:
                    q = query + query_pos
                else:
                    q = query
                attn_out, _ = self.self_attn(
                    q, query, query, key_padding_mask=None
                )
                query = identity + attn_out
                identity = query
            elif op == "norm":
                query = self.norms[norm_idx](query)
                norm_idx += 1
            elif op == "cross_attn":
                query = self.cross_attn(
                    query,
                    key=key,
                    value=value,
                    identity=identity,
                    query_pos=query_pos,
                    key_padding_mask=key_padding_mask,
                    reference_points=reference_points,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    **kwargs,
                )
                identity = query
            elif op == "ffn":
                query = self.ffn(query, identity=identity)
                identity = query

        return query


class MapTRDecoder(nn.Module):
    """MapTR Decoder：多层 DetrTransformerDecoderLayer，带 reference 迭代细化。
    与 config decoder type='MapTRDecoder' 一致。
    输入 query: (num_query, bs, embed_dims)，reference_points: (bs, num_query, 2)。
    """

    def __init__(
        self,
        num_layers: int,
        embed_dims: int,
        num_heads: int = 4,
        num_levels: int = 1,
        num_points: int = 4,
        im2col_step: int = 192,
        feedforward_channels: int = 512,
        ffn_dropout: float = 0.1,
        dropout: float = 0.1,
        return_intermediate: bool = True,
        operation_order: Tuple[str, ...] = (
            "self_attn",
            "norm",
            "cross_attn",
            "norm",
            "ffn",
            "norm",
        ),
    ):
        super().__init__()
        self.return_intermediate = return_intermediate
        self.num_layers = num_layers

        layer = DetrTransformerDecoderLayer(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            im2col_step=im2col_step,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            dropout=dropout,
            operation_order=operation_order,
        )
        self.layers = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(num_layers)]
        )

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        reference_points: Optional[torch.Tensor] = None,
        reg_branches: Optional[nn.ModuleList] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.Tensor] = None,
        level_start_index: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (num_query, bs, embed_dims)
            value: (num_value, bs, embed_dims)，即 BEV flatten 后的 memory
            reference_points: (bs, num_query, 2)，归一化 [0,1]
            reg_branches: 每层回归 head，用于更新 reference_points

        Returns:
            inter_states: (num_layers, num_query, bs, embed_dims) 或单层时 (num_query, bs, embed_dims)
            inter_references: (num_layers, bs, num_query, 2) 或 (bs, num_query, 2)
        """
        output = query
        intermediate = []
        intermediate_reference_points = []

        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points[..., :2].unsqueeze(2)
            output = layer(
                output,
                key=key,
                value=value,
                query_pos=query_pos,
                key_padding_mask=key_padding_mask,
                reference_points=reference_points_input,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                **kwargs,
            )
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                assert reference_points.shape[-1] == 2
                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = (
                    tmp[..., :2] + inverse_sigmoid(reference_points[..., :2])
                )
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points
            )
        return output, reference_points


def build_maptr_decoder(
    num_layers: int = 2,
    embed_dims: int = 256,
    num_heads: int = 4,
    num_levels: int = 1,
    num_points: int = 4,
    im2col_step: int = 192,
    feedforward_channels: int = 512,
    ffn_dropout: float = 0.1,
    dropout: float = 0.1,
    return_intermediate: bool = True,
    operation_order: Tuple[str, ...] = (
        "self_attn",
        "norm",
        "cross_attn",
        "norm",
        "ffn",
        "norm",
    ),
    **kwargs,
) -> MapTRDecoder:
    """从 config 风格参数字典构建 MapTRDecoder。"""
    return MapTRDecoder(
        num_layers=num_layers,
        embed_dims=embed_dims,
        num_heads=num_heads,
        num_levels=num_levels,
        num_points=num_points,
        im2col_step=im2col_step,
        feedforward_channels=feedforward_channels,
        ffn_dropout=ffn_dropout,
        dropout=dropout,
        return_intermediate=return_intermediate,
        operation_order=operation_order,
    )
