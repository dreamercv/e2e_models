"""
MapTR 风格地图头：纯 PyTorch，不依赖 mmcv。
输入 BEV 特征 (B, C, H, W)，输出地图元素（折线）的类别与点集。
实现论文要点：instance_pts query、分类按 instance 聚合、点集 minmax 得 bbox、
reference 迭代细化、多顺序 GT + Chamfer 匹配、Chamfer + 方向 loss、多层 aux loss。
"""
from typing import List, Optional, Tuple, Dict, Any
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment

# MapTR decoder / bbox_coder（不依赖 mmcv）
try:
    from .decoder import MapTRDecoder
    from .bbox_coder import MapTRNMSFreeCoder
    _HAS_MAPTR = True
except Exception:
    _HAS_MAPTR = False

# 匹配与损失模块（几何/匈牙利/Chamfer/方向 loss）
from .assigner_loss import (
    normalize_2d_pts,
    denormalize_2d_pts,
    normalize_2d_bbox,
    denormalize_2d_bbox,
    chamfer_cost_matrix,
    MapTRAssigner,
    chamfer_loss,
    direction_cosine_loss,
    inverse_sigmoid,
)

__all__ = [
    "MapHead",
    "build_map_head",
    "MapTRNMSFreeCoder2D",
    "normalize_2d_pts",
    "denormalize_2d_pts",
    "normalize_2d_bbox",
    "denormalize_2d_bbox",
]


# ---------- pc_range: [xmin, ymin, xmax, ymax] 与 MapTR 一致 ----------
def _get_pc_range_from_bounds(bev_bounds):
    if bev_bounds is None:
        return [-80.0, -40.0, 120.0, 40.0]
    x0, x1 = bev_bounds[0][0], bev_bounds[0][1]
    y0, y1 = bev_bounds[1][0], bev_bounds[1][1]
    return [x0, y0, x1, y1]


# ---------- bbox_coder：与 config MapTRNMSFreeCoder 一致（后处理用） ----------
class MapTRNMSFreeCoder2D:
    """纯 PyTorch，无 mmcv。归一化 bbox (cx,cy,w,h) + scores -> 物理坐标 bbox，按 score topk。"""

    def __init__(self, pc_range: List[float], max_num: int = 50):
        self.pc_range = pc_range
        self.max_num = max_num

    def decode(
        self,
        bbox_preds: torch.Tensor,
        scores: torch.Tensor,
        score_threshold: float = 0.0,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """bbox_preds: (B, N, 4) 归一化 cx,cy,w,h；scores: (B, N). 返回 list of bboxes (xyxy), list of scores."""
        x0, y0, x1, y1 = self.pc_range[0], self.pc_range[1], self.pc_range[2], self.pc_range[3]
        patch_w = x1 - x0
        patch_h = y1 - y0
        B, N, _ = bbox_preds.shape
        cx, cy, w, h = bbox_preds[..., 0], bbox_preds[..., 1], bbox_preds[..., 2], bbox_preds[..., 3]
        cx = cx * patch_w + x0
        cy = cy * patch_h + y0
        w = w * patch_w
        h = h * patch_h
        x1_ = cx - w / 2.0
        y1_ = cy - h / 2.0
        x2_ = cx + w / 2.0
        y2_ = cy + h / 2.0
        bboxes = torch.stack([x1_, y1_, x2_, y2_], dim=-1)
        out_bboxes, out_scores = [], []
        for b in range(B):
            score_b = scores[b]
            bbox_b = bboxes[b]
            keep = score_b >= score_threshold
            score_b = score_b[keep]
            bbox_b = bbox_b[keep]
            if score_b.numel() > self.max_num:
                topk = torch.topk(score_b, self.max_num)
                score_b = topk.values
                bbox_b = bbox_b[topk.indices]
            out_bboxes.append(bbox_b)
            out_scores.append(score_b)
        return out_bboxes, out_scores


# ---------- LearnedPositionalEncoding2D（与 config positional_encoding 一致） ----------
class LearnedPositionalEncoding2D(nn.Module):
    """LearnedPositionalEncoding 2D 版，纯 PyTorch，用于 BEV 特征加位置编码。"""

    def __init__(self, num_feats: int, row_num_embed: int = 80, col_num_embed: int = 40):
        super().__init__()
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, B: int, H: int, W: int, device: torch.device, dtype: torch.dtype):
        rows = torch.arange(H, device=device)
        cols = torch.arange(W, device=device)
        y_emb = self.row_embed(rows)
        x_emb = self.col_embed(cols)
        pos_y = y_emb[:, None, :].expand(H, W, -1)
        pos_x = x_emb[None, :, :].expand(H, W, -1)
        pos = torch.cat([pos_x, pos_y], dim=-1)
        pos = pos.permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1).to(dtype)
        return pos


# ---------- MapHead ----------
class MapHead(nn.Module):
    def __init__(
        self,
        bev_feat_dim: int = 256,
        embed_dims: int = 256,
        num_vec: int = 100,
        num_pts_per_vec: int = 20,
        num_pts_per_gt_vec: int = 20,
        num_classes: int = 3,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        feedforward_dims: int = 1024,
        pc_range: Optional[List[float]] = None,
        bev_bounds=None,
        with_box_refine: bool = True,
        use_instance_pts: bool = True,
        dir_interval: int = 1,
        cls_weight: float = 1.0,
        reg_weight: float = 1.0,
        pts_weight: float = 1.0,
        loss_pts_src_weight: float = 1.0,
        loss_pts_dst_weight: float = 1.0,
        loss_dir_weight: float = 0.005,
        aux_loss_weight: float = 0.5,
        row_num_embed: int = 80,
        col_num_embed: int = 40,
        use_maptr_decoder: bool = False,
        maptr_decoder_num_layers: int = 2,
        maptr_num_heads: int = 4,
        maptr_im2col_step: int = 192,
        maptr_feedforward_channels: int = 512,
        maptr_num_levels: int = 1,
        maptr_num_points: int = 4,
        post_center_range: Optional[List[float]] = None,
        bbox_coder_max_num: int = 50,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.use_maptr_decoder = use_maptr_decoder
        self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec
        self.num_pts_per_gt_vec = num_pts_per_gt_vec
        self.num_classes = num_classes
        self.num_decoder_layers = num_decoder_layers
        self.with_box_refine = with_box_refine
        self.use_instance_pts = use_instance_pts
        self.dir_interval = dir_interval
        self.aux_loss_weight = aux_loss_weight
        self.loss_dir_weight = loss_dir_weight
        self.pc_range = pc_range or _get_pc_range_from_bounds(bev_bounds)
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed

        num_query = num_vec * num_pts_per_vec
        self.num_query = num_query

        if use_instance_pts:
            self.instance_embedding = nn.Embedding(num_vec, embed_dims * 2)
            self.pts_embedding = nn.Embedding(num_pts_per_vec, embed_dims * 2)
        else:
            self.query_embedding = nn.Embedding(num_query, embed_dims * 2)

        self.reference_points = nn.Linear(embed_dims, 2)
        self.bev_proj = nn.Conv2d(bev_feat_dim, embed_dims, 1)
        self.positional_encoding = LearnedPositionalEncoding2D(
            num_feats=embed_dims // 2,
            row_num_embed=row_num_embed,
            col_num_embed=col_num_embed,
        )

        if use_maptr_decoder and _HAS_MAPTR:
            self.decoder = MapTRDecoder(
                num_layers=maptr_decoder_num_layers,
                embed_dims=embed_dims,
                num_heads=maptr_num_heads,
                num_levels=maptr_num_levels,
                num_points=maptr_num_points,
                im2col_step=maptr_im2col_step,
                feedforward_channels=maptr_feedforward_channels,
                ffn_dropout=dropout,
                dropout=dropout,
                return_intermediate=True,
                operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
            )
        else:
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=embed_dims,
                nhead=num_heads,
                dim_feedforward=feedforward_dims,
                dropout=dropout,
                activation="relu",
                batch_first=True,
                norm_first=False,
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        self.code_size = 2
        _num_dec = maptr_decoder_num_layers if use_maptr_decoder else num_decoder_layers
        num_pred = _num_dec + 1 if with_box_refine else _num_dec
        self.reg_branches = nn.ModuleList()
        self.cls_branches = nn.ModuleList()
        for _ in range(num_pred):
            self.reg_branches.append(
                nn.Sequential(
                    nn.Linear(embed_dims, embed_dims),
                    nn.ReLU(),
                    nn.Linear(embed_dims, self.code_size),
                )
            )
            self.cls_branches.append(
                nn.Sequential(
                    nn.Linear(embed_dims, embed_dims),
                    nn.LayerNorm(embed_dims),
                    nn.ReLU(),
                    nn.Linear(embed_dims, num_classes),
                )
            )

        self.assigner = MapTRAssigner(
            cls_weight=cls_weight,
            reg_weight=reg_weight,
            pts_weight=pts_weight,
            pc_range=self.pc_range,
        )
        if use_maptr_decoder and _HAS_MAPTR:
            self.bbox_coder = MapTRNMSFreeCoder(
                pc_range=self.pc_range,
                post_center_range=post_center_range,
                max_num=bbox_coder_max_num,
                num_classes=num_classes,
            )
        else:
            self.bbox_coder = MapTRNMSFreeCoder2D(pc_range=self.pc_range, max_num=bbox_coder_max_num)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.uniform_(self.reference_points.weight, 0, 1)
        nn.init.zeros_(self.reference_points.bias)
        if self.use_instance_pts:
            nn.init.normal_(self.instance_embedding.weight, std=0.01)
            nn.init.normal_(self.pts_embedding.weight, std=0.01)
        else:
            nn.init.normal_(self.query_embedding.weight, std=0.01)
        for m in self.cls_branches:
            if isinstance(m[-1], nn.Linear):
                nn.init.constant_(m[-1].bias, -2.0)

    def _get_query_embed(self, bs: int, device: torch.device, dtype: torch.dtype):
        if self.use_instance_pts:
            pts_emb = self.pts_embedding.weight.unsqueeze(0)
            inst_emb = self.instance_embedding.weight.unsqueeze(1)
            query_embed = (pts_emb + inst_emb).flatten(0, 1).to(dtype)
        else:
            query_embed = self.query_embedding.weight.to(dtype)
        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
        query_pos, query = torch.split(query_embed, self.embed_dims, dim=-1)
        return query, query_pos

    @staticmethod
    def transform_box(pts: torch.Tensor, num_vec: int, num_pts: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """pts: (B, num_vec*num_pts, 2) -> bbox (B, num_vec, 4) 归一化 cx,cy,w,h，pts_reshape (B, num_vec, num_pts, 2)."""
        B = pts.shape[0]
        pts_reshape = pts.view(B, num_vec, num_pts, 2)
        pts_x = pts_reshape[..., 0]
        pts_y = pts_reshape[..., 1]
        x1 = pts_x.min(dim=2, keepdim=True)[0]
        x2 = pts_x.max(dim=2, keepdim=True)[0]
        y1 = pts_y.min(dim=2, keepdim=True)[0]
        y2 = pts_y.max(dim=2, keepdim=True)[0]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = (x2 - x1).clamp(min=1e-6)
        h = (y2 - y1).clamp(min=1e-6)
        bbox = torch.cat([cx, cy, w, h], dim=2).squeeze(2)
        return bbox, pts_reshape

    def forward(
        self,
        bev_feature: torch.Tensor,
        metas: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        bev_feature: (B, C, H, W) 时序对齐后的 BEV 特征（如 backbone 最后一帧输出）
        Returns: dict with map_cls_scores, map_pts_preds (normalized), map_bbox_preds (normalized), init_ref, inter_refs (for aux)
        """
        B, C, H, W = bev_feature.shape
        device = bev_feature.device
        dtype = bev_feature.dtype

        bev_proj = self.bev_proj(bev_feature) # 2 256 200 80
        pos = self.positional_encoding(B, self.row_num_embed, self.col_num_embed, device, dtype)
        if (H, W) != (self.row_num_embed, self.col_num_embed):
            pos = F.interpolate(pos, size=(H, W), mode="bilinear", align_corners=False)
        memory = (bev_proj + pos).flatten(2).permute(0, 2, 1)

        query, query_pos = self._get_query_embed(B, device, dtype)
        ref = self.reference_points(query_pos).sigmoid()
        init_ref = ref

        if self.use_maptr_decoder and _HAS_MAPTR:
            num_dec_layers = self.decoder.num_layers
            query_t = query.permute(1, 0, 2)
            query_pos_t = query_pos.permute(1, 0, 2)
            memory_t = memory.permute(1, 0, 2)
            spatial_shapes = memory.new_tensor([[H, W]], dtype=torch.long)
            level_start_index = memory.new_zeros(1, dtype=torch.long)
            inter_states, inter_references = self.decoder(
                query=query_t,
                value=memory_t,
                query_pos=query_pos_t,
                reference_points=ref,
                reg_branches=self.reg_branches,
                spatial_shapes=memory.new_tensor([[H, W]], dtype=torch.long, device=device),
                level_start_index=level_start_index,
            )
            all_cls = []
            all_bbox = []
            all_pts = []
            inter_refs = []
            for lid in range(num_dec_layers):
                output = inter_states[lid].permute(1, 0, 2)
                ref_l = inter_references[lid]
                bbox, pts_reshape = self.transform_box(ref_l, self.num_vec, self.num_pts_per_vec)
                cls_out = self.cls_branches[lid](
                    output.view(B, self.num_vec, self.num_pts_per_vec, -1).mean(2)
                )
                all_cls.append(cls_out)
                all_bbox.append(bbox)
                all_pts.append(pts_reshape)
                inter_refs.append(ref_l)
            output = inter_states[-1].permute(1, 0, 2)
            last_ref = inter_references[-1]
            if self.with_box_refine:
                delta = self.reg_branches[num_dec_layers](output)
                last_ref = (inverse_sigmoid(last_ref) + delta).sigmoid()
            bbox_final, pts_final = self.transform_box(last_ref, self.num_vec, self.num_pts_per_vec)
            cls_final = self.cls_branches[-1](output.view(B, self.num_vec, self.num_pts_per_vec, -1).mean(2))
            all_cls.append(cls_final)
            all_bbox.append(bbox_final)
            all_pts.append(pts_final)
        else:# 纯decoder (self-attention)
            output = query
            inter_refs = []
            all_cls = []
            all_bbox = []
            all_pts = []

            for lid in range(self.num_decoder_layers):
                tgt = output + query_pos
                mem = memory
                output = self.decoder.layers[lid](
                    tgt,
                    mem,
                    tgt_mask=None,
                    memory_mask=None,
                    tgt_key_padding_mask=None,
                    memory_key_padding_mask=None,
                )
                output = output + query

                ref_in = ref
                if self.with_box_refine:
                    delta = self.reg_branches[lid](output)
                    ref = (inverse_sigmoid(ref_in) + delta).sigmoid()
                    ref = ref.detach()
                else:
                    delta = self.reg_branches[lid](output)
                    ref = delta.sigmoid()

                bbox, pts_reshape = self.transform_box(ref, self.num_vec, self.num_pts_per_vec)
                cls_out = self.cls_branches[lid](
                    output.view(B, self.num_vec, self.num_pts_per_vec, -1).mean(2)
                )
                all_cls.append(cls_out)
                all_bbox.append(bbox)
                all_pts.append(pts_reshape)
                inter_refs.append(ref)

            last_ref = inter_refs[-1] if inter_refs else init_ref
            if self.with_box_refine:
                delta = self.reg_branches[self.num_decoder_layers](output)
                last_ref = (inverse_sigmoid(last_ref) + delta).sigmoid()
            bbox_final, pts_final = self.transform_box(last_ref, self.num_vec, self.num_pts_per_vec)
            cls_final = self.cls_branches[-1](output.view(B, self.num_vec, self.num_pts_per_vec, -1).mean(2))
            all_cls.append(cls_final)
            all_bbox.append(bbox_final)
            all_pts.append(pts_final)

        return {
            "map_cls_scores": all_cls,
            "map_bbox_preds": all_bbox,
            "map_pts_preds": all_pts,
            "init_ref": init_ref,
            "inter_refs": inter_refs,
        }

    def loss(
        self,
        pred_dict: Dict[str, Any],
        gt_bboxes_list: List[Optional[torch.Tensor]],
        gt_labels_list: List[Optional[torch.Tensor]],
        gt_pts_list: List[Optional[torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """gt_bboxes_list / gt_labels_list: list length B; gt_pts_list: list of (N_gt, num_orders, num_pts, 2) or (N_gt, num_pts, 2)."""
        all_cls = pred_dict["map_cls_scores"]
        all_bbox = pred_dict["map_bbox_preds"]
        all_pts = pred_dict["map_pts_preds"]
        num_layers = len(all_cls)
        device = all_cls[0].device

        # 将 GT 移到与预测相同的 device，并填充缺失项
        gt_bboxes_list = [
            (g.to(device) if g is not None else all_cls[0].new_zeros(0, 4))
            for g in gt_bboxes_list
        ]
        gt_labels_list = [
            (g.to(device) if g is not None else all_cls[0].new_zeros(0, dtype=torch.long))
            for g in gt_labels_list
        ]
        gt_pts_list = [
            (g.to(device) if g is not None else all_cls[0].new_zeros(0, 1, self.num_pts_per_gt_vec, 2))
            for g in gt_pts_list
        ]

        for i, g in enumerate(gt_pts_list):
            if g.dim() == 3:
                gt_pts_list[i] = g.unsqueeze(1)

        assigned_gt_inds, order_index = self.assigner(
            all_bbox[-1],
            all_cls[-1],
            all_pts[-1],
            gt_bboxes_list,
            gt_labels_list,
            gt_pts_list,
        )

        losses = {}
        num_pos = 0
        for b in range(all_cls[0].shape[0]):
            num_pos += (assigned_gt_inds[b] > 0).sum().item()
        num_pos = max(num_pos, 1)

        for layer_idx in range(num_layers):
            suffix = "" if layer_idx == num_layers - 1 else f"_d{layer_idx}"
            w = 1.0 if layer_idx == num_layers - 1 else self.aux_loss_weight
            cls_scores = all_cls[layer_idx]
            bbox_preds = all_bbox[layer_idx]
            pts_preds = all_pts[layer_idx]

            loss_cls = self._loss_cls(cls_scores, assigned_gt_inds, gt_labels_list, num_pos)
            loss_bbox, loss_pts, loss_dir = self._loss_single(
                bbox_preds,
                pts_preds,
                assigned_gt_inds,
                order_index,
                gt_bboxes_list,
                gt_labels_list,
                gt_pts_list,
                num_pos,
            )
            losses[f"loss_map_cls{suffix}"] = loss_cls * w
            losses[f"loss_map_bbox{suffix}"] = loss_bbox * w
            losses[f"loss_map_pts{suffix}"] = loss_pts * w
            losses[f"loss_map_dir{suffix}"] = loss_dir * w

        return losses

    def _loss_cls(
        self,
        cls_scores: torch.Tensor,
        assigned_gt_inds: torch.Tensor,
        gt_labels_list: List[torch.Tensor],
        num_pos: float,
    ) -> torch.Tensor:
        """FocalLoss，与 config loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0) 一致。"""
        B, N, C = cls_scores.shape
        target = cls_scores.new_zeros((B, N, C))
        for b in range(B):
            pos = assigned_gt_inds[b] > 0
            if pos.any():
                idx = assigned_gt_inds[b][pos] - 1
                labels_b = gt_labels_list[b][idx]
                target[b, pos, labels_b] = 1.0
        prob = cls_scores.sigmoid()
        gamma = 2.0
        alpha = 0.25
        eps = 1e-6
        p_t = prob * target + (1 - prob) * (1 - target)
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        bce = -(
            target * torch.log(prob.clamp(min=eps))
            + (1 - target) * torch.log((1 - prob).clamp(min=eps))
        )
        loss = (alpha_t * (1 - p_t) ** gamma * bce).sum() / max(num_pos, 1.0)
        return loss * 2.0

    def _loss_single(
        self,
        bbox_preds: torch.Tensor,
        pts_preds: torch.Tensor,
        assigned_gt_inds: torch.Tensor,
        order_index: Optional[torch.Tensor],
        gt_bboxes_list: List[torch.Tensor],
        gt_labels_list: List[torch.Tensor],
        gt_pts_list: List[torch.Tensor],
        num_pos: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, _ = bbox_preds.shape
        norm_bbox_preds = bbox_preds
        norm_pts_preds = pts_preds

        bbox_targets = bbox_preds.new_zeros(B, N, 4)
        pts_targets = bbox_preds.new_zeros(B, N, self.num_pts_per_gt_vec, 2)
        pts_weights = bbox_preds.new_zeros(B, N, self.num_pts_per_gt_vec, 1)
        valid = bbox_preds.new_zeros(B, N, dtype=torch.bool)

        for b in range(B):
            pos = assigned_gt_inds[b] > 0
            if not pos.any():
                continue
            idx = assigned_gt_inds[b][pos] - 1
            bbox_targets[b][pos] = normalize_2d_bbox(gt_bboxes_list[b][idx], self.pc_range)
            gt_pts_b = gt_pts_list[b][idx]
            if order_index is not None:
                o = order_index[b][pos]
                pts_t = gt_pts_b[torch.arange(len(idx), device=gt_pts_b.device), o]
            else:
                pts_t = gt_pts_b[:, 0]
            pts_t = pts_t.to(pts_preds.device)
            if pts_t.shape[1] != self.num_pts_per_gt_vec:
                pts_t = F.interpolate(
                    pts_t.permute(0, 2, 1),
                    size=self.num_pts_per_gt_vec,
                    mode="linear",
                    align_corners=True,
                ).permute(0, 2, 1)
            pts_targets[b][pos] = normalize_2d_pts(pts_t, self.pc_range)
            pts_weights[b][pos] = 1.0
            valid[b][pos] = True

        pos_flat = valid.reshape(-1)
        if pos_flat.sum() == 0:
            return (
                bbox_preds.new_tensor(0.0),
                pts_preds.new_tensor(0.0),
                pts_preds.new_tensor(0.0),
            )

        bbox_pred_flat = norm_bbox_preds.reshape(-1, 4)[pos_flat]
        bbox_tgt_flat = bbox_targets.reshape(-1, 4)[pos_flat]
        loss_bbox = F.l1_loss(bbox_pred_flat, bbox_tgt_flat, reduction="sum") / max(num_pos, 1)

        pts_pred_flat = norm_pts_preds.reshape(-1, self.num_pts_per_vec, 2)[pos_flat]
        pts_tgt_flat = pts_targets.reshape(-1, self.num_pts_per_gt_vec, 2)[pos_flat]
        if pts_pred_flat.shape[1] != pts_tgt_flat.shape[1]:
            pts_pred_flat = F.interpolate(
                pts_pred_flat.permute(0, 2, 1),
                size=pts_tgt_flat.shape[1],
                mode="linear",
                align_corners=True,
            ).permute(0, 2, 1)
        diff = torch.abs(pts_pred_flat - pts_tgt_flat)
        loss_pts = (diff.sum() / max(num_pos, 1.0)) * 5.0
        w_dir = pts_weights.reshape(-1, self.num_pts_per_gt_vec, 1)[pos_flat]
        pred_denorm = denormalize_2d_pts(pts_pred_flat, self.pc_range)
        tgt_denorm = denormalize_2d_pts(pts_tgt_flat, self.pc_range)
        loss_dir = direction_cosine_loss(
            pred_denorm,
            tgt_denorm,
            weight=w_dir,
            dir_interval=self.dir_interval,
            avg_factor=num_pos,
        ) * self.loss_dir_weight

        return loss_bbox, loss_pts, loss_dir

    def decode(
        self,
        pred_dict: Dict[str, Any],
        score_threshold: float = 0.5,
    ) -> List[Dict[str, torch.Tensor]]:
        """Decode to list of polylines per sample；含 bbox_coder 后处理的 bboxes（xyxy）。"""
        if _HAS_MAPTR and isinstance(self.bbox_coder, MapTRNMSFreeCoder):
            preds_dicts = {
                "all_cls_scores": pred_dict["map_cls_scores"],
                "all_bbox_preds": pred_dict["map_bbox_preds"],
                "all_pts_preds": pred_dict["map_pts_preds"],
            }
            return self.bbox_coder.decode(preds_dicts, score_threshold=score_threshold)
        cls_scores = pred_dict["map_cls_scores"][-1]
        bbox_preds = pred_dict["map_bbox_preds"][-1]
        pts_preds = pred_dict["map_pts_preds"][-1]
        B = cls_scores.shape[0]
        probs = cls_scores.sigmoid()
        max_scores, labels = probs.max(dim=-1)
        list_bboxes, list_scores = self.bbox_coder.decode(
            bbox_preds, max_scores, score_threshold=score_threshold
        )
        out = []
        for b in range(B):
            keep = max_scores[b] >= score_threshold
            pts_b = denormalize_2d_pts(pts_preds[b][keep], self.pc_range)
            out.append({
                "labels": labels[b][keep],
                "scores": max_scores[b][keep],
                "pts": pts_b,
                "bboxes": list_bboxes[b],
            })
        return out


# def build_map_head(
#     bev_feat_dim: int = 256,
#     embed_dims: int = 256,
#     num_vec: int = 100,
#     num_pts_per_vec: int = 20,
#     num_pts_per_gt_vec: int = 20,
#     num_classes: int = 3,
#     num_decoder_layers: int = 6,
#     num_heads: int = 8,
#     dropout: float = 0.1,
#     feedforward_dims: int = 1024,
#     pc_range: Optional[List[float]] = None,
#     bev_bounds=None,
#     row_num_embed: int = 80,
#     col_num_embed: int = 40,
#     **kwargs,
# ) -> MapHead:
#     """通用 MapHead 构建函数。"""
#     return MapHead(
#         bev_feat_dim=bev_feat_dim,
#         embed_dims=embed_dims,
#         num_vec=num_vec,
#         num_pts_per_vec=num_pts_per_vec,
#         num_pts_per_gt_vec=num_pts_per_gt_vec,
#         num_classes=num_classes,
#         num_decoder_layers=num_decoder_layers,
#         num_heads=num_heads,
#         dropout=dropout,
#         feedforward_dims=feedforward_dims,
#         pc_range=pc_range,
#         bev_bounds=bev_bounds,
#         row_num_embed=row_num_embed,
#         col_num_embed=col_num_embed,
#         **kwargs,
#     )


def build_map_head(
    bev_feat_dim: int = 256,
    embed_dims: int = 256,
    num_vec: int = 50,
    num_pts_per_vec: int = 20,
    num_pts_per_gt_vec: int = 20,
    num_classes: int = 3,
    pc_range: Optional[List[float]] = None,
    bev_bounds=None,
    bev_h: int = 80,
    bev_w: int = 40,
    decoder_num_layers: int = 2,
    decoder_num_heads: int = 4,
    decoder_im2col_step: int = 192,
    decoder_feedforward_channels: int = 512,
    decoder_ffn_dropout: float = 0.1,
    bbox_coder_post_center_range: Optional[List[float]] = None,
    bbox_coder_max_num: int = 50,
    voxel_size: Optional[List[float]] = None,
    **kwargs,
) -> MapHead:
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