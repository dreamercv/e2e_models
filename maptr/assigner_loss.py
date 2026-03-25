"""
MapTR 匹配与损失模块（匈牙利 + Chamfer/CD + 方向 loss），纯 PyTorch，不依赖 mmcv。
从原来的 `map_head.py` 中拆出，便于维护与复用。
"""
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

__all__ = [
    "normalize_2d_pts",
    "denormalize_2d_pts",
    "normalize_2d_bbox",
    "denormalize_2d_bbox",
    "chamfer_cost_matrix",
    "MapTRAssigner",
    "chamfer_loss",
    "direction_cosine_loss",
]


# ---------- 坐标归一化 / 反归一化 ----------
def normalize_2d_pts(pts: torch.Tensor, pc_range: List[float]) -> torch.Tensor:
    """将 2D 点从物理坐标归一化到 [0,1] 区间。

    Args:
        pts: (..., 2) 物理坐标 (x, y)
        pc_range: [xmin, ymin, xmax, ymax]
    """
    x0, y0, x1, y1 = pc_range[0], pc_range[1], pc_range[2], pc_range[3]
    patch_w = x1 - x0
    patch_h = y1 - y0
    out = pts.clone()
    out[..., 0:1] = (pts[..., 0:1] - x0) / (patch_w + 1e-6)
    out[..., 1:2] = (pts[..., 1:2] - y0) / (patch_h + 1e-6)
    return out


def denormalize_2d_pts(pts: torch.Tensor, pc_range: List[float]) -> torch.Tensor:
    """将归一化到 [0,1] 的 2D 点还原到物理坐标。"""
    x0, y0, x1, y1 = pc_range[0], pc_range[1], pc_range[2], pc_range[3]
    out = pts.clone()
    out[..., 0:1] = pts[..., 0:1] * (x1 - x0) + x0
    out[..., 1:2] = pts[..., 1:2] * (y1 - y0) + y0
    return out


def normalize_2d_bbox(bboxes: torch.Tensor, pc_range: List[float]) -> torch.Tensor:
    """xyxy bbox -> 归一化 cx,cy,w,h（[0,1]），与官方 config 一致。

    Args:
        bboxes: (..., 4) [x1,y1,x2,y2]
        pc_range: [xmin, ymin, xmax, ymax]
    """
    x0, y0, x1, y1 = pc_range[0], pc_range[1], pc_range[2], pc_range[3]
    patch_w = x1 - x0
    patch_h = y1 - y0
    x1_, y1_, x2_, y2_ = (
        bboxes[..., 0:1],
        bboxes[..., 1:2],
        bboxes[..., 2:3],
        bboxes[..., 3:4],
    )
    cx = (x1_ + x2_) / 2.0
    cy = (y1_ + y2_) / 2.0
    w = (x2_ - x1_).clamp(min=1e-6)
    h = (y2_ - y1_).clamp(min=1e-6)
    cx = (cx - x0) / (patch_w + 1e-6)
    cy = (cy - y0) / (patch_h + 1e-6)
    w = w / (patch_w + 1e-6)
    h = h / (patch_h + 1e-6)
    return torch.cat([cx, cy, w, h], dim=-1)


def denormalize_2d_bbox(bboxes: torch.Tensor, pc_range: List[float]) -> torch.Tensor:
    """归一化 cx,cy,w,h [0,1] -> 物理坐标 xyxy。"""
    x0, y0, x1, y1 = pc_range[0], pc_range[1], pc_range[2], pc_range[3]
    patch_w = x1 - x0
    patch_h = y1 - y0
    cx, cy, w, h = (
        bboxes[..., 0:1],
        bboxes[..., 1:2],
        bboxes[..., 2:3],
        bboxes[..., 3:4],
    )
    cx = cx * patch_w + x0
    cy = cy * patch_h + y0
    w = w * patch_w
    h = h * patch_h
    x1_ = cx - w / 2.0
    y1_ = cy - h / 2.0
    x2_ = cx + w / 2.0
    y2_ = cy + h / 2.0
    return torch.cat([x1_, y1_, x2_, y2_], dim=-1)


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Sigmoid 的反函数，用于 reference refinement。"""
    x = x.clamp(min=eps, max=1 - eps)
    return torch.log(x / (1 - x))


# ---------- Chamfer cost: (Nq, M, 2), (Ngt, N, 2) -> (Nq, Ngt) ----------
def chamfer_cost_matrix(
    pred_pts: torch.Tensor,
    gt_pts: torch.Tensor,
    loss_src_weight: float = 1.0,
    loss_dst_weight: float = 1.0,
) -> torch.Tensor:
    """用于匈牙利匹配的 Chamfer 距离代价矩阵。"""
    Nq, M, _ = pred_pts.shape
    Ngt, N, _ = gt_pts.shape
    pred_flat = pred_pts.unsqueeze(1).expand(-1, Ngt, -1, -1).reshape(Nq * Ngt, M, 2)
    gt_flat = gt_pts.unsqueeze(0).expand(Nq, -1, -1, -1).reshape(Nq * Ngt, N, 2)
    dist = torch.cdist(pred_flat, gt_flat, p=2)
    src2dst = dist.min(dim=2)[0].mean(dim=1)
    dst2src = dist.min(dim=1)[0].mean(dim=1)
    cost = (src2dst * loss_src_weight + dst2src * loss_dst_weight).view(Nq, Ngt)
    return cost


# ---------- Assigner: Hungarian（FocalCost + BBoxL1 + OrderedPtsL1Cost） ----------
class MapTRAssigner(nn.Module):
    """MapTR 的匈牙利匹配器，对应官方 `MapTRAssigner`。"""

    def __init__(
        self,
        cls_weight: float = 2.0,
        reg_weight: float = 0.0,
        pts_weight: float = 5.0,
        pc_range: Optional[List[float]] = None,
    ):
        super().__init__()
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.pts_weight = pts_weight
        self.pc_range = pc_range or [-80.0, -40.0, 120.0, 40.0]

    def forward(
        self,
        bbox_pred: torch.Tensor,
        cls_pred: torch.Tensor,
        pts_pred: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_pts: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        bbox_pred: (bs, num_vec, 4) normalized [0,1] cx,cy,w,h or x1,y1,x2,y2
        cls_pred: (bs, num_vec, num_cls)
        pts_pred: (bs, num_vec, num_pts, 2) normalized
        gt_bboxes: list of (N_gt, 4) x1,y1,x2,y2 in pc_range
        gt_labels: list of (N_gt,)
        gt_pts: list of (N_gt, num_orders, num_pts, 2) or (N_gt, num_pts, 2) in pc_range
        Returns:
            assigned_gt_inds: (bs, num_vec)  0=bg, 1-based gt index
            order_index: (bs, num_vec) or None，记录多顺序 GT 的选择
        """
        bs, num_vec, _ = bbox_pred.shape
        device = bbox_pred.device
        assigned_gt_inds = bbox_pred.new_zeros(bs, num_vec, dtype=torch.long)
        order_index = None
        has_multi_order = False

        for b in range(bs):
            gt_b = gt_bboxes[b]
            lb_b = gt_labels[b]
            pts_b = gt_pts[b]
            if gt_b is None or gt_b.numel() == 0:
                continue
            num_gts = gt_b.shape[0]
            bbox_p = bbox_pred[b]
            cls_p = cls_pred[b]
            pts_p = pts_pred[b]

            if pts_b.dim() == 3:
                pts_b = pts_b.unsqueeze(1)
            _, num_orders, num_pts_gt, _ = pts_b.shape
            has_multi_order = num_orders > 1

            norm_gt_bboxes = normalize_2d_bbox(gt_b, self.pc_range)
            norm_gt_pts = normalize_2d_pts(pts_b, self.pc_range)

            num_pts_pred = pts_p.shape[1]
            if num_pts_pred != num_pts_gt:
                pts_p = F.interpolate(
                    pts_p.permute(0, 2, 1),
                    size=num_pts_gt,
                    mode="linear",
                    align_corners=True,
                ).permute(0, 2, 1)

            cls_cost = self._cls_cost(cls_p, lb_b)
            reg_cost = torch.cdist(
                bbox_p[:, :4].float().reshape(num_vec, -1),
                norm_gt_bboxes[:, :4].float().reshape(num_gts, -1),
                p=1,
            )
            pts_cost_all = []
            for o in range(num_orders):
                gt_o = norm_gt_pts[:, o]
                pred_flat = pts_p.reshape(num_vec, -1)
                gt_flat = gt_o.reshape(num_gts, -1)
                c = torch.cdist(pred_flat, gt_flat, p=1)
                pts_cost_all.append(c)
            pts_cost_stack = torch.stack(pts_cost_all, dim=2)
            pts_cost, order_idx = pts_cost_stack.min(dim=2)
            cost = (
                self.cls_weight * cls_cost
                + self.reg_weight * reg_cost
                + self.pts_weight * pts_cost
            )

            cost_np = cost.detach().cpu().numpy()
            cost_np = np.nan_to_num(cost_np, nan=1e8, posinf=1e8, neginf=1e8)
            row_idx, col_idx = linear_sum_assignment(cost_np)
            inds = assigned_gt_inds[b]
            inds[:] = 0
            for r, c in zip(row_idx, col_idx):
                inds[r] = c + 1
            if has_multi_order:
                if order_index is None:
                    order_index = bbox_pred.new_zeros(
                        bs, num_vec, dtype=torch.long
                    )
                order_index[b, row_idx] = order_idx[row_idx, col_idx]

        return assigned_gt_inds, order_index

    def _cls_cost(self, cls_pred: torch.Tensor, gt_labels: torch.Tensor) -> torch.Tensor:
        """FocalLossCost：与 config cls_cost=dict(type='FocalLossCost', weight=2.0) 一致。"""
        num_vec, num_cls = cls_pred.shape
        num_gts = gt_labels.shape[0]
        prob = cls_pred.sigmoid()
        p = prob[:, gt_labels]
        gamma = 2.0
        alpha = 0.25
        eps = 1e-6
        p = p.clamp(min=eps, max=1 - eps)
        cost = -alpha * (1.0 - p) ** gamma * torch.log(p)
        return cost


# ---------- Chamfer loss & Direction loss ----------
def chamfer_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    avg_factor: float = 1.0,
    loss_src_weight: float = 1.0,
    loss_dst_weight: float = 1.0,
) -> torch.Tensor:
    """标准 Chamfer 距离损失，pred/target: (N, num_pts, 2)。"""
    dist = torch.cdist(pred, target, p=2)
    src2dst = dist.min(dim=2)[0].mean(dim=1)
    dst2src = dist.min(dim=1)[0].mean(dim=1)
    loss = (src2dst * loss_src_weight + dst2src * loss_dst_weight)
    if weight is not None:
        loss = loss * weight.squeeze(-1).mean(-1)
    return loss.sum() / max(avg_factor, 1e-6)


def direction_cosine_loss(
    pred_pts: torch.Tensor,
    target_pts: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    dir_interval: int = 1,
    avg_factor: float = 1.0,
) -> torch.Tensor:
    """方向 cos 损失，用于约束线段方向一致性。"""
    pred_dir = pred_pts[:, dir_interval:] - pred_pts[:, :-dir_interval]
    tgt_dir = target_pts[:, dir_interval:] - target_pts[:, :-dir_interval]
    pred_norm = F.normalize(pred_dir.reshape(-1, 2), dim=-1)
    tgt_norm = F.normalize(tgt_dir.reshape(-1, 2), dim=-1)
    cos = (pred_norm * tgt_norm).sum(dim=-1).clamp(-1, 1)
    loss = (1 - cos).reshape(pred_pts.shape[0], -1)
    if weight is not None:
        w = weight[:, dir_interval:].reshape(loss.shape[0], -1)
        loss = loss * w
    return loss.sum() / max(avg_factor, 1e-6)