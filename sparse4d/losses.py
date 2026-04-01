"""
Sparse4D 检测分支用到的 loss 实现，参考官方 Sparse4D 中
`mmdet3d_plugin/models/detection3d/losses.py` 与 mmdet 的 Focal/L1/CE/GaussianFocal 接口，
做了纯 PyTorch 版本，便于在当前工程中直接使用。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .box3d import X, Y, Z,W, L, H, SIN_YAW, COS_YAW, CNS, YNS
from .box3d import decode_box
__all__ = [
    "FocalLoss",
    "L1Loss",
    "CrossEntropyLoss",
    "GaussianFocalLoss",
    "SparseBox3DLoss",
]


def reduce_mean(x: torch.Tensor) -> torch.Tensor:
    """简单的 reduce_mean，用于 avg_factor 计算。"""
    return x.sum() / max(x.numel(), 1)


class FocalLoss(nn.Module):
    """
    多类 Focal Loss，接口对齐 mmdet:
    - pred: (N, C) logits
    - target: (N,) long，类别 id；负值(除 ignore_index)表示背景
    - avg_factor: 用于归一化
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
        ignore_index: int = -1,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, pred, target, avg_factor=None):
        # pred: (N, C), target: (N,)
        # keep background negatives (target < 0) for dense cls supervision.
        valid = target != self.ignore_index
        if valid.sum() == 0:
            return pred.new_tensor(0.0)

        pred = pred[valid]  # (N', C)
        target = target[valid]  # (N',)
        num_cls = pred.shape[-1]
        target_one_hot = pred.new_zeros(pred.shape)  # (N', C)
        pos_mask = target >= 0
        if pos_mask.any():
            pos_target = target[pos_mask].clamp(0, num_cls - 1)
            target_one_hot[pos_mask] = F.one_hot(pos_target, num_cls).float()
        pred_sigmoid = pred.sigmoid()

        # p_t
        pt = pred_sigmoid * target_one_hot + (1 - pred_sigmoid) * (1 - target_one_hot)

        # alpha_t：对正类用 alpha，对负类用 1-alpha（不再对类做 sum 放大）
        if self.alpha is not None:
            alpha_factor = (
                self.alpha * target_one_hot + (1 - self.alpha) * (1 - target_one_hot)
            )
        else:
            alpha_factor = 1.0

        modulating_factor = (1 - pt).pow(self.gamma)

        # BCE with logits
        bce_loss = F.binary_cross_entropy_with_logits(
            pred, target_one_hot, reduction="none"
        )  # (N', C)

        loss = bce_loss * alpha_factor * modulating_factor
        loss = loss.sum(dim=-1)  # (N',)

        if avg_factor is not None:
            loss = loss.sum() / max(float(avg_factor), 1.0)
        else:
            if self.reduction == "mean":
                loss = loss.mean()
            else:
                loss = loss.sum()


        return loss


class L1Loss(nn.Module):
    """
    L1 Loss，接口对齐 mmdet:
    - pred, target: 任意形状
    - weight: 同 shape 或可 broadcast
    - avg_factor: 归一化因子
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        assert reduction in ("sum", "mean")
        self.reduction = reduction

    def forward(self, pred, target, weight=None, avg_factor=None):
        loss = torch.abs(pred - target)
        if weight is not None:
            loss = loss * weight

        if avg_factor is not None:
            loss = loss.sum() / max(float(avg_factor), 1.0)
        else:
            if self.reduction == "sum":
                loss = loss.sum()
            else:
                loss = loss.mean()

        return loss


class CrossEntropyLoss(nn.Module):
    """
    极简版 CrossEntropyLoss，当前用于：
    - use_sigmoid=True 的二值情况（例如 centerness），即 BCEWithLogits。
    """

    def __init__(self, use_sigmoid: bool = True, reduction: str = "mean"):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction

    def forward(self, pred, target, avg_factor=None, weight=None):
        if self.use_sigmoid:
            # pred: logits, target: float in [0,1]
            loss = F.binary_cross_entropy_with_logits(
                pred, target, reduction="none"
            )
        else:
            # 多类 softmax CE（目前工程里暂时用不到）
            loss = F.cross_entropy(pred, target.long(), reduction="none")

        if weight is not None:
            loss = loss * weight


        if avg_factor is not None:
            loss = loss.sum() / max(float(avg_factor), 1.0)
        else:
            if self.reduction == "sum":
                loss = loss.sum()
            else:
                loss = loss.mean()

        return loss


class GaussianFocalLoss(nn.Module):
    """
    GaussianFocalLoss 近似实现，接口对齐 mmdet:
    - pred: logits
    - target: [0,1]，通常是高斯分布或 0/1
    这里使用类似 CenterNet / mmdet 的实现。
    """

    def __init__(self, alpha: float = 2.0, gamma: float = 4.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target, avg_factor=None, weight=None):
        # pred: logits, target: [0,1]
        pred_sigmoid = pred.sigmoid()
        pos_inds = target.eq(1).float()
        neg_inds = target.lt(1).float()

        pos_loss = -torch.log(pred_sigmoid + 1e-12) * torch.pow(
            1 - pred_sigmoid, self.alpha
        ) * pos_inds

        neg_weight = torch.pow(1 - target, self.gamma)
        neg_loss = -torch.log(1 - pred_sigmoid + 1e-12) * torch.pow(
            pred_sigmoid, self.alpha
        ) * neg_weight * neg_inds

        loss = pos_loss + neg_loss

        if weight is not None:
            loss = loss * weight


        if avg_factor is not None:
            loss = loss.sum() / max(float(avg_factor), 1.0)
        else:
            if self.reduction == "sum":
                loss = loss.sum()
            else:
                loss = loss.mean()

        return loss

class SparseBox3DLoss(nn.Module):
    """
    SparseBox3DLoss：参考官方实现的包装器。
    - 内部使用 L1Loss 作为 box_loss
    - 可选 CrossEntropyLoss / GaussianFocalLoss 作为 centerness / yawness
    - 支持 barrier 等类别 yaw 反向等价（cls_allow_reverse）
    """

    def __init__(
        self,
        reg_weights=None,
        loss_centerness: bool = False,
        loss_yawness: bool = False,
        loss_giou: bool = False,
        cls_allow_reverse=None,
    ):
        super().__init__()
        self.reg_weights = reg_weights if reg_weights is not None else [1.0] * 11
        self.loss_box = L1Loss(reduction="mean")
        self.loss_cns = CrossEntropyLoss(use_sigmoid=True, reduction="mean") if loss_centerness else None
        self.loss_yns = GaussianFocalLoss(reduction="mean") if loss_yawness else None
        self.loss_giou = GIoU() if loss_giou else None
        self.cls_allow_reverse = cls_allow_reverse

    def forward(
        self,
        box,
        box_target,
        weight=None,
        avg_factor=None,
        suffix="",
        quality=None,
        cls_target=None,
    ):
        # yaw 反向处理（如 barrier 类别，正反方向等价）
        if self.cls_allow_reverse is not None and cls_target is not None:
            cos_sim = F.cosine_similarity(
                box_target[..., [SIN_YAW, COS_YAW]],
                box[..., [SIN_YAW, COS_YAW]],
                dim=-1,
            )
            if_reverse = cos_sim < 0
            if_reverse = (
                torch.isin(
                    cls_target, cls_target.new_tensor(self.cls_allow_reverse)
                )
                & if_reverse
            )
            box_target[..., [SIN_YAW, COS_YAW]] = torch.where(
                if_reverse[..., None],
                -box_target[..., [SIN_YAW, COS_YAW]],
                box_target[..., [SIN_YAW, COS_YAW]],
            )

        output = {}

        # 盒子回归：L1 + reg_weights
        ndim = min(len(self.reg_weights), box.shape[-1], box_target.shape[-1])
        box = box[..., :ndim]
        box_target = box_target[..., :ndim]
        weight = torch.where(box_target.isnan(), weight.new_tensor(0.0), weight)
        weight = torch.where(box.isnan(), weight.new_tensor(0.0), weight)
        box = torch.where(box.isnan(), box.new_tensor(0.0), box)
        box_target = torch.where(box_target.isnan(), box_target.new_tensor(0.0), box_target)
        
        if weight is not None:
            w = weight
            if w.dim() == box.dim() - 1:
                # (B, N) -> (B, N, 1)
                w = w.unsqueeze(-1)
            w = w[..., :ndim]
        else:
            w = box.new_ones(box.shape)

        reg_w = box.new_tensor(self.reg_weights[:ndim])
        w = w * reg_w  # (B, N, D)
        # sub_num = reg_w = box.new_tensor(self.reg_weights[:ndim])
        # sub_num[sub_num!=0] = 1
        # avg_factor *= sub_num.to(torch.int).sum()

        box_loss = self.loss_box(
            box,
            box_target,
            weight=w,
            avg_factor=avg_factor,
        )
        output[f"loss_box{suffix}"] = box_loss

        # 可选的 centerness / yawness
        if quality is not None and (self.loss_cns is not None or self.loss_yns is not None):
            cns = quality[..., CNS]
            yns = quality[..., YNS]#.sigmoid()

            # centerness target: 距离越近越大，exp(-||Δxyz||)
            if self.loss_cns is not None:
                cns_target = torch.norm(
                    box_target[..., [X, Y, Z]] - box[..., [X, Y, Z]],
                    p=2,
                    dim=-1,
                )
                cns_target = torch.exp(-cns_target)
                cns_loss = self.loss_cns(
                    cns,
                    cns_target,
                    avg_factor=avg_factor,
                )
                output[f"loss_cns{suffix}"] = cns_loss

            if self.loss_yns is not None:
                yns_target = (
                    F.cosine_similarity(
                        box_target[..., [SIN_YAW, COS_YAW]],
                        box[..., [SIN_YAW, COS_YAW]],
                        dim=-1,
                    )
                    > 0
                )
                yns_target = yns_target.float()
                yns_loss = self.loss_yns(
                    yns,
                    yns_target,
                    avg_factor=avg_factor,
                )
                output[f"loss_yns{suffix}"] = yns_loss


        if self.loss_giou is not None:
            giou_loss = self.loss_giou(box,box_target,avg_factor)
            output[f"loss_iou{suffix}"] = giou_loss
        return output











class GIoU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,box, box_target,avg_factor):
        
        pred_dec = decode_box(box)    # (num_pos, 10)
        tgt_dec = decode_box(box_target)
        pred_bev = torch.stack([pred_dec[..., 0], pred_dec[..., 1],
                                pred_dec[..., 3], pred_dec[..., 4],
                                pred_dec[..., 6]], dim=-1)
        tgt_bev = torch.stack([tgt_dec[..., 0], tgt_dec[..., 1],
                                tgt_dec[..., 3], tgt_dec[..., 4],
                                tgt_dec[..., 6]], dim=-1)
        giou = self.rotated_giou(pred_bev, tgt_bev)   # (num_pos,)
        loss_giou = (1 - giou).sum() / max(float(avg_factor), 1.0)
        return loss_giou

    def rotated_giou(self,boxes1, boxes2, eps=1e-8):
        """
        计算 BEV 旋转矩形的 GIoU。
        boxes1, boxes2: (N, 5) 或 (..., 5)，最后一维为 [cx, cy, w, l, yaw]
        返回 (N,) GIoU 值。
        """
        # 提取参数
        cx1, cy1, w1, l1, yaw1 = boxes1.unbind(-1)
        cx2, cy2, w2, l2, yaw2 = boxes2.unbind(-1)

        # 获取顶点
        vert1 = self.get_rect_vertices(cx1, cy1, w1, l1, yaw1)  # (N,4,2)
        vert2 = self.get_rect_vertices(cx2, cy2, w2, l2, yaw2)  # (N,4,2)

        # 计算交集多边形
        inters = self.intersect_convex_polygons(vert1, vert2)  # list of (K_i,2)

        # 计算交集面积
        inter_area = torch.zeros_like(cx1)
        for i, poly in enumerate(inters):
            if poly.shape[0] >= 3:
                inter_area[i] = self.polygon_area(poly.unsqueeze(0)).squeeze()

        # 计算两个矩形自身面积
        area1 = w1 * l1
        area2 = w2 * l2
        union_area = area1 + area2 - inter_area

        # 计算最小外接矩形（轴对齐）的顶点
        all_pts = torch.cat([vert1, vert2], dim=1)  # (N,8,2)
        xmin = all_pts[..., 0].min(dim=-1)[0]
        xmax = all_pts[..., 0].max(dim=-1)[0]
        ymin = all_pts[..., 1].min(dim=-1)[0]
        ymax = all_pts[..., 1].max(dim=-1)[0]
        enclosing_area = (xmax - xmin) * (ymax - ymin)

        iou = inter_area / (union_area + eps)
        giou = iou - (enclosing_area - union_area) / (enclosing_area + eps)
        return giou

    
    def get_rect_vertices(self,cx, cy, w, l, yaw):
        """
        获取旋转矩形的四个顶点（顺时针或逆时针，首尾相连）。
        cx, cy: (...,) 中心坐标
        w, l: (...,) 宽度（x 方向）和长度（y 方向）
        yaw: (...,) 旋转角（弧度）
        返回顶点张量 (..., 4, 2)
        """
        dx = w / 2
        dy = l / 2
        # 未旋转时的四个角点（顺序：左下、右下、右上、左上）
        corners = torch.stack([
            torch.stack([-dx, -dy], dim=-1),  # 左下
            torch.stack([ dx, -dy], dim=-1),  # 右下
            torch.stack([ dx,  dy], dim=-1),  # 右上
            torch.stack([-dx,  dy], dim=-1),  # 左上
        ], dim=-2)  # (..., 4, 2)
        center = torch.stack([cx, cy], dim=-1)
        sin, cos = torch.sin(yaw), torch.cos(yaw)
        vertices = self.rotate_points(corners, center, sin, cos)
        return vertices

    
    def intersect_convex_polygons(self,poly1, poly2):
        """
        计算两个凸多边形的交集，返回交集顶点列表（每个 batch 独立）。
        poly1, poly2: (..., M, 2), (..., N, 2)
        """
        batch_size = poly1.shape[0]
        out_polys = []
        for b in range(batch_size):
            subject = poly1[b]  # (M,2)
            clip = poly2[b]     # (N,2)
            # 用 clip 的每条边裁剪 subject
            result = subject
            for i in range(clip.shape[0]):
                a = clip[i]
                b = clip[(i+1) % clip.shape[0]]
                result_list = self.clip_polygon_by_halfplane(result.unsqueeze(0), a.unsqueeze(0), b.unsqueeze(0))
                result = result_list[0] if len(result_list[0]) > 0 else torch.zeros(0, 2, device=poly1.device)
                if result.shape[0] < 3:
                    result = torch.zeros(0, 2, device=poly1.device)
                    break
            out_polys.append(result)
        return out_polys

    def polygon_area(self,poly):
        """
        计算多边形面积（鞋带公式），poly: (..., N, 2)
        """
        N = poly.shape[-2]
        shift = torch.roll(poly, shifts=-1, dims=-2)
        cross = poly[..., 0] * shift[..., 1] - poly[..., 1] * shift[..., 0]
        area = cross.sum(dim=-1) / 2.0
        return area.abs()
    
    def rotate_points(self, points, center, sin, cos):
        """绕中心旋转点集 points: (..., N, 2)；center (..., 2)；sin/cos (...) 与 batch 维对齐。"""
        x, y = points[..., 0], points[..., 1]
        cx = center[..., 0].unsqueeze(-1)
        cy = center[..., 1].unsqueeze(-1)
        s = sin.unsqueeze(-1)
        c = cos.unsqueeze(-1)
        x_rel = x - cx
        y_rel = y - cy
        x_rot = x_rel * c - y_rel * s
        y_rot = x_rel * s + y_rel * c
        return torch.stack([x_rot + cx, y_rot + cy], dim=-1)

    

    def clip_polygon_by_halfplane(self,poly, a, b):
        """
        使用 Sutherland-Hodgman 算法，将多边形 poly 裁剪到半平面左侧（含边界）。
        poly: (..., N, 2)
        a, b: (..., 2) 定义半平面的边（从 a 到 b）
        返回裁剪后的多边形顶点列表（每个 batch 独立）。
        """
        # 对每个 batch 独立处理（可并行，但为清晰采用循环）
        batch_shape = poly.shape[:-2]
        N = poly.shape[-2]
        out_polys = []
        for idx in range(poly.shape[0]):  # 假设 poly 至少有 batch 维度
            p = poly[idx]  # (N, 2)
            a_ = a[idx]
            b_ = b[idx]
            output = [p[-1]]  # 从最后一个点开始，便于循环
            for i in range(N):
                cur = p[i]
                prev = p[i-1]
                # 判断点是否在左侧（叉积 >= 0）
                def is_left(p):
                    return (b_[0]-a_[0])*(p[1]-a_[1]) - (b_[1]-a_[1])*(p[0]-a_[0]) >= 0
                cur_inside = is_left(cur)
                prev_inside = is_left(prev)
                if prev_inside and cur_inside:
                    output.append(cur)
                elif prev_inside and not cur_inside:
                    # 求交点
                    # 线段 prev-cur 与边 a-b 的交点
                    # 参数方程解
                    t = ((a_[0]-prev[0])*(b_[1]-a_[1]) - (a_[1]-prev[1])*(b_[0]-a_[0])) / \
                        ((cur[0]-prev[0])*(b_[1]-a_[1]) - (cur[1]-prev[1])*(b_[0]-a_[0]) + 1e-8)
                    ip = prev + t * (cur - prev)
                    output.append(ip)
                elif not prev_inside and cur_inside:
                    t = ((a_[0]-prev[0])*(b_[1]-a_[1]) - (a_[1]-prev[1])*(b_[0]-a_[0])) / \
                        ((cur[0]-prev[0])*(b_[1]-a_[1]) - (cur[1]-prev[1])*(b_[0]-a_[0]) + 1e-8)
                    ip = prev + t * (cur - prev)
                    output.append(ip)
                    output.append(cur)
            # 去掉最后一个重复点（起始点重复）
            if len(output) > 0:
                output = output[1:]  # 去掉初始的 prev
            out_polys.append(torch.stack(output, dim=0) if output else torch.zeros(0, 2, device=poly.device, dtype=poly.dtype))
        # 返回 list of tensors
        return out_polys