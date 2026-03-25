"""
Sparse4D 检测分支用到的 loss 实现，参考官方 Sparse4D 中
`mmdet3d_plugin/models/detection3d/losses.py` 与 mmdet 的 Focal/L1/CE/GaussianFocal 接口，
做了纯 PyTorch 版本，便于在当前工程中直接使用。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .box3d import X, Y, Z, SIN_YAW, COS_YAW, CNS, YNS

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
    - target: (N,) long，类别 id，ignore_index 表示忽略
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
        valid = (target >= 0) & (target != self.ignore_index)
        if valid.sum() == 0:
            return pred.new_tensor(0.0)

        pred = pred[valid]  # (N', C)
        target = target[valid]  # (N',)
        num_cls = pred.shape[-1]
        target = target.clamp(0, num_cls - 1)

        target_one_hot = F.one_hot(target, num_cls).float()  # (N', C)
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
        cls_allow_reverse=None,
    ):
        super().__init__()
        self.reg_weights = reg_weights if reg_weights is not None else [1.0] * 11
        self.loss_box = L1Loss(reduction="mean")
        self.loss_cns = CrossEntropyLoss(use_sigmoid=True, reduction="mean") if loss_centerness else None
        self.loss_yns = GaussianFocalLoss(reduction="mean") if loss_yawness else None
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

        # 新增：清理 NaN，避免 0 * NaN -> NaN
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
        sub_num = reg_w = box.new_tensor(self.reg_weights[:ndim])
        sub_num[sub_num!=0] = 1
        avg_factor *= sub_num.to(torch.int).sum()

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
            yns = quality[..., YNS].sigmoid()

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

        return output