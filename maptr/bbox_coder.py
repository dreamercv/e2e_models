"""
MapTR 风格 bbox_coder：MapTRNMSFreeCoder，纯 PyTorch，不依赖 mmcv。
支持 post_center_range（8 元：min x,y,x,y 与 max x,y,x,y）、max_num、pts 解码。
"""
from typing import List, Tuple, Optional, Dict, Any
import torch

__all__ = [
    "MapTRNMSFreeCoder",
    "build_maptr_bbox_coder",
]


def _denormalize_2d_bbox_from_pc_range(
    bboxes: torch.Tensor, pc_range: List[float]
) -> torch.Tensor:
    """bboxes: (..., 4) 归一化 cx,cy,w,h [0,1]。pc_range 可为 4 元 [x0,y0,x1,y1] 或 6 元 [x0,y0,z0,x1,y1,z1]。"""
    if len(pc_range) >= 6:
        x0, y0, x1, y1 = pc_range[0], pc_range[1], pc_range[3], pc_range[4]
    else:
        x0, y0, x1, y1 = pc_range[0], pc_range[1], pc_range[2], pc_range[3]
    patch_w = x1 - x0
    patch_h = y1 - y0
    cx = bboxes[..., 0:1] * patch_w + x0
    cy = bboxes[..., 1:2] * patch_h + y0
    w = bboxes[..., 2:3] * patch_w
    h = bboxes[..., 3:4] * patch_h
    x1_ = cx - w / 2.0
    y1_ = cy - h / 2.0
    x2_ = cx + w / 2.0
    y2_ = cy + h / 2.0
    return torch.cat([x1_, y1_, x2_, y2_], dim=-1)


def _denormalize_2d_pts_from_pc_range(
    pts: torch.Tensor, pc_range: List[float]
) -> torch.Tensor:
    """pts: (..., 2) 归一化 [0,1]。pc_range 同 _denormalize_2d_bbox_from_pc_range。"""
    if len(pc_range) >= 6:
        x0, y0, x1, y1 = pc_range[0], pc_range[1], pc_range[3], pc_range[4]
    else:
        x0, y0, x1, y1 = pc_range[0], pc_range[1], pc_range[2], pc_range[3]
    patch_w = x1 - x0
    patch_h = y1 - y0
    out = pts.clone()
    out[..., 0:1] = pts[..., 0:1] * patch_w + x0
    out[..., 1:2] = pts[..., 1:2] * patch_h + y0
    return out


class MapTRNMSFreeCoder:
    """MapTR 无 NMS 解码器：按 score topk，post_center_range 过滤，输出 bboxes + pts。
    与 config bbox_coder type='MapTRNMSFreeCoder' 一致。
    post_center_range: 8 元 [xmin, ymin, xmin, ymin, xmax, ymax, xmax, ymax] 用于 bbox 四维比较。
    """

    def __init__(
        self,
        pc_range: List[float],
        voxel_size: Optional[List[float]] = None,
        post_center_range: Optional[List[float]] = None,
        max_num: int = 50,
        score_threshold: Optional[float] = None,
        num_classes: int = 3,
    ):
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes

    def decode_single(
        self,
        cls_scores: torch.Tensor,
        bbox_preds: torch.Tensor,
        pts_preds: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        cls_scores: (num_query, num_classes)
        bbox_preds: (num_query, 4) 归一化 cx,cy,w,h
        pts_preds: (num_query, fixed_num_pts, 2) 归一化
        """
        max_num = self.max_num
        cls_scores = cls_scores.sigmoid()
        scores, indexs = cls_scores.view(-1).topk(
            min(max_num, cls_scores.numel())
        )
        labels = indexs % self.num_classes
        bbox_index = torch.div(indexs, self.num_classes, rounding_mode="floor")
        bbox_preds = bbox_preds[bbox_index]
        pts_preds = pts_preds[bbox_index]

        final_box_preds = _denormalize_2d_bbox_from_pc_range(
            bbox_preds, self.pc_range
        )
        final_pts_preds = _denormalize_2d_pts_from_pc_range(
            pts_preds, self.pc_range
        )
        final_scores = scores
        final_preds = labels

        thresh_mask = None
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold
            tmp_score = self.score_threshold
            while thresh_mask.sum() == 0:
                tmp_score *= 0.9
                if tmp_score < 0.01:
                    thresh_mask = final_scores > -1
                    break
                thresh_mask = final_scores >= tmp_score

        if self.post_center_range is not None:
            pr = torch.tensor(
                self.post_center_range,
                device=final_box_preds.device,
                dtype=final_box_preds.dtype,
            )
            if pr.numel() >= 8:
                mask = (final_box_preds[..., :4] >= pr[:4]).all(dim=1)
                mask &= (final_box_preds[..., :4] <= pr[4:8]).all(dim=1)
            else:
                mask = (final_box_preds[..., :4] >= pr[:4]).all(dim=1)
                mask &= (final_box_preds[..., :4] <= pr[4:]).all(dim=1)
            if thresh_mask is not None:
                mask = mask & thresh_mask
            boxes = final_box_preds[mask]
            scores = final_scores[mask]
            pts = final_pts_preds[mask]
            labels = final_preds[mask]
        else:
            if thresh_mask is not None:
                boxes = final_box_preds[thresh_mask]
                scores = final_scores[thresh_mask]
                pts = final_pts_preds[thresh_mask]
                labels = final_preds[thresh_mask]
            else:
                boxes = final_box_preds
                scores = final_scores
                pts = final_pts_preds
                labels = final_preds

        return {
            "bboxes": boxes,
            "scores": scores,
            "labels": labels,
            "pts": pts,
        }

    def decode(
        self,
        preds_dicts: Dict[str, Any],
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        preds_dicts: 至少包含 'all_cls_scores', 'all_bbox_preds', 'all_pts_preds'，
            取最后一层 [-1]。每个 shape: (bs, num_query, ...)。
        """
        all_cls_scores = preds_dicts["all_cls_scores"][-1]
        all_bbox_preds = preds_dicts["all_bbox_preds"][-1]
        all_pts_preds = preds_dicts["all_pts_preds"][-1]
        batch_size = all_cls_scores.size(0)
        predictions_list = []
        th = score_threshold if score_threshold is not None else self.score_threshold
        for i in range(batch_size):
            pred = self.decode_single(
                all_cls_scores[i],
                all_bbox_preds[i],
                all_pts_preds[i],
            )
            predictions_list.append(pred)
        return predictions_list


def build_maptr_bbox_coder(
    pc_range: List[float],
    voxel_size: Optional[List[float]] = None,
    post_center_range: Optional[List[float]] = None,
    max_num: int = 50,
    score_threshold: Optional[float] = None,
    num_classes: int = 3,
    **kwargs,
) -> MapTRNMSFreeCoder:
    """从 config 风格参数字典构建 MapTRNMSFreeCoder。"""
    return MapTRNMSFreeCoder(
        pc_range=pc_range,
        voxel_size=voxel_size,
        post_center_range=post_center_range,
        max_num=max_num,
        score_threshold=score_threshold,
        num_classes=num_classes,
    )