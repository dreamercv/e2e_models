import torch
import torch.nn.functional as F
import numpy as np
import cv2


def nms(heatmap, kernel=3):
    """非极大值抑制"""
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heatmap, kernel, stride=1, padding=pad)
    keep = (hmax == heatmap).float()
    return heatmap * keep


def topk(heatmap, K=100):
    """获取top K个峰值点"""
    B, C, H, W = heatmap.shape
    heatmap = heatmap.view(B, C, -1)
    topk_scores, topk_inds = torch.topk(heatmap, K)
    topk_inds = topk_inds % (H * W)
    topk_ys = topk_inds // W
    topk_xs = topk_inds % W
    
    topk_score = topk_scores.view(B, C, K)
    topk_ind = topk_inds.view(B, C, K)
    topk_y = topk_ys.view(B, C, K).float()
    topk_x = topk_xs.view(B, C, K).float()
    
    return topk_score, topk_ind, topk_y, topk_x


def nms_boxes(boxes, scores, iou_threshold=0.5):
    """
    对bbox进行NMS
    
    Args:
        boxes: [N, 4] (x1, y1, x2, y2)
        scores: [N]
        iou_threshold: IoU阈值
    
    Returns:
        keep: [N] bool tensor, True表示保留
    """
    if len(boxes) == 0:
        return torch.zeros(0, dtype=torch.bool, device=boxes.device)
    
    # 计算面积
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    # 按分数排序
    _, order = scores.sort(0, descending=True)
    
    keep = []
    while len(order) > 0:
        i = order[0].item()  # 转换为Python标量
        keep.append(i)
        
        if len(order) == 1:
            break
        
        # 计算IoU
        xx1 = torch.max(boxes[i, 0], boxes[order[1:], 0])
        yy1 = torch.max(boxes[i, 1], boxes[order[1:], 1])
        xx2 = torch.min(boxes[i, 2], boxes[order[1:], 2])
        yy2 = torch.min(boxes[i, 3], boxes[order[1:], 3])
        
        w = torch.clamp(xx2 - xx1, min=0)
        h = torch.clamp(yy2 - yy1, min=0)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        # 保留IoU小于阈值的
        mask = iou <= iou_threshold
        order = order[1:][mask]
    
    keep_tensor = torch.zeros(len(boxes), dtype=torch.bool, device=boxes.device)
    if len(keep) > 0:
        keep_tensor[keep] = True
    return keep_tensor


def decode_centernet(heatmap, offset, size, K=100, score_thresh=0.3, nms_kernel=3, nms_iou_thresh=0.5, max_detections=100):
    """
    解码CenterNet预测结果
    
    Args:
        heatmap: [B, C, H, W] 热图
        offset: [B, 2, H, W] 偏移量
        size: [B, 2, H, W] 尺寸
        K: top K个检测结果（每个类别）
        score_thresh: 分数阈值
        nms_kernel: heatmap NMS核大小
        nms_iou_thresh: bbox NMS的IoU阈值
        max_detections: 最大检测数量
    
    Returns:
        detections: List of [N, 6] (x1, y1, x2, y2, score, class_id) for each batch
    """
    B, C, H, W = heatmap.shape
    
    # 对heatmap做NMS
    heatmap = nms(heatmap, kernel=nms_kernel)
    
    # 获取对应的offset和size
    offset = offset.permute(0, 2, 3, 1).contiguous()
    size = size.permute(0, 2, 3, 1).contiguous()
    
    batch_detections = []
    
    for b in range(B):
        detections = []
        
        # 方法1：每个类别独立取top K（原始方法）
        # 但先过滤低分，减少候选数量
        for c in range(C):
            # 获取该类别所有位置的分数
            class_heatmap = heatmap[b, c]  # [H, W]
            
            # 只处理分数超过阈值的点
            valid_mask = class_heatmap > score_thresh
            if valid_mask.sum() == 0:
                continue
            
            # 获取有效位置的坐标和分数
            valid_y, valid_x = torch.where(valid_mask)
            valid_scores = class_heatmap[valid_y, valid_x]
            
            # 如果有效点太多，只取top K
            if len(valid_scores) > K:
                topk_scores, topk_indices = torch.topk(valid_scores, K)
                valid_y = valid_y[topk_indices]
                valid_x = valid_x[topk_indices]
                valid_scores = topk_scores
            else:
                topk_indices = torch.arange(len(valid_scores), device=valid_scores.device)
            
            # 获取对应的offset和size
            offsets = offset[b, valid_y, valid_x]  # [N, 2]
            sizes = size[b, valid_y, valid_x]  # [N, 2]
            
            # 计算最终的中心点
            center_xs = valid_x.float() + offsets[:, 0]
            center_ys = valid_y.float() + offsets[:, 1]
            
            # 计算边界框
            x1s = center_xs - sizes[:, 0] / 2
            y1s = center_ys - sizes[:, 1] / 2
            x2s = center_xs + sizes[:, 0] / 2
            y2s = center_ys + sizes[:, 1] / 2
            
            # 过滤无效的size（太小或太大，可能是背景误检）
            min_size = 2.0  # 最小尺寸（在输出特征图尺度上）
            max_size = max(H, W) * 0.8  # 最大尺寸（不超过特征图的80%）
            size_valid = (sizes[:, 0] > min_size) & (sizes[:, 1] > min_size) & \
                        (sizes[:, 0] < max_size) & (sizes[:, 1] < max_size)
            
            if size_valid.sum() == 0:
                continue
            
            x1s = x1s[size_valid]
            y1s = y1s[size_valid]
            x2s = x2s[size_valid]
            y2s = y2s[size_valid]
            valid_scores = valid_scores[size_valid]
            
            # 组合结果 [x1, y1, x2, y2, score, class_id]
            boxes = torch.stack([
                x1s, y1s, x2s, y2s, valid_scores,
                torch.full_like(valid_scores, c, dtype=torch.float32)
            ], dim=1)
            
            detections.append(boxes)
        
        if len(detections) > 0:
            detections = torch.cat(detections, dim=0)
            
            # 按分数排序
            _, indices = torch.sort(detections[:, 4], descending=True)
            detections = detections[indices]
            
            # 对bbox做NMS（跨类别）
            boxes_nms = detections[:, :4]
            scores_nms = detections[:, 4]
            keep = nms_boxes(boxes_nms, scores_nms, iou_threshold=nms_iou_thresh)
            detections = detections[keep]
            
            # 限制最大检测数量
            if len(detections) > max_detections:
                detections = detections[:max_detections]
        else:
            detections = torch.zeros((0, 6), device=heatmap.device)
        
        batch_detections.append(detections)
    
    return batch_detections


def visualize_detections(image, detections, class_names=None, score_thresh=0.3, show_labels=False):
    """
    可视化检测结果
    
    Args:
        image: numpy array [H, W, 3] RGB格式
        detections: [N, 6] (x1, y1, x2, y2, score, class_id)
        class_names: 类别名称列表
        score_thresh: 分数阈值
        show_labels: 是否显示类别标签（默认False，不显示）
    
    Returns:
        vis_image: 可视化后的图片
    """
    vis_image = image.copy()
    h, w = image.shape[:2]
    
    # 颜色列表
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
    ]
    
    for det in detections:
        x1, y1, x2, y2, score, class_id = det
        
        if score < score_thresh:
            continue
        
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        class_id = int(class_id)
        
        # 限制在图片范围内
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        # 选择颜色
        color = colors[class_id % len(colors)]
        
        # 绘制边界框
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        
        # 只在需要时绘制标签
        if show_labels:
            label = f"{class_id}"
            if class_names and class_id < len(class_names):
                label = f"{class_names[class_id]}: {score:.2f}"
            else:
                label = f"cls{class_id}: {score:.2f}"
            
            # 计算文字位置
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_image, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1)
            cv2.putText(vis_image, label, (x1, y1 - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return vis_image

