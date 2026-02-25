import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for heatmap - 改进版，减少虚景"""
    
    def __init__(self, alpha=2, beta=4, neg_weight=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.neg_weight = neg_weight  # 负样本权重，增加可以更好地抑制背景
    
    def forward(self, pred, target, mask):
        """
        Args:
            pred: [B, C, H, W] 预测的heatmap
            target: [B, C, H, W] 目标的heatmap
            mask: [B, H, W] 有效区域mask（这里不使用mask限制负样本，计算所有位置的负样本损失）
        """
        pos_mask = (target > 0).float()
        # 负样本：所有target=0的位置都是负样本
        # 注意：这里不使用mask来限制负样本，因为我们需要计算所有背景区域的负样本损失
        # 以抑制背景误检
        neg_mask = (target == 0).float()
        
        # 正样本损失：鼓励预测接近目标值
        pos_loss = -torch.pow(1 - pred, self.alpha) * torch.log(pred + 1e-10) * target * pos_mask
        
        # 负样本损失：强烈惩罚背景区域的预测
        # 对于背景区域，pred应该接近0，如果pred较大则损失很大
        # 计算所有背景位置的负样本损失，不使用mask限制
        neg_loss = -torch.pow(pred, self.alpha) * torch.pow(1 - target, self.beta) * torch.log(1 - pred + 1e-10) * neg_mask
        
        num_pos = pos_mask.sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        
        if num_pos > 0:
            # 增加负样本权重，更强烈地抑制背景预测
            loss = (pos_loss + self.neg_weight * neg_loss) / num_pos
        else:
            # 如果没有正样本，只计算负样本损失
            loss = self.neg_weight * neg_loss / (neg_mask.sum() + 1e-10)
        
        return loss


class L1Loss(nn.Module):
    """L1 Loss for offset and size"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target, mask):
        """
        Args:
            pred: [B, 2, H, W] 预测的offset或size
            target: [B, 2, H, W] 目标的offset或size
            mask: [B, H, W] 有效区域mask
        """
        mask = mask.unsqueeze(1).expand_as(pred)
        loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-10)
        return loss


class CenterNetLoss(nn.Module):
    """CenterNet总损失函数"""
    
    def __init__(self, heatmap_weight=1.0, offset_weight=1.0, size_weight=0.1, neg_weight=2.0):
        super().__init__()
        # 增加neg_weight，更强烈地抑制背景预测
        self.heatmap_loss = FocalLoss(neg_weight=neg_weight)
        self.offset_loss = L1Loss()
        self.size_loss = L1Loss()
        
        self.heatmap_weight = heatmap_weight
        self.offset_weight = offset_weight
        self.size_weight = size_weight
    
    def forward(self, pred_heatmap, pred_offset, pred_size,
                target_heatmap, target_offset, target_size, mask):
        """
        Args:
            pred_heatmap: [B, C, H, W]
            pred_offset: [B, 2, H, W]
            pred_size: [B, 2, H, W]
            target_heatmap: [B, C, H, W]
            target_offset: [B, 2, H, W]
            target_size: [B, 2, H, W]
            mask: [B, H, W]
        """
        loss_heatmap = self.heatmap_loss(pred_heatmap, target_heatmap, mask)
        loss_offset = self.offset_loss(pred_offset, target_offset, mask)
        loss_size = self.size_loss(pred_size, target_size, mask)
        
        total_loss = (self.heatmap_weight * loss_heatmap +
                     self.offset_weight * loss_offset +
                     self.size_weight * loss_size)
        
        return {
            'total_loss': total_loss,
            'heatmap_loss': loss_heatmap,
            'offset_loss': loss_offset,
            'size_loss': loss_size
        }

