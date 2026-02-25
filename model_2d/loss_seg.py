import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        mask = 1 - (target > 0.25) * (target < 0.75).float() # 全样本不参与计算
        num = torch.sum(2 * torch.mul(predict, target) * mask,dim=1) + self.smooth
        if self.p == 2:
            den = torch.sum((predict.pow(self.p) + target.pow(self.p))*mask,dim=1) + self.smooth
        else:
            den = torch.sum((predict + target)*mask,dim=1) + self.smooth
        loss = 1 - num / den
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss for multi-class segmentation
    
    Args:
        weight: An array of shape [num_classes,] for class weighting
        ignore_index: class index to ignore (not used currently)
        predict: A tensor of shape [B, C, H, W] - logits
        target: A tensor of shape [B, C, H, W] - one-hot encoded or binary masks
        other args pass to BinaryDiceLoss
    
    Returns:
        Dice loss (scalar)
    
    Note:
        For multi-class segmentation, Dice loss is computed per-class and then averaged.
        This is the standard approach for multi-class Dice loss.
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        """
        Args:
            predict: [B, C, H, W] - logits from model
            target: [B, C, H, W] - one-hot encoded masks (0 or 1)
        """
        assert predict.shape == target.shape, f'predict & target shape do not match: {predict.shape} vs {target.shape}'
        
        # Apply sigmoid to get probabilities
        predict = predict.sigmoid()
        
        B, C, H, W = predict.shape
        
        # Compute Dice loss for each class separately, then average
        # This is the standard approach for multi-class Dice loss
        dice = BinaryDiceLoss(p=1, reduction='none', **self.kwargs)
        
        class_losses = []
        for c in range(C):
            # Extract single class: [B, H, W]
            pred_c = predict[:, c, :, :]
            tgt_c = target[:, c, :, :]
            
            # Compute Dice loss for this class: [B]
            loss_c = dice(pred_c, tgt_c)
            class_losses.append(loss_c)
        
        # Stack: [B, C]
        class_losses = torch.stack(class_losses, dim=1)
        
        # Apply class weights if provided
        if self.weight is not None:
            weight_tensor = torch.tensor(self.weight, device=class_losses.device, dtype=class_losses.dtype)
            weight_tensor = weight_tensor.view(1, -1)  # [1, C]
            class_losses = class_losses * weight_tensor
        
        # Average over classes, then over batch
        dice_loss = class_losses.mean()
        
        return dice_loss

class BCELoss(torch.nn.Module):
    """Binary Cross Entropy Loss for multi-class segmentation
    
    Args:
        pos_weight: Weight for positive samples. Can be:
            - Scalar: Same weight for all classes
            - List/Tensor of shape [num_classes]: Different weight for each class
    
    Note:
        For multi-class segmentation with one-hot encoding, BCEWithLogitsLoss
        computes loss independently for each class and then averages.
        pos_weight can be per-class to handle class imbalance.
    """
    def __init__(self, pos_weight):
        super(BCELoss, self).__init__()
        # Store pos_weight (will be converted to tensor in forward)
        self.pos_weight = pos_weight

    def forward(self, ypred, ytgt):
        """
        Args:
            ypred: [B, C, H, W] - logits from model
            ytgt: [B, C, H, W] - one-hot encoded masks (0 or 1)
        
        Returns:
            BCE loss (scalar)
        """
        ytgt = ytgt.float()
        ypred = ypred.float()
        
        B, C, H, W = ypred.shape
        
        # Create valid mask: pixels with certain labels (0 or 1)
        # Ignore uncertain pixels (0.25 < value < 0.75)
        # valid_mask: True for pixels we want to compute loss on
        valid_mask = (ytgt <= 0.25) | (ytgt >= 0.75)
        
        # Setup pos_weight based on number of classes
        if isinstance(self.pos_weight, (int, float)):
            # Scalar: use same weight for all classes
            pos_weight_tensor = torch.tensor([self.pos_weight] * C, device=ypred.device, dtype=ypred.dtype)
        else:
            # List or tensor: per-class weights
            pos_weight_tensor = torch.tensor(self.pos_weight, device=ypred.device, dtype=ypred.dtype) if not isinstance(self.pos_weight, torch.Tensor) else self.pos_weight.to(ypred.device)
            if len(pos_weight_tensor) == 1:
                pos_weight_tensor = pos_weight_tensor.expand(C)
            assert len(pos_weight_tensor) == C, f"pos_weight length {len(pos_weight_tensor)} must match num_classes {C}"
        
        # Compute loss per class separately (standard approach for multi-class BCE)
        losses_per_class = []
        for c in range(C):
            # Get valid pixels for this class
            valid_c = valid_mask[:, c, :, :]
            if valid_c.sum() > 0:
                pred_c_valid = ypred[:, c, :, :][valid_c]
                tgt_c_valid = ytgt[:, c, :, :][valid_c]
                
                # Compute BCE loss for this class
                loss_c = F.binary_cross_entropy_with_logits(
                    pred_c_valid,
                    tgt_c_valid,
                    pos_weight=pos_weight_tensor[c],
                    reduction='mean'
                )
                losses_per_class.append(loss_c)
        
        # Average over classes
        if len(losses_per_class) > 0:
            loss = torch.stack(losses_per_class).mean()
        else:
            # No valid pixels, return zero loss (with gradient)
            loss = torch.tensor(0.0, device=ypred.device, requires_grad=True)
        
        return loss

class SegLoss(torch.nn.Module):
    def __init__(self, pos_weight):
        super(SegLoss, self).__init__()
        self.bce_loss = BCELoss(pos_weight)
        self.dice_loss = DiceLoss()

    def forward(self, ypred, ytgt):
        bce_loss = self.bce_loss(ypred, ytgt)
        dice_loss = self.dice_loss(ypred, ytgt)
        return {
            'bce_loss': bce_loss,
            'dice_loss': dice_loss
        }