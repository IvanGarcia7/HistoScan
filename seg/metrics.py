"""
Semantic segmentation metrics computation module.

Provides standard evaluation metrics for binary image segmentation tasks.
All metrics operate on predicted probability maps (post-sigmoid) and thresholded
binary ground truth masks.

Implemented metrics:
- Dice coefficient (F1 score): Measures overlap between predicted and target masks
- Intersection over Union (IoU/Jaccard): Measures ratio of intersection to union
- Pixel accuracy: Computes per-pixel classification accuracy

All functions include numerical stability epsilon to prevent division by zero.
Predictions are automatically thresholded at 0.5 to create binary masks.
"""

import torch


def dice_score(pred, target, eps=1e-6):
    """
    Calculate Dice coefficient (F1 score) for binary segmentation.
    
    Measures the overlap between predicted and target binary masks using the
    Dice similarity coefficient, which ranges from 0 (no overlap) to 1 (perfect match).
    Formula: (2 * |X ∩ Y|) / (|X| + |Y|)
    
    Args:
        pred (torch.Tensor): Predicted probability map, shape arbitrary.
            Values should be in [0, 1] (post-sigmoid output).
        target (torch.Tensor): Ground truth binary mask, same shape as pred.
            Values should be 0 or 1.
        eps (float): Epsilon for numerical stability to prevent division by zero.
            Default is 1e-6.
    
    Returns:
        torch.Tensor: Scalar tensor containing Dice coefficient in range [0, 1].
        Scalar can be converted to Python float with .item()
    
    Example:
        >>> pred = torch.rand(1, 256, 256)  # Post-sigmoid probability
        >>> target = torch.randint(0, 2, (1, 256, 256)).float()
        >>> dice = dice_score(pred, target)
        >>> print(f"Dice: {dice.item():.4f}")
    """
    pred = (pred > 0.5).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2 * inter + eps) / (union + eps)


def iou_score(pred, target, eps=1e-6):
    """
    Calculate Intersection over Union (IoU/Jaccard index) for binary segmentation.
    
    Measures the ratio of intersection to union of predicted and target masks.
    Also known as Jaccard index. Ranges from 0 (no overlap) to 1 (perfect match).
    Formula: |X ∩ Y| / (|X| ∪ |Y|) = |X ∩ Y| / (|X| + |Y| - |X ∩ Y|)
    
    Args:
        pred (torch.Tensor): Predicted probability map, shape arbitrary.
            Values should be in [0, 1] (post-sigmoid output).
        target (torch.Tensor): Ground truth binary mask, same shape as pred.
            Values should be 0 or 1.
        eps (float): Epsilon for numerical stability to prevent division by zero.
            Default is 1e-6.
    
    Returns:
        torch.Tensor: Scalar tensor containing IoU value in range [0, 1].
        Scalar can be converted to Python float with .item()
    
    Note:
        IoU is generally more conservative than Dice coefficient. A gap between
        Dice and IoU may indicate scattered false positives or false negatives.
    
    Example:
        >>> pred = torch.rand(1, 256, 256)  # Post-sigmoid probability
        >>> target = torch.randint(0, 2, (1, 256, 256)).float()
        >>> iou = iou_score(pred, target)
        >>> print(f"IoU: {iou.item():.4f}")
    """
    pred = (pred > 0.5).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return (inter + eps) / (union + eps)


def pixel_accuracy(pred, target, thresh=0.5):
    """
    Calculate pixel-level classification accuracy.
    
    Computes the fraction of correctly classified pixels. Simple metric but can be
    misleading for imbalanced datasets where one class dominates (e.g., 95% background).
    More suitable as supplementary metric alongside Dice/IoU.
    
    Args:
        pred (torch.Tensor): Predicted probability map, shape arbitrary.
            Values should be in [0, 1] (post-sigmoid output).
        target (torch.Tensor): Ground truth binary mask, same shape as pred.
            Values should be 0 or 1.
        thresh (float): Threshold for binarizing predictions. Default is 0.5.
            Predictions > thresh are classified as positive (1), otherwise negative (0).
    
    Returns:
        torch.Tensor: Scalar tensor containing accuracy in range [0, 1].
        Scalar can be converted to Python float with .item()
    
    Warning:
        This metric can be misleading on imbalanced datasets. For example,
        a model predicting all zeros on 99% background data would achieve
        99% accuracy despite being useless. Use Dice or IoU for more reliable
        evaluation on imbalanced segmentation tasks.
    
    Example:
        >>> pred = torch.rand(1, 256, 256)  # Post-sigmoid probability
        >>> target = torch.randint(0, 2, (1, 256, 256)).float()
        >>> acc = pixel_accuracy(pred, target, thresh=0.5)
        >>> print(f"Accuracy: {acc.item():.4f}")
    """
    pred = (pred > thresh).float()
    correct = (pred == target).float().sum()
    total = target.numel()
    return correct / total

