"""Evaluation metrics for visual grounding."""

import torch
import numpy as np


def compute_iou(pred_box: torch.Tensor, gt_box: torch.Tensor) -> torch.Tensor:
    """Compute IoU between predicted and ground truth bounding boxes.
    
    Args:
        pred_box: (N, 4) or (4,) in [x1, y1, x2, y2] normalized format
        gt_box: (N, 4) or (4,) in same format
    
    Returns:
        IoU values, shape (N,) or scalar
    """
    if pred_box.dim() == 1:
        pred_box = pred_box.unsqueeze(0)
    if gt_box.dim() == 1:
        gt_box = gt_box.unsqueeze(0)

    # Intersection
    inter_x1 = torch.max(pred_box[:, 0], gt_box[:, 0])
    inter_y1 = torch.max(pred_box[:, 1], gt_box[:, 1])
    inter_x2 = torch.min(pred_box[:, 2], gt_box[:, 2])
    inter_y2 = torch.min(pred_box[:, 3], gt_box[:, 3])

    inter_w = torch.clamp(inter_x2 - inter_x1, min=0.0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0.0)
    inter_area = inter_w * inter_h

    # Union
    pred_area = (pred_box[:, 2] - pred_box[:, 0]) * (pred_box[:, 3] - pred_box[:, 1])
    gt_area = (gt_box[:, 2] - gt_box[:, 0]) * (gt_box[:, 3] - gt_box[:, 1])
    union_area = pred_area + gt_area - inter_area

    iou = inter_area / (union_area + 1e-8)
    return iou


def compute_giou(pred_box: torch.Tensor, gt_box: torch.Tensor) -> torch.Tensor:
    """Compute generalized IoU for aligned xyxy box pairs."""
    if pred_box.dim() == 1:
        pred_box = pred_box.unsqueeze(0)
    if gt_box.dim() == 1:
        gt_box = gt_box.unsqueeze(0)

    inter_x1 = torch.maximum(pred_box[:, 0], gt_box[:, 0])
    inter_y1 = torch.maximum(pred_box[:, 1], gt_box[:, 1])
    inter_x2 = torch.minimum(pred_box[:, 2], gt_box[:, 2])
    inter_y2 = torch.minimum(pred_box[:, 3], gt_box[:, 3])
    intersection = (
        (inter_x2 - inter_x1).clamp(min=0)
        * (inter_y2 - inter_y1).clamp(min=0)
    )

    pred_area = (
        (pred_box[:, 2] - pred_box[:, 0]).clamp(min=0)
        * (pred_box[:, 3] - pred_box[:, 1]).clamp(min=0)
    )
    gt_area = (
        (gt_box[:, 2] - gt_box[:, 0]).clamp(min=0)
        * (gt_box[:, 3] - gt_box[:, 1]).clamp(min=0)
    )
    union = pred_area + gt_area - intersection
    iou = intersection / union.clamp(min=1e-8)

    enclosing_x1 = torch.minimum(pred_box[:, 0], gt_box[:, 0])
    enclosing_y1 = torch.minimum(pred_box[:, 1], gt_box[:, 1])
    enclosing_x2 = torch.maximum(pred_box[:, 2], gt_box[:, 2])
    enclosing_y2 = torch.maximum(pred_box[:, 3], gt_box[:, 3])
    enclosing_area = (
        (enclosing_x2 - enclosing_x1).clamp(min=0)
        * (enclosing_y2 - enclosing_y1).clamp(min=0)
    )
    return iou - (enclosing_area - union) / enclosing_area.clamp(min=1e-8)


def compute_acc_at_05(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> float:
    """Compute ACC@0.5: percentage of predictions with IoU >= 0.5.
    
    Args:
        pred_boxes: (N, 4)
        gt_boxes: (N, 4)
    
    Returns:
        Accuracy as a float between 0 and 1.
    """
    ious = compute_iou(pred_boxes, gt_boxes)
    correct = (ious >= 0.5).float()
    return correct.mean().item()
