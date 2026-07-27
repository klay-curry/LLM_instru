"""Bounding box post-processing utilities."""

import torch
import numpy as np


def postprocess_bbox(x1: float, y1: float, x2: float, y2: float):
    """Ensure bbox is valid: x1<x2, y1<y2, values in [0,1].
    
    Returns:
        Valid [x1, y1, x2, y2] or None if invalid (zero area).
    """
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    x1, y1 = max(0.0, min(1.0, x1)), max(0.0, min(1.0, y1))
    x2, y2 = max(0.0, min(1.0, x2)), max(0.0, min(1.0, y2))
    area = (x2 - x1) * (y2 - y1)
    if area < 1e-6:
        return None
    return [x1, y1, x2, y2]


def validate_bbox(bbox) -> bool:
    """Check if a bbox is valid for submission.
    
    Returns True if bbox follows [x1,y1,x2,y2], normalized to [0,1],
    x1<x2, y1<y2, no NaN.
    """
    if bbox is None or len(bbox) != 4:
        return False
    x1, y1, x2, y2 = bbox
    if any(np.isnan([x1, y1, x2, y2])):
        return False
    if x1 >= x2 or y1 >= y2:
        return False
    if not (0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1):
        return False
    return True


def batch_postprocess(pred_boxes: torch.Tensor) -> torch.Tensor:
    """Post-process a batch of bbox predictions.
    
    Args:
        pred_boxes: (N, 4) tensor of [x1, y1, x2, y2]
    
    Returns:
        (N, 4) tensor with corrected coordinates.
    """
    batch = []
    for i in range(pred_boxes.shape[0]):
        b = pred_boxes[i].detach().cpu().tolist()
        valid = postprocess_bbox(*b)
        if valid is None:
            valid = [0.0, 0.0, 0.1, 0.1]  # fallback small box
        batch.append(valid)
    return torch.tensor(batch, device=pred_boxes.device, dtype=pred_boxes.dtype)
