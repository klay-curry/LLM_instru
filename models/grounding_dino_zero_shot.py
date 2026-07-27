"""
Grounding DINO zero-shot visual grounding.

Wraps HuggingFace GroundingDinoForObjectDetection with a simple API.
Returns one bounding box per (image, query) pair by:
    1. Run inference with a low threshold.
    2. Apply IoU-based NMS to remove duplicates.
    3. Return the box with the highest confidence.

Input image is the visible (RGB) view only — Grounding DINO is an RGB model.
The depth/infrared information is ignored by this branch (the VLM branch
already uses them in the hybrid pipeline).

Pixel coordinates are converted to normalized [0, 1] using the original image
dimensions reported by `processor.post_process_grounded_object_detection`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pair-wise IoU of two sets of boxes in xyxy format. Returns (N, M) array."""
    a_x1, a_y1, a_x2, a_y2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    b_x1, b_y1, b_x2, b_y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter_x1 = np.maximum(a_x1[:, None], b_x1[None, :])
    inter_y1 = np.maximum(a_y1[:, None], b_y1[None, :])
    inter_x2 = np.minimum(a_x2[:, None], b_x2[None, :])
    inter_y2 = np.minimum(a_y2[:, None], b_y2[None, :])
    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter = inter_w * inter_h
    area_a = (a_x2 - a_x1) * (a_y2 - a_y1)
    area_b = (b_x2 - b_x1) * (b_y2 - b_y1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> List[int]:
    """Classic greedy NMS. Returns indices to keep, sorted by score desc."""
    if len(boxes) == 0:
        return []
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        rest = order[1:]
        ious = iou_xyxy(boxes[i:i+1], boxes[rest])[0]
        order = rest[ious <= iou_threshold]
    return keep


@dataclass
class GroundingDINOConfig:
    model_path: str
    box_threshold: float = 0.25
    text_threshold: float = 0.25
    nms_iou_threshold: float = 0.5
    device: str = "cuda:0"


class GroundingDINOZeroShot:
    """Wraps HuggingFace Grounding DINO for zero-shot visual grounding."""

    def __init__(self, cfg: GroundingDINOConfig):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        self.cfg = cfg
        self.processor = AutoProcessor.from_pretrained(cfg.model_path)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            cfg.model_path, device_map=cfg.device,
        ).eval()
        self.device = next(self.model.parameters()).device
        logger.info(
            "[gdino] loaded %s on %s, dtype=%s",
            cfg.model_path, self.device, next(self.model.parameters()).dtype,
        )

    @torch.inference_mode()
    def predict_bbox(
        self,
        rgb: Image.Image,
        query: str,
        fallback_box: Tuple[float, float, float, float] = (0.45, 0.45, 0.55, 0.55),
    ) -> Tuple[List[float], float]:
        """Return one (bbox_xyxy_normalized, score) pair.

        bbox is length-4 floats in [0, 1] with x1<x2, y1<y2. Returns fallback
        and 0.0 if no detection.
        """
        inputs = self.processor(images=rgb, text=query, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        target_size = rgb.size[::-1]  # (H, W)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            threshold=self.cfg.box_threshold,
            text_threshold=self.cfg.text_threshold,
            target_sizes=[target_size],
        )[0]
        boxes = results["boxes"].cpu().numpy()  # (N, 4) in xyxy pixels
        scores = results["scores"].cpu().numpy()  # (N,)

        if len(boxes) == 0:
            return list(fallback_box), 0.0

        # NMS
        keep = nms(boxes, scores, iou_threshold=self.cfg.nms_iou_threshold)
        boxes = boxes[keep]
        scores = scores[keep]

        # Take highest-confidence box (the most likely "the target")
        best = int(scores.argmax())
        x1, y1, x2, y2 = boxes[best]
        W, H = rgb.size  # PIL: (W, H)
        bbox = [
            float(np.clip(x1 / W, 0, 1)),
            float(np.clip(y1 / H, 0, 1)),
            float(np.clip(x2 / W, 0, 1)),
            float(np.clip(y2 / H, 0, 1)),
        ]
        # enforce x1<x2, y1<y2
        bbox[0], bbox[2] = min(bbox[0], bbox[2]), max(bbox[0], bbox[2])
        bbox[1], bbox[3] = min(bbox[1], bbox[3]), max(bbox[1], bbox[3])
        if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) < 1e-6:
            return list(fallback_box), 0.0
        return bbox, float(scores[best])

    def close(self) -> None:
        del self.model
        del self.processor
        torch.cuda.empty_cache()