"""
Dataset for multi-modal visual grounding.
Reads JSON annotation files and loads RGB/Depth/IR images + text query + bbox.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
import numpy as np

from .preprocessing import read_rgb, read_depth, read_ir, MultiModalPreprocessor


class VisualGroundingDataset(Dataset):
    """Multi-modal visual grounding dataset.
    
    Each sample: visible + infrared + depth images + text query + bbox annotation.
    """

    def __init__(
        self,
        json_path: str,
        image_root: str,
        preprocessor: MultiModalPreprocessor,
        is_train: bool = True,
    ):
        """
        Args:
            json_path: Path to annotation JSON file.
            image_root: Root directory containing "visible/", "infrared/", "depth/" subdirs.
            preprocessor: Multi-modal image preprocessor.
            is_train: Whether this is training set (affects augmentations).
        """
        with open(json_path, "r") as f:
            self.data = json.load(f)

        self.image_root = image_root
        self.preprocessor = preprocessor
        self.is_train = is_train
        self.query_ids = list(self.data.keys())

    def __len__(self) -> int:
        return len(self.query_ids)

    def __getitem__(self, idx: int) -> Dict:
        query_id = self.query_ids[idx]
        ann = self.data[query_id]

        # Read image paths (FIX: previously depth and IR paths were swapped).
        rgb_path = os.path.join(self.image_root, ann["visible"])
        depth_path = os.path.join(self.image_root, ann["depth"]) if "depth" in ann else None
        ir_path = os.path.join(self.image_root, ann["infrared"]) if "infrared" in ann else None

        # Handle path naming (different JSON formats)
        if not os.path.exists(rgb_path):
            # Try alternative key names
            rgb_path = os.path.join(self.image_root, ann.get("visible", ann.get("rgb", "")))
            depth_path = os.path.join(self.image_root, ann.get("depth", ""))
            ir_path = os.path.join(self.image_root, ann.get("infrared", ann.get("ir", "")))

        # Read images
        rgb = read_rgb(rgb_path)
        depth = read_depth(depth_path)
        ir = read_ir(ir_path)

        # Preprocess
        images = self.preprocessor(rgb, depth, ir)

        # Text query
        query_text = ann.get("query", ann.get("text", ann.get("caption", "")))

        # BBox (normalized, [x1, y1, x2, y2])
        bbox = ann.get("bbox", None)
        if bbox is not None:
            bbox = torch.tensor(bbox, dtype=torch.float32)

        return {
            "query_id": query_id,
            "visible": images["visible"],        # (3, H, W)
            "infrared": images["infrared"],      # (3, H, W)
            "depth": images["depth"],            # (3, H, W)
            "query_text": query_text,
            "bbox": bbox,                        # (4,)
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for DataLoader."""
    keys = ["visible", "infrared", "depth"]
    result = {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}
    result["query_text"] = [b["query_text"] for b in batch]
    result["query_id"] = [b["query_id"] for b in batch]

    # Filter out samples without bbox (test set may not have labels)
    has_bbox = all(b["bbox"] is not None for b in batch)
    if has_bbox:
        result["bbox"] = torch.stack([b["bbox"] for b in batch], dim=0)
    else:
        result["bbox"] = None

    return result
