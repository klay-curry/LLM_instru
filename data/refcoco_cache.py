"""Offline RefCOCO dataset backed by precomputed Qwen semantic tokens."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

_IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGE_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class CachedRefCOCODataset(Dataset):
    """Load RGB images, normalized boxes, and cached Qwen tokens."""

    def __init__(
        self,
        cache_path: str | Path,
        image_size: int = 224,
        max_samples: int | None = None,
    ) -> None:
        self.cache_path = Path(cache_path)
        if not self.cache_path.is_file():
            raise FileNotFoundError(f"RefCOCO cache not found: {self.cache_path}")

        logger.info("Loading feature cache: %s", self.cache_path)
        cache = torch.load(self.cache_path, map_location="cpu", weights_only=False)
        required = {"semantic_tokens", "bbox", "queries", "image_paths"}
        missing = required.difference(cache)
        if missing:
            raise KeyError(f"Cache is missing fields: {sorted(missing)}")

        lengths = {key: len(cache[key]) for key in required}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Cache fields have different lengths: {lengths}")

        size = next(iter(lengths.values()))
        if max_samples is not None:
            size = min(size, max_samples)
        self.semantic_tokens = cache["semantic_tokens"][:size]
        self.boxes_xyxy_pixels = cache["bbox"][:size].float()
        self.queries = cache["queries"][:size]
        self.image_paths = cache["image_paths"][:size]
        self.image_size = image_size

        token_dims = {tokens.shape[-1] for tokens in self.semantic_tokens}
        if len(token_dims) != 1:
            raise ValueError(f"Inconsistent semantic token dimensions: {token_dims}")
        self.semantic_dim = token_dims.pop()
        logger.info(
            "Loaded %d samples with semantic_dim=%d", size, self.semantic_dim
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path = Path(self.image_paths[index])
        if not image_path.is_file():
            raise FileNotFoundError(f"COCO image not found: {image_path}")

        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
            width, height = image.size
            resized = image.resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )

        image_tensor = torch.from_numpy(np.asarray(resized).copy())
        image_tensor = image_tensor.permute(2, 0, 1).float().div_(255.0)
        image_tensor = (image_tensor - _IMAGE_MEAN) / _IMAGE_STD

        scale = torch.tensor([width, height, width, height], dtype=torch.float32)
        bbox = self.boxes_xyxy_pixels[index] / scale
        if not torch.isfinite(bbox).all():
            raise ValueError(f"Non-finite bbox at sample {index}: {bbox.tolist()}")
        if not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
            raise ValueError(f"Invalid xyxy bbox at sample {index}: {bbox.tolist()}")
        if not ((bbox >= 0).all() and (bbox <= 1).all()):
            raise ValueError(f"Out-of-range bbox at sample {index}: {bbox.tolist()}")

        query = self.queries[index]
        if isinstance(query, np.ndarray):
            query = " | ".join(str(part) for part in query.tolist())
        elif isinstance(query, (list, tuple)):
            query = " | ".join(str(part) for part in query)
        else:
            query = str(query)

        return {
            "image": image_tensor,
            "semantic_tokens": self.semantic_tokens[index].float(),
            "bbox": bbox,
            "query": query,
            "image_path": str(image_path),
        }


def collate_cached_refcoco(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable-length semantic tokens and create their padding mask."""
    max_tokens = max(item["semantic_tokens"].shape[0] for item in batch)
    semantic_dim = batch[0]["semantic_tokens"].shape[1]
    semantic_tokens = torch.zeros(len(batch), max_tokens, semantic_dim)
    semantic_padding_mask = torch.ones(len(batch), max_tokens, dtype=torch.bool)

    for index, item in enumerate(batch):
        length = item["semantic_tokens"].shape[0]
        semantic_tokens[index, :length] = item["semantic_tokens"]
        semantic_padding_mask[index, :length] = False

    return {
        "image": torch.stack([item["image"] for item in batch]),
        "semantic_tokens": semantic_tokens,
        "semantic_padding_mask": semantic_padding_mask,
        "bbox": torch.stack([item["bbox"] for item in batch]),
        "query": [item["query"] for item in batch],
        "image_path": [item["image_path"] for item in batch],
    }
