"""
Multi-modal image preprocessing pipeline.
Handles RGB, Depth (16-bit single-channel), and Infrared (3-channel pseudo-color).
"""

import torch
import numpy as np
import cv2


def read_rgb(path: str) -> np.ndarray:
    """Read RGB image. Returns HxWx3 uint8 array in [0, 255]."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"RGB image not found: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_depth(path: str) -> np.ndarray:
    """Read depth image. Returns HxW float32 array in mm.
    Valid range: ~300mm to ~20000mm. 0 = invalid.
    Uses IMREAD_UNCHANGED so the 16-bit values are preserved.
    """
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Depth image not found: {path}")
    # cv2 with IMREAD_UNCHANGED on a 16-bit PNG returns uint16; on 8-bit PNG returns uint8.
    # Either way, cast to float32 (units are mm).
    return depth.astype(np.float32)


def read_ir(path: str) -> np.ndarray:
    """Read infrared image. Returns HxWx3 uint8.
    3 channels are identical (pseudo-color stack).
    """
    ir = cv2.imread(path, cv2.IMREAD_COLOR)
    if ir is None:
        raise FileNotFoundError(f"IR image not found: {path}")
    return cv2.cvtColor(ir, cv2.COLOR_BGR2RGB)


class DepthNormalizer:
    """Normalize depth image to [0, 1] range with outlier clipping."""

    def __init__(self, clip_max: float = 20000.0):
        self.clip_max = clip_max

    def __call__(self, depth: np.ndarray) -> np.ndarray:
        # Remove invalid depths (0 or very small values)
        depth = np.clip(depth, 0, self.clip_max)
        # Normalize to [0, 1]
        depth = depth / self.clip_max
        # Convert to 3 channels (repeat) for ViT compatibility
        depth_3ch = np.stack([depth, depth, depth], axis=-1)
        return depth_3ch.astype(np.float32)


class IRNormalizer:
    """Normalize infrared image."""

    def __init__(self, use_single_channel: bool = True):
        self.use_single_channel = use_single_channel

    def __call__(self, ir: np.ndarray) -> np.ndarray:
        if self.use_single_channel:
            # All 3 channels are identical, extract one
            gray = ir[:, :, 0].astype(np.float32)
            # Simple min-max or mean-std normalization
            gray = (gray - gray.mean()) / (gray.std() + 1e-8)
            gray = np.clip(gray, -3.0, 3.0)  # clip outliers
            gray = (gray + 3.0) / 6.0  # map to [0, 1]
            ir_norm = np.stack([gray, gray, gray], axis=-1)
        else:
            ir_norm = ir.astype(np.float32) / 255.0
        return ir_norm.astype(np.float32)


class RGBPreprocessor:
    """Normalize RGB image to [0, 1]."""

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        return rgb.astype(np.float32) / 255.0


class MultiModalPreprocessor:
    """Unified preprocessor for all three visual modalities.
    
    Output: dict with keys "visible", "infrared", "depth",
    each being a CHW tensor normalized to [0,1].
    """

    def __init__(self, image_size: int = 448, depth_clip_max: float = 20000.0):
        self.image_size = image_size
        self.rgb_proc = RGBPreprocessor()
        self.depth_proc = DepthNormalizer(clip_max=depth_clip_max)
        self.ir_proc = IRNormalizer(use_single_channel=True)

    def __call__(self, rgb: np.ndarray, depth: np.ndarray, ir: np.ndarray):
        # Preprocess each modality
        rgb = self.rgb_proc(rgb)
        depth = self.depth_proc(depth)
        ir = self.ir_proc(ir)

        # Resize to target size
        rgb = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        ir = cv2.resize(ir, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)

        # Convert to CHW tensors
        rgb = torch.from_numpy(rgb).permute(2, 0, 1).float()  # (3, H, W)
        depth = torch.from_numpy(depth).permute(2, 0, 1).float()  # (3, H, W)
        ir = torch.from_numpy(ir).permute(2, 0, 1).float()  # (3, H, W)

        return {
            "visible": rgb,
            "infrared": ir,
            "depth": depth,
        }
