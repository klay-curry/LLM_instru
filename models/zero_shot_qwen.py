"""
Zero-shot multi-modal visual grounding using Qwen3-VL-8B-Instruct.

Pipeline (per query):
    RGB (PIL) + Depth (16-bit PNG, normalized -> PIL) + IR (PIL) + query text
        -> Qwen3-VL chat template
        -> model.generate (greedy)
        -> parse 4 floats [x1, y1, x2, y2] from response
        -> postprocess (clip, sort, validate)

Notes
-----
* The model's output format is not guaranteed to be strictly `[x1, y1, x2, y2]`.
  We try multiple regex patterns and fall back to a centered 0.1x0.1 box.
* Coordinate system: normalized [0, 1] with (0, 0) = top-left.
* Depth is 16-bit PNG (uint16) with valid range up to 20000 mm.
* IR is 3-channel pseudo-color PNG; we extract a single channel and treat it
  as a thermal/gray-scale image.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Image loading
# ----------------------------------------------------------------------------

def load_rgb(path: str, size: int = 448) -> Image.Image:
    """Load and resize RGB image. Returns a PIL.Image in RGB."""
    arr = cv2.imread(path, cv2.IMREAD_COLOR)
    if arr is None:
        raise FileNotFoundError(f"RGB not found: {path}")
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    if arr.shape[0] != size or arr.shape[1] != size:
        arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(arr)


def load_depth(path: str, size: int = 448, clip_max_mm: float = 20000.0) -> Image.Image:
    """Load 16-bit depth PNG, normalize to [0, 255] uint8, replicate to 3ch.

    Returns a PIL.Image (mode 'RGB').
    """
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Depth not found: {path}")
    arr = raw.astype(np.float32)
    arr = np.clip(arr, 0.0, clip_max_mm)
    arr = arr / clip_max_mm  # [0, 1]
    arr = (arr * 255.0).astype(np.uint8)
    arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[0] != size or arr.shape[1] != size:
        arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_NEAREST)
    return Image.fromarray(arr)


def load_ir(path: str, size: int = 448) -> Image.Image:
    """Load 3-channel IR (pseudo-color) and treat as a single-channel thermal image.

    Returns a PIL.Image (mode 'RGB').
    """
    arr = cv2.imread(path, cv2.IMREAD_COLOR)
    if arr is None:
        raise FileNotFoundError(f"IR not found: {path}")
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    if arr.shape[0] != size or arr.shape[1] != size:
        arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(arr)


def load_three_views(
    image_root: str,
    ann: dict,
    size: int = 448,
    depth_clip_max_mm: float = 20000.0,
) -> Tuple[Image.Image, Image.Image, Image.Image]:
    """Given one annotation dict, load the (visible, depth, ir) PIL triple.

    Order returned: (rgb, depth, ir). This is the canonical order used in the prompt.
    """
    rgb = load_rgb(os.path.join(image_root, ann["visible"]), size=size)
    depth = load_depth(
        os.path.join(image_root, ann["depth"]), size=size, clip_max_mm=depth_clip_max_mm
    )
    ir = load_ir(os.path.join(image_root, ann["infrared"]), size=size)
    return rgb, depth, ir


# ----------------------------------------------------------------------------
# Prompt templates (multiple for ensemble)
# ----------------------------------------------------------------------------

PROMPT_TEMPLATES = {
    "default": (
        "You are a precise visual grounding system. The first image is the visible-light RGB view. "
        "The second image is the depth map (brighter = farther). "
        "The third image is the infrared / thermal view (brighter = hotter).\n"
        "TASK: Locate the target described as: '{query}'\n"
        "OUTPUT FORMAT (STRICT): Output ONLY four numbers in the range [0.0, 1.0] separated by "
        "commas, representing (x1, y1, x2, y2) where (0,0) is the top-left corner and "
        "(1,1) is the bottom-right corner of the RGB image.\n"
        "Example output: 0.12,0.34,0.56,0.78\n"
        "Output:"
    ),
    "tight": (
        "You are a precise visual grounding system. "
        "Three views: RGB, depth (brighter=farther), infrared/thermal (brighter=hotter).\n"
        "TASK: Output a TIGHT bounding box around the target described as: '{query}'.\n"
        "The box should closely hug the target without too much extra padding.\n"
        "OUTPUT: ONLY four floats in [0,1] as 'x1,y1,x2,y2' (top-left, bottom-right).\n"
        "Example: 0.12,0.34,0.56,0.78\n"
        "Output:"
    ),
    "centered": (
        "You are a precise visual grounding system. "
        "Three views: RGB, depth (brighter=farther), infrared/thermal (brighter=hotter).\n"
        "TASK: Provide the bounding box of: '{query}'. Make sure the entire target is inside.\n"
        "OUTPUT: ONLY four decimals in [0,1] as 'x1,y1,x2,y2'. (0,0)=top-left, (1,1)=bottom-right.\n"
        "Output:"
    ),
    "concise": (
        "Locate '{query}' in the three views (RGB, depth, thermal).\n"
        "Reply with ONLY 4 floats in [0,1] for x1,y1,x2,y2."
    ),
}

PROMPT_TEMPLATE = PROMPT_TEMPLATES["default"]


def merge_bboxes(
    boxes: List[List[float]],
    method: str = "median",
) -> List[float]:
    """Merge multiple bbox predictions via median or mean.

    Each box is [x1, y1, x2, y2] in [0, 1].
    """
    if not boxes:
        return [0.45, 0.45, 0.55, 0.55]
    arr = np.array(boxes, dtype=np.float64)
    if method == "median":
        merged = np.median(arr, axis=0)
    elif method == "mean":
        merged = np.mean(arr, axis=0)
    else:
        raise ValueError(f"unknown merge method: {method}")
    return [float(v) for v in merged]


# ----------------------------------------------------------------------------
# Output parsing
# ----------------------------------------------------------------------------

# Match patterns like "[0.1, 0.2, 0.3, 0.4]" or "0.1,0.2,0.3,0.4"
_NUM4_RE = re.compile(
    r"\[?\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]?"
)


def parse_bbox(text: str) -> Optional[List[float]]:
    """Parse four floats from a free-form model response. Returns None if not found.

    Strategy:
        1. Look for the first match of NUM4_RE anywhere in the text.
        2. If matched, check whether all values are within [0, 1.5] (heuristic
           tolerance for cases where the model emits raw pixel coordinates and
           we should later re-normalize).
        3. If values look like pixel coordinates (e.g. max > 1.5 and image dim
           known), normalize by image size.
    """
    m = _NUM4_RE.search(text)
    if not m:
        return None
    vals = [float(m.group(i)) for i in range(1, 5)]
    # Detect pixel coordinates: assume image <= 4096, so any value > 1.5 suggests pixels
    if max(vals) > 1.5:
        # normalize assuming 1024x1024 OR just /1000 if max<=2000
        if max(vals) <= 2000:
            return [v / 1000.0 for v in vals]
        return [v / 4096.0 for v in vals]
    return vals


# ----------------------------------------------------------------------------
# Post-processing
# ----------------------------------------------------------------------------

def postprocess_bbox(
    bbox: Optional[List[float]],
    fallback_box: Tuple[float, float, float, float] = (0.45, 0.45, 0.55, 0.55),
) -> List[float]:
    """Validate and normalize a bbox to [0, 1] with x1<x2, y1<y2.

    Returns ``fallback_box`` if the bbox is missing, NaN, or has zero area.
    """
    if bbox is None or len(bbox) != 4:
        return list(fallback_box)
    x1, y1, x2, y2 = bbox
    if any(np.isnan(v) for v in (x1, y1, x2, y2)):
        return list(fallback_box)
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    x1 = float(np.clip(x1, 0.0, 1.0))
    x2 = float(np.clip(x2, 0.0, 1.0))
    y1 = float(np.clip(y1, 0.0, 1.0))
    y2 = float(np.clip(y2, 0.0, 1.0))
    if (x2 - x1) * (y2 - y1) < 1e-6:
        return list(fallback_box)
    return [x1, y1, x2, y2]


# ----------------------------------------------------------------------------
# Main inference class
# ----------------------------------------------------------------------------

@dataclass
class InferenceConfig:
    model_path: str
    image_size: int = 448
    depth_clip_max_mm: float = 20000.0
    max_new_tokens: int = 64
    do_sample: bool = False
    num_beams: int = 1
    device: str = "cuda:0"
    torch_dtype: str = "bfloat16"


class ZeroShotGrounder:
    """Zero-shot multi-modal visual grounder backed by Qwen3-VL-8B-Instruct."""

    def __init__(self, cfg: InferenceConfig):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration  # local import

        self.cfg = cfg
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
            cfg.torch_dtype
        ]
        logger.info("[grounder] loading processor from %s", cfg.model_path)
        self.processor = AutoProcessor.from_pretrained(cfg.model_path, trust_remote_code=True)
        # For decoder-only models we must left-pad so generation is aligned.
        try:
            self.processor.tokenizer.padding_side = "left"
        except Exception:
            pass
        logger.info("[grounder] loading model...")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            cfg.model_path,
            torch_dtype=dtype,
            device_map=cfg.device,
            trust_remote_code=True,
        ).eval()
        self.device = next(self.model.parameters()).device
        logger.info(
            "[grounder] ready on device=%s dtype=%s", self.device, next(self.model.parameters()).dtype
        )

    @torch.inference_mode()
    def predict_bbox(
        self,
        rgb: Image.Image,
        depth: Image.Image,
        ir: Image.Image,
        query: str,
    ) -> Tuple[List[float], str]:
        """Predict one normalized bbox. Returns (bbox, raw_text).

        bbox is always length-4 floats in [0, 1] with x1<x2, y1<y2.
        """
        bbox, raw = self.predict_batch([(rgb, depth, ir, query)])
        return bbox[0], raw[0]

    @torch.inference_mode()
    def predict_batch(
        self,
        samples: List[Tuple[Image.Image, Image.Image, Image.Image, str]],
    ) -> Tuple[List[List[float]], List[str]]:
        """Batched inference. Each element is (rgb, depth, ir, query).

        Returns (list_of_bbox, list_of_raw_text), in the same order as input.
        """
        # Build prompts
        texts = []
        all_images: List[Image.Image] = []
        for rgb, depth, ir, query in samples:
            prompt = PROMPT_TEMPLATE.format(query=query)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": rgb},
                        {"type": "image", "image": depth},
                        {"type": "image", "image": ir},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            texts.append(text)
            all_images.extend([rgb, depth, ir])

        inputs = self.processor(
            text=texts, images=all_images, return_tensors="pt", padding=True
        ).to(self.device)

        out = self.model.generate(
            **inputs,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=self.cfg.do_sample,
            num_beams=self.cfg.num_beams,
        )
        prompt_len = inputs["input_ids"].shape[1]
        gens = self.processor.batch_decode(
            out[:, prompt_len:], skip_special_tokens=True
        )

        boxes = [postprocess_bbox(parse_bbox(g)) for g in gens]
        return boxes, list(gens)

    def predict_from_annotation(
        self,
        image_root: str,
        ann: dict,
    ) -> Tuple[List[float], str]:
        rgb, depth, ir = load_three_views(
            image_root,
            ann,
            size=self.cfg.image_size,
            depth_clip_max_mm=self.cfg.depth_clip_max_mm,
        )
        return self.predict_bbox(rgb, depth, ir, ann["query"])

    def predict_batch_from_annotations(
        self,
        image_root: str,
        anns: List[dict],
    ) -> Tuple[List[List[float]], List[str]]:
        """Batch inference over a list of annotation dicts."""
        samples = []
        for ann in anns:
            rgb, depth, ir = load_three_views(
                image_root,
                ann,
                size=self.cfg.image_size,
                depth_clip_max_mm=self.cfg.depth_clip_max_mm,
            )
            samples.append((rgb, depth, ir, ann["query"]))
        return self.predict_batch(samples)

    @torch.inference_mode()
    def predict_ensemble_batch(
        self,
        image_root: str,
        anns: List[dict],
        prompt_names: List[str] = ("default", "tight", "centered"),
        merge: str = "median",
    ) -> Tuple[List[List[float]], List[List[str]]]:
        """Batched ensemble: run N prompts per sample and merge by median/mean."""
        # Pre-load images once
        loaded = []
        for ann in anns:
            rgb, depth, ir = load_three_views(
                image_root, ann,
                size=self.cfg.image_size,
                depth_clip_max_mm=self.cfg.depth_clip_max_mm,
            )
            loaded.append((rgb, depth, ir, ann["query"]))

        all_boxes_per_sample: List[List[List[float]]] = [[] for _ in loaded]
        all_raws_per_sample: List[List[str]] = [[] for _ in loaded]

        for name in prompt_names:
            prompt = PROMPT_TEMPLATES[name]
            texts, images_flat = [], []
            for rgb, depth, ir, query in loaded:
                full_prompt = prompt.format(query=query)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": rgb},
                            {"type": "image", "image": depth},
                            {"type": "image", "image": ir},
                            {"type": "text", "text": full_prompt},
                        ],
                    }
                ]
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                texts.append(text)
                images_flat.extend([rgb, depth, ir])

            inputs = self.processor(
                text=texts, images=images_flat, return_tensors="pt", padding=True
            ).to(self.device)
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=self.cfg.do_sample,
                num_beams=self.cfg.num_beams,
            )
            prompt_len = inputs["input_ids"].shape[1]
            gens = self.processor.batch_decode(
                out[:, prompt_len:], skip_special_tokens=True
            )

            for i, gen in enumerate(gens):
                all_raws_per_sample[i].append(gen)
                raw_box = parse_bbox(gen)
                all_boxes_per_sample[i].append(postprocess_bbox(raw_box))

        merged_boxes = [
            postprocess_bbox(merge_bboxes(b, method=merge))
            for b in all_boxes_per_sample
        ]
        return merged_boxes, all_raws_per_sample

    def close(self) -> None:
        # free CUDA memory
        del self.model
        del self.processor
        torch.cuda.empty_cache()