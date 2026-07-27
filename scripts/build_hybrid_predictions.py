"""
Build hybrid predictions by fusing Qwen3-VL and Grounding DINO outputs.

Strategy: use Qwen3-VL's bbox as the primary output. When Grounding DINO
has a high-confidence detection AND it agrees (IoU > 0.3) with Qwen3-VL,
take the average (mean) bbox. Otherwise keep Qwen3-VL's.

This is conservative: it only changes Qwen3-VL's output when there's
strong agreement between the two models, which empirically tends to
improve precision (averaging reduces individual model bias).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import List, Optional, Tuple


def iou_xyxy(a: List[float], b: List[float]) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def merge_mean(a: List[float], b: List[float]) -> List[float]:
    return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2, (a[3] + b[3]) / 2]


def build_hybrid(
    qwen_preds_path: str,
    gdino_preds_path: str,
    output_path: str,
    iou_threshold: float = 0.3,
    min_gdino_score: float = 0.25,
) -> None:
    with open(qwen_preds_path) as f:
        qwen = json.load(f)
    with open(gdino_preds_path) as f:
        gdino = json.load(f)

    n_total = 0
    n_kept = 0
    n_merged = 0
    n_gdino_only = 0  # not used currently

    out = {}
    for qid in sorted(qwen.keys()):
        n_total += 1
        qb = qwen[qid]["bbox"]
        # Default: use Qwen
        out_bbox = qb
        if qid in gdino:
            gdata = gdino[qid]
            gb = gdata["bbox"]
            gs = gdata.get("_score", 0.0)
            iou_val = iou_xyxy(qb, gb)
            if gs >= min_gdino_score and iou_val >= iou_threshold:
                # Strong agreement -> take mean (precision usually improves)
                out_bbox = merge_mean(qb, gb)
                n_merged += 1
            else:
                n_kept += 1
        out[qid] = {
            "visible": qwen[qid]["visible"],
            "infrared": qwen[qid]["infrared"],
            "depth": qwen[qid]["depth"],
            "query": qwen[qid]["query"],
            "bbox": out_bbox,
        }

    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(
        f"[hybrid] wrote {output_path}: {n_total} total, "
        f"{n_merged} merged (IoU>={iou_threshold}), {n_kept} kept Qwen3-VL"
    )


if __name__ == "__main__":
    qwen_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/predictions.json"
    gdino_path = sys.argv[2] if len(sys.argv) > 2 else "outputs/gdino_full.json"
    output_path = sys.argv[3] if len(sys.argv) > 3 else "outputs/predictions_hybrid.json"
    iou_thr = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3
    min_score = float(sys.argv[5]) if len(sys.argv) > 5 else 0.25
    build_hybrid(qwen_path, gdino_path, output_path, iou_thr, min_score)