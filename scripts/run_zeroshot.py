"""
Run zero-shot multi-modal visual grounding inference over queries.json
and write predictions.json in the required submission format.

Usage:
    python scripts/run_zeroshot.py --config configs/zeroshot.yaml [--limit N]

Output:
    predictions.json - {query_id: {visible, infrared, depth, query, bbox}}
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional

import yaml

# Make `code/LLM/` importable so we can `import models.zero_shot_qwen`
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models.zero_shot_qwen import InferenceConfig, ZeroShotGrounder  # noqa: E402


def setup_logger(log_path: Optional[str]) -> logging.Logger:
    log = logging.getLogger("run_zeroshot")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(sh)
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        fh = logging.FileHandler(log_path, mode="a")
        fh.setFormatter(fmt)
        log.addHandler(fh)
    return log


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_inference_config(cfg: dict) -> InferenceConfig:
    return InferenceConfig(
        model_path=cfg["model"]["name"],
        image_size=int(cfg["image"]["size"]),
        depth_clip_max_mm=float(cfg["image"]["depth_clip_max_mm"]),
        max_new_tokens=int(cfg["generation"]["max_new_tokens"]),
        do_sample=bool(cfg["generation"]["do_sample"]),
        num_beams=int(cfg["generation"]["num_beams"]),
        device=cfg["model"].get("device_map", "cuda:0"),
        torch_dtype=cfg["model"].get("torch_dtype", "bfloat16"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot Qwen3-VL visual grounding.")
    parser.add_argument("--config", default="configs/zeroshot.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Override config inference.limit")
    parser.add_argument("--start", type=int, default=None, help="Override config inference.start_index")
    parser.add_argument("--end", type=int, default=None, help="Override config inference.end_index")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--output", default=None, help="Override output json path")
    parser.add_argument("--log", default=None, help="Override log path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    log_path = args.log or cfg["paths"].get("output_log", "outputs/run.log")
    log = setup_logger(log_path)

    queries_path = cfg["paths"]["queries_json"]
    image_root = cfg["paths"]["image_root"]
    output_json = args.output or cfg["paths"]["output_json"]
    limit = args.limit if args.limit is not None else int(cfg["inference"].get("limit", -1))
    start = args.start if args.start is not None else int(cfg["inference"].get("start_index", 0))
    end = args.end if args.end is not None else int(cfg["inference"].get("end_index", -1))
    batch_size = args.batch_size if args.batch_size is not None else int(cfg["inference"].get("batch_size", 4))

    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    log.info("[run] queries: %s", queries_path)
    log.info("[run] image_root: %s", image_root)
    log.info("[run] output: %s", output_json)
    with open(queries_path, "r") as f:
        queries: Dict[str, dict] = json.load(f)
    qids = sorted(queries.keys())
    if start > 0:
        qids = qids[start:]
    if end and end > 0:
        qids = qids[:end - start if start else end]
    if limit and limit > 0:
        qids = qids[:limit]
    log.info("[run] processing %d queries (start=%d, end=%s, limit=%s, batch_size=%d)", len(qids), start, end, limit, batch_size)

    infer_cfg = build_inference_config(cfg)
    grounder = ZeroShotGrounder(infer_cfg)

    # Resume from existing prediction file if present (idempotent restart).
    predictions: Dict[str, dict] = {}
    if os.path.exists(output_json):
        try:
            with open(output_json, "r") as f:
                predictions = json.load(f)
            log.info("[run] resuming from %s with %d existing entries", output_json, len(predictions))
        except Exception as e:
            log.warning("[run] could not resume from %s: %s; starting fresh", output_json, e)
            predictions = {}

    t0 = time.time()
    flush_every = int(cfg.get("inference", {}).get("flush_every", 5))

    # Only process queries that haven't been predicted yet (or whose bbox is None)
    pending = [qid for qid in qids if qid not in predictions or not predictions[qid].get("bbox")]
    log.info("[run] %d queries pending (already have %d)", len(pending), len(predictions))

    # Batched inference loop
    success = 0
    parse_fail = 0
    for batch_start in range(0, len(pending), batch_size):
        batch_qids = pending[batch_start: batch_start + batch_size]
        anns = [queries[qid] for qid in batch_qids]
        t1 = time.time()
        try:
            boxes, raws = grounder.predict_batch_from_annotations(image_root, anns)
            success += len(batch_qids)
        except Exception as e:
            log.warning("[run] batch exception: %s; falling back to per-sample", e)
            boxes, raws = [], []
            for ann in anns:
                try:
                    b, r = grounder.predict_from_annotation(image_root, ann)
                    boxes.append(b); raws.append(r)
                except Exception as e2:
                    log.warning("[run] sample exception: %s", e2)
                    boxes.append([0.45, 0.45, 0.55, 0.55]); raws.append(f"ERROR: {e2}")
                    parse_fail += 1

        elapsed = time.time() - t1
        for qid, bbox, raw in zip(batch_qids, boxes, raws):
            ann = queries[qid]
            predictions[qid] = {
                "visible": ann["visible"],
                "infrared": ann["infrared"],
                "depth": ann["depth"],
                "query": ann["query"],
                "bbox": bbox,
            }

        total_done = len(predictions)
        done_now = total_done
        total_elapsed = time.time() - t0
        avg = total_elapsed / max(batch_start // batch_size + 1, 1)
        per_q = elapsed / max(len(batch_qids), 1)

        log.info(
            "[run] batch done: %d/%d (this batch=%d in %.2fs, %.2fs/q, batch_avg=%.2fs) last_qid=%s bbox=%s",
            done_now, len(qids), len(batch_qids), elapsed, per_q, avg, batch_qids[-1], boxes[-1],
        )

        # Periodic flush so progress survives an interruption
        if ((batch_start // batch_size) + 1) % flush_every == 0:
            try:
                tmp_path = output_json + ".tmp"
                with open(tmp_path, "w") as f:
                    json.dump(predictions, f, indent=2)
                os.replace(tmp_path, output_json)
                log.info("[run] flushed %d entries to %s", len(predictions), output_json)
            except Exception as e:
                log.warning("[run] flush failed: %s", e)

    total_time = time.time() - t0
    log.info(
        "[run] done. %d queries in %.1fs (avg %.2fs/q, success=%d, errors=%d)",
        len(qids), total_time, total_time / max(len(qids), 1), success, parse_fail,
    )

    with open(output_json, "w") as f:
        json.dump(predictions, f, indent=2)
    log.info("[run] wrote %s with %d entries", output_json, len(predictions))


if __name__ == "__main__":
    main()