"""Benchmark RefCOCO training throughput across candidate batch sizes."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.refcoco_cache import CachedRefCOCODataset, collate_cached_refcoco
from models.semantic_grounding import SemanticGroundingModel
from utils.metrics import compute_giou

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        default="/root/autodl-fs/datasets/precomputed_features/refcoco_train.pt",
    )
    parser.add_argument(
        "--dinov2-checkpoint",
        default="/root/autodl-fs/weights/dinov2_vitb14_reg4_pretrain.pth",
    )
    parser.add_argument(
        "--output",
        default="/root/autodl-tmp/experiments/exp008_refcoco_pilot/batch_probe.json",
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--timed-steps", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = CachedRefCOCODataset(
        args.cache,
        image_size=args.image_size,
        max_samples=max(args.batch_sizes),
    )
    device = torch.device("cuda")
    model = SemanticGroundingModel(
        dinov2_checkpoint=args.dinov2_checkpoint,
        semantic_dim=dataset.semantic_dim,
        use_semantics=True,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-4,
    )
    results = []

    for batch_size in args.batch_sizes:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_cached_refcoco,
            pin_memory=True,
        )
        cpu_batch = next(iter(loader))
        batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in cpu_batch.items()
        }
        logger.info("Probing batch=%d", batch_size)

        try:
            model.train()
            timings = []
            torch.cuda.reset_peak_memory_stats()
            total_steps = args.warmup_steps + args.timed_steps
            for step in range(total_steps):
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    predictions = model(
                        batch["image"],
                        batch["semantic_tokens"],
                        batch["semantic_padding_mask"],
                    )
                    loss = (
                        5.0 * F.l1_loss(predictions, batch["bbox"])
                        + 2.0
                        * (1.0 - compute_giou(predictions, batch["bbox"])).mean()
                    )
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                if step >= args.warmup_steps:
                    timings.append(elapsed)

            mean_step_seconds = sum(timings) / len(timings)
            result = {
                "batch_size": batch_size,
                "status": "ok",
                "mean_step_seconds": mean_step_seconds,
                "samples_per_second": batch_size / mean_step_seconds,
                "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
                "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
                "loss": loss.item(),
            }
        except torch.OutOfMemoryError as error:
            result = {
                "batch_size": batch_size,
                "status": "oom",
                "error": str(error),
            }
        results.append(result)
        logger.info("Result: %s", result)
        output_path.write_text(json.dumps(results, indent=2))

        del batch, cpu_batch, loader
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()

    logger.info("Wrote %s", output_path)


if __name__ == "__main__":
    main()
