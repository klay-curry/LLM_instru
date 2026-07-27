"""Train a semantic-guided RGB locator from offline RefCOCO caches."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.refcoco_cache import CachedRefCOCODataset, collate_cached_refcoco
from models.semantic_grounding import SemanticGroundingModel
from utils.metrics import compute_acc_at_05, compute_giou, compute_iou

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-cache",
        default="/root/autodl-fs/datasets/precomputed_features/refcoco_train.pt",
    )
    parser.add_argument(
        "--val-cache",
        default="/root/autodl-fs/datasets/precomputed_features/refcoco_val.pt",
    )
    parser.add_argument(
        "--dinov2-checkpoint",
        default="/root/autodl-fs/weights/dinov2_vitb14_reg4_pretrain.pth",
    )
    parser.add_argument(
        "--output-dir",
        default="/root/autodl-tmp/experiments/exp007_refcoco_locator",
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--decoder-layers", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--l1-weight", type=float, default=5.0)
    parser.add_argument("--giou-weight", type=float, default=2.0)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--no-semantics", action="store_true")
    return parser.parse_args()


def configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "train.log"),
        ],
        force=True,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loader(
    dataset: CachedRefCOCODataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_cached_refcoco,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        **batch,
        "image": batch["image"].to(device, non_blocking=True),
        "semantic_tokens": batch["semantic_tokens"].to(
            device, non_blocking=True
        ),
        "semantic_padding_mask": batch["semantic_padding_mask"].to(
            device, non_blocking=True
        ),
        "bbox": batch["bbox"].to(device, non_blocking=True),
    }


def compute_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    l1_weight: float,
    giou_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    l1_loss = F.l1_loss(predictions, targets)
    giou_loss = (1.0 - compute_giou(predictions, targets)).mean()
    total = l1_weight * l1_loss + giou_weight * giou_loss
    return total, l1_loss, giou_loss


def save_checkpoint(
    path: Path,
    model: SemanticGroundingModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_acc: float,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "best_acc": best_acc,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "args": vars(args),
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: SemanticGroundingModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint["epoch"] + 1, checkpoint.get("best_acc", 0.0)


@torch.no_grad()
def evaluate(
    model: SemanticGroundingModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    predictions = []
    targets = []
    for batch in loader:
        batch = move_batch(batch, device)
        output = model(
            batch["image"],
            batch["semantic_tokens"],
            batch["semantic_padding_mask"],
        )
        predictions.append(output.float().cpu())
        targets.append(batch["bbox"].float().cpu())

    pred_boxes = torch.cat(predictions)
    gt_boxes = torch.cat(targets)
    iou = compute_iou(pred_boxes, gt_boxes)
    return {
        "acc_at_05": compute_acc_at_05(pred_boxes, gt_boxes),
        "mean_iou": iou.mean().item(),
        "median_iou": iou.median().item(),
        "invalid_boxes": float(
            (~(
                (pred_boxes[:, 0] < pred_boxes[:, 2])
                & (pred_boxes[:, 1] < pred_boxes[:, 3])
                & (pred_boxes >= 0).all(dim=1)
                & (pred_boxes <= 1).all(dim=1)
            )).sum().item()
        ),
    }


def run_smoke_test(
    model: SemanticGroundingModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    model.train()
    batch = move_batch(next(iter(loader)), device)
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        predictions = model(
            batch["image"],
            batch["semantic_tokens"],
            batch["semantic_padding_mask"],
        )
        loss, _, _ = compute_loss(
            predictions, batch["bbox"], args.l1_weight, args.giou_weight
        )
    loss.backward()
    optimizer.step()

    smoke_path = output_dir / "smoke_checkpoint.pt"
    torch.save(model.state_dict(), smoke_path)
    reloaded = SemanticGroundingModel(
        dinov2_checkpoint=args.dinov2_checkpoint,
        semantic_dim=loader.dataset.semantic_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.decoder_layers,
        use_semantics=not args.no_semantics,
    ).to(device)
    reloaded.load_state_dict(torch.load(smoke_path, map_location=device, weights_only=True))
    reloaded.eval()
    model.eval()
    with torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        before = model(
            batch["image"],
            batch["semantic_tokens"],
            batch["semantic_padding_mask"],
        )
        after = reloaded(
            batch["image"],
            batch["semantic_tokens"],
            batch["semantic_padding_mask"],
        )
    max_difference = (before - after).abs().max().item()
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3
    logger.info(
        "Smoke test passed | loss=%.4f | prediction_shape=%s | "
        "checkpoint_max_diff=%.3g | peak_memory=%.1fGB",
        loss.item(),
        tuple(predictions.shape),
        max_difference,
        peak_memory,
    )
    if max_difference > 1e-5:
        raise RuntimeError(f"Checkpoint reload changed predictions: {max_difference}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    configure_logging(output_dir)
    set_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training entrypoint")
    if args.gradient_accumulation < 1:
        raise ValueError("gradient_accumulation must be at least 1")

    if args.smoke_test:
        args.max_train_samples = args.max_train_samples or max(args.batch_size, 8)
        args.max_val_samples = args.max_val_samples or max(args.batch_size, 8)
        args.num_workers = 0

    with (output_dir / "args.json").open("w") as file:
        json.dump(vars(args), file, indent=2)

    train_dataset = CachedRefCOCODataset(
        args.train_cache,
        image_size=args.image_size,
        max_samples=args.max_train_samples,
    )
    val_dataset = CachedRefCOCODataset(
        args.val_cache,
        image_size=args.image_size,
        max_samples=args.max_val_samples,
    )
    train_loader = build_loader(
        train_dataset, args.batch_size, True, args.num_workers
    )
    val_loader = build_loader(
        val_dataset, args.batch_size, False, args.num_workers
    )

    device = torch.device("cuda")
    model = SemanticGroundingModel(
        dinov2_checkpoint=args.dinov2_checkpoint,
        semantic_dim=train_dataset.semantic_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.decoder_layers,
        use_semantics=not args.no_semantics,
    ).to(device)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    logger.info(
        "Train=%d Val=%d Trainable=%d Semantics=%s",
        len(train_dataset),
        len(val_dataset),
        sum(parameter.numel() for parameter in trainable_parameters),
        not args.no_semantics,
    )

    if args.smoke_test:
        run_smoke_test(model, train_loader, optimizer, device, args, output_dir)
        return

    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        start_epoch, best_acc = load_checkpoint(
            args.resume, model, optimizer, scheduler
        )
        logger.info("Resumed at epoch %d with best ACC %.4f", start_epoch, best_acc)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        samples_seen = 0
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()

        for step, batch in enumerate(train_loader):
            batch = move_batch(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                predictions = model(
                    batch["image"],
                    batch["semantic_tokens"],
                    batch["semantic_padding_mask"],
                )
                loss, l1_loss, giou_loss = compute_loss(
                    predictions,
                    batch["bbox"],
                    args.l1_weight,
                    args.giou_weight,
                )
                scaled_loss = loss / args.gradient_accumulation
            scaled_loss.backward()

            should_step = (
                (step + 1) % args.gradient_accumulation == 0
                or step + 1 == len(train_loader)
            )
            if should_step:
                torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            batch_size = batch["image"].shape[0]
            epoch_loss += loss.item() * batch_size
            samples_seen += batch_size
            if (step + 1) % args.log_interval == 0:
                elapsed = time.perf_counter() - started
                logger.info(
                    "Epoch %d Step %d/%d | loss=%.4f l1=%.4f giou=%.4f "
                    "| %.1f samples/s | mem=%.1fGB",
                    epoch + 1,
                    step + 1,
                    len(train_loader),
                    loss.item(),
                    l1_loss.item(),
                    giou_loss.item(),
                    samples_seen / elapsed,
                    torch.cuda.max_memory_allocated() / 1024**3,
                )

        metrics = evaluate(model, val_loader, device)
        scheduler.step()
        elapsed = time.perf_counter() - started
        logger.info(
            "Epoch %d/%d | loss=%.4f | ACC@0.5=%.4f meanIoU=%.4f "
            "medianIoU=%.4f invalid=%d | %.1fs | peak_mem=%.1fGB",
            epoch + 1,
            args.epochs,
            epoch_loss / samples_seen,
            metrics["acc_at_05"],
            metrics["mean_iou"],
            metrics["median_iou"],
            int(metrics["invalid_boxes"]),
            elapsed,
            torch.cuda.max_memory_allocated() / 1024**3,
        )

        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            max(best_acc, metrics["acc_at_05"]),
            args,
        )
        if metrics["acc_at_05"] >= best_acc:
            best_acc = metrics["acc_at_05"]
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_acc,
                args,
            )


if __name__ == "__main__":
    main()
