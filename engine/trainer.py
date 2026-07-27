"""Training loop for the multi-modal grounding model."""

import os
import time
import logging
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
from tqdm import tqdm

from utils.metrics import compute_acc_at_05


logger = logging.getLogger(__name__)


def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps: int, num_training_steps: int, min_lr_ratio: float = 0.01
):
    """Create a schedule with a constant warmup and then cosine decay."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


class Trainer:
    """Trainer for MultiModalGroundingModel."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: torch.device,
        output_dir: str = "outputs",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer
        train_cfg = config["training"]
        param_groups = model.get_trainable_params()
        self.optimizer = AdamW(
            [
                {"params": param_groups["backbone"], "lr": train_cfg["optimizer"]["lr_backbone"]},
                {"params": param_groups["lora"], "lr": train_cfg["optimizer"]["lr_lora"]},
                {"params": param_groups["decoder"], "lr": train_cfg["optimizer"]["lr_decoder"]},
                {"params": param_groups["vlm_fusion"], "lr": train_cfg["optimizer"]["lr_vlm_fusion"]},
            ],
            weight_decay=train_cfg["optimizer"]["weight_decay"],
            betas=tuple(train_cfg["optimizer"]["betas"]),
        )

        # Scheduler
        total_steps = len(train_loader) * train_cfg["num_epochs"]
        warmup_steps = int(total_steps * train_cfg["scheduler"]["warmup_ratio"])
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps
        )

        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=(train_cfg["mixed_precision"] == "fp16"))
        self.amp_dtype = torch.bfloat16 if train_cfg["mixed_precision"] == "bf16" else torch.float16

        # Loss config
        self.loss_cfg = train_cfg["loss"]

        # Training state
        self.start_epoch = 0
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.early_stop_patience = train_cfg.get("early_stopping_patience", 10)

    def train(self, num_epochs: int = None):
        """Run full training loop."""
        if num_epochs is None:
            num_epochs = self.config["training"]["num_epochs"]

        logger.info(f"Starting training for {num_epochs} epochs...")
        logger.info(f"Output dir: {self.output_dir}")

        for epoch in range(self.start_epoch, num_epochs):
            # Train
            train_loss = self._train_epoch(epoch)
            
            # Evaluate
            val_acc = self._evaluate(epoch)

            # Save checkpoint
            self._save_checkpoint(epoch, val_acc)

            # Early stopping
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                # Save best model
                self._save_checkpoint(epoch, val_acc, is_best=True)
            else:
                self.patience_counter += 1
                logger.info(
                    f"Early stopping patience: {self.patience_counter}/{self.early_stop_patience}"
                )
                if self.patience_counter >= self.early_stop_patience:
                    logger.info("Early stopping triggered!")
                    break

        logger.info(f"Training complete! Best validation ACC@0.5: {self.best_val_acc:.4f}")
        return self.best_val_acc

    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            visible = batch["visible"].to(self.device)
            infrared = batch["infrared"].to(self.device)
            depth = batch["depth"].to(self.device)
            target_bbox = batch["bbox"].to(self.device)
            query_texts = batch["query_text"]

            # Forward
            with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                pred_dict = self.model(
                    visible=visible,
                    infrared=infrared,
                    depth=depth,
                    query_texts=query_texts,
                )
                loss = self.model.compute_loss(pred_dict, target_bbox, self.loss_cfg)

            # Backward
            self.optimizer.zero_grad()
            if self.amp_dtype == torch.float16:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config["training"]["clip_grad_norm"]
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config["training"]["clip_grad_norm"]
                )
                self.optimizer.step()

            self.scheduler.step()

            # Logging
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch} - Train Loss: {avg_loss:.4f}")
        return avg_loss

    def _evaluate(self, epoch: int) -> float:
        """Evaluate on validation set."""
        self.model.eval()
        all_pred_boxes = []
        all_gt_boxes = []

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
        with torch.no_grad():
            for batch in pbar:
                visible = batch["visible"].to(self.device)
                infrared = batch["infrared"].to(self.device)
                depth = batch["depth"].to(self.device)
                target_bbox = batch["bbox"].to(self.device)
                query_texts = batch["query_text"]

                with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                    pred_dict = self.model(
                        visible=visible,
                        infrared=infrared,
                        depth=depth,
                        query_texts=query_texts,
                    )
                    pred_bbox = pred_dict["pred_bbox"]

                all_pred_boxes.append(pred_bbox.cpu())
                all_gt_boxes.append(target_bbox.cpu())

        pred_boxes = torch.cat(all_pred_boxes, dim=0)
        gt_boxes = torch.cat(all_gt_boxes, dim=0)
        val_acc = compute_acc_at_05(pred_boxes, gt_boxes)

        logger.info(f"Epoch {epoch} - Val ACC@0.5: {val_acc:.4f}")
        return val_acc

    def _save_checkpoint(self, epoch: int, val_acc: float, is_best: bool = False):
        """Save model checkpoint."""
        filename = "best_model.pth" if is_best else f"checkpoint_epoch_{epoch}.pth"
        save_path = self.output_dir / filename

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_acc": val_acc,
            "best_val_acc": self.best_val_acc,
            "config": self.config,
        }
        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved: {save_path} (val_acc={val_acc:.4f})")

        # Also save as latest
        latest_path = self.output_dir / "latest.pth"
        torch.save(checkpoint, latest_path)
