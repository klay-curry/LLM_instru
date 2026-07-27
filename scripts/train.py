"""
Training script for multi-modal visual grounding model.

Usage:
    python scripts/train.py --config configs/default.yaml
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import VisualGroundingDataset, collate_fn
from data.preprocessing import MultiModalPreprocessor
from models.grounding_model import MultiModalGroundingModel, build_model
from engine.trainer import Trainer
from engine.evaluator import Evaluator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log"),
    ],
)
logger = logging.getLogger(__name__)


def main(args):
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Output dir
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Preprocessor
    preprocessor = MultiModalPreprocessor(
        image_size=config["data"]["image_size"],
        depth_clip_max=config["preprocessing"]["depth"]["clip_max"],
    )

    # Datasets
    train_dataset = VisualGroundingDataset(
        json_path=config["data"]["train_json_path"],
        image_root=config["data"]["image_root"],
        preprocessor=preprocessor,
        is_train=True,
    )
    val_dataset = VisualGroundingDataset(
        json_path=config["data"]["val_json_path"],
        image_root=config["data"]["image_root"],
        preprocessor=preprocessor,
        is_train=False,
    )
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Model
    model = build_model(config)
    model = model.to(device)

    # Log trainable params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params: {total_params:,}, Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

    # Resume from checkpoint
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        output_dir=str(output_dir),
    )

    # Train
    best_acc = trainer.train(num_epochs=config["training"]["num_epochs"])
    logger.info(f"Training completed. Best ACC@0.5: {best_acc:.4f}")

    # Final evaluation
    logger.info("Running final evaluation on validation set...")
    evaluator = Evaluator(model, device)
    final_acc = evaluator.evaluate(val_loader)
    logger.info(f"Final validation ACC@0.5: {final_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multi-modal visual grounding model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file path")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    args = parser.parse_args()
    main(args)
