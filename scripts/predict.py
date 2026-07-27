"""
Inference script for multi-modal visual grounding.

Usage:
    python scripts/predict.py --config configs/default.yaml --checkpoint outputs/best_model.pth --output predictions.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import VisualGroundingDataset, collate_fn
from data.preprocessing import MultiModalPreprocessor
from models.grounding_model import build_model
from engine.evaluator import Evaluator


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main(args):
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Preprocessor
    preprocessor = MultiModalPreprocessor(
        image_size=config["data"]["image_size"],
        depth_clip_max=config["preprocessing"]["depth"]["clip_max"],
    )

    # Dataset (test)
    test_dataset = VisualGroundingDataset(
        json_path=args.test_json,
        image_root=config["data"]["image_root"],
        preprocessor=preprocessor,
        is_train=False,
    )
    logger.info(f"Test samples: {len(test_dataset)}")

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Model
    model = build_model(config)
    model = model.to(device)

    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Checkpoint loaded (epoch {checkpoint.get('epoch', 0)}, val_acc={checkpoint.get('val_acc', 'N/A')})")

    # Predict
    evaluator = Evaluator(model, device)
    predictions = evaluator.predict_and_save(
        dataloader=test_loader,
        save_path=args.output,
        ref_json=args.test_json,
    )
    logger.info(f"Predictions saved to {args.output}")
    logger.info(f"Total predictions: {len(predictions)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with multi-modal grounding model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--test-json", type=str, required=True, help="Test JSON annotation file")
    parser.add_argument("--output", type=str, default="outputs/predictions.json", help="Output JSON path")
    parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    args = parser.parse_args()
    main(args)
