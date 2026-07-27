"""
Prepare submission ZIP file from prediction JSON.

Usage:
    python scripts/prepare_submit.py --prediction outputs/predictions.json --output submit.zip
"""

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

# Make project root importable so `from utils.postprocess import ...` works
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.postprocess import validate_bbox


def main(args):
    # Load predictions
    with open(args.prediction, "r") as f:
        predictions = json.load(f)

    # Validate predictions
    valid_count = 0
    invalid_count = 0
    for qid, data in predictions.items():
        bbox = data.get("bbox", [])
        if validate_bbox(bbox):
            valid_count += 1
        else:
            invalid_count += 1
            logger.warning(f"Invalid bbox for {qid}: {bbox}")

    logger.info(f"Validation: {valid_count} valid, {invalid_count} invalid")

    # Create ZIP
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(args.prediction, arcname="predictions.json")

    logger.info(f"Submission ZIP saved to {args.output}")
    logger.info(f"Size: {Path(args.output).stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Prepare submission ZIP")
    parser.add_argument("--prediction", type=str, required=True, help="Prediction JSON path")
    parser.add_argument("--output", type=str, default="submit.zip", help="Output ZIP path")
    args = parser.parse_args()
    main(args)
