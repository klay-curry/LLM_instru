"""
Split queries.json into train/val partitions by image id.
This is a *very simple* splitter used to verify the data-loading pipeline;
we keep all queries tied to the same image in the same split so that
evaluation leakage is impossible.
"""

import argparse
import json
import os
import random
from typing import Dict, List, Tuple


def parse_image_id(query_id: str) -> str:
    """query_id format: '000002_001' -> image_id '000002'."""
    return query_id.split("_")[0]


def split_by_image(
    queries: Dict[str, dict],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """Group queries by image id and split 8:2 (train:val)."""
    image_to_qids: Dict[str, List[str]] = {}
    for qid in queries:
        image_to_qids.setdefault(parse_image_id(qid), []).append(qid)

    image_ids = sorted(image_to_qids.keys())
    rng = random.Random(seed)
    rng.shuffle(image_ids)

    val_n = int(len(image_ids) * val_ratio)
    val_images = set(image_ids[:val_n])

    train, val = {}, {}
    for qid, ann in queries.items():
        if parse_image_id(qid) in val_images:
            val[qid] = ann
        else:
            train[qid] = ann

    return train, val


def main() -> None:
    parser = argparse.ArgumentParser(description="Split queries.json by image id (8:2).")
    parser.add_argument("--queries", required=True, help="Path to queries.json")
    parser.add_argument("--out-dir", required=True, help="Output directory for train.json/val.json")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.queries, "r") as f:
        queries = json.load(f)
    print(f"[split] total queries: {len(queries)}")

    train, val = split_by_image(queries, val_ratio=args.val_ratio, seed=args.seed)
    print(f"[split] train: {len(train)} queries, val: {len(val)} queries")

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "train.json")
    val_path = os.path.join(args.out_dir, "val.json")
    with open(train_path, "w") as f:
        json.dump(train, f, indent=2)
    with open(val_path, "w") as f:
        json.dump(val, f, indent=2)
    print(f"[split] wrote {train_path}")
    print(f"[split] wrote {val_path}")


if __name__ == "__main__":
    main()