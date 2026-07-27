"""
Pre-compute Qwen3-VL semantic tokens for RefCOCO from local parquet.
Single-image processing (faster than batched due to variable image sizes).

Usage:
    python scripts/precompute_refcoco_features.py
"""

import logging
import os
import sys
import time
from pathlib import Path

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = "/root/autodl-fs/weights/Qwen3-VL-8B-Instruct"
COCO_ROOT = "/root/autodl-fs/datasets/coco/train2014"
OUTPUT_DIR = "/root/autodl-fs/datasets/precomputed_features"

SPLIT_FILES = {
    "refcoco_train": "/root/autodl-fs/datasets/refcoco/data/train-00000-of-00001-94431d5f4bd5b93f.parquet",
    "refcoco_val": "/root/autodl-fs/datasets/refcoco/data/validation-00000-of-00001-bfeafdc84ca37aa2.parquet",
    "refcocoplus_train": "/root/autodl-fs/datasets/refcocoplus/data/train-00000-of-00001-7294665695c630ee.parquet",
    "refcocoplus_val": "/root/autodl-fs/datasets/refcocoplus/data/validation-00000-of-00001-8c57d66282bc60c9.parquet",
    "refcocog_train": "/root/autodl-fs/datasets/refcocog/data/train-00000-of-00001-4fe3e6340cfb69ed.parquet",
    "refcocog_val": "/root/autodl-fs/datasets/refcocog/data/validation-00000-of-00001-15168dfe7b5961e5.parquet",
}


def load_samples(parquet_path: str):
    df = pd.read_parquet(parquet_path)
    samples = []
    for _, row in df.iterrows():
        fname = row["file_name"]
        img_path = os.path.join(COCO_ROOT, fname)
        if not os.path.exists(img_path):
            simpler = fname.rsplit("_", 1)[0] + ".jpg"
            img_path = os.path.join(COCO_ROOT, simpler)
        samples.append({
            "image_path": img_path,
            "query": row["captions"][0] if isinstance(row["captions"], list) else row["captions"],
            "bbox": list(row["bbox"]),
        })
    return samples


def main():
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info("Loading Qwen3-VL...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    ).eval()
    logger.info(f"GPU: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    for key, parquet_path in SPLIT_FILES.items():
        out_path = os.path.join(OUTPUT_DIR, f"{key}.pt")
        if os.path.exists(out_path):
            logger.info(f"Skip {key}")
            continue

        logger.info(f"\n=== {key} ===")
        samples = load_samples(parquet_path)
        logger.info(f"Samples: {len(samples)}")

        all_tokens, all_bbox, all_queries, all_paths = [], [], [], []
        errors = 0
        t0 = time.time()

        for i, s in enumerate(tqdm(samples, desc=key)):
            if not os.path.exists(s["image_path"]):
                errors += 1
                continue

            try:
                img = Image.open(s["image_path"]).convert("RGB")
                msgs = [{"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": f"Locate: {s['query']}"}
                ]}]
                text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], images=[img], return_tensors="pt").to(model.device)

                with torch.no_grad():
                    out = model(**inputs, output_hidden_states=True, return_dict=True)

                hidden = out.hidden_states[-1][0]
                mm = inputs.get("mm_token_type_ids")
                if mm is not None:
                    hidden = hidden[mm[0] == 0]

                all_tokens.append(hidden.cpu())
                all_bbox.append(torch.tensor(s["bbox"], dtype=torch.float32))
                all_queries.append(s["query"])
                all_paths.append(s["image_path"])

            except Exception as e:
                errors += 1
                if errors <= 3:
                    logger.warning(f"Error [{i}]: {e}")

            if i % 1000 == 999:
                torch.cuda.empty_cache()
                elapsed = time.time() - t0
                speed = (i + 1) / elapsed
                remaining = (len(samples) - i - 1) / speed
                logger.info(f"  {i+1}/{len(samples)} | {speed:.1f} img/s | ETA {remaining/60:.0f}min")

        elapsed = time.time() - t0
        speed = len(all_tokens) / elapsed if elapsed > 0 else 0
        logger.info(f"Done: {len(all_tokens)}/{len(samples)} (errors={errors}) in {elapsed/60:.0f}min ({speed:.1f} img/s)")

        torch.save({
            "semantic_tokens": all_tokens,
            "bbox": torch.stack(all_bbox),
            "queries": all_queries,
            "image_paths": all_paths,
        }, out_path)
        torch.cuda.empty_cache()

    logger.info("All done!")


if __name__ == "__main__":
    main()
