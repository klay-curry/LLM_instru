"""Evaluation and inference utilities."""

import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.metrics import compute_acc_at_05
from utils.postprocess import batch_postprocess


class Evaluator:
    """Evaluator for visual grounding predictions."""

    def __init__(self, model: torch.nn.Module, device: torch.device, amp_dtype=torch.bfloat16):
        self.model = model
        self.device = device
        self.amp_dtype = amp_dtype

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate model on a labeled dataset. Returns ACC@0.5."""
        self.model.eval()
        all_pred = []
        all_gt = []

        pbar = tqdm(dataloader, desc="Evaluating")
        for batch in pbar:
            visible = batch["visible"].to(self.device)
            infrared = batch["infrared"].to(self.device)
            depth = batch["depth"].to(self.device)
            query_texts = batch["query_text"]
            gt_boxes = batch["bbox"].to(self.device)

            with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                pred_dict = self.model(
                    visible=visible,
                    infrared=infrared,
                    depth=depth,
                    query_texts=query_texts,
                )
                pred_boxes = pred_dict["pred_bbox"]

            # Post-process
            pred_boxes = batch_postprocess(pred_boxes)

            all_pred.append(pred_boxes.cpu())
            all_gt.append(gt_boxes.cpu())

        pred = torch.cat(all_pred, dim=0)
        gt = torch.cat(all_gt, dim=0)
        acc = compute_acc_at_05(pred, gt)
        return acc

    @torch.no_grad()
    def predict(self, dataloader: DataLoader) -> Dict:
        """Run inference and return predictions dict for submission.
        
        Returns dict mapping query_id -> {visible, infrared, depth, query, bbox}
        """
        self.model.eval()
        results = {}

        pbar = tqdm(dataloader, desc="Predicting")
        for batch in pbar:
            visible = batch["visible"].to(self.device)
            infrared = batch["infrared"].to(self.device)
            depth = batch["depth"].to(self.device)
            query_texts = batch["query_text"]
            query_ids = batch["query_id"]

            with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                pred_dict = self.model(
                    visible=visible,
                    infrared=infrared,
                    depth=depth,
                    query_texts=query_texts,
                )
                pred_boxes = pred_dict["pred_bbox"]

            # Post-process
            pred_boxes = batch_postprocess(pred_boxes)

            # Build submission dict
            for i, qid in enumerate(query_ids):
                bbox = pred_boxes[i].cpu().tolist()
                results[qid] = {
                    "visible": "",  # These will be filled from original JSON
                    "infrared": "",
                    "depth": "",
                    "query": query_texts[i],
                    "bbox": [round(v, 6) for v in bbox],
                }

        return results

    def predict_and_save(self, dataloader: DataLoader, save_path: str, ref_json: str = None):
        """Run inference, fill metadata from reference JSON, save to file.
        
        Args:
            dataloader: Test dataloader
            save_path: Output JSON path
            ref_json: Reference JSON with image paths and queries
        """
        # Load reference to fill metadata
        if ref_json:
            with open(ref_json, "r") as f:
                ref_data = json.load(f)
        else:
            ref_data = {}

        # Run prediction
        raw_results = self.predict(dataloader)

        # Fill metadata from reference
        for qid in raw_results:
            if qid in ref_data:
                raw_results[qid]["visible"] = ref_data[qid].get("visible", "")
                raw_results[qid]["infrared"] = ref_data[qid].get("infrared", "")
                raw_results[qid]["depth"] = ref_data[qid].get("depth", "")
                raw_results[qid]["query"] = ref_data[qid].get("query", raw_results[qid]["query"])

        # Save
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(raw_results, f, indent=2)

        print(f"Predictions saved to {save_path}")
        return raw_results
