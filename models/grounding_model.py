"""
Full multi-modal grounding model combining:
  Branch 1: Multi-modal Vision Encoder (Stem + Backbone + FPN)
  Branch 2: VLM Semantic Branch (Qwen3-VL with LoRA)
  Fusion: Cross-Attention Decoder + BBox Head
"""

import torch
import torch.nn as nn
from .vision_encoder import MultiModalVisionEncoder
from .vlm_branch import VLMSemanticBranch
from .fusion_decoder import CrossAttentionDecoder, BBoxHead


class MultiModalGroundingModel(nn.Module):
    """End-to-end multi-modal visual grounding model."""

    def __init__(
        self,
        vision_encoder_cfg: dict = None,
        vlm_cfg: dict = None,
        decoder_cfg: dict = None,
    ):
        super().__init__()

        # Branch 1: Visual encoding
        if vision_encoder_cfg is None:
            vision_encoder_cfg = {}
        self.vision_encoder = MultiModalVisionEncoder(
            stem_channels=vision_encoder_cfg.get("stem_channels", 256),
            backbone=vision_encoder_cfg.get("backbone", "dummy"),
            fpn_channels=vision_encoder_cfg.get("fpn_channels", [256, 256, 256, 256]),
            pretrained=vision_encoder_cfg.get("pretrained", True),
            freeze_backbone=vision_encoder_cfg.get("freeze_backbone", False),
        )

        # Branch 2: VLM semantic
        if vlm_cfg is None:
            vlm_cfg = {}
        self.vlm_branch = VLMSemanticBranch(
            model_name=vlm_cfg.get("name", "Qwen/Qwen3-VL-2B-Instruct"),
            lora_rank=vlm_cfg.get("lora_rank", 16),
            lora_alpha=vlm_cfg.get("lora_alpha", 32),
            lora_dropout=vlm_cfg.get("lora_dropout", 0.05),
            lora_target_modules=vlm_cfg.get("lora_target_modules", None),
            freeze_vision_encoder=vlm_cfg.get("freeze_vision_encoder", True),
            freeze_llm_embedding=vlm_cfg.get("freeze_llm_embedding", True),
            output_semantic_dim=vlm_cfg.get("output_semantic_dim", 768),
            use_coarse_bbox=True,
        )
        # Get the actual semantic token dimension
        vlm_out_dim = vlm_cfg.get("output_semantic_dim", 768)

        # Fusion: Decoder
        if decoder_cfg is None:
            decoder_cfg = {}
        self.decoder = CrossAttentionDecoder(
            d_model=decoder_cfg.get("d_model", 256),
            nhead=decoder_cfg.get("nhead", 8),
            num_layers=decoder_cfg.get("num_layers", 6),
            dim_feedforward=decoder_cfg.get("dim_feedforward", 1024),
            dropout=decoder_cfg.get("dropout", 0.1),
            num_queries=decoder_cfg.get("num_queries", 1),
        )

        # BBox head
        self.bbox_head = BBoxHead(d_model=decoder_cfg.get("d_model", 256))

    def forward(
        self,
        visible: torch.Tensor,
        infrared: torch.Tensor,
        depth: torch.Tensor,
        query_texts: list,
    ):
        # Branch 1: Visual encoding -> multi-scale features
        visual_features = self.vision_encoder(
            rgb=visible,
            depth=depth,
            ir=infrared,
        )

        # Branch 2: VLM understanding -> semantic tokens + coarse bbox
        vlm_output = self.vlm_branch(
            rgb=visible,
            depth=depth,
            ir=infrared,
            query_texts=query_texts,
        )
        semantic_tokens = vlm_output["semantic_tokens"]
        coarse_bbox = vlm_output["coarse_bbox"]

        # Fusion: Decoder
        decoder_queries = self.decoder(
            visual_features=visual_features,
            semantic_tokens=semantic_tokens,
            coarse_bbox=coarse_bbox,
        )

        # BBox prediction
        pred_bbox = self.bbox_head(decoder_queries)  # (B, 4)

        return {
            "pred_bbox": pred_bbox,
            "coarse_bbox": coarse_bbox,
            "semantic_tokens": semantic_tokens,
        }

    def compute_loss(self, pred_dict: dict, target_bbox: torch.Tensor, loss_cfg: dict = None):
        """Compute training loss."""
        from utils.metrics import compute_giou

        if loss_cfg is None:
            loss_cfg = {"giou_weight": 2.0, "l1_weight": 1.0, "coarse_weight": 0.5}

        pred = pred_dict["pred_bbox"]
        coarse = pred_dict["coarse_bbox"]

        # GIoU loss
        giou = compute_giou(pred, target_bbox)  # higher is better
        loss_giou = (1.0 - giou).mean()  # convert to loss

        # L1 loss
        loss_l1 = torch.nn.functional.l1_loss(pred, target_bbox)

        loss = (
            loss_cfg["giou_weight"] * loss_giou
            + loss_cfg["l1_weight"] * loss_l1
        )

        # Coarse bbox loss (optional)
        if coarse is not None and loss_cfg.get("coarse_weight", 0) > 0:
            giou_coarse = compute_giou(coarse, target_bbox)
            loss_coarse_giou = (1.0 - giou_coarse).mean()
            loss_l1_coarse = torch.nn.functional.l1_loss(coarse, target_bbox)
            loss += loss_cfg["coarse_weight"] * (loss_coarse_giou + loss_l1_coarse)

        return loss

    def get_trainable_params(self):
        """Get parameter groups with different learning rates."""
        backbone_params = []
        lora_params = []
        decoder_params = []
        vlm_fusion_params = []

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if "vlm_branch.model" in name and "lora" in name:
                lora_params.append(p)
            elif "vlm_branch" in name:
                vlm_fusion_params.append(p)
            elif "vision_encoder" in name:
                backbone_params.append(p)
            else:
                decoder_params.append(p)

        return {
            "backbone": backbone_params,
            "lora": lora_params,
            "decoder": decoder_params,
            "vlm_fusion": vlm_fusion_params,
        }


def build_model(config: dict) -> MultiModalGroundingModel:
    """Build full grounding model from config dictionary."""
    model = MultiModalGroundingModel(
        vision_encoder_cfg=config["model"]["backbone"],
        vlm_cfg=config["model"]["vlm"],
        decoder_cfg=config["model"]["decoder"],
    )
    return model
