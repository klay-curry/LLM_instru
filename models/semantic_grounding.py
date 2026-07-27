"""Semantic-guided RGB grounding model for supervised RefCOCO training."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.vision_encoder import DINOV2Backbone


class SemanticDecoderLayer(nn.Module):
    """Fuse a learned object query with semantic and spatial memories."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.semantic_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.visual_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, hidden_dim),
        )
        self.semantic_norm = nn.LayerNorm(hidden_dim)
        self.visual_norm = nn.LayerNorm(hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        semantic_memory: torch.Tensor,
        visual_memory: torch.Tensor,
        semantic_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        semantic_output = self.semantic_attention(
            query,
            semantic_memory,
            semantic_memory,
            key_padding_mask=semantic_padding_mask,
            need_weights=False,
        )[0]
        query = self.semantic_norm(query + self.dropout(semantic_output))

        visual_output = self.visual_attention(
            query, visual_memory, visual_memory, need_weights=False
        )[0]
        query = self.visual_norm(query + self.dropout(visual_output))
        return self.output_norm(query + self.dropout(self.feedforward(query)))


class SemanticGroundingModel(nn.Module):
    """Predict a normalized bbox from RGB features and cached Qwen tokens."""

    def __init__(
        self,
        dinov2_checkpoint: str,
        semantic_dim: int = 4096,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        feedforward_dim: int = 1024,
        dropout: float = 0.1,
        use_semantics: bool = True,
    ) -> None:
        super().__init__()
        self.use_semantics = use_semantics
        self.backbone = DINOV2Backbone(
            ckpt_path=dinov2_checkpoint,
            out_channels=768,
            freeze=True,
        )
        self.visual_projection = nn.Conv2d(768, hidden_dim, kernel_size=1)
        self.semantic_projection = nn.Sequential(
            nn.LayerNorm(semantic_dim),
            nn.Linear(semantic_dim, hidden_dim),
        )
        self.null_semantic = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.object_query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.layers = nn.ModuleList(
            SemanticDecoderLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                feedforward_dim=feedforward_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        )
        self.bbox_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
        )
        nn.init.normal_(self.object_query, std=0.02)
        nn.init.normal_(self.null_semantic, std=0.02)

    def train(self, mode: bool = True) -> "SemanticGroundingModel":
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(
        self,
        images: torch.Tensor,
        semantic_tokens: torch.Tensor,
        semantic_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            visual_features = self.backbone(images)

        visual_memory = self.visual_projection(visual_features)
        visual_memory = visual_memory.flatten(2).transpose(1, 2)

        batch_size = images.shape[0]
        if self.use_semantics:
            semantic_memory = self.semantic_projection(semantic_tokens)
            padding_mask = semantic_padding_mask
        else:
            semantic_memory = self.null_semantic.expand(batch_size, -1, -1)
            padding_mask = torch.zeros(
                batch_size, 1, dtype=torch.bool, device=images.device
            )

        query = self.object_query.expand(batch_size, -1, -1)
        for layer in self.layers:
            query = layer(
                query,
                semantic_memory=semantic_memory,
                visual_memory=visual_memory,
                semantic_padding_mask=padding_mask,
            )

        raw_box = self.bbox_head(query[:, 0]).sigmoid()
        center = raw_box[:, :2]
        size = raw_box[:, 2:]
        top_left = center * (1.0 - size)
        bottom_right = top_left + size
        return torch.cat([top_left, bottom_right], dim=-1)
