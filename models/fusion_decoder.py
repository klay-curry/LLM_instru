"""
Cross-Attention Decoder (Fusion stage).
- Takes multi-scale visual features (from Branch 1) + semantic tokens (from Branch 2)
- Cross-attention decoder to produce final BBox
- Replicates Grounding DINO's decoder structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TransformerDecoderLayer(nn.Module):
    """Single decoder layer with self-attn, text-cross-attn, image-cross-attn, FFN."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Cross-attention: text-guided (semantic tokens as K,V)
        self.text_cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        # Cross-attention: image-guided (visual features as K,V)
        self.image_cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm4 = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt: torch.Tensor,
        text_memory: torch.Tensor,
        image_memory: torch.Tensor,
        text_key_padding_mask=None,
    ):
        # 1. Self-attention
        tgt2 = self.self_attn(tgt, tgt, tgt, need_weights=False)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # 2. Text-guided cross-attention
        tgt2 = self.text_cross_attn(
            tgt, text_memory, text_memory,
            key_padding_mask=text_key_padding_mask,
            need_weights=False,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # 3. Image-guided cross-attention
        tgt2 = self.image_cross_attn(tgt, image_memory, image_memory, need_weights=False)[0]
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        # 4. FFN
        tgt2 = self.ffn(tgt)
        tgt = tgt + tgt2
        tgt = self.norm4(tgt)

        return tgt


class CrossAttentionDecoder(nn.Module):
    """Decoder that fuses semantic tokens + visual features to predict BBox."""

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_queries: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries

        # Learnable object queries (or initialized from coarse bbox)
        self.query_embed = nn.Embedding(num_queries, d_model)

        # Project semantic tokens from VLM to decoder d_model
        self.text_proj = nn.Linear(d_model, d_model)  # assuming same dim; adjust if needed

        # Project visual features to decoder d_model
        self.vis_proj = nn.Conv2d(d_model, d_model, kernel_size=1)

        # Decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        visual_features: dict,
        semantic_tokens: torch.Tensor,
        coarse_bbox: torch.Tensor = None,
    ):
        """
        Args:
            visual_features: dict of {"p2", "p3", "p4", "p5"}, each (B, 256, H_i, W_i)
            semantic_tokens: (B, N_sem, d_sem) from VLM branch
            coarse_bbox: (B, 4) optional coarse bbox to initialize queries
        
        Returns:
            output_queries: (B, num_queries, d_model)
        """
        batch_size = semantic_tokens.shape[0]

        # Project visual features: use concatenated multi-scale
        # Upsample all to P2 resolution and concatenate
        p2 = visual_features["p2"]  # (B, 256, H/4, W/4)
        p3 = F.interpolate(visual_features["p3"], size=p2.shape[-2:], mode="bilinear", align_corners=False)
        p4 = F.interpolate(visual_features["p4"], size=p2.shape[-2:], mode="bilinear", align_corners=False)
        p5 = F.interpolate(visual_features["p5"], size=p2.shape[-2:], mode="bilinear", align_corners=False)

        cat_vis = torch.cat([p2, p3, p4, p5], dim=1)  # (B, 1024, H/4, W/4)
        vis_proj = self.vis_proj(cat_vis)  # (B, 256, H/4, W/4)
        B, C, H, W = vis_proj.shape
        image_memory = vis_proj.flatten(2).permute(0, 2, 1)  # (B, H*W, 256)

        # Project semantic tokens
        text_memory = self.text_proj(semantic_tokens)  # (B, N_sem, 256)

        # Object queries
        if coarse_bbox is not None:
            # Initialize from coarse bbox using learned positional encoding
            query_pos = self._bbox_to_pos_encoding(coarse_bbox)  # (B, d_model)
            query_embed = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)  # (B, 1, d_model)
            tgt = query_embed + query_pos.unsqueeze(1)
        else:
            tgt = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)  # (B, 1, d_model)

        # Run decoder layers
        for layer in self.layers:
            tgt = layer(
                tgt,
                text_memory=text_memory,
                image_memory=image_memory,
            )

        return tgt  # (B, num_queries, d_model)

    def _bbox_to_pos_encoding(self, bbox: torch.Tensor) -> torch.Tensor:
        """Convert bbox coordinates to a positional encoding.
        
        Args:
            bbox: (B, 4) normalized [x1, y1, x2, y2]
        Returns:
            (B, d_model) sinusoidal positional encoding
        """
        B = bbox.shape[0]
        pe = torch.zeros(B, self.d_model, device=bbox.device)
        
        # Simple approach: MLP on bbox
        # In practice, use sinusoidal or learned PE
        pos_mlp = nn.Sequential(
            nn.Linear(4, self.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, self.d_model),
        ).to(bbox.device)
        
        return pos_mlp(bbox)


class BBoxHead(nn.Module):
    """BBox prediction head on top of decoder queries."""

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.reg = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 4),
            nn.Sigmoid(),  # normalize to [0, 1]
        )

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        """Predict bbox from decoder output queries.
        
        Args:
            queries: (B, num_queries, d_model)
        Returns:
            bbox: (B, 4) [x1, y1, x2, y2]
        """
        # Take the first (and only) query
        query = queries[:, 0, :]  # (B, d_model)
        bbox = self.reg(query)  # (B, 4)
        return bbox
