"""
Minimal DINOv2 ViT-B/14 implementation for loading local .pth weights.

Architecture matches the official DINOv2 ViT-B/14 with 4 register tokens.
Reference: https://github.com/facebookresearch/dinov2
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 12, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4, use_layer_scale: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), dim)
        # LayerScale (learnable gamma per channel)
        if use_layer_scale:
            self.ls1 = nn.Parameter(torch.ones(dim) * 1e-5)
            self.ls2 = nn.Parameter(torch.ones(dim) * 1e-5)
        else:
            self.ls1 = nn.Parameter(torch.ones(dim))
            self.ls2 = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x)) * self.ls1
        x = x + self.mlp(self.norm2(x)) * self.ls2
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 14, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class DINOv2(nn.Module):
    """DINOv2 ViT-B/14 with register tokens.

    Matches the official 'dinov2_vitb14_reg4_pretrain.pth' checkpoint.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 14,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4,
        use_register: bool = True,
        num_register_tokens: int = 4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_register_tokens = num_register_tokens if use_register else 0

        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_tokens = nn.Parameter(torch.zeros(1, self.num_register_tokens, embed_dim)) if use_register else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1 + self.num_register_tokens, embed_dim))

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim)) if use_register else None

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning all tokens (cls + registers + patches)."""
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, N, C)

        # Concatenate CLS, register tokens
        cls_token = self.cls_token.expand(B, -1, -1)
        if self.register_tokens is not None:
            reg_tokens = self.register_tokens.expand(B, -1, -1)
            x = torch.cat([cls_token, reg_tokens, x], dim=1)
        else:
            x = torch.cat([cls_token, x], dim=1)

        # Add position embedding (interpolate if size mismatch)
        if x.shape[1] != self.pos_embed.shape[1]:
            # Interpolate pos_embed for different image size
            cls_pos = self.pos_embed[:, 0:1]
            reg_pos = self.pos_embed[:, 1:1+self.num_register_tokens] if self.register_tokens is not None else None
            patch_pos = self.pos_embed[:, 1+self.num_register_tokens:]  # (1, N, C)
            grid_size = int(patch_pos.shape[1] ** 0.5)
            patch_pos = patch_pos.reshape(1, grid_size, grid_size, -1).permute(0, 3, 1, 2)
            # Target grid size
            tgt_size = x.shape[1] - 1 - (self.num_register_tokens if self.register_tokens is not None else 0)
            tgt_grid = int(tgt_size ** 0.5)
            patch_pos = F.interpolate(patch_pos, size=(tgt_grid, tgt_grid), mode='bicubic', align_corners=False)
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, tgt_grid * tgt_grid, -1)
            if self.register_tokens is not None:
                x = x + torch.cat([cls_pos, reg_pos, patch_pos], dim=1)
            else:
                x = x + torch.cat([cls_pos, patch_pos], dim=1)
        else:
            x = x + self.pos_embed

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x  # (B, 1+4+N, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: return patch tokens reshaped as spatial feature map."""
        B, _, H, W = x.shape
        out = self.forward_features(x)
        patch_tokens = out[:, 1 + self.num_register_tokens:, :]  # (B, N, C)
        grid_h = H // self.patch_embed.patch_size
        grid_w = W // self.patch_embed.patch_size
        features = patch_tokens.permute(0, 2, 1).reshape(B, -1, grid_h, grid_w)
        return features
