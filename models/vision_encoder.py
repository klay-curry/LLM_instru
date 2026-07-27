"""
Multi-modal visual encoding branch (Branch 1).
- Separate Conv Stems for RGB / Depth / IR
- Feature Pyramid Network backbone (DINOv2 or Swin)
- Outputs multi-scale feature maps (P2-P5) for the decoder
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvStem(nn.Module):
    """Lightweight convolutional stem for a single modality.
    
    Maps input channels to a shared feature dimension.
    """

    def __init__(self, in_channels: int, out_channels: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiModalFusionStem(nn.Module):
    """Three separate Conv Stems for RGB/Depth/IR + 1x1 Conv fusion."""

    def __init__(self, stem_channels: int = 256):
        super().__init__()
        self.rgb_stem = ConvStem(in_channels=3, out_channels=stem_channels)
        self.depth_stem = ConvStem(in_channels=3, out_channels=stem_channels)  # depth repeated to 3ch
        self.ir_stem = ConvStem(in_channels=3, out_channels=stem_channels)
        self.fusion = nn.Sequential(
            nn.Conv2d(stem_channels * 3, stem_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        f_rgb = self.rgb_stem(rgb)
        f_depth = self.depth_stem(depth)
        f_ir = self.ir_stem(ir)
        fused = torch.cat([f_rgb, f_depth, f_ir], dim=1)
        fused = self.fusion(fused)
        return fused


class FeaturePyramid(nn.Module):
    """Lightweight Feature Pyramid Network.
    
    Takes a single-scale feature from the backbone and produces
    multi-scale features P2-P5.
    """

    def __init__(self, in_channels: int, out_channels: int = 256):
        super().__init__()
        # Lateral connections
        self.lateral = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # Top-down
        self.topdown = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        # Extra layers for P4, P5
        self.extra = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(32, out_channels),
                nn.ReLU(inplace=True),
            ) for _ in range(2)  # P4, P5
        ])

    def forward(self, x: torch.Tensor):
        # x: (B, C, H/16, W/16) from backbone
        c5 = self.lateral(x)  # (B, 256, H/16, W/16)
        p5 = c5

        # Top-down: upsample and add
        p4 = F.interpolate(p5, scale_factor=2, mode="nearest")
        p4 = self.topdown(p4)

        p3 = F.interpolate(p4, scale_factor=2, mode="nearest")
        p3 = self.topdown(p3)

        p2 = F.interpolate(p3, scale_factor=2, mode="nearest")
        p2 = self.topdown(p2)

        return {
            "p2": p2,  # (B, 256, H/4, W/4)
            "p3": p3,  # (B, 256, H/8, W/8)
            "p4": p4,  # (B, 256, H/16, W/16)
            "p5": p5,  # (B, 256, H/32, W/32)
        }


class DummyBackbone(nn.Module):
    """A simple CNN backbone that produces single-scale features.
    
    Used as a placeholder when DINOv2/Swin is not available.
    For actual training, replace with:
      - DINOv2: https://huggingface.co/facebook/dinov2-base
      - Swin: https://huggingface.co/microsoft/swin-base-patch4-window7-224
    """

    def __init__(self, in_channels: int = 256):
        super().__init__()
        self.stages = nn.Sequential(
            self._make_stage(in_channels, in_channels, stride=1),  # 1/4 -> 1/8
            self._make_stage(in_channels, in_channels, stride=1),  # 1/8 -> 1/16
        )

    def _make_stage(self, in_ch, out_ch, stride):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 256, H/4, W/4)
        return self.stages(x)  # (B, 256, H/16, W/16)


class DINOV2Backbone(nn.Module):
    """DINOv2 ViT-B/14 backbone loaded from local .pth file.

    Wraps the official DINOv2 ViT-B/14 with register tokens.
    Outputs a single-scale spatial feature map (B, C, H/14, W/14).
    """

    def __init__(
        self,
        ckpt_path: str = "",
        out_channels: int = 768,
        freeze: bool = False,
    ):
        super().__init__()
        self.out_channels = out_channels

        # Build DINOv2 ViT-B/14 architecture
        from .dinov2_arch import DINOv2  # local implementation
        self.model = DINOv2(
            img_size=224,
            patch_size=14,
            embed_dim=out_channels,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            use_register=True,  # reg4 variant
        )

        if ckpt_path and os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            # Rename LayerScale keys: ls1.gamma -> ls1
            rename_keys = {}
            for k in state:
                if k.endswith('.ls1.gamma') or k.endswith('.ls2.gamma'):
                    new_k = k[:-6]  # remove '.gamma'
                    rename_keys[k] = new_k
            for old_k, new_k in rename_keys.items():
                state[new_k] = state.pop(old_k)
            # Handle pos_embed size mismatch
            if 'pos_embed' in state and state['pos_embed'].shape != self.model.pos_embed.shape:
                old_pos = state.pop('pos_embed')
                new_pos = self._interpolate_pos_embed(old_pos, self.model.pos_embed.shape[1])
                state['pos_embed'] = new_pos
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if missing:
                print(f"[DINOv2] missing keys: {len(missing)}")
            if unexpected:
                print(f"[DINOv2] unexpected keys: {len(unexpected)}")
            print(f"[DINOv2] loaded from {ckpt_path}")
        else:
            print(f"[DINOv2] No checkpoint at '{ckpt_path}', using random init")

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    @staticmethod
    def _interpolate_pos_embed(pos_embed: torch.Tensor, new_len: int) -> torch.Tensor:
        B, L, C = pos_embed.shape
        n_register = 4
        cls_pos = pos_embed[:, 0:1]
        reg_pos = pos_embed[:, 1:1 + n_register]
        patch_pos = pos_embed[:, 1 + n_register:]
        n_patches = patch_pos.shape[1]
        # Find closest factor pair for 2D reshape
        g = int(round(n_patches ** 0.5))
        while n_patches % g != 0:
            g -= 1
        h, w = g, n_patches // g
        patch_pos_2d = patch_pos.reshape(B, h, w, C).permute(0, 3, 1, 2)
        new_n = new_len - 1 - n_register
        new_g = int(round(new_n ** 0.5))
        while new_n % new_g != 0:
            new_g -= 1
        new_h, new_w = new_g, new_n // new_g
        patch_pos_2d = F.interpolate(patch_pos_2d, size=(new_h, new_w), mode='bicubic', align_corners=False)
        patch_pos_2d = patch_pos_2d.permute(0, 2, 3, 1).reshape(B, new_h * new_w, C)
        return torch.cat([cls_pos, reg_pos, patch_pos_2d], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial features.

        Args:
            x: (B, 3, H, W) input image, normalized
        Returns:
            features: (B, out_channels, H/14, W/14) spatial feature map
        """
        # DINOv2 forward: returns (B, 1+4+N, out_dim) = cls + reg_tokens + patch_tokens
        B, _, H, W = x.shape
        out = self.model.forward_features(x)
        # Remove CLS token (first) and register tokens (next 4)
        patch_tokens = out[:, 5:, :]  # (B, N, C)
        # Reshape to spatial grid
        grid_h = H // 14
        grid_w = W // 14
        features = patch_tokens.permute(0, 2, 1).reshape(B, -1, grid_h, grid_w)
        return features


class MultiModalVisionEncoder(nn.Module):
    """Branch 1: Multi-modal visual encoding.

    Input: RGB(3,H,W), Depth(3,H,W), IR(3,H,W)
    Output: Multi-scale feature maps {p2, p3, p4, p5} each (B, 256, H_i, W_i)
    """

    def __init__(
        self,
        stem_channels: int = 256,
        backbone: str = "dummy",  # "dummy", "dinov2", "swin"
        fpn_channels: int = 256,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dinov2_ckpt: str = "",
    ):
        super().__init__()
        self.stem = MultiModalFusionStem(stem_channels=stem_channels)

        # Backbone
        if backbone == "dinov2":
            self.backbone_dim = 768
            self.backbone = DINOV2Backbone(
                ckpt_path=dinov2_ckpt,
                out_channels=self.backbone_dim,
                freeze=freeze_backbone,
            )
            self._backbone_proj = nn.Conv2d(self.backbone_dim, stem_channels, kernel_size=1)
        elif backbone == "swin":
            raise NotImplementedError(
                "Swin integration: use HuggingFace transformers. "
                "Set backbone='dummy' for now."
            )
        else:
            self.backbone = DummyBackbone(in_channels=stem_channels)
            self._backbone_proj = nn.Identity()

        self.fpn = FeaturePyramid(in_channels=stem_channels, out_channels=fpn_channels)

    def forward(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        ir: torch.Tensor,
    ) -> dict:
        # DINOv2: process raw RGB directly (it expects 3-channel input)
        if hasattr(self, 'backbone') and isinstance(self.backbone, DINOV2Backbone):
            features = self.backbone(rgb)  # (B, 768, H/14, W/14)
            features = self._backbone_proj(features)  # (B, stem_channels, H/14, W/14)
        else:
            # Stem: separate convs + fusion -> (B, stem_channels, H/4, W/4)
            fused = self.stem(rgb, depth, ir)
            features = self.backbone(fused)
            features = self._backbone_proj(features)

        # FPN: -> multi-scale {p2, p3, p4, p5}
        pyramid = self.fpn(features)
        return pyramid
