"""
VLM Semantic Branch (Branch 2).
- Tri-modal fusion adapter (Conv 9->3) to feed RGB+Depth+IR into Qwen3-VL
- Qwen3-VL LLM with LoRA fine-tuning
- Output: Semantic Tokens (hidden states) + Coarse BBox Prior
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    Qwen3VLForConditionalGeneration,
)
from peft import LoraConfig, get_peft_model, TaskType


class TriModalFusionAdapter(nn.Module):
    """Lightweight adapter: concatenate RGB(3ch) + Depth(3ch) + IR(3ch) -> 9ch -> 3ch.
    
    This is placed BEFORE Qwen3-VL's ViT, so that the ViT receives a fused 3-channel input.
    """

    def __init__(self):
        super().__init__()
        self.fusion = nn.Conv2d(9, 3, kernel_size=1, bias=False)

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        # rgb/depth/ir: each (B, 3, H, W), normalized to [0, 1]
        fused = torch.cat([rgb, depth, ir], dim=1)  # (B, 9, H, W)
        fused = self.fusion(fused)  # (B, 3, H, W)
        return fused


class VLMSemanticBranch(nn.Module):
    """VLM branch for multi-modal semantic understanding.
    
    Takes RGB/Depth/IR images + text query, outputs:
    1. Semantic tokens (hidden states from LLM's last layer)
    2. Coarse bbox prior (optional, from an MLP on semantic tokens)
    
    Uses LoRA for parameter-efficient fine-tuning of the LLM.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: list = None,
        freeze_vision_encoder: bool = True,
        freeze_llm_embedding: bool = True,
        output_semantic_dim: int = 768,
        use_coarse_bbox: bool = True,
    ):
        super().__init__()
        self.use_coarse_bbox = use_coarse_bbox

        # Tri-modal fusion adapter
        self.fusion_adapter = TriModalFusionAdapter()

        # Qwen3-VL model
        # NOTE: This loads a potentially large model. For testing, use the 2B variant.
        # For final training, use the 7B variant.
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Freeze vision encoder (ViT + Connector)
        if freeze_vision_encoder:
            self.model.visual = self.model.visual.eval()
            for p in self.model.visual.parameters():
                p.requires_grad = False

        # Freeze embeddings
        if freeze_llm_embedding:
            if hasattr(self.model.model, 'embed_tokens'):
                for p in self.model.model.embed_tokens.parameters():
                    p.requires_grad = False

        # Apply LoRA
        if lora_target_modules is None:
            lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        # Semantic token extraction: we'll take hidden states from the last layer
        # and project them to output_semantic_dim
        self.semantic_proj = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, output_semantic_dim),
            nn.LayerNorm(output_semantic_dim),
        )

        # Optional coarse bbox head
        if use_coarse_bbox:
            self.coarse_bbox_head = nn.Sequential(
                nn.Linear(output_semantic_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 4),
                nn.Sigmoid(),
            )

    def forward(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        ir: torch.Tensor,
        query_texts: list,
    ):
        """
        Args:
            rgb: (B, 3, H, W) normalized RGB image
            depth: (B, 3, H, W) normalized depth
            ir: (B, 3, H, W) normalized IR
            query_texts: list of strings, batch of text queries
        
        Returns:
            dict with:
                - semantic_tokens: (B, num_semantic_tokens, output_semantic_dim)
                - coarse_bbox: (B, 4) if use_coarse_bbox else None
        """
        # Fuse multi-modal images
        fused_image = self.fusion_adapter(rgb, depth, ir)  # (B, 3, H, W)

        # Process inputs for Qwen3-VL
        batch_size = fused_image.shape[0]
        
        # Build conversation format for each sample in the batch
        # Qwen3-VL uses a chat template
        texts = []
        for query in query_texts:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": None},  # placeholder, image will be passed separately
                        {"type": "text", "text": f"Locate the target: {query}"},
                    ],
                },
            ]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            texts.append(text)

        # Prepare images and text for the processor
        # Note: In a real implementation, you'd pass the actual fused_image tensors.
        # The processor expects PIL images or tensor format.
        # For now, this is a simplified structure.
        inputs = self.processer_handler(fused_image, texts)

        # Forward through Qwen3-VL
        with torch.set_grad_enabled(self.training):
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        # Extract semantic tokens
        # Use the last hidden states from the LLM
        hidden_states = outputs.hidden_states[-1]  # (B, seq_len, hidden_size)
        
        # Take the hidden states corresponding to visual tokens
        # In Qwen3-VL, visual tokens are at the beginning of the sequence
        # We need to determine how many visual tokens there are
        # This depends on the image size and ViT configuration
        num_visual_tokens = self._get_num_visual_tokens(fused_image.shape[2:])
        
        visual_hidden = hidden_states[:, :num_visual_tokens, :]  # (B, N_v, hidden)
        semantic_tokens = self.semantic_proj(visual_hidden)  # (B, N_v, out_dim)

        # Mean pooling over visual tokens for coarse bbox
        if self.use_coarse_bbox:
            pooled = semantic_tokens.mean(dim=1)  # (B, out_dim)
            coarse_bbox = self.coarse_bbox_head(pooled)  # (B, 4)
        else:
            coarse_bbox = None

        return {
            "semantic_tokens": semantic_tokens,
            "coarse_bbox": coarse_bbox,
        }

    def _get_num_visual_tokens(self, image_size):
        """Estimate number of visual tokens for a given image size.
        
        This is a simplified estimation. In practice, check Qwen3-VL's ViT config.
        """
        h, w = image_size
        # Typical ViT patch size = 14, and Qwen3-VL uses a resampler
        # that reduces ~256 patches to a fixed number (e.g., 64-256)
        patch_size = 14
        num_patches = (h // patch_size) * (w // patch_size)
        # After resampler, typically ~256 tokens
        return min(num_patches, 256)

    def processor_handler(self, fused_image: torch.Tensor, texts: list):
        """Convert fused image tensor and texts to processor inputs.
        
        NOTE: This is a simplified handler. For actual training, adapt to 
        Qwen3-VL's specific processor API which expects either PIL images
        or specific tensor formats.
        """
        # In practice, you'd do something like:
        # images = [ToPILImage()(img) for img in fused_image]
        # inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        # 
        # For the code skeleton, we return a placeholder structure.
        # This needs to be adapted based on Qwen3-VL's actual processor API.
        
        # Placeholder: just pass through to model
        # The actual implementation depends on the Qwen3-VL version.
        inputs = self.processor(
            text=texts,
            images=fused_image,
            return_tensors="pt",
            padding=True,
        )
        # Move to same device as model
        inputs = {k: v.to(self.model.device) if hasattr(v, 'to') else v
                  for k, v in inputs.items()}
        return inputs


def create_vlm_branch(config: dict) -> VLMSemanticBranch:
    """Factory function to create VLM branch from config."""
    vlm_cfg = config["model"]["vlm"]
    return VLMSemanticBranch(
        model_name=vlm_cfg["name"],
        lora_rank=vlm_cfg["lora_rank"],
        lora_alpha=vlm_cfg["lora_alpha"],
        lora_dropout=vlm_cfg["lora_dropout"],
        lora_target_modules=vlm_cfg["lora_target_modules"],
        freeze_vision_encoder=vlm_cfg["freeze_vision_encoder"],
        freeze_llm_embedding=vlm_cfg["freeze_llm_embedding"],
        output_semantic_dim=vlm_cfg["output_semantic_dim"],
        use_coarse_bbox=True,
    )
