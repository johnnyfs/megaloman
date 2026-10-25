"""Bidirectional pixel transformer: B,H,W inputs -> B,H,W,4 logits."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import Config, DatasetConfig, ModelConfig
from .palette import MASK, PAD
from .rope2d import apply_axial_rope, pixel_coordinates


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads, self.dropout = cfg.heads, cfg.dropout
        self.attention_dropout = cfg.dropout if cfg.attention_dropout is None else cfg.attention_dropout
        self.norm1, self.norm2 = nn.LayerNorm(cfg.d_model), nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.mlp = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_model * cfg.mlp_ratio), nn.GELU(),
                                 nn.Dropout(cfg.dropout), nn.Linear(cfg.d_model * cfg.mlp_ratio, cfg.d_model))

    def forward(self, x, coordinates, attention_mask):
        b, n, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = apply_axial_rope(q, coordinates), apply_axial_rope(k, coordinates)
        h = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask,
                                          dropout_p=self.attention_dropout if self.training else 0.0, is_causal=False)
        x = x + F.dropout(self.proj(h.transpose(1, 2).reshape(b, n, d)), self.dropout, self.training)
        return x + F.dropout(self.mlp(self.norm2(x)), self.dropout, self.training)


class PixelTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig, resolutions=(16,)):
        super().__init__()
        Config(model=cfg, dataset=DatasetConfig(resolutions=list(resolutions))).validate()
        self.cfg, self.resolutions = cfg, tuple(resolutions)
        self.embedding = nn.Embedding(cfg.num_colors + 2, cfg.d_model, padding_idx=PAD)
        self.pixel_type = nn.Parameter(torch.zeros(cfg.d_model))
        self.resolution_type = nn.Parameter(torch.zeros(cfg.d_model))
        # Metadata interface can later accept color-depth conditioning. v0 stays 4-class.
        self.resolution_mlp = nn.Sequential(nn.Linear(2, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, cfg.d_model))
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.layers)])
        self.norm, self.head = nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.num_colors)

    def forward(self, pixels, *, height=None, width=None, object_ids=None, valid_mask=None):
        if pixels.ndim != 3 or pixels.dtype != torch.long:
            raise ValueError("pixels must be LongTensor[B,H,W]")
        b, h, w = pixels.shape
        if (height is not None and height != h) or (width is not None and width != w):
            raise ValueError("metadata and pixel dimensions disagree")
        if h != w or h not in self.resolutions:
            raise ValueError(f"unsupported size {h}x{w}; configured sizes: {self.resolutions}")
        if object_ids is not None:
            raise ValueError("object conditioning is disabled in v0")
        if torch.any((pixels < 0) | (pixels > PAD)):
            raise ValueError("input IDs must be classes 0..3, MASK=4, or PAD=5")
        valid = pixels != PAD
        if valid_mask is not None:
            if valid_mask.shape != pixels.shape or valid_mask.dtype != torch.bool:
                raise ValueError("valid_mask must be BoolTensor[B,H,W]")
            valid = valid & valid_mask
        pixels = pixels.masked_fill(~valid, PAD)
        metadata = pixels.new_tensor([[math.log(h), math.log(w)]], dtype=self.embedding.weight.dtype)
        prefix = (self.resolution_mlp(metadata) + self.resolution_type).expand(b, -1).unsqueeze(1)
        x = torch.cat((prefix, self.embedding(pixels.flatten(1)) + self.pixel_type), 1)
        # SDPA bool True means allowed. All queries can see valid keys + prefix;
        # PAD queries are discarded by the objective and never serve as keys.
        keys = torch.cat((torch.ones(b, 1, dtype=torch.bool, device=pixels.device), valid.flatten(1)), 1)
        attention_mask = keys[:, None, None, :]
        coords = pixel_coordinates(h, w, device=pixels.device, num_colors=self.cfg.num_colors)
        for block in self.blocks:
            x = block(x, coords, attention_mask)
        return self.head(self.norm(x[:, 1:])).reshape(b, h, w, self.cfg.num_colors)
