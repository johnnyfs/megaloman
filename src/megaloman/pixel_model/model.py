"""Bidirectional pixel transformer: B,H,W inputs -> B,H,W,4 logits."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import Config, DatasetConfig, ModelConfig
from .palette import MASK, PAD
from .rope2d import apply_axial_rope, pixel_coordinates


class ConvResidual(nn.Module):
    """Stride-1 3x3 residual convs on [B,HW,D] token features; invalid cells zeroed first."""

    def __init__(self, d, layers):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        convs = []
        for i in range(layers):
            convs += [nn.Conv2d(d, d, 3, padding=1)] + ([nn.GELU()] if i < layers - 1 else [])
        self.convs = nn.Sequential(*convs)
        nn.init.zeros_(convs[-1].weight); nn.init.zeros_(convs[-1].bias)

    def forward(self, x, h, w, valid):
        b, n, d = x.shape
        y = (self.norm(x) * valid.flatten(1)[..., None]).transpose(1, 2).reshape(b, d, h, w)
        return x + self.convs(y).reshape(b, d, n).transpose(1, 2)


class RelativeBias(nn.Module):
    """Per-head learned bias over 2D offsets (shared by all layers); prefix rows/cols get 0."""

    def __init__(self, heads, max_size=64):
        super().__init__()
        self.max_size = max_size
        self.table = nn.Parameter(torch.zeros(heads, 2 * max_size - 1, 2 * max_size - 1))

    def forward(self, h, w, device):
        y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
        y, x = y.flatten(), x.flatten()
        dy = y[:, None] - y[None, :] + self.max_size - 1
        dx = x[:, None] - x[None, :] + self.max_size - 1
        bias = self.table[:, dy, dx]
        return F.pad(bias, (1, 0, 1, 0))


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
        self.stem = ConvResidual(cfg.d_model, cfg.conv_stem) if cfg.conv_stem else None
        self.decoder = ConvResidual(cfg.d_model, cfg.conv_decoder) if cfg.conv_decoder else None
        self.rel_bias = RelativeBias(cfg.heads) if cfg.rel_bias else None
        self.noise_mlp = (nn.Sequential(nn.Linear(16, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, cfg.d_model))
                          if cfg.noise_conditioning else None)
        self.vocab = {}
        if cfg.conditioning:
            from .data import FACTORS
            self.vocab = {f: getattr(cfg, {"category": "cond_categories", "identity": "cond_identities"}.get(f, f"cond_{f}s"))
                          for f in FACTORS}
            self.label_embeddings = nn.ModuleList([nn.Embedding(len(v) + 1, cfg.d_model) for v in self.vocab.values()])
            for e in self.label_embeddings:
                nn.init.normal_(e.weight, std=.02)

    def encode_labels(self, labels, *, drop=()):
        """list of label dicts -> LongTensor[B,factors]; unknown or dropped factors -> 0 (null)."""
        rows = []
        for item in labels:
            rows.append([0 if (f in drop or item is None or item.get(f) not in v) else v.index(item[f]) + 1
                         for f, v in self.vocab.items()])
        return torch.tensor(rows, dtype=torch.long, device=self.head.weight.device)

    def forward(self, pixels, *, height=None, width=None, object_ids=None, valid_mask=None, labels=None, noise_level=None):
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
        prefix = (self.resolution_mlp(metadata) + self.resolution_type).expand(b, -1)
        if labels is not None and self.vocab:
            prefix = prefix + sum(e(labels[:, j]) for j, e in enumerate(self.label_embeddings))
        if self.noise_mlp is not None:
            t = torch.zeros(b, device=pixels.device) if noise_level is None else torch.as_tensor(
                noise_level, dtype=torch.float32, device=pixels.device).expand(b)
            freqs = torch.arange(8, device=pixels.device, dtype=torch.float32)
            angles = t.float()[:, None] * math.pi * (2 ** freqs)[None] / 2
            prefix = prefix + self.noise_mlp(torch.cat((angles.sin(), angles.cos()), -1).to(prefix.dtype))
        prefix = prefix.unsqueeze(1)
        tokens = self.embedding(pixels.flatten(1)) + self.pixel_type
        if self.stem is not None:  # inputs are already corrupted (MASK), so no hidden pixel can leak
            tokens = self.stem(tokens, h, w, valid)
        x = torch.cat((prefix, tokens), 1)
        # SDPA bool True means allowed. All queries can see valid keys + prefix;
        # PAD queries are discarded by the objective and never serve as keys.
        keys = torch.cat((torch.ones(b, 1, dtype=torch.bool, device=pixels.device), valid.flatten(1)), 1)
        attention_mask = keys[:, None, None, :]
        if self.rel_bias is not None:
            attention_mask = self.rel_bias(h, w, pixels.device)[None].masked_fill(~attention_mask, -torch.inf)
        coords = pixel_coordinates(h, w, device=pixels.device, num_colors=self.cfg.num_colors)
        for block in self.blocks:
            x = block(x, coords, attention_mask)
        features = x[:, 1:]
        if self.decoder is not None:
            features = self.decoder(features, h, w, valid)
        return self.head(self.norm(features)).reshape(b, h, w, self.cfg.num_colors)
