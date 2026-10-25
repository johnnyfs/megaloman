"""Per-example mask mixtures and masked conditional NLL (not joint likelihood)."""
import math

import torch
from torch.nn import functional as F

from .config import MaskConfig
from .palette import MASK, PAD


def make_mask(valid, cfg=None, *, generator=None, ratio=None, contiguous=False, return_info=False):
    """Seedable CPU masks, copied to input device. Each example masks >=1 cell.

    Non-full fractions use cos(pi*u/2), u~Uniform(0,1). Contiguous masks
    choose the nearest valid cells to a seeded center in Chebyshev distance;
    ties use row-major order, producing a compact expanding rectangular hole.
    """
    cfg = cfg or MaskConfig()
    if ratio is not None and not 0 <= ratio <= 1:
        raise ValueError("mask ratio must be in [0,1]")
    if valid.ndim != 3 or valid.dtype != torch.bool:
        raise ValueError("valid must be BoolTensor[B,H,W]")
    valid_cpu = valid.cpu()
    out = torch.zeros_like(valid_cpu)
    kinds = []
    _, h, w = valid.shape
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    for b in range(len(valid)):
        indices = valid_cpu[b].flatten().nonzero().flatten()
        n = len(indices)
        if not n:
            raise ValueError("each example must have at least one valid pixel")
        hole = contiguous
        if ratio is None:
            full = torch.rand((), generator=generator).item() < cfg.full_probability
            fraction = 1.0 if full else math.cos(math.pi * torch.rand((), generator=generator).item() / 2)
            hole = not full and torch.rand((), generator=generator).item() < cfg.contiguous_probability
        else:
            fraction = ratio
        kinds.append("full" if fraction == 1 else "contiguous" if hole else "random")
        count = max(1, min(n, math.ceil(n * fraction)))
        if hole:
            center = indices[torch.randint(n, (), generator=generator)]
            distance = torch.maximum((x - center % w).abs(), (y - center // w).abs()).flatten()[indices]
            order = torch.argsort(distance, stable=True)
        else:
            order = torch.randperm(n, generator=generator)
        out[b].view(-1)[indices[order[:count]]] = True
    result = out.to(valid.device)
    return (result, kinds) if return_info else result


def corrupt(targets, mask):
    return targets.masked_fill(mask & (targets != PAD), MASK)


def masked_loss(logits, targets, mask, valid_mask=None):
    """Mean over selected cells within each example, then mean across examples."""
    selected = mask & (targets != PAD)
    if valid_mask is not None:
        selected = selected & valid_mask
    counts = selected.flatten(1).sum(1)
    if torch.any(counts == 0):
        raise ValueError("each example needs at least one masked valid target")
    if torch.any((targets[selected] < 0) | (targets[selected] >= 4)):
        raise ValueError("masked targets must be true categorical IDs, never MASK/PAD")
    # Ignore unselected positions before CE; their values must not affect loss.
    safe_targets = targets.masked_fill(~selected, -100)
    ce = F.cross_entropy(logits.movedim(-1, 1), safe_targets, reduction="none", ignore_index=-100)
    return (ce.flatten(1).sum(1) / counts).mean()
