"""Parallel decoding with cumulative cosine commits and exact categorical locks."""
import math

import torch

from .config import SampleConfig
from .palette import MASK


@torch.inference_mode()
def sample(model, *, size, count=1, seed=42, cfg=None, initial=None, editable=None, return_trace=False, labels=None, allowed=None):
    """allowed: optional BoolTensor[count,size,size,4]; masked cells only sample allowed classes
    (e.g. forbid TRANSPARENT inside a silhouette for shape-conditioned infill)."""
    cfg = cfg or SampleConfig()
    if size not in model.resolutions or count < 1:
        raise ValueError("unsupported resolution or invalid count")
    if cfg.steps < 1 or cfg.temperature < 0 or cfg.ranking_temperature < 0 or not 0 <= cfg.top_k <= 4:
        raise ValueError("invalid sampling settings")
    if not math.isfinite(cfg.temperature) or not math.isfinite(cfg.ranking_temperature):
        raise ValueError("sampler temperatures must be finite")
    device = next(model.parameters()).device
    shape = (count, size, size)
    if initial is None:
        if editable is not None:
            raise ValueError("editable requires an initial categorical grid")
        pixels = torch.full(shape, MASK, dtype=torch.long, device=device)
        remaining = torch.ones(shape, dtype=torch.bool, device=device)
    else:
        if initial.shape != shape or initial.dtype != torch.long or torch.any((initial < 0) | (initial >= 4)):
            raise ValueError("initial must be LongTensor[count,size,size] of classes 0..3")
        if editable is None or editable.shape != shape or editable.dtype != torch.bool:
            raise ValueError("editable must be BoolTensor with the same shape")
        pixels = initial.to(device).clone()
        remaining = editable.to(device).clone()
        pixels[remaining] = MASK
    if allowed is not None:
        if allowed.shape != (*shape, 4) or allowed.dtype != torch.bool or not allowed.any(-1).all():
            raise ValueError("allowed must be BoolTensor[count,size,size,4] with a class per cell")
        allowed = allowed.to(device)
    initial_counts = remaining.flatten(1).sum(1)
    editable_cells = remaining.clone()
    committed_confidence = torch.zeros(shape, device=device)
    # Independent, explicitly seeded categorical and Gumbel generators.
    categorical_rng = torch.Generator(device=device).manual_seed(seed)
    ranking_rng = torch.Generator(device=device).manual_seed(seed + 1)
    trace = [pixels.cpu().clone()] if return_trace else None
    was_training = model.training
    model.eval()
    try:
        for t in range(1, cfg.steps + 1):
            if not remaining.any():
                break
            if labels is None:
                logits = model(pixels).float()
            else:
                logits = model(pixels, labels=labels).float()
                if cfg.guidance:
                    uncond = model(pixels, labels=torch.zeros_like(labels)).float()
                    logits = uncond + (1 + cfg.guidance) * (logits - uncond)
            if not torch.isfinite(logits).all():
                raise FloatingPointError("nonfinite sampling logits")
            filtered = logits / cfg.temperature if cfg.temperature > 0 else logits
            if allowed is not None:
                filtered = filtered.masked_fill(~allowed, -torch.inf)
            if cfg.top_k:
                # Stable class-ID tie ordering; select exactly k classes.
                order = torch.argsort(filtered, dim=-1, descending=True, stable=True)
                keep = torch.zeros_like(filtered, dtype=torch.bool).scatter_(-1, order[..., :cfg.top_k], True)
                filtered = filtered.masked_fill(~keep, -torch.inf)
            probabilities = filtered.softmax(-1)
            if cfg.temperature == 0:
                candidates = filtered.argmax(-1)
            else:
                candidates = torch.multinomial(probabilities.reshape(-1, 4), 1,
                                               generator=categorical_rng).reshape(shape)
            # Confidence is probability of the sampled class, not max-class probability.
            confidence = probabilities.gather(-1, candidates.unsqueeze(-1)).squeeze(-1).clamp_min(1e-20).log()
            raw_confidence = confidence.clone()
            revise = cfg.remask > 0 and t <= cfg.remask_until * cfg.steps and t < cfg.steps
            if cfg.ranking_temperature:
                u = torch.rand(shape, device=device, generator=ranking_rng).clamp(1e-7, 1 - 1e-7)
                amplitude = cfg.ranking_temperature
                if cfg.ranking_schedule == "linear":
                    amplitude *= 1 - (t - 1) / cfg.steps  # full noise first, ~0 by the last step
                confidence += amplitude * (-(-u.log()).log())
            for b in range(count):
                m0 = initial_counts[b].item()
                target = m0 if t == cfg.steps else m0 - math.floor(m0 * math.cos(math.pi * t / (2 * cfg.steps)))
                already = m0 - remaining[b].sum().item()
                needed = max(0, target - already)
                indices = remaining[b].flatten().nonzero().flatten()
                ranks = torch.argsort(confidence[b].flatten()[indices], descending=True, stable=True)
                chosen = indices[ranks[:needed]]  # row-major tie breaking
                old = (editable_cells[b] & ~remaining[b]).flatten().nonzero().flatten()
                pixels[b].view(-1)[chosen] = candidates[b].flatten()[chosen]
                remaining[b].view(-1)[chosen] = False
                committed_confidence[b].view(-1)[chosen] = raw_confidence[b].flatten()[chosen]
                if revise and len(old):
                    k = math.floor(cfg.remask * len(old))
                    if k:
                        low = old[torch.argsort(committed_confidence[b].flatten()[old], stable=True)[:k]]
                        pixels[b].view(-1)[low] = MASK
                        remaining[b].view(-1)[low] = True
            if return_trace:
                trace.append(pixels.cpu().clone())
    finally:
        model.train(was_training)
    if remaining.any() or torch.any(pixels >= 4):
        raise RuntimeError("sampler failed to complete")
    return (pixels, trace) if return_trace else pixels


def infill(model, canvas, known_mask, *, seed=42, cfg=None, keep_foreground=False):
    """Fill the unknown cells of one canvas (LongTensor[size,size]); known cells stay fixed.

    keep_foreground: unknown cells may not become TRANSPARENT (silhouette-conditioned infill).
    """
    from .palette import TRANSPARENT
    canvas = torch.as_tensor(canvas, dtype=torch.long)
    editable = ~torch.as_tensor(known_mask, dtype=torch.bool)
    allowed = None
    if keep_foreground:
        allowed = torch.ones((*canvas.shape, 4), dtype=torch.bool)
        allowed[..., TRANSPARENT] = False
        allowed[~editable] = True
        allowed = allowed[None]
    initial = canvas.clamp(0, 3)[None]
    return sample(model, size=canvas.shape[-1], count=1, seed=seed, cfg=cfg,
                  initial=initial, editable=editable[None], allowed=allowed)[0]
