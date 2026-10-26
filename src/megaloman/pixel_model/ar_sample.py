"""Ancestral sampling and exact NLL for the causal (autoregressive) pixel model."""
import torch
from torch.nn import functional as F


def order_indices(size, order):
    """Flat (row-major) cell indices in generation order."""
    rows = range(size - 1, -1, -1) if order == "bottom" else range(size)
    return [y * size + x for y in rows for x in range(size)]


@torch.inference_mode()
def ar_sample(model, *, size, count=1, seed=42, temperature=1.0, top_p=1.0, known=None, known_mask=None):
    """Sample pixel by pixel. known/known_mask (LongTensor/BoolTensor [count,size,size]) give infill:
    known cells are teacher-forced when their turn comes (only earlier known cells condition each step)."""
    device = next(model.parameters()).device
    g = torch.Generator(device=device).manual_seed(seed)
    pixels = torch.zeros(count, size, size, dtype=torch.long, device=device)
    if known is not None:
        known, known_mask = known.to(device), known_mask.to(device)
    was = model.training
    model.eval()
    try:
        for idx in order_indices(size, model.cfg.ar_order):
            y, x = divmod(idx, size)
            # causal: logits at (y,x) only depend on earlier cells, so later placeholders are harmless
            logits = model(pixels)[:, y, x].float()
            if temperature == 0:
                choice = logits.argmax(-1)
            else:
                probs = (logits / temperature).softmax(-1)
                if top_p < 1:
                    sp, si = probs.sort(-1, descending=True)
                    drop = sp.cumsum(-1) - sp > top_p
                    sp = sp.masked_fill(drop, 0)
                    probs = torch.zeros_like(probs).scatter(-1, si, sp)
                choice = torch.multinomial(probs, 1, generator=g).squeeze(-1)
            if known is not None:
                choice = torch.where(known_mask[:, y, x], known[:, y, x], choice)
            pixels[:, y, x] = choice
    finally:
        model.train(was)
    return pixels


@torch.inference_mode()
def ar_nll(model, grids, batch=64):
    """Per-frame mean NLL (nats/pixel) of exact teacher-forced likelihood."""
    device = next(model.parameters()).device
    model.eval()
    out = []
    for i in range(0, len(grids), batch):
        g = grids[i:i + batch].to(device)
        logits = model(g).float()
        out.append(F.cross_entropy(logits.permute(0, 3, 1, 2), g, reduction="none").flatten(1).mean(1).cpu())
    return torch.cat(out)
