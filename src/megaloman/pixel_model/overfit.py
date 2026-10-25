"""Optional optimization diagnostics, separate from artistic/generalization gates."""
import time
from pathlib import Path

import torch

from .checkpoint import atomic_json
from .data import ApprovedDataset
from .masking import corrupt, make_mask, masked_loss
from .model import PixelTransformer
from .render import export_samples
from .sample import sample
from .train import autocast_context, resolve_runtime


def overfit_checks(cfg, *, steps, output):
    if not 1 <= steps <= 10000:
        raise ValueError("overfit diagnostic steps must be 1..10000")
    cfg.validate()
    runtime = resolve_runtime(cfg.train)
    torch.set_num_threads(cfg.train.cpu_threads)
    dataset = ApprovedDataset(cfg.dataset.manifest, cfg.dataset.resolutions)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    start = time.monotonic()
    for mode in ("single_full", "few_partial"):
        torch.manual_seed(cfg.train.seed)
        model = PixelTransformer(cfg.model, cfg.dataset.resolutions).to(runtime["device"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=runtime["precision"] == "fp16")
        indices = dataset.indices("train", cfg.dataset.resolutions[0])[:1 if mode == "single_full" else 4]
        targets = torch.stack([dataset.grids[i] for i in indices]).to(runtime["device"])
        generator = torch.Generator().manual_seed(cfg.train.seed)
        losses = []
        for step in range(steps + 1):
            if time.monotonic() - start > cfg.train.max_seconds:
                break
            # Fixed evaluation masks allow a before/after loss comparison.
            fixed_generator = torch.Generator().manual_seed(cfg.train.seed + 99)
            fixed = make_mask(torch.ones_like(targets, dtype=torch.bool), ratio=1 if mode == "single_full" else .5,
                              generator=fixed_generator)
            model.eval()
            with torch.no_grad():
                loss = masked_loss(model(corrupt(targets, fixed)), targets, fixed).item()
            losses.append(loss)
            if step == steps:
                break
            model.train()
            mask = make_mask(torch.ones_like(targets, dtype=torch.bool), ratio=1 if mode == "single_full" else .5,
                             generator=generator)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(runtime):
                objective = masked_loss(model(corrupt(targets, mask)).float(), targets, mask)
            if not torch.isfinite(objective):
                raise FloatingPointError("nonfinite overfit objective")
            scaler.scale(objective).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            scaler.step(optimizer); scaler.update()
        if not losses:
            results[mode] = {"status": "time guard before first step"}
            continue
        generated = sample(model, size=targets.shape[-1], cfg=cfg.sample, seed=cfg.train.seed)
        export_samples(generated, output / mode)
        results[mode] = {"examples": len(targets), "updates": len(losses) - 1, "fixed_mask_losses": losses,
                         "loss_decreased": losses[-1] < losses[0], "memorization_gate": "not declared by loss decrease alone"}
    atomic_json(output / "overfit.json", results)
    return results
