"""Bounded single-process training; identical core for CPU, CUDA, and Modal."""
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from .checkpoint import atomic_json, checkpoint_path, provenance, restore_training, save_checkpoint
from .data import ApprovedDataset, BalancedBatchStream
from .masking import corrupt, make_mask, masked_loss
from .model import PixelTransformer
from .palette import PAD


def resolve_runtime(cfg):
    device = "cuda" if cfg.device == "auto" and torch.cuda.is_available() else cfg.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; use device=cpu or fix the driver")
    precision = cfg.precision
    if precision == "auto":
        precision = ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if device == "cuda" else "fp32"
    if device == "cpu" and precision == "fp16":
        raise ValueError("CPU fp16 training is unsupported; use fp32 or bf16")
    if device == "cuda" and precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("CUDA BF16 requested but unsupported")
    return {"device": device, "precision": precision,
            "torch": torch.__version__, "threads": cfg.cpu_threads,
            "gpu": torch.cuda.get_device_name() if device == "cuda" else None}


def autocast_context(runtime):
    if runtime["precision"] == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if runtime["precision"] == "bf16" else torch.float16
    return torch.autocast(runtime["device"], dtype=dtype)


def lr_multiplier(step, cfg):
    if step < cfg.warmup_steps:
        return (step + 1) / max(1, cfg.warmup_steps)
    progress = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps))
    floor = cfg.min_lr / cfg.lr
    return floor + (1 - floor) * .5 * (1 + math.cos(math.pi * progress))


def train(cfg, *, resume=None, stop_after=None, durable_commit=None):
    """stop_after interrupts an existing schedule for reproducibility tests/smoke.

    max_steps remains the fixed scheduler horizon on resume. Runtime guards are
    checked between optimizer updates; an in-progress update is allowed to finish.
    """
    cfg.validate()
    t = cfg.train
    runtime = resolve_runtime(t)
    torch.set_num_threads(t.cpu_threads)
    random.seed(t.seed); np.random.seed(t.seed); torch.manual_seed(t.seed)
    if runtime["device"] == "cuda":
        torch.cuda.manual_seed_all(t.seed)
        torch.cuda.reset_peak_memory_stats()
    dataset = ApprovedDataset(cfg.dataset.manifest, cfg.dataset.resolutions)
    model = PixelTransformer(cfg.model, cfg.dataset.resolutions).to(runtime["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=t.lr, weight_decay=t.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_multiplier(s, t))
    scaler = torch.amp.GradScaler("cuda", enabled=runtime["precision"] == "fp16")
    stream = BalancedBatchStream(dataset, t.batch_size, t.seed + 1)
    mask_generator = torch.Generator().manual_seed(t.seed + 2)
    step = 0
    if resume:
        step = restore_training(resume, cfg=cfg, dataset=dataset, runtime=runtime, model=model,
                                optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                                stream=stream, mask_generator=mask_generator)
    run = Path(t.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    if not resume and (run / "metrics.jsonl").exists():
        raise ValueError("run already has metrics; resume its checkpoint or choose a fresh run_dir")
    if resume:
        later = [p for p in (run / "checkpoints").glob("step-*") if int(p.name.removeprefix("step-")) > step]
        if later:
            raise ValueError("run contains later checkpoints; resume latest or choose a fresh run_dir")
        metrics_path = run / "metrics.jsonl"
        if metrics_path.exists():
            # A killed run may have logged steps after its last durable checkpoint.
            # Keep only completed checkpoint history; tolerate a torn final line.
            rows = metrics_path.read_text().splitlines()
            kept = []
            for i, line in enumerate(rows):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    if i == len(rows) - 1:
                        break
                    raise
                if record["step"] <= step:
                    kept.append(line)
            metrics_path.write_text("\n".join(kept) + ("\n" if kept else ""))
    atomic_json(run / "config.json", cfg.to_dict())
    atomic_json(run / "dataset.json", dataset.summary())
    atomic_json(run / "environment.json", {**runtime, **provenance(), "parameters": sum(p.numel() for p in model.parameters())})
    start = time.monotonic()
    checkpoint = checkpoint_path(resume) if resume else None
    last_saved = step if resume else -1
    reason = "max_steps"
    best_val, best_step, stale = math.inf, None, 0
    optimizer.zero_grad(set_to_none=True)
    try:
        while step < t.max_steps:
            if (stop_after is not None and step >= stop_after) or time.monotonic() - start >= t.max_seconds:
                reason = "stop_after" if stop_after is not None and step >= stop_after else "max_seconds"
                break
            tick = time.monotonic()
            losses, ratios, tokens = [], [], 0
            kinds, diagnostics, ratio_losses = [], [], {}
            predicted_counts = torch.zeros(4, dtype=torch.long)
            for _ in range(t.grad_accum_steps):
                targets = stream.next().to(runtime["device"])
                mask, mask_kinds = make_mask(targets != PAD, cfg.masking, generator=mask_generator, return_info=True)
                with autocast_context(runtime):
                    logits = model(corrupt(targets, mask))
                    loss = masked_loss(logits.float(), targets, mask)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite loss at step {step}")
                scaler.scale(loss / t.grad_accum_steps).backward()
                losses.append(loss.detach().item())
                ratios.extend(mask.float().flatten(1).mean(1).tolist())
                kinds.extend(mask_kinds)
                from .evaluate import categorical_metrics
                for j in range(len(targets)):
                    metrics = categorical_metrics(logits[j].detach().cpu(), targets[j].cpu(), mask[j].cpu())
                    diagnostics.append(metrics)
                    ratio = metrics["actual_mask_ratio"]
                    band = "100%" if ratio == 1 else "75-100%" if ratio > .75 else "50-75%" if ratio > .5 else "25-50%" if ratio > .25 else "0-25%"
                    ratio_losses.setdefault(band, []).append(metrics["masked_conditional_nll"])
                predicted_counts += torch.bincount(logits.detach().argmax(-1)[mask].cpu(), minlength=4)
                tokens += targets.numel()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip_norm, error_if_nonfinite=True)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if runtime["device"] == "cuda":
                torch.cuda.synchronize()
            duration = time.monotonic() - tick
            row = {"step": step, "loss": sum(losses) / len(losses), "lr": optimizer.param_groups[0]["lr"],
                   "gradient_norm": norm.item(), "mask_ratio": sum(ratios) / len(ratios),
                   "fully_masked_fraction": sum(r == 1 for r in ratios) / len(ratios),
                   "seconds": duration, "tokens_per_second": tokens / duration,
                   "effective_dataset_passes": stream.position / len(dataset.indices("train")),
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated() if runtime["device"] == "cuda" else 0,
                   "peak_reserved_bytes": torch.cuda.max_memory_reserved() if runtime["device"] == "cuda" else 0}
            row["mask_kinds"] = {kind: kinds.count(kind) / len(kinds) for kind in ("full", "random", "contiguous")}
            row["loss_by_mask_ratio"] = {band: sum(values) / len(values) for band, values in ratio_losses.items()}
            row["predicted_class_balance"] = (predicted_counts.float() / predicted_counts.sum()).tolist()
            row["masked_metrics"] = {key: sum(values) / len(values) if values else None for key in diagnostics[0]
                for values in [[r[key] for r in diagnostics if r[key] is not None]]}
            if t.eval_every and step % t.eval_every == 0:
                from .evaluate import conditional_report
                row["validation"] = conditional_report(model, dataset, split="val", ratios=(.5, .75, 1.0),
                                                         seed=t.seed + 100)
                # Scalar val loss: model masked NLL averaged over every size/kind/ratio row.
                nlls = [r["model"]["metrics"]["masked_conditional_nll"] for r in row["validation"]["rows"]]
                row["val_loss"] = sum(nlls) / len(nlls)
                if row["val_loss"] < best_val - t.min_delta:
                    best_val, best_step, stale = row["val_loss"], step, 0
                else:
                    stale += 1
                row["best_val_loss"], row["best_val_step"] = best_val, best_step
            if t.sample_every and step % t.sample_every == 0:
                from .sample import sample
                from .render import export_samples
                for size in cfg.dataset.resolutions:
                    samples = sample(model, size=size, count=4, seed=t.seed + 200, cfg=cfg.sample)
                    export_samples(samples, run / "samples" / f"step-{step:08d}-{size}",
                                   metadata={"step": step, "size": size, "seed": t.seed + 200})
            with (run / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps({k: v for k, v in row.items() if k != "validation"}), flush=True)
            if t.checkpoint_every and step % t.checkpoint_every == 0:
                checkpoint = save_checkpoint(run, model=model, optimizer=optimizer, scheduler=scheduler,
                    scaler=scaler, stream=stream, mask_generator=mask_generator, step=step,
                    cfg=cfg, dataset=dataset, runtime=runtime, durable_commit=durable_commit)
                last_saved = step
            if t.patience and stale >= t.patience:
                reason = "plateau"
                break
    except KeyboardInterrupt:
        # Never serialize partial gradients/RNG as a completed update.
        raise KeyboardInterrupt(f"resume the latest completed checkpoint: {checkpoint}") from None
    if step != last_saved:
        # Save only completed optimizer updates, including time-guard exits.
        existing = run / "checkpoints" / f"step-{step:08d}"
        if existing.exists():
            checkpoint = existing
        else:
            checkpoint = save_checkpoint(run, model=model, optimizer=optimizer, scheduler=scheduler,
                scaler=scaler, stream=stream, mask_generator=mask_generator, step=step,
                cfg=cfg, dataset=dataset, runtime=runtime, durable_commit=durable_commit)
    result = {"step": step, "reason": reason, "elapsed_seconds": time.monotonic() - start,
              "checkpoint": str(checkpoint),
              "best_val_loss": best_val if best_step is not None else None, "best_val_step": best_step, "device": runtime["device"], "synthetic": dataset.raw.get("synthetic", False)}
    atomic_json(run / "result.json", result)
    if durable_commit:
        durable_commit()
    return result
