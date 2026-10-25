"""Bounded single-process training; identical core for CPU, CUDA, and Modal."""
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from .checkpoint import atomic_json, checkpoint_path, provenance, restore_training, save_best, save_checkpoint
from .data import ApprovedDataset, BalancedBatchStream
from .masking import balanced_loss, corrupt, make_mask, masked_loss
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
    dataset = ApprovedDataset(cfg.dataset.manifest, cfg.dataset.resolutions, cfg.dataset.skip_other_sizes)
    if cfg.model.conditioning and not cfg.model.cond_games:
        train_labels = [dataset.labels(i) for i in dataset.indices("train")]
        for factor, name in (("game", "cond_games"), ("category", "cond_categories"), ("view", "cond_views"),
                             ("identity", "cond_identities")):
            setattr(cfg.model, name, sorted({l[factor] for l in train_labels if l[factor]}))
    model = PixelTransformer(cfg.model, cfg.dataset.resolutions).to(runtime["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=t.lr, weight_decay=t.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_multiplier(s, t))
    scaler = torch.amp.GradScaler("cuda", enabled=runtime["precision"] == "fp16")
    stream = BalancedBatchStream(dataset, t.batch_size, t.seed + 1,
                                 bucket_sampling=cfg.dataset.bucket_sampling, hshift=cfg.dataset.hshift, hflip=cfg.dataset.hflip)
    mask_generator = torch.Generator().manual_seed(t.seed + 2)
    ema_model = None
    if t.ema:
        import copy
        ema_model = copy.deepcopy(model).eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)
    step, early = 0, None
    if resume:
        step, early = restore_training(resume, cfg=cfg, dataset=dataset, runtime=runtime, model=model,
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
    if early:  # early-stop state survives resume
        best_val = math.inf if early["best"] is None else early["best"]
        best_step, stale = early["best_step"], early["stale"]
    early_state = lambda: {"best": None if best_val == math.inf else best_val, "best_step": best_step,
                           "stale": stale, "metric": t.select_metric}
    optimizer.zero_grad(set_to_none=True)
    try:
        while step < t.max_steps:
            if (stop_after is not None and step >= stop_after) or time.monotonic() - start >= t.max_seconds:
                reason = "stop_after" if stop_after is not None and step >= stop_after else "max_seconds"
                break
            tick = time.monotonic()
            losses, ce_losses, ratios, tokens = [], [], [], 0
            kinds, diagnostics, ratio_losses, fg_by_band = [], [], {}, {}
            predicted_counts = torch.zeros(4, dtype=torch.long)
            for _ in range(t.grad_accum_steps):
                targets = stream.next().to(runtime["device"])
                mask, mask_kinds = make_mask(targets != PAD, cfg.masking, generator=mask_generator, return_info=True)
                with autocast_context(runtime):
                    labels = None
                    if cfg.model.conditioning:
                        labels = model.encode_labels([dataset.labels(i) for i in stream.last_indices])
                        keep = torch.rand(labels.shape, generator=mask_generator).to(labels.device) >= cfg.model.cond_dropout
                        labels = labels * keep
                    logits = model(corrupt(targets, mask), labels=labels)
                    ce = masked_loss(logits.float(), targets, mask)
                    loss = balanced_loss(logits.float(), targets, mask) if cfg.dataset.loss == "fg_bg_balanced" else ce
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite loss at step {step}")
                scaler.scale(loss / t.grad_accum_steps).backward()
                losses.append(loss.detach().item())
                ce_losses.append(ce.detach().item())
                ratios.extend(mask.float().flatten(1).mean(1).tolist())
                kinds.extend(mask_kinds)
                from .evaluate import categorical_metrics
                for j in range(len(targets)):
                    metrics = categorical_metrics(logits[j].detach().cpu(), targets[j].cpu(), mask[j].cpu())
                    diagnostics.append(metrics)
                    ratio = metrics["actual_mask_ratio"]
                    band = "100%" if ratio == 1 else "75-100%" if ratio > .75 else "50-75%" if ratio > .5 else "25-50%" if ratio > .25 else "0-25%"
                    ratio_losses.setdefault(band, []).append(metrics["masked_conditional_nll"])
                    if metrics["foreground_nll"] is not None:
                        fg_by_band.setdefault(band, []).append((metrics["foreground_nll"], metrics["foreground_accuracy"]))
                predicted_counts += torch.bincount(logits.detach().argmax(-1)[mask].cpu(), minlength=4)
                tokens += targets.numel()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip_norm, error_if_nonfinite=True)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if ema_model is not None:
                with torch.no_grad():
                    decay = min(t.ema, (1 + step) / (10 + step))
                    for e, p in zip(ema_model.parameters(), model.parameters()):
                        e.lerp_(p, 1 - decay)
            if runtime["device"] == "cuda":
                torch.cuda.synchronize()
            duration = time.monotonic() - tick
            row = {"step": step, "loss": sum(losses) / len(losses), "loss_ce": sum(ce_losses) / len(ce_losses),
                   "objective": cfg.dataset.loss, "lr": optimizer.param_groups[0]["lr"],
                   "gradient_norm": norm.item(), "mask_ratio": sum(ratios) / len(ratios),
                   "fully_masked_fraction": sum(r == 1 for r in ratios) / len(ratios),
                   "seconds": duration, "tokens_per_second": tokens / duration,
                   "effective_dataset_passes": stream.position / len(dataset.indices("train")),
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated() if runtime["device"] == "cuda" else 0,
                   "peak_reserved_bytes": torch.cuda.max_memory_reserved() if runtime["device"] == "cuda" else 0}
            row["mask_kinds"] = {kind: kinds.count(kind) / len(kinds) for kind in ("full", "random", "contiguous")}
            row["loss_by_mask_ratio"] = {band: sum(values) / len(values) for band, values in ratio_losses.items()}
            row["foreground_by_mask_ratio"] = {band: {"nll": sum(v[0] for v in values) / len(values),
                                                      "accuracy": sum(v[1] for v in values) / len(values)}
                                               for band, values in fg_by_band.items()}
            row["predicted_class_balance"] = (predicted_counts.float() / predicted_counts.sum()).tolist()
            row["masked_metrics"] = {key: sum(values) / len(values) if values else None for key in diagnostics[0]
                for values in [[r[key] for r in diagnostics if r[key] is not None]]}
            if t.eval_every and step % t.eval_every == 0:
                from .evaluate import MASK_RATIOS, conditional_report, summarize
                eval_model = ema_model if ema_model is not None else model
                row["validation"] = conditional_report(eval_model, dataset, split="val", ratios=MASK_RATIOS,
                                                         seed=t.seed + 100)
                summary = row["val_summary"] = summarize(row["validation"])
                # Scalar val loss: plain model masked NLL averaged over every size/kind/ratio row.
                row["val_loss"] = summary["model"]["masked_conditional_nll"]
                row["val_balanced_nll"] = summary["balanced_nll"]
                score = row[t.select_metric]
                if score < best_val - t.min_delta:
                    best_val, best_step, stale = score, step, 0
                    save_best(run, model=eval_model, step=step, score=score, metric=t.select_metric,
                              cfg=cfg, dataset=dataset, runtime=runtime)
                else:
                    stale += 1
                row["best_val_loss"], row["best_val_step"] = best_val, best_step
                row["select_metric"], row["stale_evals"] = t.select_metric, stale
            if t.sample_every and step % t.sample_every == 0:
                from .sample import sample
                from .render import export_samples
                for size in cfg.dataset.resolutions:
                    samples = sample(model, size=size, count=4, seed=t.seed + 200, cfg=cfg.sample)
                    export_samples(samples, run / "samples" / f"step-{step:08d}-{size}",
                                   metadata={"step": step, "size": size, "seed": t.seed + 200})
            with (run / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps({k: v for k, v in row.items() if k not in ("validation", "val_summary")}), flush=True)
            if t.checkpoint_every and step % t.checkpoint_every == 0:
                checkpoint = save_checkpoint(run, model=model, optimizer=optimizer, scheduler=scheduler,
                    scaler=scaler, stream=stream, mask_generator=mask_generator, step=step,
                    cfg=cfg, dataset=dataset, runtime=runtime, durable_commit=durable_commit,
                    early_stop=early_state())
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
                cfg=cfg, dataset=dataset, runtime=runtime, durable_commit=durable_commit,
                early_stop=early_state())
    result = {"step": step, "reason": reason, "elapsed_seconds": time.monotonic() - start,
              "checkpoint": str(checkpoint),
              "best_val_loss": best_val if best_step is not None else None, "best_val_step": best_step,
              "select_metric": t.select_metric, "best_dir": str(run / "best") if best_step is not None else None,
              "exposure_by_identity": dict(stream.exposure), "bucket_policy": stream.policy, "device": runtime["device"], "synthetic": dataset.raw.get("synthetic", False)}
    atomic_json(run / "result.json", result)
    if durable_commit:
        durable_commit()
    return result
