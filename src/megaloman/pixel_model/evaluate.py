"""Frozen-mask metrics, training-only controls, novelty panels, and human hooks."""
import csv
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from .checkpoint import atomic_json
from .data import ApprovedDataset
from .masking import corrupt, make_mask
from .render import contact_sheet, export_samples
from .sample import sample

MASK_RATIOS = (.25, .50, .75, .95, 1.0)


def _divide(a, b):
    return float(a / b) if b else None


def categorical_metrics(logits, target, mask):
    """One image; conditional metrics at masked cells and full reconstruction IoU."""
    probs = logits.float().softmax(-1)
    predicted = probs.argmax(-1)
    true, pred = target[mask], predicted[mask]
    ce = -probs[mask].gather(-1, true[:, None]).clamp_min(1e-30).log().squeeze(-1)
    fg = true != 0
    predicted_fg = pred != 0
    tp = (fg & predicted_fg).sum().item()
    reconstructed = torch.where(mask, predicted, target)
    union = ((reconstructed != 0) | (target != 0)).sum().item()
    metrics = {"masked_conditional_nll": ce.mean().item(), "masked_accuracy": (true == pred).float().mean().item(),
               "foreground_nll": ce[fg].mean().item() if fg.any() else None,
               "background_nll": ce[~fg].mean().item() if (~fg).any() else None,
               "foreground_accuracy": (true[fg] == pred[fg]).float().mean().item() if fg.any() else None,
               "foreground_precision": _divide(tp, predicted_fg.sum().item()),
               "foreground_recall": _divide(tp, fg.sum().item()),
               "reconstruction_foreground_iou": _divide(((reconstructed != 0) & (target != 0)).sum().item(), union),
               "actual_mask_ratio": mask.float().mean().item()}
    for cls in range(4):
        chosen = true == cls
        metrics[f"class_{cls}_accuracy"] = (pred[chosen] == cls).float().mean().item() if chosen.any() else None
    return metrics


def global_prior(dataset, smoothing=1.0):
    """One 4-class distribution over every training pixel at every configured size."""
    counts = torch.bincount(torch.cat([dataset.grids[i].flatten() for i in dataset.indices("train")]), minlength=4)
    counts = counts.double() + smoothing
    return (counts / counts.sum()).float()


def fit_baselines(dataset, size, smoothing=1.0, global_frequency=None):
    """Training-only priors: true global, per-size bucket, and per-(size, y, x) position."""
    grids = torch.stack([dataset.grids[i] for i in dataset.indices("train", size)])
    one_hot = F.one_hot(grids, num_classes=4).float()
    position = (one_hot.sum(0) + smoothing)
    position = position / position.sum(-1, keepdim=True)
    bucket = one_hot.sum((0, 1, 2)) + smoothing
    bucket /= bucket.sum()
    if global_frequency is None:
        global_frequency = global_prior(dataset, smoothing)
    return {"grids": grids, "global": global_frequency.log().expand(size, size, 4),
            "bucket": bucket.log().expand(size, size, 4), "position": position.log()}


def visible_nearest_neighbor(training, target, mask):
    visible = ~mask
    mismatch = ((training != target) & visible).flatten(1).sum(1)
    # First training record wins ties. At 100% masking, all candidates tie.
    return training[mismatch.argmin()]


def _means(rows):
    out = {}
    for key in rows[0]:
        values = [r[key] for r in rows if r[key] is not None]
        out[key] = sum(values) / len(values) if values else None
    return out


def identity_aggregate(rows, seed):
    identities = {}
    for identity, metrics in rows:
        identities.setdefault(identity, []).append(metrics)
    per_identity = {key: _means(metrics) for key, metrics in identities.items()}
    aggregate = _means(list(per_identity.values()))
    values = torch.tensor([r["masked_conditional_nll"] for r in per_identity.values()])
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(len(values), (200, len(values)), generator=generator)
    bootstrap = values[indices].mean(1)
    return {"metrics": aggregate, "identities": len(values), "frames": len(rows),
            "identity_nll_bootstrap_95pct": torch.quantile(bootstrap, torch.tensor([.025, .975])).tolist(),
            "per_identity": per_identity}


@torch.inference_mode()
def conditional_report(model, dataset, *, split="val", ratios=MASK_RATIOS, seed=1729, include_contiguous=True):
    if split not in ("val", "test"):
        raise ValueError("evaluate frozen val or test splits; choose validation during development")
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    results = []
    try:
        prior = global_prior(dataset)
        for size in model.resolutions:
            baselines = fit_baselines(dataset, size, global_frequency=prior)
            for kind in (("random", "contiguous") if include_contiguous else ("random",)):
                for ratio in ratios:
                    if kind == "contiguous" and ratio == 1:
                        continue  # identical to the random full mask; count it once
                    generator = torch.Generator().manual_seed(seed + size + int(ratio * 10000))
                    rows = {key: [] for key in ("model", "global_frequency", "bucket_frequency", "position_frequency",
                                                    "visible_nearest_neighbor")}
                    for i in dataset.indices(split, size):
                        target = dataset.grids[i]
                        mask = make_mask(torch.ones_like(target, dtype=torch.bool)[None], ratio=ratio,
                                         contiguous=kind == "contiguous", generator=generator)[0]
                        logits = model(corrupt(target[None].to(device), mask[None].to(device)))[0].cpu()
                        neighbor = visible_nearest_neighbor(baselines["grids"], target, mask)
                        # Retrieval has no likelihood; report a declared .01-smoothed categorical surrogate.
                        neighbor_logits = ((F.one_hot(neighbor, 4).float() + .01) / 1.04).log()
                        for key, values in (("model", logits), ("global_frequency", baselines["global"]),
                                            ("bucket_frequency", baselines["bucket"]),
                                            ("position_frequency", baselines["position"]),
                                            ("visible_nearest_neighbor", neighbor_logits)):
                            rows[key].append((dataset.records[i]["identity"], categorical_metrics(values, target, mask)))
                    grouped = {key: identity_aggregate(value, seed) for key, value in rows.items()}
                    baseline_nll = grouped["position_frequency"]["metrics"]["masked_conditional_nll"]
                    model_nll = grouped["model"]["metrics"]["masked_conditional_nll"]
                    results.append({"size": size, "mask_kind": kind, "mask_ratio": ratio, **grouped,
                                    "position_baseline_relative_nll_improvement": 1 - model_nll / baseline_nll})
    finally:
        model.train(was_training)
    return {"split": split, "mask_seed": seed, "rows": results,
            "notes": ["Metrics average frames within identities, then identities equally.",
                      "Bootstrap intervals are descriptive; tiny fixture cohorts cannot support generalization claims.",
                      "Visible-neighbor NLL is a .01-smoothed surrogate, not retrieval likelihood.",
                      "100%-mask accuracy is not unconditional sample quality.",
                      "global_frequency is one prior over all sizes; bucket_frequency is per size.",
                      "Contiguous 100% equals random 100% and is reported once."]}


SUMMARY_KEYS = ("masked_conditional_nll", "foreground_nll", "background_nll", "foreground_accuracy",
                "foreground_precision", "foreground_recall", "reconstruction_foreground_iou", "masked_accuracy")


def summarize(report):
    """Equal-weight means over rows: overall, full mask only, and per ratio, for model and priors."""
    def mean(rows, who, key):
        values = [r[who]["metrics"][key] for r in rows if r[who]["metrics"][key] is not None]
        return sum(values) / len(values) if values else None
    rows = report["rows"]
    full = [r for r in rows if r["mask_ratio"] == 1]
    out = {"model": {k: mean(rows, "model", k) for k in SUMMARY_KEYS},
           "model_full_mask": {k: mean(full, "model", k) for k in SUMMARY_KEYS},
           "priors": {who: {"nll": mean(rows, who, "masked_conditional_nll"),
                            "full_mask_nll": mean(full, who, "masked_conditional_nll"),
                            "foreground_nll": mean(rows, who, "foreground_nll")}
                      for who in ("global_frequency", "bucket_frequency", "position_frequency")},
           "by_ratio": {str(q): {k: mean([r for r in rows if r["mask_ratio"] == q], "model", k)
                                 for k in ("masked_conditional_nll", "foreground_nll", "foreground_accuracy",
                                           "reconstruction_foreground_iou")}
                        for q in sorted({r["mask_ratio"] for r in rows})}}
    m = out["model"]
    out["balanced_nll"] = (.5 * m["foreground_nll"] + .5 * m["background_nll"]
                           if m["foreground_nll"] is not None and m["background_nll"] is not None else None)
    return out


def foreground_distance(a, b):
    """Categorical disagreements over foreground union; blank-blank distance is zero."""
    union = (a != 0) | (b != 0)
    return ((a != b) & union).flatten(-2).sum(-1).float() / union.flatten(-2).sum(-1).clamp_min(1)


def novelty_report(samples, training, *, near_threshold=.10):
    nearest_indices, distances, exact = [], [], []
    for sprite in samples:
        distance = foreground_distance(training, sprite)
        index = distance.argmin().item()
        nearest_indices.append(index)
        distances.append(distance[index].item())
        exact.append(bool((training == sprite).flatten(1).all(1).any()))
    pairs = [foreground_distance(samples[i + 1:], samples[i]) for i in range(len(samples) - 1)]
    pair_mean = torch.cat(pairs).mean().item() if pairs else None
    unique = len({g.numpy().tobytes() for g in samples.cpu()})
    return {"count": len(samples), "exact_training_copy_rate": sum(exact) / len(samples),
            "near_training_copy_rate": sum(d <= near_threshold for d in distances) / len(samples),
            "near_threshold": near_threshold, "near_threshold_calibrated": False,
            "nearest_training_indices": nearest_indices, "nearest_foreground_distances": distances,
            "unique_fraction": unique / len(samples), "mean_pairwise_foreground_distance": pair_mean,
            "blank_fraction": (samples.flatten(1) == 0).all(1).float().mean().item(),
            "class_balance": torch.bincount(samples.flatten(), minlength=4).float().div(samples.numel()).tolist(),
            "noisy_sample_rate": None, "noisy_sample_rate_status": "requires human ratings; no calibrated classifier"}


def write_human_hooks(samples, controls, output, seed):
    output = Path(output)
    n = len(samples)
    generator = torch.Generator().manual_seed(seed)
    control_indices = torch.randint(len(controls), (n,), generator=generator)
    all_grids = torch.cat((samples, controls[control_indices]))
    order = torch.randperm(2 * n, generator=generator)
    contact_sheet(all_grids[order], output / "blind-sheet.png")
    atomic_json(output / "blind-key.json", [
        {"sheet_index": j, "source": "generated" if i < n else "held_out_control",
         "source_index": i if i < n else control_indices[i - n].item()} for j, i in enumerate(order.tolist())])
    with (output / "human-ratings.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["sheet_index", "silhouette_1_to_5", "coherence_1_to_5", "readability_1_to_5",
                         "usefulness_1_to_5", "useful_without_structural_redraw", "blank_or_noisy", "notes"])
        for j in range(2 * n):
            writer.writerow([j, "", "", "", "", "", "", ""])


def evaluate(model, cfg, *, output, split="val", count=64, seed=1729, near_threshold=.10):
    if count < 1 or not 0 <= near_threshold <= 1:
        raise ValueError("sample count must be positive and near-copy threshold in [0,1]")
    dataset = ApprovedDataset(cfg.dataset.manifest, cfg.dataset.resolutions, cfg.dataset.skip_other_sizes)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    conditional = conditional_report(model, dataset, split=split, seed=seed)
    report = {"dataset": dataset.summary(), "conditional": conditional, "summary": summarize(conditional),
              "generation_seed": seed, "generation": {}, "artistic_gate": "unassessed: human ratings required",
              "baseline_gate": "provisional >=5% lower position-baseline NLL at random 50% and 75% masks",
              "quality_claim": "scaffold/fixture results establish execution only"}
    for size in cfg.dataset.resolutions:
        directory = output / str(size)
        directory.mkdir(exist_ok=True)
        # Sequential samples bound attention memory independently of report size.
        grids = torch.cat([sample(model, size=size, seed=seed + i, cfg=cfg.sample).cpu() for i in range(count)])
        train_grids = torch.stack([dataset.grids[i] for i in dataset.indices("train", size)])
        controls = torch.stack([dataset.grids[i] for i in dataset.indices(split, size)])
        novelty = novelty_report(grids, train_grids, near_threshold=near_threshold)
        report["generation"][str(size)] = novelty
        export_samples(grids, directory / "samples", metadata={"seed": seed, "unselected": True})
        nearest = torch.stack([train_grids[i] for i in novelty["nearest_training_indices"]])
        contact_sheet(torch.stack((grids, nearest), 1).flatten(0, 1), directory / "nearest-training.png")
        generator = torch.Generator().manual_seed(seed)
        resampled = train_grids[torch.randint(len(train_grids), (count,), generator=generator)]
        contact_sheet(resampled, directory / "training-resampling-control.png")
        novelty["training_resampling_control"] = novelty_report(resampled, train_grids, near_threshold=near_threshold)
        write_human_hooks(grids, controls, directory, seed)
    atomic_json(output / "report.json", report)
    return report
