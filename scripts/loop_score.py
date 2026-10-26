"""Fixed improvement-loop score for one checkpoint (plans/run2-diagnosis.md, queue item 0).

Reports val balanced NLL and 50%-mask NLL (frozen val masks, same seed as training),
plus the unconditional sample score: mean over sizes of role-histogram L1 vs train
+ |foreground-fraction gap|, from 64 samples per size at fixed seeds. Also NN novelty.

usage: PYTHONPATH=src python scripts/loop_score.py CHECKPOINT OUT_DIR [--device cuda]
       [--steps N] [--temperature T] [--ranking-temperature R] [--remask F] [--count 64]
"""
import argparse
import json
from pathlib import Path

import torch

from megaloman.pixel_model.checkpoint import load_model
from megaloman.pixel_model.data import ApprovedDataset
from megaloman.pixel_model.evaluate import conditional_report, foreground_distance, novelty_report, summarize
from megaloman.pixel_model.render import contact_sheet
from megaloman.pixel_model.sample import sample

SEED = 1729
LOCAL_MANIFEST = Path.home() / "Projects/megaloman-ptv0/workspace/pixel_model/datasets/mm-rm-enemies-v2/manifest.json"


def histogram(grids):
    return torch.bincount(grids.flatten(), minlength=4).double().div(grids.numel())


def blobs(grid):
    """4-connected foreground components."""
    fg = (grid != 0).tolist()
    h, w, seen, n = len(fg), len(fg[0]), set(), 0
    for y in range(h):
        for x in range(w):
            if fg[y][x] and (y, x) not in seen:
                n += 1
                stack = [(y, x)]
                seen.add((y, x))
                while stack:
                    cy, cx = stack.pop()
                    for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                        if 0 <= ny < h and 0 <= nx < w and fg[ny][nx] and (ny, nx) not in seen:
                            seen.add((ny, nx))
                            stack.append((ny, nx))
    return n


def border_ink(grids):
    """Fraction of silhouette-border pixels (fg touching transparent/edge, 4-nbr) that are INK."""
    fg = grids != 0
    pad = torch.nn.functional.pad(fg, (1, 1, 1, 1), value=False)
    touch = ~pad[:, :-2, 1:-1] | ~pad[:, 2:, 1:-1] | ~pad[:, 1:-1, :-2] | ~pad[:, 1:-1, 2:]
    edge = fg & touch
    return ((grids == 1) & edge).sum().item() / max(1, edge.sum().item())


def patterns(grids):
    """Normalised histogram of 3x3 role patterns (base-4 codes; zero padding)."""
    pad = torch.nn.functional.pad(grids, (1, 1, 1, 1))
    code = torch.zeros_like(grids)
    for dy in range(3):
        for dx in range(3):
            code = code * 4 + pad[:, dy:dy + grids.shape[1], dx:dx + grids.shape[2]]
    counts = torch.bincount(code.flatten(), minlength=4 ** 9).double()
    return counts / counts.sum()


def structure(samples, train):
    sb = sum(blobs(g) for g in samples) / len(samples)
    tb = sum(blobs(g) for g in train) / len(train)
    return {"blobs": sb, "train_blobs": tb, "outline": border_ink(samples), "train_outline": border_ink(train),
            "pattern_l1": (patterns(samples) - patterns(train)).abs().sum().item()}


def nn_distances(queries, training):
    return torch.tensor([foreground_distance(training, q).min().item() for q in queries])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("output")
    p.add_argument("--device", default="cuda")
    p.add_argument("--count", type=int, default=64)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--manifest", help="relocated snapshot (remote runs)")
    p.add_argument("--late", help="hybrid sampling: hand off to this checkpoint's model")
    p.add_argument("--handoff", type=float, default=.25)
    p.add_argument("--no-nll", action="store_true", help="skip conditional NLL (sampler-only change)")
    for name in ("steps", "temperature", "ranking_temperature", "remask", "remask_until", "guidance"):
        p.add_argument("--" + name.replace("_", "-"), type=float)
    a = p.parse_args()
    model, cfg, meta = load_model(a.checkpoint, a.device)
    late = load_model(a.late, a.device)[0].eval() if a.late else None
    for name in ("steps", "temperature", "ranking_temperature", "remask", "remask_until", "guidance"):
        value = getattr(a, name)
        if value is not None:
            setattr(cfg.sample, name, int(value) if name == "steps" else value)
    if a.manifest:
        cfg.dataset.manifest = a.manifest
    else:  # always score against the fixed v2 snapshot (same val; train reference = v2 train)
        cfg.dataset.manifest = str(LOCAL_MANIFEST)
    dataset = ApprovedDataset(cfg.dataset.manifest, cfg.dataset.resolutions, cfg.dataset.skip_other_sizes)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    report = {"checkpoint": str(a.checkpoint), "step": meta["step"], "sample_config": cfg.to_dict()["sample"]}
    if not a.no_nll:
        summary = summarize(conditional_report(model, dataset, split="val", seed=cfg.train.seed + 100))
        report["val_balanced_nll"] = summary["balanced_nll"]
        report["val_nll_50"] = summary["by_ratio"]["0.5"]["masked_conditional_nll"]
        report["val_summary"] = summary
    sizes, grid_rows = {}, []
    for size in cfg.dataset.resolutions:
        labels = [None] * a.count
        if getattr(model, "vocab", None):  # game/category of a seeded random train frame; identity null
            pool = [dataset.labels(i) for i in dataset.indices("train", size)]
            pick = torch.randint(len(pool), (a.count,), generator=torch.Generator().manual_seed(a.seed + size))
            labels = [model.encode_labels([pool[j]], drop=("identity",)) for j in pick.tolist()]
        grids = torch.cat([sample(model, size=size, seed=a.seed + i, cfg=cfg.sample, labels=labels[i],
                                  late_model=late, handoff=a.handoff).cpu()
                           for i in range(a.count)])
        train = torch.stack([dataset.grids[i] for i in dataset.indices("train", size)])
        val = torch.stack([dataset.grids[i] for i in dataset.indices("val", size)])
        hs, ht = histogram(grids), histogram(train)
        l1 = (hs - ht).abs().sum().item()
        fg_gap = abs((1 - hs[0]) - (1 - ht[0])).item()
        novelty = novelty_report(grids, train)
        sizes[str(size)] = {"role_l1": l1, "fg_gap": fg_gap, "score": l1 + fg_gap,
                            "sample_hist": hs.tolist(), "train_hist": ht.tolist(),
                            "exact_copy_rate": novelty["exact_training_copy_rate"],
                            "near_copy_rate": novelty["near_training_copy_rate"],
                            "nn_mean": nn_distances(grids, train).mean().item(),
                            "val_nn_mean": nn_distances(val, train).mean().item(),
                            "unique_fraction": novelty["unique_fraction"], **structure(grids, train)}
        contact_sheet(grids, out / f"samples-{size}.png", scale=4)
        grid_rows.append(grids[:4])
    report["sizes"] = sizes
    report["sample_score"] = sum(s["score"] for s in sizes.values()) / len(sizes)
    report["role_l1_mean"] = sum(s["role_l1"] for s in sizes.values()) / len(sizes)
    for key in ("pattern_l1", "blobs", "outline"):
        report[key] = sum(s[key] for s in sizes.values()) / len(sizes)
    report["copy_rate"] = max(s["exact_copy_rate"] for s in sizes.values())
    contact_sheet([g for row in grid_rows for g in row], out / "grid.png", scale=8, columns=4)
    (out / "score.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report.get(k) for k in ("val_balanced_nll", "val_nll_50", "sample_score", "role_l1_mean", "pattern_l1", "blobs", "outline",
                                                 "copy_rate")}))
    for size, s in sizes.items():
        print(size, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()
                     if k not in ("sample_hist", "train_hist")},
              "hist", [round(x, 3) for x in s["sample_hist"]], "train", [round(x, 3) for x in s["train_hist"]])


if __name__ == "__main__":
    main()
