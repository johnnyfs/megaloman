"""Fair score for silhouette/outline-conditioned tone infill on real val frames.

Given each val frame's TRANSPARENT and OUTLINE (INK on the silhouette border) as known, the
interior is hidden. Model vs three baselines that get the same shape:
  position  per-(size,y,x) prior over INK/TONE_1/TONE_2 from train interiors
  nn        nearest train frame by silhouette IoU (same size); its classes where it has
            foreground, else its most common interior class (".01-smoothed")
  random    iid fill from the train interior role histogram
Metrics on interior pixels only: NLL (model renormalised over the 3 fg roles), accuracy,
role-histogram L1 vs truth, coherence (same-class share of interior 4-neighbour pairs),
speckles (interior pixels with no same-class 4-neighbour) per frame, vs the true frames.

usage: PYTHONPATH=src python scripts/interior_eval.py TONE_CKPT OUT.json [--min-ink .8]
"""
import argparse
import json
import math
from pathlib import Path

import torch

from megaloman.pixel_model.checkpoint import load_model
from megaloman.pixel_model.data import ApprovedDataset
from megaloman.pixel_model.palette import INK, MASK
from megaloman.pixel_model.sample import infill
from megaloman.pixel_model.snapshot_outline import border

V2 = Path.home() / "Projects/megaloman-ptv0/workspace/pixel_model/datasets/mm-rm-enemies-v2/manifest.json"


def known_of(g):
    edge = torch.from_numpy(border(g.numpy()))
    return (g == 0) | (edge & (g == INK))


def structure(g, interior):
    same, pairs = 0, 0
    speck = 0
    h, w = g.shape
    for y in range(h):
        for x in range(w):
            if not interior[y, x]:
                continue
            nb = [(y + dy, x + dx) for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))
                  if 0 <= y + dy < h and 0 <= x + dx < w and g[y + dy, x + dx] != 0]
            match = [g[p] == g[y, x] for p in nb]
            if not any(match):
                speck += 1
            for p, m in zip(nb, match):
                if interior[p] and p > (y, x):
                    pairs += 1
                    same += int(m)
    return same / max(1, pairs), speck


def summarise(rows):
    return {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("out")
    p.add_argument("--min-ink", type=float, default=.8)
    a = p.parse_args()
    model, cfg, _ = load_model(a.checkpoint, "cuda")
    ds = ApprovedDataset(V2, cfg.dataset.resolutions, True)
    gen = torch.Generator().manual_seed(7)
    results = {}
    for size in cfg.dataset.resolutions:
        train = [ds.grids[i] for i in ds.indices("train", size)]
        inter_train = [((g != 0) & ~known_of(g)) for g in train]
        counts = torch.zeros(size, size, 4)
        hist = torch.zeros(4)
        for g, m in zip(train, inter_train):
            counts += torch.nn.functional.one_hot(g, 4).float() * m[..., None]
            hist += torch.bincount(g[m], minlength=4).float()
        pos = (counts[..., 1:] + 1) / (counts[..., 1:] + 1).sum(-1, keepdim=True)
        hist = hist[1:] / hist[1:].sum()
        rows = {k: [] for k in ("model", "position", "nn", "random", "truth")}
        for i in ds.indices("val", size):
            g = ds.grids[i]
            edge = torch.from_numpy(border(g.numpy()))
            if edge.any() and (g[edge] == INK).float().mean() < a.min_ink:
                continue
            known = known_of(g)
            interior = ~known
            if not interior.any():
                continue
            t = g[interior] - 1  # 0..2 over INK/T1/T2
            # model: one pass with the whole interior masked, renormalised over fg roles
            x = g.clone()
            x[interior] = MASK
            with torch.inference_mode():
                logits = model(x[None].cuda())[0].float().cpu()
            pm = logits[..., 1:].softmax(-1)[interior]
            filled = infill(model, torch.where(known, g, 0), known, seed=1729 + i, cfg=cfg.sample,
                            keep_foreground=True).cpu()
            # nearest train silhouette
            fg = g != 0
            ious = torch.tensor([((fg & (h != 0)).sum() / (fg | (h != 0)).sum()).item() for h in train])
            nn = train[ious.argmax()]
            mode = torch.bincount(nn[nn != 0], minlength=4)[1:].argmax() + 1
            nn_fill = torch.where(nn != 0, nn, mode)
            pn = (torch.nn.functional.one_hot(nn_fill[interior] - 1, 3).float() + .01) / 1.03
            pp = pos[interior]
            rnd = g.clone()
            rnd[interior] = torch.multinomial(hist, int(interior.sum()), replacement=True, generator=gen) + 1
            pos_fill = g.clone()
            pos_fill[interior] = torch.multinomial(pp, 1, generator=gen).flatten() + 1
            nn_grid = torch.where(interior, nn_fill, g)
            true_hist = torch.bincount(t, minlength=3).float() / len(t)
            for name, probs, grid in (("model", pm, filled), ("position", pp, pos_fill), ("nn", pn, nn_grid),
                                      ("random", hist.expand(len(t), 3), rnd), ("truth", None, g)):
                coherence, speck = structure(grid, interior)
                row = {"coherence": coherence, "speckles": speck}
                if probs is not None:
                    row["nll"] = -probs.gather(1, t[:, None]).clamp_min(1e-9).log().mean().item()
                    row["accuracy"] = (probs.argmax(1) == t).float().mean().item()
                    sh = torch.bincount(grid[interior] - 1, minlength=3).float() / len(t)
                    row["hist_l1"] = (sh - true_hist).abs().sum().item()
                    row["sample_accuracy"] = (grid[interior] - 1 == t).float().mean().item()
                rows[name].append(row)
        results[str(size)] = {k: summarise(v) for k, v in rows.items()} | {"frames": len(rows["truth"])}
    mean = {k: {m: sum(results[s][k][m] for s in results) / len(results) for m in results["16"][k]}
            for k in ("model", "position", "nn", "random", "truth")}
    report = {"checkpoint": a.checkpoint, "by_size": results, "mean": mean}
    Path(a.out).write_text(json.dumps(report, indent=2) + "\n")
    for k, v in mean.items():
        print(k, {m: round(x, 3) for m, x in v.items()})


if __name__ == "__main__":
    main()
