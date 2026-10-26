"""Score an autoregressive checkpoint with the loop protocol: exact val NLL/pixel per size
(vs position prior), sample score, structure metrics and NN novelty (64 samples/size, seed 1729).
Optionally also the masked model's full-mask NLL on the same val frames (--masked CKPT).

usage: PYTHONPATH=src python scripts/ar_score.py CKPT OUT [--temperature 1] [--top-p 1] [--masked CKPT] [--no-nll]
"""
import argparse
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).parent))
from loop_score import LOCAL_MANIFEST, SEED, histogram, nn_distances, structure  # noqa: E402
from megaloman.pixel_model.ar_sample import ar_nll, ar_sample  # noqa: E402
from megaloman.pixel_model.checkpoint import load_model  # noqa: E402
from megaloman.pixel_model.data import ApprovedDataset  # noqa: E402
from megaloman.pixel_model.evaluate import fit_baselines, novelty_report  # noqa: E402
from megaloman.pixel_model.palette import MASK  # noqa: E402
from megaloman.pixel_model.render import contact_sheet  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint"); p.add_argument("output")
    p.add_argument("--device", default="cuda"); p.add_argument("--count", type=int, default=64)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--temperature", type=float, default=1.0); p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--masked"); p.add_argument("--no-nll", action="store_true")
    p.add_argument("--manifest", help="relocated snapshot (remote runs)")
    a = p.parse_args()
    model, cfg, meta = load_model(a.checkpoint, a.device)
    dataset = ApprovedDataset(a.manifest or str(LOCAL_MANIFEST), cfg.dataset.resolutions, cfg.dataset.skip_other_sizes)
    masked = load_model(a.masked, a.device)[0].eval() if a.masked else None
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    report = {"checkpoint": str(a.checkpoint), "step": meta["step"], "order": cfg.model.ar_order,
              "temperature": a.temperature, "top_p": a.top_p, "sizes": {}}
    rows = []
    for size in cfg.dataset.resolutions:
        train = torch.stack([dataset.grids[i] for i in dataset.indices("train", size)])
        val = torch.stack([dataset.grids[i] for i in dataset.indices("val", size)])
        s = {}
        if not a.no_nll:
            s["ar_nll"] = ar_nll(model, val, batch=16).mean().item()
            pos = fit_baselines(dataset, size)["position"]
            s["position_prior_nll"] = F.nll_loss(pos.expand(len(val), -1, -1, -1).permute(0, 3, 1, 2), val).item()
            if masked is not None:
                with torch.inference_mode():
                    logits = torch.cat([masked(torch.full_like(v, MASK).to(a.device)).float().cpu() for v in val.split(16)])
                s["masked_full_nll"] = F.cross_entropy(logits.permute(0, 3, 1, 2), val).item()
        grids = torch.cat([ar_sample(model, size=size, count=min(16, a.count - j), seed=a.seed + size + j,
                                     temperature=a.temperature, top_p=a.top_p).cpu() for j in range(0, a.count, 16)])
        hs, ht = histogram(grids), histogram(train)
        l1 = (hs - ht).abs().sum().item(); fg_gap = abs(hs[0] - ht[0]).item()
        nov = novelty_report(grids, train)
        s.update({"role_l1": l1, "fg_gap": fg_gap, "score": l1 + fg_gap, "sample_hist": hs.tolist(),
                  "train_hist": ht.tolist(), "exact_copy_rate": nov["exact_training_copy_rate"],
                  "near_copy_rate": nov["near_training_copy_rate"], "nn_mean": nn_distances(grids, train).mean().item(),
                  "val_nn_mean": nn_distances(val, train).mean().item(), **structure(grids, train)})
        report["sizes"][str(size)] = s
        contact_sheet(grids, out / f"samples-{size}.png", scale=4)
        rows.append(grids[:4])
    sz = report["sizes"].values()
    mean = lambda k: sum(s[k] for s in sz) / len(sz)
    for k in ("score", "role_l1", "pattern_l1", "blobs", "train_blobs", "outline", "train_outline", "nn_mean", "val_nn_mean"):
        report[k] = mean(k)
    report["sample_score"] = report.pop("score")
    if not a.no_nll:
        for k in ("ar_nll", "position_prior_nll") + (("masked_full_nll",) if masked else ()):
            report[k] = mean(k)
    report["copy_rate"] = max(s["exact_copy_rate"] for s in sz)
    contact_sheet([g for r in rows for g in r], out / "grid.png", scale=8, columns=4)
    (out / "score.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "sizes"}))


if __name__ == "__main__":
    main()
