"""Train a discrete-denoising pixel model, then score it like the masked loop (loop_score.py).

usage: PYTHONPATH=src python scripts/denoise_run.py train NAME [--seed S] [--steps N] [--no-uniform]
           [--no-tile] [--no-outline] [--no-blur] [--mask-share F] [--device cuda]
       PYTHONPATH=src python scripts/denoise_run.py score NAME [--steps K] [--temperature T]
           [--renoise confidence|random] [--count 64] [--tag X] [--no-nll]
Runs land in ~/Projects/megaloman-ptv0/workspace/pixel_model/runs/denoise/NAME.
Base model/optimizer = loop-21 best (plain CE, d192 x6, bs 8x4, lr 3e-4 cosine, 3000 steps, final weights).
"""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).parent))
from loop_score import LOCAL_MANIFEST, SEED, histogram, nn_distances, structure  # noqa: E402
from megaloman.pixel_model.config import ModelConfig, TrainConfig  # noqa: E402
from megaloman.pixel_model.data import ApprovedDataset, BalancedBatchStream  # noqa: E402
from megaloman.pixel_model.denoise import (DenoiseSampleConfig, MaskLevelWrapper, NoiseConfig, corrupt,  # noqa: E402
                                           denoise_loss, denoise_sample, sample_t)
from megaloman.pixel_model.evaluate import conditional_report, novelty_report, summarize  # noqa: E402
from megaloman.pixel_model.model import PixelTransformer  # noqa: E402
from megaloman.pixel_model.render import contact_sheet  # noqa: E402
from megaloman.pixel_model.train import lr_multiplier  # noqa: E402

RUNS = Path.home() / "Projects/megaloman-ptv0/workspace/pixel_model/runs/denoise"
SIZES = [16, 24, 32]


def load(run, device):
    meta = json.loads((run / "config.json").read_text())
    model = PixelTransformer(ModelConfig(**meta["model"]), SIZES).to(device)
    model.load_state_dict(torch.load(run / "model.pt", map_location=device))
    return model.eval(), NoiseConfig(**meta["noise"])


def train(a):
    run = RUNS / a.name
    run.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    noise = NoiseConfig(mask_share=a.mask_share, uniform=not a.no_uniform, tile_shift=not a.no_tile,
                        outline_break=not a.no_outline, blur_patch=not a.no_blur,
                        corrupted_weight=a.corrupted_weight)
    mcfg = ModelConfig(d_model=192, layers=6, heads=6, dropout=0.1, attention_dropout=0.0, noise_conditioning=True)
    tcfg = TrainConfig(seed=a.seed, max_steps=a.steps, warmup_steps=100, lr=3e-4, min_lr=1e-5)
    ds = ApprovedDataset(LOCAL_MANIFEST, SIZES, True)
    model = PixelTransformer(mcfg, SIZES).to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_multiplier(s, tcfg))
    stream = BalancedBatchStream(ds, 8, a.seed + 1, bucket_sampling="sqrt_count", hshift=1)
    gen = torch.Generator().manual_seed(a.seed + 2)
    (run / "config.json").write_text(json.dumps({"model": asdict(mcfg), "noise": asdict(noise), "seed": a.seed,
                                                  "steps": a.steps}, indent=2))
    log = (run / "metrics.jsonl").open("w")
    start = time.time()
    for step in range(1, a.steps + 1):
        total = 0.0
        for _ in range(4):
            x0 = stream.next()
            t = sample_t(len(x0), noise, gen)
            xt, hit = corrupt(x0, t, noise, gen)
            x0, xt, hit = x0.to(a.device), xt.to(a.device), hit.to(a.device)
            with torch.autocast(a.device, dtype=torch.bfloat16):
                logits = torch.cat([model(xt[i:i + 1], noise_level=float(t[i])) for i in range(len(xt))]) \
                    if a.per_example_t else model(xt, noise_level=t.to(a.device))
            loss = denoise_loss(logits, x0, hit, noise)
            (loss / 4).backward()
            total += loss.item() / 4
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if step % 50 == 0 or step == 1:
            row = {"step": step, "loss": total, "seconds": time.time() - start}
            log.write(json.dumps(row) + "\n"); log.flush()
            print(json.dumps(row), flush=True)
    torch.save(model.state_dict(), run / "model.pt")


def score(a):
    run = RUNS / a.name
    model, noise = load(run, a.device)
    ds = ApprovedDataset(LOCAL_MANIFEST, SIZES, True)
    scfg = DenoiseSampleConfig(steps=a.steps, temperature=a.temperature, renoise=a.renoise,
                               ranking_temperature=a.ranking_temperature)
    out = run / ("score" + (f"-{a.tag}" if a.tag else ""))
    out.mkdir(exist_ok=True)
    report = {"run": a.name, "sample_config": asdict(scfg)}
    if not a.no_nll:
        s = summarize(conditional_report(MaskLevelWrapper(model, a.nll_scale or 1.0), ds, split="val", seed=1729 + 100))
        report["val_balanced_nll"], report["val_nll_50"] = s["balanced_nll"], s["by_ratio"]["0.5"]["masked_conditional_nll"]
    sizes, rows = {}, []
    for size in SIZES:
        grids = torch.cat([denoise_sample(model, size=size, count=1, seed=a.seed + i, cfg=scfg, noise=noise)
                           for i in range(a.count)])
        train = torch.stack([ds.grids[i] for i in ds.indices("train", size)])
        hs, ht = histogram(grids), histogram(train)
        l1 = (hs - ht).abs().sum().item()
        fg = abs(hs[0] - ht[0]).item()
        nov = novelty_report(grids, train)
        sizes[str(size)] = {"role_l1": l1, "fg_gap": fg, "score": l1 + fg, "sample_hist": hs.tolist(),
                            "exact_copy_rate": nov["exact_training_copy_rate"],
                            "nn_mean": nn_distances(grids, train).mean().item(), **structure(grids, train)}
        contact_sheet(grids, out / f"samples-{size}.png", scale=4)
        rows.append(grids[:4])
    report["sizes"] = sizes
    report["sample_score"] = sum(s["score"] for s in sizes.values()) / 3
    for k in ("role_l1", "pattern_l1", "blobs", "outline", "nn_mean"):
        report[k] = sum(s[k] for s in sizes.values()) / 3
    report["copy_rate"] = max(s["exact_copy_rate"] for s in sizes.values())
    torch.save(rows, out / "grid_rows.pt")
    contact_sheet([g for r in rows for g in r], out / "grid.png", scale=8, columns=4)
    (out / "score.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "sizes"}))
    for size, s in sizes.items():
        print(size, {k: round(v, 3) for k, v in s.items() if isinstance(v, float)})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "score"])
    p.add_argument("name")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--steps", type=int)
    p.add_argument("--mask-share", type=float, default=0.5)
    p.add_argument("--corrupted-weight", type=float, default=2.0)
    for f in ("no-uniform", "no-tile", "no-outline", "no-blur", "no-nll", "per-example-t"):
        p.add_argument("--" + f, action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--ranking-temperature", type=float, default=1.0)
    p.add_argument("--renoise", default="random")
    p.add_argument("--count", type=int, default=64)
    p.add_argument("--tag", default="")
    p.add_argument("--nll-scale", type=float, help="t = MASK fraction * this (default 1; 1/mask_share measured worse)")
    a = p.parse_args()
    if a.mode == "train":
        a.steps = a.steps or 3000
        train(a)
    else:
        a.steps = a.steps or 32
        score(a)


if __name__ == "__main__":
    main()
