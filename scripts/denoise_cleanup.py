"""Cleanup demo: damage val sprites (structured + per-pixel noise, no MASK), then denoise from t0.

usage: PYTHONPATH=src python scripts/denoise_cleanup.py RUN_NAME [--t0 0.25] [--steps 4] [--temperature 0] [--renoise none]
Writes runs/denoise/RUN/cleanup/{cleanup.png, snipe.png, cleanup.json}.
Rows of cleanup.png: damaged | cleaned | original. snipe.png: rough | cleaned (no ground truth).
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).parent))
from denoise_run import RUNS, SIZES, load  # noqa: E402
from loop_score import LOCAL_MANIFEST  # noqa: E402
from megaloman.pixel_model.data import ApprovedDataset  # noqa: E402
from megaloman.pixel_model.denoise import DenoiseSampleConfig, corrupt, denoise_sample  # noqa: E402
from megaloman.pixel_model.render import contact_sheet  # noqa: E402

SNIPE = Path.home() / "Projects/megaloman-ptv0/workspace/experiments/regions/snipe/v3tight/render"


def snipe_roles(path):
    """RGBA 4-5 colour render -> roles by luminance: alpha 0 -> 0, darkest -> INK, then tones."""
    a = np.array(Image.open(path).convert("RGBA")).astype(float)
    out = np.zeros(a.shape[:2], dtype=np.int64)
    lum = a[..., :3] @ [.299, .587, .114]
    opaque = a[..., 3] > 0
    out[opaque & (lum < 20)] = 1
    out[opaque & (lum >= 20) & (lum < 80)] = 2
    out[opaque & (lum >= 80)] = 3
    return torch.from_numpy(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("name")
    p.add_argument("--t0", type=float, default=0.25)
    p.add_argument("--damage", type=float, default=0.25)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--renoise", default="none")
    p.add_argument("--damage-seed", type=int, default=7)
    a = p.parse_args()
    model, noise = load(RUNS / a.name, a.device)
    damage_noise = replace(noise, mask_share=0.0)  # no MASK tokens: a rough sprite, not a hole
    scfg = DenoiseSampleConfig(steps=a.steps, temperature=a.temperature, renoise=a.renoise, ranking_temperature=0.0)
    ds = ApprovedDataset(LOCAL_MANIFEST, SIZES, True)
    gen = torch.Generator().manual_seed(a.damage_seed)
    out = RUNS / a.name / "cleanup"
    out.mkdir(exist_ok=True)
    picks = []
    for size, n in ((32, 4), (24, 2), (16, 2)):
        idx = ds.indices("val", size)
        picks += [idx[i] for i in torch.randperm(len(idx), generator=gen)[:n].tolist()]
    damaged, cleaned, original, stats = [], [], [], []
    for j, i in enumerate(picks):
        x0 = ds.grids[i][None]
        xd, _ = corrupt(x0, torch.tensor([a.damage]), damage_noise, gen)
        xc = denoise_sample(model, start=xd, t0=a.t0, cfg=scfg, noise=noise, seed=j)
        fg = (x0 != 0) | (xd != 0)
        stats.append({"id": ds.records[i]["id"], "size": x0.shape[-1],
                      "damaged_err": (xd != x0).float().mean().item(), "cleaned_err": (xc != x0).float().mean().item(),
                      "damaged_fg_err": (xd != x0)[fg].float().mean().item(),
                      "cleaned_fg_err": (xc != x0)[fg].float().mean().item()})
        damaged.append(xd[0]); cleaned.append(xc[0]); original.append(x0[0])
    pad = lambda g: torch.nn.functional.pad(g, (0, 32 - g.shape[1], 0, 32 - g.shape[0]))
    contact_sheet([pad(g) for row in (damaged, cleaned, original) for g in row], out / "cleanup.png",
                  scale=6, columns=len(picks))
    snipes = [snipe_roles(SNIPE / "render_32_free.png"), snipe_roles(SNIPE / "render_16_free.png")]
    rows = [[], []]
    for k, g in enumerate(snipes):
        for t0 in (0.15, 0.3):
            c = denoise_sample(model, start=g[None], t0=t0, cfg=scfg, noise=noise, seed=100 + k)[0]
            rows[0].append(pad(g)); rows[1].append(pad(c))
    contact_sheet(rows[0] + rows[1], out / "snipe.png", scale=6, columns=4)
    summary = {"t0": a.t0, "damage": a.damage, "steps": a.steps, "temperature": a.temperature, "frames": stats,
               "mean_damaged_err": sum(s["damaged_err"] for s in stats) / len(stats),
               "mean_cleaned_err": sum(s["cleaned_err"] for s in stats) / len(stats),
               "mean_damaged_fg_err": sum(s["damaged_fg_err"] for s in stats) / len(stats),
               "mean_cleaned_fg_err": sum(s["cleaned_fg_err"] for s in stats) / len(stats)}
    (out / "cleanup.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "frames"}))


if __name__ == "__main__":
    main()
