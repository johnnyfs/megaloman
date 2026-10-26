"""Shape-then-tones chain: sample shapes from the 3-role outline model, then infill the tones
with a 4-class model given TRANSPARENT and OUTLINE (as INK) as known, interior unknown and
kept foreground. Writes OUT/grid.png: per size, 4 shapes then their 4 infilled results.

usage: PYTHONPATH=src python scripts/outline_chain.py OUTLINE_CKPT TONE_CKPT OUT [--steps 64]
"""
import argparse
import json
from pathlib import Path

import torch

from megaloman.pixel_model.checkpoint import load_model
from megaloman.pixel_model.palette import INK, TRANSPARENT
from megaloman.pixel_model.render import contact_sheet
from megaloman.pixel_model.sample import infill, sample
from megaloman.pixel_model.snapshot_outline import OUTLINE, OUTLINE_PALETTE


def main():
    p = argparse.ArgumentParser()
    p.add_argument("outline")
    p.add_argument("tone")
    p.add_argument("out")
    p.add_argument("--steps", type=int)
    p.add_argument("--seeds", type=int, default=4)
    a = p.parse_args()
    shape_model, shape_cfg, _ = load_model(a.outline, "cuda")
    tone_model, tone_cfg, _ = load_model(a.tone, "cuda")
    if a.steps:
        shape_cfg.sample.steps = tone_cfg.sample.steps = a.steps
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    shapes, tones, stats = [], [], {}
    for size in (16, 24, 32):
        s_row, t_row = [], []
        for i in range(a.seeds):
            shape = sample(shape_model, size=size, seed=1729 + i, cfg=shape_cfg.sample)[0].cpu()
            shape = shape.clamp(max=2)  # role 3 is unused in the outline data
            known = (shape == TRANSPARENT) | (shape == OUTLINE)
            canvas = torch.where(shape == OUTLINE, INK, TRANSPARENT)
            filled = infill(tone_model, canvas, known, seed=1729 + i, cfg=tone_cfg.sample, keep_foreground=True).cpu()
            s_row.append(shape)
            t_row.append(filled)
        shapes.append(s_row)
        tones.append(t_row)
        stats[size] = {"shape_hist": torch.bincount(torch.stack(s_row).flatten(), minlength=3).float()
                       .div(sum(g.numel() for g in s_row)).tolist()}
    contact_sheet([g for row in shapes for g in row], out / "shapes.png", scale=8, columns=a.seeds,
                  palette=OUTLINE_PALETTE)
    contact_sheet([g for row in tones for g in row], out / "infilled.png", scale=8, columns=a.seeds)
    from PIL import Image
    left, right = Image.open(out / "shapes.png"), Image.open(out / "infilled.png")
    grid = Image.new("RGB", (left.width + right.width + 24, max(left.height, right.height)), (40, 40, 48))
    grid.paste(left, (0, 0))
    grid.paste(right, (left.width + 24, 0))
    grid.save(out / "grid.png")
    (out / "chain.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
