"""Consistent categorical exports for model outputs and authentic controls."""
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .palette import DISPLAY_PALETTE, indexed_image, write_indexed


def contact_sheet(grids, path, *, scale=6, columns=8, palette=DISPLAY_PALETTE):
    arrays = [g.detach().cpu().numpy() if isinstance(g, torch.Tensor) else np.asarray(g) for g in grids]
    if not arrays:
        raise ValueError("cannot render an empty sheet")
    cell = max(max(g.shape) for g in arrays) * scale + 8
    columns = min(columns, len(arrays))
    sheet = Image.new("RGB", (columns * cell, math.ceil(len(arrays) / columns) * cell), (90, 90, 90))
    for i, grid in enumerate(arrays):
        im = indexed_image(grid, palette).convert("RGBA")
        # Identical checkerboard for generated and authentic transparent pixels.
        yy, xx = np.mgrid[:grid.shape[0], :grid.shape[1]]
        gray = np.where((xx + yy) % 2 == 0, 180, 210).astype(np.uint8)
        background = Image.fromarray(np.stack((gray, gray, gray, np.full_like(gray, 255)), -1))
        im = Image.alpha_composite(background, im).convert("RGB").resize(
            (grid.shape[1] * scale, grid.shape[0] * scale), Image.Resampling.NEAREST)
        sheet.paste(im, ((i % columns) * cell + 4, (i // columns) * cell + 4))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return path


def export_samples(grids, output, *, metadata=None, trace=None, palette=DISPLAY_PALETTE):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    grids = grids.detach().cpu()
    for i, grid in enumerate(grids):
        array = grid.numpy()
        write_indexed(output / f"{i:04d}.png", array, palette)
        indexed_image(array, palette).resize((grid.shape[1] * 12, grid.shape[0] * 12),
                                             Image.Resampling.NEAREST).save(output / f"{i:04d}-large.png", transparency=0)
    (output / "pixels.json").write_text(json.dumps({"metadata": metadata or {}, "pixels": grids.tolist()}, indent=2) + "\n")
    contact_sheet(grids, output / "sheet.png", palette=palette)
    if trace is not None:
        # JSON retains MASK=4 in unfinished states, rather than rendering it as a color.
        (output / "trace.json").write_text(json.dumps([t.tolist() for t in trace]) + "\n")
    return output
