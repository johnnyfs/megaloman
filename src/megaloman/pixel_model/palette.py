"""The input is already approved categorical art; no recoloring or quantization."""
from pathlib import Path

import numpy as np
from PIL import Image

NUM_COLORS = 4
TRANSPARENT, INK, TONE_1, TONE_2 = range(NUM_COLORS)
MASK, PAD = NUM_COLORS, NUM_COLORS + 1
DISPLAY_PALETTE = ((0, 0, 0), (30, 28, 40), (94, 108, 162), (240, 225, 180))
# Future depth conditioning belongs in metadata/position encoding, never in pixel IDs.
ENCODING = {"version": "megaloman-indexed-v1", "num_colors": NUM_COLORS,
            "roles": ["TRANSPARENT", "INK", "TONE_1", "TONE_2"],
            "transparent_index": 0, "geometry": "native-approved", "palette_mapping": "approved-role-ids"}


def read_indexed(path, size=None):
    with Image.open(path) as im:
        if im.format != "PNG" or im.mode != "P":
            raise ValueError(f"{path}: requires indexed (P-mode) PNG, with approved role IDs")
        if size is not None and im.size != (size, size):
            raise ValueError(f"{path}: expected native {size}x{size}, got {im.size}")
        ids = np.array(im, dtype=np.int64)
        if ids.max() >= NUM_COLORS:
            raise ValueError(f"{path}: pixel indices must be 0..3")
        # Validate alpha per ID, including absent roles; allow unused opaque entries.
        alpha = im.info.get("transparency")
        if isinstance(alpha, int):
            valid_alpha = alpha == TRANSPARENT
        elif isinstance(alpha, bytes):
            valid_alpha = bool(alpha) and alpha[0] == 0 and all(a == 255 for a in alpha[1:])
        else:
            valid_alpha = False
        if not valid_alpha:
            raise ValueError(f"{path}: index 0 must be transparent and other indices fully opaque")
        return ids


def indexed_image(ids, palette=DISPLAY_PALETTE):
    ids = np.asarray(ids)
    if ids.ndim != 2 or not np.issubdtype(ids.dtype, np.integer) or np.any((ids < 0) | (ids >= NUM_COLORS)):
        raise ValueError("expected a 2D integer array of classes 0..3")
    if len(palette) != NUM_COLORS or any(len(c) != 3 for c in palette):
        raise ValueError("display palette must contain four RGB triples")
    im = Image.fromarray(ids.astype(np.uint8)).convert("P")
    im.putpalette([v for c in palette for v in c] + [0] * (768 - NUM_COLORS * 3))
    im.info["transparency"] = 0
    return im


def write_indexed(path, ids, palette=DISPLAY_PALETTE):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    indexed_image(ids, palette).save(path, transparency=0)
