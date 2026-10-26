"""Derived shape snapshot: 3 roles per pixel from a 4-tone snapshot (same frames, buckets, splits).

TRANSPARENT=0, OUTLINE=1 (INK pixel 4-adjacent to transparent or the canvas edge),
INTERIOR=2 (every other non-transparent pixel). Frames whose silhouette border (non-transparent
pixels 4-adjacent to transparent/edge) is under `min_ink` INK are dropped and counted. Role 3 is
unused, so the unchanged 4-class model/sampler trains on it (class 3 never appears).
Frames that collapse to an exact duplicate of a frame in another split are dropped (leak guard).

usage: python -m megaloman.pixel_model.snapshot_outline BASE_MANIFEST OUT_DIR [--min-ink .8]
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

from .data import sha256
from .palette import INK, TRANSPARENT, write_indexed
from .snapshot_megaman import read_ids

OUTLINE, INTERIOR = 1, 2
OUTLINE_PALETTE = ((0, 0, 0), (20, 20, 28), (200, 190, 160), (255, 0, 255))


def border(ids):
    fg = ids != TRANSPARENT
    padded = np.pad(fg, 1, constant_values=False)
    touches = ~padded[:-2, 1:-1] | ~padded[2:, 1:-1] | ~padded[1:-1, :-2] | ~padded[1:-1, 2:]
    return fg & touches


def to_shape(ids):
    edge = border(ids)
    out = np.where(ids == TRANSPARENT, TRANSPARENT, INTERIOR)
    out[edge & (ids == INK)] = OUTLINE
    ink_fraction = (ids[edge] == INK).mean() if edge.any() else 0.0
    return out, float(ink_fraction)


def build(base_manifest, out, min_ink=.8):
    base_manifest, out = Path(base_manifest), Path(out)
    if (out / "manifest.json").exists():
        raise ValueError(f"{out} already holds a snapshot; snapshots are frozen")
    base = json.loads(base_manifest.read_text())
    stats, records, owner = collections.Counter(), [], {}
    fractions = []
    for r in base["records"]:
        shape, fraction = to_shape(read_ids(base_manifest.parent / r["path"]))
        fractions.append(fraction)
        key = shape.tobytes() + bytes(shape.shape)
        if fraction < min_ink:
            stats[f"dropped_low_ink/{r['split']}"] += 1
            continue
        if key in owner and owner[key] != r["split"]:
            stats["dropped_cross_split_duplicate"] += 1
            continue
        owner[key] = r["split"]
        name = f"frames/{Path(r['path']).name}"
        write_indexed(out / name, shape, OUTLINE_PALETTE)
        records.append({**r, "path": name, "sha256": sha256(out / name), "border_ink_fraction": fraction})
        stats[f"kept/{r['split']}"] += 1
    manifest = {**{k: v for k, v in base.items() if k != "records"},
                "approval_id": base["approval_id"] + "+outline3",
                "source": {**base.get("source", {}), "derived": "outline3: transparent/outline/interior",
                           "min_border_ink": min_ink, "base_manifest": str(base_manifest)}, "records": records}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    summary = {"stats": dict(stats), "frames": len(records), "base_frames": len(base["records"]),
               "border_ink_fraction_quartiles": np.quantile(fractions, [.1, .25, .5, .75]).tolist(),
               "by_size_split": dict(collections.Counter(f"{r['split']}/{r['size']}" for r in records))}
    (out / "build.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("out")
    ap.add_argument("--min-ink", type=float, default=.8)
    a = ap.parse_args()
    print(json.dumps(build(a.base, a.out, a.min_ink), indent=2))
