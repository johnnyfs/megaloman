"""Freeze a schema-v1 snapshot from the sprites-4tone review queue.

Cohort: approved Mega Man robot-masters and enemies (projectiles/effects dropped,
same filter as the counting script). Frames are deduplicated by exact 4-tone pixels
of the tight crop, with horizontal mirrors counted as duplicates. Each frame is
placed on the smallest square canvas in SIZES (horizontally centred, bottom aligned,
as the cards are grounded); larger frames are dropped. Splits are by animation.

usage: python -m megaloman.pixel_model.snapshot_megaman OUT_DIR [--decisions PATH]
"""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import random
import re

import numpy as np
from PIL import Image

from .data import sha256
from .palette import ENCODING, NUM_COLORS, write_indexed

REVIEW = Path(os.path.expanduser("~/datasets/sprites/_review/sprites-4tone"))
DECISIONS = Path(os.path.expanduser("~/Projects/cockpit/reviews/sprites-4tone.decisions.jsonl"))
CATEGORIES = ("robot-masters", "enemies")
SIZES = (16, 24, 32, 64)
DROP = re.compile(r"explo|(?<!no)flash$|(?<!no)flash-|spark$|^spark-|bullet|projectile|fireball|^shot$"
                  r"|smoke|debris|particle|splash$|impact", re.I)


def read_ids(path):
    with Image.open(path) as im:
        if im.mode != "P":
            raise ValueError(f"{path}: expected canonical indexed PNG")
        ids = np.array(im, dtype=np.int64)
    if ids.max() >= NUM_COLORS:
        raise ValueError(f"{path}: class ids must be 0..3")
    return ids


def crop(ids):
    ys, xs = np.nonzero(ids)
    if not len(ys):
        return None
    return ids[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def place(ids):
    """Square canvas, bottom aligned, horizontally centred; keeps the card's own padding when it fits."""
    h, w = ids.shape
    size = next((s for s in SIZES if s >= max(h, w)), None)
    if size is None:
        c = crop(ids)
        h, w = c.shape
        size = next((s for s in SIZES if s >= max(h, w)), None)
        if size is None:
            return None
        ids = c
    out = np.zeros((size, size), dtype=np.int64)
    x = (size - w) // 2
    out[size - h:, x:x + w] = ids
    return out


def approved_items(decisions):
    variant = {}
    for line in decisions.read_text().splitlines():
        d = json.loads(line)
        if d.get("verdict") == "approved":
            variant[(d["item"], d.get("content_hash"))] = d.get("variant")
            variant[d["item"]] = d.get("variant")
    state = json.loads((REVIEW / "state.json").read_text())
    for key, s in sorted(state.items()):
        if s.get("status") != "approved":
            continue
        parts = key.split("/")
        if parts[0] == "megaman":
            off = 1
        elif parts[:2] == ["_derived", "megaman"]:
            off = 2
        else:
            continue
        if parts[off + 1] not in CATEGORIES or DROP.search(parts[-1]):
            continue
        card = json.loads(Path(s["approved_card"]).read_text())
        variants = card["meta"]["variants"]
        vid = variant.get((key, card["content_hash"])) or variant.get(key)
        if vid not in variants:
            vid = next(iter(variants))
        yield key, parts[off], parts[off + 1], variants[vid]["frames"]


def build(out, decisions=DECISIONS, seed=1729, val=0.08, test=0.05):
    out = Path(out)
    if (out / "manifest.json").exists():
        raise ValueError(f"{out} already holds a snapshot; snapshots are frozen")
    seen, frames, stats = set(), [], collections.Counter()
    for key, game, category, paths in approved_items(decisions):
        stats["items"] += 1
        for i, path in enumerate(paths):
            ids = read_ids(path)
            c = crop(ids)
            if c is None:
                stats["empty"] += 1
                continue
            k1 = (c.shape, c.tobytes())
            k2 = (c.shape, np.ascontiguousarray(c[:, ::-1]).tobytes())
            if k1 in seen or k2 in seen:
                stats["duplicate"] += 1
                continue
            seen.add(k1)
            grid = place(ids)
            if grid is None:
                stats["too_large"] += 1
                continue
            frames.append(dict(key=key, game=game, category=category, index=i, grid=grid))
    # Split by animation; retry seeds until every size has val and test frames.
    animations = sorted({f["key"] for f in frames})
    for attempt in range(1000):
        rng = random.Random(seed + attempt)
        order = animations[:]
        rng.shuffle(order)
        nv, nt = round(len(order) * val), round(len(order) * test)
        split = {a: "val" for a in order[:nv]} | {a: "test" for a in order[nv:nv + nt]}
        have = {(f["grid"].shape[0], split.get(f["key"], "train")) for f in frames}
        if all((s, sp) in have for s in {f["grid"].shape[0] for f in frames} for sp in ("train", "val", "test")):
            break
    else:
        raise RuntimeError("could not find a split covering every size")
    records = []
    for f in frames:
        name = f"frames/{f['key'].replace('/', '__')}__{f['index']:03d}.png"
        write_indexed(out / name, f["grid"])
        records.append({"id": f"{f['key']}#{f['index']}", "path": name, "sha256": sha256(out / name),
                        "size": int(f["grid"].shape[0]), "identity": f["key"], "animation": f["key"],
                        "split": split.get(f["key"], "train"), "game": f["game"], "category": f["category"]})
    approval = hashlib.sha256((REVIEW / "state.json").read_bytes() + decisions.read_bytes()).hexdigest()[:16]
    manifest = {"schema_version": 1, "approval_id": f"sprites-4tone:{approval}", "synthetic": False,
                "encoding": ENCODING, "source": {"queue": "sprites-4tone", "categories": list(CATEGORIES),
                "dedup": "exact crop + horizontal mirror", "placement": "square, bottom aligned, centred",
                "split": "by animation", "split_seed": seed + attempt}, "records": records}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    counts = collections.Counter(f"{r['split']}/{r['size']}" for r in records)
    summary = {"stats": dict(stats), "frames": len(records), "animations": len(animations),
               "splits": dict(collections.Counter(r["split"] for r in records)),
               "split_animations": dict(collections.Counter(split.get(a, "train") for a in animations)),
               "counts": dict(sorted(counts.items()))}
    (out / "build.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--decisions", default=str(DECISIONS))
    a = ap.parse_args()
    print(json.dumps(build(a.out, Path(a.decisions)), indent=2))
