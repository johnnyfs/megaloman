"""Freeze a schema-v1 snapshot from the sprites-4tone review queue.

Cohort: approved Mega Man robot-masters and enemies (projectiles/effects dropped,
same filter as the counting script). Frames are deduplicated by exact 4-tone pixels
of the tight crop, with horizontal mirrors counted as duplicates. Preprocessing v2:
every frame is tight-cropped, then padded bottom-centre onto the square canvas whose
side is max(h, w) rounded up to 8 (minimum 16, maximum 64); larger frames are dropped.
Crop box, original size and placement offset are recorded per record. Splits are by
canonical character (all games, animations and _derived variants of one name together).

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
from .config import ALLOWED_SIZES
from .palette import ENCODING, NUM_COLORS, write_indexed

REVIEW = Path(os.path.expanduser("~/datasets/sprites/_review/sprites-4tone"))
DECISIONS = Path(os.path.expanduser("~/Projects/cockpit/reviews/sprites-4tone.decisions.jsonl"))
CATEGORIES = ("robot-masters", "enemies")
SIZES = ALLOWED_SIZES
VARIANT_SUFFIXES = ("plus", "alt", "edit", "clean", "b", "c")
PREPROCESSING = "tight-crop+ceil8-bottom-centre@v2"
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


def bbox(ids):
    ys, xs = np.nonzero(ids)
    if not len(ys):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def crop(ids):
    box = bbox(ids)
    return None if box is None else ids[box[1]:box[3], box[0]:box[2]]


def bucket(h, w):
    """Smallest allowed side >= max(h, w): round up to 8, minimum 16; None if over 64."""
    return next((s for s in SIZES if s >= max(h, w)), None)


def place(c):
    """Tight crop c onto its bucket canvas, bottom aligned, horizontally centred (left-biased)."""
    h, w = c.shape
    size = bucket(h, w)
    if size is None:
        return None, None
    out = np.zeros((size, size), dtype=np.int64)
    x, y = (size - w) // 2, size - h
    out[y:, x:x + w] = c
    return out, (x, y)


def _norm(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def character(key, known=()):
    """Canonical character: the name folder, normalised (case, - and _ dropped), across games.

    A variant folder whose leading hyphen-separated words name another known character
    joins it (polarwoman-whiteskates -> polarwoman).
    """
    parts = key.split("/")
    off = 2 if parts[0] == "_derived" else 1
    name = parts[off + 2]
    words = name.split("-")
    for n in range(len(words) - 1, 0, -1):
        if _norm("-".join(words[:n])) in known:
            return _norm("-".join(words[:n]))
    flat = _norm(name)
    for k in known:  # beetleladyplus -> beetlelady
        if len(k) >= 6 and flat != k and flat.startswith(k) and flat[len(k):] in VARIANT_SUFFIXES:
            return k
    return flat


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


def build(out, decisions=DECISIONS, seed=1729, val=0.10, test=0.10, require=(16, 24, 32), min_eval=8):
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
            grid, offset = place(c)
            if grid is None:
                stats["too_large"] += 1
                continue
            frames.append(dict(key=key, game=game, category=category, index=i, grid=grid,
                               crop=bbox(ids), orig=[int(ids.shape[1]), int(ids.shape[0])], offset=offset,
                               character=None))
    known = {character(f["key"]) for f in frames}
    for f in frames:
        f["character"] = character(f["key"], known)
    # Split by canonical character, ~80/10/10 by frames; retry seeds until every required
    # size has train/val/test frames.
    characters = sorted({f["character"] for f in frames})
    animations = sorted({f["key"] for f in frames})
    per = collections.Counter(f["character"] for f in frames)
    total = sum(per.values())
    for attempt in range(1000):
        rng = random.Random(seed + attempt)
        order = characters[:]
        rng.shuffle(order)
        split, nv, nt = {}, 0, 0
        for ch in order:
            if nv < val * total:
                split[ch], nv = "val", nv + per[ch]
            elif nt < test * total:
                split[ch], nt = "test", nt + per[ch]
        have = collections.Counter((f["grid"].shape[0], split.get(f["character"], "train")) for f in frames)
        if all(have[z, "train"] and have[z, "val"] >= min_eval and have[z, "test"] >= min_eval for z in require):
            break
    else:
        raise RuntimeError("could not find a split covering every required size")
    records = []
    for f in frames:
        name = f"frames/{f['key'].replace('/', '__')}__{f['index']:03d}.png"
        write_indexed(out / name, f["grid"])
        records.append({"id": f"{f['key']}#{f['index']}", "path": name, "sha256": sha256(out / name),
                        "size": int(f["grid"].shape[0]), "identity": f["character"], "animation": f["key"],
                        "split": split.get(f["character"], "train"), "game": f["game"], "category": f["category"],
                        "crop": list(f["crop"]), "orig_size": f["orig"], "placement": list(f["offset"]),
                        "anchor": "bottom-centre", "preprocessing": PREPROCESSING})
    approval = hashlib.sha256((REVIEW / "state.json").read_bytes() + decisions.read_bytes()).hexdigest()[:16]
    manifest = {"schema_version": 1, "approval_id": f"sprites-4tone:{approval}", "synthetic": False,
                "encoding": ENCODING, "source": {"queue": "sprites-4tone", "categories": list(CATEGORIES),
                "dedup": "exact crop + horizontal mirror", "placement": "tight crop, ceil-8 square, bottom aligned, centred",
                "preprocessing": PREPROCESSING, "split": "by canonical character", "split_seed": seed + attempt},
                "records": records}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    counts = collections.Counter(f"{r['split']}/{r['size']}" for r in records)
    by_size = collections.Counter(r["size"] for r in records)
    summary = {"stats": dict(stats), "frames": len(records), "animations": len(animations),
               "characters": len(characters),
               "splits": dict(collections.Counter(r["split"] for r in records)),
               "split_characters": dict(collections.Counter(split.get(c, "train") for c in characters)),
               "split_animations": dict(collections.Counter(split.get(character(a, known), "train") for a in animations)),
               "by_size": {str(k): by_size[k] for k in sorted(by_size)},
               "counts": dict(sorted(counts.items())),
               "characters_by_split": {sp: sorted(c for c in characters if split.get(c, "train") == sp)
                                       for sp in ("val", "test")}}
    (out / "build.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--decisions", default=str(DECISIONS))
    a = ap.parse_args()
    print(json.dumps(build(a.out, Path(a.decisions)), indent=2))
