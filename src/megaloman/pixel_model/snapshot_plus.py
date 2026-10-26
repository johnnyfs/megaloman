"""Frozen "plus" snapshot: a base snapshot's records unchanged (same val/test), plus every
other approved sprites-4tone character frame (all games and packs) added as train-only.

Same v2 preprocessing (tight crop, ceil-8 bottom-centre canvas, exact + mirror dedup against
the base and each other). Projectiles, effects, items and text are excluded. Every record
gets a `labels` field {game, category, view, identity} from the shared vocabulary
(data.record_labels): category singular across games, view from the animation name
(walk-down/-up -> overhead-down/-up, walk-side/left/right -> side, else side, flagged
`view_guessed`), identity game-scoped for non-Mega-Man packs so unknown-NNN ids stay distinct.
Approvals and review queues are only read.

usage: python -m megaloman.pixel_model.snapshot_plus BASE_MANIFEST OUT_DIR
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np

from .data import record_labels, sha256
from .palette import write_indexed
from .snapshot_megaman import DECISIONS, DROP, REVIEW, character, crop, place, read_ids, bbox

CATEGORIES = {"robot-masters": "robot-master", "enemies": "enemy", "bosses": "boss", "npcs": "npc",
              "player": "player"}
PREPROCESSING = "tight-crop+ceil8-bottom-centre@v2"


def view_of(animation):
    a = animation.lower()
    if "down" in a or "front" in a:
        return "overhead-down", False
    if a.endswith("-up") or "walk-up" in a or "back" in a:
        return "overhead-up", False
    if any(w in a for w in ("side", "left", "right")):
        return "side", False
    return "side", True


def approved(decisions):
    chosen = {}
    for line in decisions.read_text().splitlines():
        d = json.loads(line)
        if d.get("verdict") == "approved":
            chosen[(d["item"], d.get("content_hash"))] = d.get("variant")
            chosen[d["item"]] = d.get("variant")
    for key, s in sorted(json.loads((REVIEW / "state.json").read_text()).items()):
        if s.get("status") != "approved":
            continue
        parts = key.split("/")
        off = 1 if parts[0] == "_derived" else 0
        if len(parts) < off + 5:
            continue
        franchise, game, category, name, animation = parts[off:off + 5]
        if category not in CATEGORIES or DROP.search(parts[-1]) or DROP.search(name):
            continue
        card = json.loads(Path(s["approved_card"]).read_text())
        variants = card["meta"]["variants"]
        vid = chosen.get((key, card["content_hash"])) or chosen.get(key)
        if vid not in variants:
            vid = next(iter(variants))
        yield key, franchise, game, category, name, animation, variants[vid]["frames"]


def build(base_manifest, out, decisions=DECISIONS):
    base_manifest, out = Path(base_manifest), Path(out)
    if (out / "manifest.json").exists():
        raise ValueError(f"{out} already holds a snapshot; snapshots are frozen")
    base = json.loads(base_manifest.read_text())
    base_keys = {r["animation"] for r in base["records"]}
    held_out = {r["identity"] for r in base["records"] if r["split"] != "train"}
    seen, records, stats = set(), [], collections.Counter()
    for r in base["records"]:
        ids = read_ids(base_manifest.parent / r["path"])
        c = crop(ids)
        seen.add((c.shape, c.tobytes()))
        seen.add((c.shape, np.ascontiguousarray(c[:, ::-1]).tobytes()))
        name = f"frames/{Path(r['path']).name}"
        (out / "frames").mkdir(parents=True, exist_ok=True)
        (out / name).write_bytes((base_manifest.parent / r["path"]).read_bytes())
        row = {**r, "path": name}
        row["labels"] = record_labels(r)
        records.append(row)
    known = {r["identity"] for r in base["records"]}
    for key, franchise, game, category, name, animation, paths in approved(decisions):
        if key in base_keys:
            continue
        if franchise == "megaman":
            identity = character(key, known)
        else:
            identity = f"{franchise}/{game}/{name}"
        if identity in held_out:
            stats["held_out_identity_skipped"] += 1
            continue
        view, guessed = view_of(animation)
        for i, path in enumerate(paths):
            ids = read_ids(path)
            c = crop(ids)
            if c is None:
                stats["empty"] += 1
                continue
            k1, k2 = (c.shape, c.tobytes()), (c.shape, np.ascontiguousarray(c[:, ::-1]).tobytes())
            if k1 in seen or k2 in seen:
                stats["duplicate"] += 1
                continue
            seen.add(k1)
            grid, offset = place(c)
            if grid is None:
                stats["too_large"] += 1
                continue
            fname = f"frames/{key.replace('/', '__')}__{i:03d}.png"
            write_indexed(out / fname, grid)
            stats["added"] += 1
            stats[f"added/{franchise}"] += 1
            records.append({"id": f"{key}#{i}", "path": fname, "sha256": sha256(out / fname),
                            "size": int(grid.shape[0]), "identity": identity, "animation": key, "split": "train",
                            "game": game, "category": category, "crop": list(bbox(ids)),
                            "orig_size": [int(ids.shape[1]), int(ids.shape[0])], "placement": list(offset),
                            "anchor": "bottom-centre", "preprocessing": PREPROCESSING, "extra": True,
                            "labels": {"game": game, "category": CATEGORIES[category], "view": view,
                                       "identity": identity, "view_guessed": guessed}})
    approval = hashlib.sha256(base_manifest.read_bytes() + (REVIEW / "state.json").read_bytes()
                              + decisions.read_bytes()).hexdigest()[:16]
    manifest = {**{k: v for k, v in base.items() if k != "records"},
                "approval_id": f"{base['approval_id']}+plus:{approval}",
                "source": {**base.get("source", {}), "plus": "all other approved character frames, train-only",
                           "base_manifest": str(base_manifest)}, "records": records}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    summary = {"stats": dict(stats), "frames": len(records),
               "by_size_split": dict(collections.Counter(f"{r['split']}/{r['size']}" for r in records)),
               "guessed_views": sorted({r["animation"] for r in records if r.get("labels", {}).get("view_guessed")
                                        and r.get("extra")})}
    (out / "build.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("out")
    a = ap.parse_args()
    print(json.dumps(build(a.base, a.out), indent=2))
