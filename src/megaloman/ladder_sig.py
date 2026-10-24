"""Ladder v3: preserve significance. As shapes shrink they lose significance; some parts must stay LARGER than their
proportional size to keep their character (the squid's side fins), others only need a FLOOR (the eyes: never less
than two separate rectangles).

  1. guidance (one cached VLM call, prompt in <out>/prompts/significance.md): per part a significance weight, its
     essential shape, mode (exaggerate | floor | none), max exaggeration, anchor, container and a minimum footprint
     per rung. Explicit user rules (OVERRIDES) are applied on top and recorded.
  2. deterministic, per rung s (source mode, built on ladder.py v1):
     - exaggerate: each source piece of the part is scaled about its anchor (centroid, or the centroid of where it
       touches its parent) by k(s) in the FULL-SIZE region map, painting over lower-significance parts (and the
       background only if allowed), then v1's coverage/priority assign shrinks it. k(s) is the larger of a schedule
       (1 at >= 64 px rising to max_exaggeration at 16 px) and the smallest k meeting the footprint/shape template,
       subject to: no adjacency v1 has at this rung is lost and no other part loses a piece. If the cap or the
       constraints stop it, the best admissible k is used and the shortfall recorded.
     - floor: after the shrink, any source piece below its minimum footprint is redrawn as its template (a w x h
       rectangle at its scaled centroid, inside its container, kept apart from its sibling pieces).
  3. same health measures as v1 (ladder.health) + footprint checks; pairwise judge v1 vs v3 at 32 and 16.

uv run python -m megaloman.ladder_sig [--no-judge] [--budget 1.5]
"""
from __future__ import annotations

import io
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from . import ladder as L
from . import models, store

VERSION = "v3"
MODEL = "openai/gpt-5.6-sol"           # the repo's judge / legend vision model
SAMPLES = 3                            # judge samples per presentation order
JUDGE_SIZES = (32, 16)
SCHEDULE_FROM = 64                     # exaggeration starts below this rung
K_STEP = 0.05
FRAME_MARGIN = 0.5                     # a scaled piece must stay this many rung px inside the canvas (keeps its tip)
FOOT_SIZES = ("64", "32", "24", "20", "16")
SHAPES = ("triangle", "pair_of_rects", "rect", "ellipse", "blob", "stroke")
# explicit rules from the user, applied on top of the VLM guidance (recorded in significance.json)
OVERRIDES = {
    "eyes": {"mode": "floor", "max_exaggeration": 1.0, "essential_shape": "pair_of_rects",
             "_why": "user: eyes are fine at proportional size; never enlarge, only floor at two separate >= 1x2 rects"},
}
ROOT = Path(__file__).resolve().parents[2]
REG = ROOT / "workspace/experiments/regions"
OUT = REG / "ladder/v3_significance"


# ------------------------------------------------------------------ guidance (VLM, cached)

def _png(im: Image.Image) -> bytes:
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def _legend_text(legend) -> str:
    return "\n".join(f"- {p['color']} {p['name']} ({p['role']}): {p['where']}" for p in legend["parts"])


def _v1_strip(src, legend, sizes=(64, 32, 16), h=256) -> Image.Image:
    pr = np.array([p["priority"] for p in legend["parts"]], float)
    n = len(legend["parts"])
    img = Image.new("RGB", (len(sizes) * (h + 16) + 16, h + 44), (90, 90, 96))
    d = ImageDraw.Draw(img)
    for k, s in enumerate(sizes):
        x = 16 + k * (h + 16)
        d.text((x, 4), f"{s}px", fill=(255, 255, 255), font=L._font(26))
        img.paste(L.colorize(L.assign(L.coverage(src, n, s), pr), legend).resize((h, h), Image.NEAREST), (x, 40))
    return img


def guidance(src, legend, concept: Image.Image, region_png: Path, ledger, prompt_dir: Path) -> dict:
    prompt = (prompt_dir / "significance.md").read_text().format(legend=_legend_text(legend))
    c = concept.convert("RGB")
    c.thumbnail((768, 768))
    r = Image.open(region_png).convert("RGB")
    r.thumbnail((768, 768))
    imgs = [_png(c), _png(r), _png(_v1_strip(src, legend))]
    body = {"model": MODEL, "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}] +
                          [{"type": "image_url", "image_url": {"url": models.data_uri(b)}} for b in imgs]}]}
    ck = store.key({"step": "ladder_sig.guidance", "model": MODEL, "prompt": prompt,
                    "images": [store.sha(b) for b in imgs]})
    resp, _, cost, cached = models.call(body, ledger, ck)
    m = re.search(r"\{.*\}", resp["choices"][0]["message"].get("content") or "", re.S)
    raw = json.loads(m.group(0)) if m else {"parse_error": (resp["choices"][0]["message"].get("content") or "")[:500]}
    return {"model": MODEL, "call": ck, "cost": cost, "cached": cached, "raw": raw}


def specs_from(g: dict, legend) -> list[dict]:
    """Normalise the VLM output into one spec per legend part (defaults: 'none'), then apply OVERRIDES."""
    by = {str(p.get("name")): p for p in (g["raw"].get("parts") or [])}
    out = []
    for p in legend["parts"]:
        v = dict(by.get(p["name"], {}))
        s = {"name": p["name"],
             "significance": float(np.clip(float(v.get("significance", 0.3) or 0.3), 0, 1)),
             "character": v.get("character"),
             "essential_shape": v.get("essential_shape") if v.get("essential_shape") in SHAPES else "blob",
             "mode": v.get("mode") if v.get("mode") in ("exaggerate", "floor", "none") else "none",
             "max_exaggeration": float(np.clip(float(v.get("max_exaggeration", 1.0) or 1.0), 1.0, 2.5)),
             "anchor": str(v.get("anchor") or "centroid"),
             "may_claim_background": bool(v.get("may_claim_background", False)),
             "container": v.get("container") if v.get("container") in [q["name"] for q in legend["parts"]] else None,
             "min_footprint": {}, "reason": v.get("reason"), "overrides": None}
        mf = v.get("min_footprint") or {}
        for k in FOOT_SIZES:
            f = mf.get(k) or {}
            s["min_footprint"][k] = {q: int(f.get(q, 0) or 0) for q in ("w", "h", "area")}
        if s["mode"] != "exaggerate":
            s["max_exaggeration"] = 1.0
        if p["name"] in OVERRIDES:
            o = OVERRIDES[p["name"]]
            s["overrides"] = {"from": {k: s[k] for k in o if not k.startswith("_")}, "why": o["_why"]}
            s.update({k: val for k, val in o.items() if not k.startswith("_")})
        out.append(s)
    return out


def footprint_at(spec, s: int) -> dict:
    """Minimum footprint at rung s: the guidance's value for that size; sizes in between use the next-smaller key."""
    if s > 64:
        return {"w": 0, "h": 0, "area": 0}
    keys = sorted(int(k) for k in FOOT_SIZES)
    k = max((x for x in keys if x <= s), default=keys[0])
    return spec["min_footprint"][str(k)]


def schedule(spec, s: int) -> float:
    t = float(np.clip(np.log2(SCHEDULE_FROM / s) / np.log2(SCHEDULE_FROM / 16), 0, 1))
    return 1 + (spec["max_exaggeration"] - 1) * t


# ------------------------------------------------------------------ source pieces and anchors

def pieces(src, i, min_frac=0.02) -> list[dict]:
    """Source components of part i (>= min_frac of the part), each with centroid and bbox."""
    m = src == i + 1
    lab, k = ndimage.label(m, L.EIGHT)
    if not k:
        return []
    areas = ndimage.sum(m, lab, range(1, k + 1))
    out = []
    for c in range(1, k + 1):
        if areas[c - 1] < min_frac * m.sum():
            continue
        cm = lab == c
        out.append({"mask": cm, "area": int(areas[c - 1]), "centroid": tuple(float(v) for v in ndimage.center_of_mass(cm))})
    return out


def anchor_of(piece, src, spec, names) -> tuple[float, float]:
    a = spec["anchor"]
    if a.startswith("attach:") and a[7:] in names:
        par = src == names.index(a[7:]) + 1
        touch = piece["mask"] & ndimage.binary_dilation(par, iterations=3)
        if touch.sum() >= 5:
            return tuple(float(v) for v in ndimage.center_of_mass(touch))
    return piece["centroid"]


def scale_mask(mask, anchor, k) -> np.ndarray:
    if k == 1.0:
        return mask
    ay, ax = anchor
    return ndimage.affine_transform(mask.astype(np.uint8), [1 / k, 1 / k], offset=[ay * (1 - 1 / k), ax * (1 - 1 / k)],
                                    order=0, mode="constant").astype(bool)


def clipped(P_i, k, side, s) -> bool:
    """True if any scaled piece's bbox would come within FRAME_MARGIN rung px of the canvas edge."""
    m = FRAME_MARGIN * side / s
    for pc in P_i:
        ys, xs = np.nonzero(pc["mask"])
        ay, ax = pc["anchor"]
        y0, y1 = ay + k * (ys.min() - ay), ay + k * (ys.max() - ay)
        x0, x1 = ax + k * (xs.min() - ax), ax + k * (xs.max() - ax)
        if k > 1 and (min(y0, x0) < m or max(y1, x1) > side - 1 - m):
            return True
    return False


def build_source(src, specs, P, ks: dict) -> np.ndarray:
    """Full-size map with each exaggerated part's pieces scaled by ks[i], lowest significance painted first."""
    sig = np.array([0.0] + [s["significance"] for s in specs])
    out = src.copy()
    for i in sorted(ks, key=lambda j: specs[j]["significance"]):
        if ks[i] == 1.0:
            continue
        sp = specs[i]
        grown = np.zeros_like(src, bool)
        for pc in P[i]:
            grown |= scale_mask(pc["mask"], pc["anchor"], ks[i])
        allowed = (sig[out] < sp["significance"]) & (out > 0)
        if sp["may_claim_background"]:
            allowed |= out == 0
        if sp["container"]:
            cont = src == [s["name"] for s in specs].index(sp["container"]) + 1
            allowed &= cont | (src == i + 1)
        out[grown & allowed] = i + 1
    return out


def shrink(m, legend, s) -> np.ndarray:
    pr = np.array([p["priority"] for p in legend["parts"]], float)
    return L.assign(L.coverage(m, len(legend["parts"]), s), pr)


# ------------------------------------------------------------------ footprint / template checks at a rung

def rung_pieces(lab, i, P_i, s, side) -> list[dict | None]:
    """Match each source piece to the rung component of part i nearest its scaled centroid (None if absent)."""
    cl, k = ndimage.label(lab == i + 1, L.EIGHT)
    out, used = [], set()
    for pc in P_i:
        cy, cx = (np.array(pc["centroid"]) + 0.5) * s / side - 0.5
        best, bd = None, 1e9
        for c in range(1, k + 1):
            if c in used:
                continue
            ys, xs = np.nonzero(cl == c)
            dd = np.min((ys - cy) ** 2 + (xs - cx) ** 2)
            if dd < bd:
                best, bd = c, dd
        if best is None or bd > (0.12 * s) ** 2 + 2:
            out.append(None)
            continue
        used.add(best)
        ys, xs = np.nonzero(cl == best)
        w, h = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
        out.append({"w": w, "h": h, "area": int(len(ys)), "fill": round(len(ys) / (w * h), 2), "comp": best})
    return out


def check(lab, i, spec, P_i, s, side) -> dict:
    f = footprint_at(spec, s)
    rp = rung_pieces(lab, i, P_i, s, side)
    deficit, fails = 0.0, []
    for j, r in enumerate(rp):
        if r is None:
            deficit += 1 + f["w"] + f["h"]
            fails.append(f"piece {j} missing")
            continue
        d = max(0, f["w"] - r["w"]) + max(0, f["h"] - r["h"]) + max(0, f["area"] - r["area"]) / max(f["area"], 1)
        if spec["essential_shape"] == "triangle" and f["area"] >= 3 and not 0.3 <= r["fill"] <= 0.8:
            d += 0.5
            fails.append(f"piece {j} fill {r['fill']} not triangular")
        if d:
            fails.append(f"piece {j} {r['w']}x{r['h']}/{r['area']}px < {f['w']}x{f['h']}/{f['area']}")
        deficit += d
    return {"ok": deficit == 0, "deficit": round(deficit, 3), "need": f,
            "pieces": [r and {k: v for k, v in r.items() if k != "comp"} for r in rp], "fails": fails}


def comp_counts(lab, n) -> list[int]:
    return [L.components(lab == i + 1) for i in range(n)]


# ------------------------------------------------------------------ floor templates (rung level)

def floor_fix(lab, i, spec, P_i, s, side, specs) -> tuple[np.ndarray, list[str]]:
    f = footprint_at(spec, s)
    if not (f["w"] or f["h"]):
        return lab, []
    w, h = max(1, f["w"]), max(1, f["h"])
    sig = np.array([0.0] + [q["significance"] for q in specs])
    names = [q["name"] for q in specs]
    out, notes = lab.copy(), []
    rp = rung_pieces(out, i, P_i, s, side)
    cl, _ = ndimage.label(out == i + 1, L.EIGHT)
    for j, (pc, r) in enumerate(zip(P_i, rp)):
        if r is not None and r["w"] >= w and r["h"] >= h and r["area"] >= f["area"]:
            continue
        mine = (cl == r["comp"]) if r else np.zeros_like(out, bool)
        others = (out == i + 1) & ~mine
        fill = spec["container"] and names.index(spec["container"]) + 1
        if mine.any():                         # give the undersized piece back to its surroundings first
            out[mine] = fill or 0
        cy, cx = (np.array(pc["centroid"]) + 0.5) * s / side - 0.5
        best = None
        for dy in (0, -1, 1, -2, 2):
            for dx in (0, -1, 1, -2, 2):
                y0, x0 = int(round(cy - (h - 1) / 2)) + dy, int(round(cx - (w - 1) / 2)) + dx
                if y0 < 0 or x0 < 0 or y0 + h > s or x0 + w > s:
                    continue
                rect = np.zeros_like(out, bool)
                rect[y0:y0 + h, x0:x0 + w] = True
                under = out[rect]
                okc = (under == fill).all() if fill else ((sig[under] < spec["significance"]) & (under > 0)).all()
                apart = not (ndimage.binary_dilation(rect, L.EIGHT) & others).any()
                if okc and apart:
                    best = rect
                    break
            if best is not None:
                break
        if best is None:
            out[mine] = i + 1
            notes.append(f"piece {j}: no room for {w}x{h} template")
            continue
        out[best] = i + 1
        notes.append(f"piece {j}: {r and str(r['w']) + 'x' + str(r['h'])} -> {w}x{h} template")
        cl, _ = ndimage.label(out == i + 1, L.EIGHT)
    return out, notes


# ------------------------------------------------------------------ one rung

def rung(src, legend, specs, P, s, cache: dict) -> dict:
    if s in cache:
        return cache[s]
    side, n = src.shape[0], len(specs)
    names = [q["name"] for q in specs]
    v1 = shrink(src, legend, s)
    ks, log = {}, {}
    exag = sorted((i for i, q in enumerate(specs) if q["mode"] == "exaggerate" and q["max_exaggeration"] > 1),
                  key=lambda i: -specs[i]["significance"])
    for i in exag:
        sp = specs[i]
        base = shrink(build_source(src, specs, P, {**ks, i: 1.0}), legend, s)
        adj0, cc0 = L.adjacency(base), comp_counts(base, n)
        k_sched = schedule(sp, s)
        grid = [round(x, 3) for x in np.arange(1.0, sp["max_exaggeration"] + 1e-9, K_STEP)]
        trials, chosen = [], None
        for k in grid:
            if clipped(P[i], k, side, s):
                trials.append({"k": k, "clipped_by_frame": True})
                break
            lab = shrink(build_source(src, specs, P, {**ks, i: k}), legend, s)
            lost = sorted(f"{names[a - 1]}|{names[b - 1]}" for a, b in adj0 - L.adjacency(lab))
            cc = comp_counts(lab, n)
            dropped = [names[j] for j in range(n) if j != i and cc[j] < cc0[j]]
            c = check(lab, i, sp, P[i], s, side)
            admissible = not lost and not dropped
            trials.append({"k": k, "deficit": c["deficit"], "lost_adjacency": lost, "pieces_lost": dropped})
            if admissible and c["ok"] and k >= k_sched - 1e-9:
                chosen = k
                break
        if chosen is None:              # cap or constraints: the admissible k closest to the schedule with least deficit
            adm = [t for t in trials if "deficit" in t and not t["lost_adjacency"] and not t["pieces_lost"]]
            chosen = max(adm, key=lambda t: (-t["deficit"], -abs(t["k"] - k_sched)))["k"] if adm else 1.0
            why = ("frame margin" if trials[-1].get("clipped_by_frame") else "cap/constraints") + ": best admissible"
        else:
            why = "schedule" if chosen < k_sched + K_STEP - 1e-9 else "footprint"
        ks[i] = chosen
        blocked = [t for t in trials if t.get("clipped_by_frame") or t.get("lost_adjacency") or t.get("pieces_lost")]
        log[names[i]] = {"k": chosen, "k_schedule": round(k_sched, 3), "chosen_by": why,
                         "first_blocked": blocked[0] if blocked else None, "trials": len(trials)}
    mod = build_source(src, specs, P, ks) if ks else src
    lab = shrink(mod, legend, s)
    floors = {}
    for i, q in enumerate(specs):
        if q["mode"] == "floor":
            lab, notes = floor_fix(lab, i, q, P[i], s, side, specs)
            if notes:
                floors[q["name"]] = notes
    checks = {q["name"]: check(lab, i, q, P[i], s, side) for i, q in enumerate(specs) if q["mode"] != "none"}
    checks_v1 = {q["name"]: check(v1, i, q, P[i], s, side) for i, q in enumerate(specs) if q["mode"] != "none"}
    changed = int((lab != v1).sum())
    rec = {"lab": lab, "v1": v1, "mod": mod, "k": log, "floors": floors, "checks": checks, "checks_v1": checks_v1,
           "changed_px_vs_v1": changed,
           "adjacency_lost_vs_v1": sorted(f"{names[a - 1]}|{names[b - 1]}" for a, b in L.adjacency(v1) - L.adjacency(lab)),
           "adjacency_added_vs_v1": sorted(f"{names[a - 1]}|{names[b - 1]}" for a, b in L.adjacency(lab) - L.adjacency(v1))}
    cache[s] = rec
    return rec


# ------------------------------------------------------------------ health / scoring

def score(hs) -> int:
    return sum({"green": 2, "amber": 1, "red": 0}[r["status"]] for h in hs for r in h)


def intended(h, s, recs_k) -> list[dict]:
    """Copy of a health list where amber flags that are the intended effect of exaggeration (area above expected
    within k^2 headroom, IoU vs the plain resample) are marked, and the status recomputed without them."""
    out = []
    for r in h:
        r = dict(r)
        k = (recs_k.get(r["part"]) or {}).get("k", 1.0)
        if k > 1.0 and r["status"] == "amber":
            keep = [x for x in r["reasons"] if not ((x.startswith("area x") and float(x[6:]) <= 1.6 * k * k * 1.15)
                                                    or x.startswith("iou "))]
            r["intended_flags"] = [x for x in r["reasons"] if x not in keep]
            r["status"] = "amber" if keep else "green"
        out.append(r)
    return out


# ------------------------------------------------------------------ judge (pairwise v1 vs v3)

def pair_panel(a, b, legend, h=320) -> bytes:
    img = Image.new("RGB", (2 * (h + 24) + 24, h + 56), (90, 90, 96))
    d = ImageDraw.Draw(img)
    for k, lab in enumerate((a, b)):
        x = 24 + k * (h + 24)
        d.text((x, 6), "AB"[k], fill=(255, 255, 255), font=L._font(30))
        img.paste(L.colorize(lab, legend).resize((h, h), Image.NEAREST), (x, 46))
    return _png(img)


def judge_pair(v1, v3, s, legend, concept_png, ledger, prompt_dir: Path) -> dict:
    prompt = (prompt_dir / "judge_pairwise.md").read_text().format(legend=_legend_text(legend), s=s)
    recs = []
    for order in (("v1", "v3"), ("v3", "v1")):
        labs = {"v1": v1, "v3": v3}
        png = pair_panel(labs[order[0]], labs[order[1]], legend)
        body = {"model": MODEL, "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": models.data_uri(concept_png)}},
                    {"type": "image_url", "image_url": {"url": models.data_uri(png)}}]}]}
        pblob, cblob = store.put_blob(png), store.put_blob(concept_png)
        for i in range(SAMPLES):
            ck = store.key({"ladder_sig_judge": VERSION, "prompt": prompt, "images": [cblob, pblob], "model": MODEL, "sample": i})
            try:
                resp, _, cost, cached = models.call(body, ledger, ck)
                m = json.loads(re.search(r"\{.*\}", resp["choices"][0]["message"].get("content") or "", re.S).group(0))
                w = str(m.get("winner", "")).strip().upper()
                win = {"A": order[0], "B": order[1]}.get(w, "tie" if w == "TIE" else None)
                recs.append({"order": "/".join(order), "sample": i, "winner": win, "letter": w, "confidence": m.get("confidence"),
                             "reason": m.get("reason"), "cost": cost, "cached": cached, "panel_blob": pblob})
            except Exception as e:     # budget or parse failure: the sample abstains
                recs.append({"order": "/".join(order), "sample": i, "winner": None, "error": str(e)[:200]})
    votes = Counter(r["winner"] for r in recs if r.get("winner"))
    return {"samples": recs, "votes": dict(votes), "n": len(recs),
            "v3_win_rate": round(votes.get("v3", 0) / max(1, sum(votes.values())), 3),
            "cost": round(sum(r.get("cost") or 0 for r in recs), 5)}


# ------------------------------------------------------------------ run

def compare_strip(rs_v1, rs_v3, legend, path: Path, h=160):
    sizes = [s for s, _ in rs_v3]
    img = Image.new("RGB", (len(sizes) * (h + 12) + 70, 2 * (h + 30) + 10), (24, 24, 28))
    d = ImageDraw.Draw(img)
    f = L._font(18)
    for r, (tag, rs) in enumerate((("v1", rs_v1), ("v3", rs_v3))):
        y = 6 + r * (h + 30)
        d.text((8, y + h // 2), tag, fill=(230, 230, 230), font=f)
        for k, (s, lab) in enumerate(rs):
            x = 60 + k * (h + 12)
            if r == 0:
                d.text((x, y), f"{s}px", fill=(230, 230, 230), font=f)
            img.paste(L.colorize(lab, legend).resize((h, h), Image.NEAREST), (x, y + 24))
    img.save(path)


def run(region_path: Path = REG / "m1/gpt-s1/regions.png", legend_path: Path = REG / "m1/legend.json",
        concept_path: Path = ROOT / "workspace/shared/sources/squid_concept_gpt_front.png", out: Path = OUT,
        targets=(32, 24, 20, 16), budget=1.5, use_judge=True) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    prompt_dir = out / "prompts"
    legend = json.loads(legend_path.read_text())
    src = L.load_labels(region_path, legend)
    n, side = len(legend["parts"]), src.shape[0]
    names = [p["name"] for p in legend["parts"]]
    sst = L.source_stats(src, n)
    ledger = models.Ledger(budget)
    concept = Image.open(concept_path).convert("RGBA")
    flat = Image.new("RGB", concept.size, (255, 255, 255))
    flat.paste(concept, mask=concept.split()[3])

    g = guidance(src, legend, flat, region_path, ledger, prompt_dir)
    specs = specs_from(g, legend)
    P = [pieces(src, i) for i in range(n)]
    for i, q in enumerate(specs):
        for pc in P[i]:
            pc["anchor"] = anchor_of(pc, src, q, names)

    cache: dict[int, dict] = {}
    res = {"version": VERSION, "base": "ladder.py v1 (source mode) + significance-preserving exaggeration",
           "source": str(region_path), "legend": str(legend_path), "fg_threshold": L.FG_THRESHOLD,
           "min_share": L.MIN_SHARE,
           "source_components": {p["name"]: sst["parts"][i]["comps"] for i, p in enumerate(legend["parts"])},
           "targets": {}}
    for t in targets:
        sizes = L.derive_rungs(side, t)
        recs = [(s, rung(src, legend, specs, P, s, cache)) for s in sizes]
        rs = [(s, r["lab"]) for s, r in recs]
        rs1 = [(s, r["v1"]) for s, r in recs]
        hs = [L.health(lab, src, legend, sst) for _, lab in rs]
        hs1 = [L.health(lab, src, legend, sst) for _, lab in rs1]
        hsi = [intended(h, s, r["k"]) for h, (s, r) in zip(hs, recs)]
        d = out / f"t{t}"
        d.mkdir(parents=True, exist_ok=True)
        for s, lab in rs:
            L.colorize(lab, legend).save(d / f"rung_{s}.png")
        L.filmstrip(rs, hs, legend, out / f"filmstrip_t{t}.png")
        compare_strip(rs1, rs, legend, out / f"compare_t{t}.png")
        first = {p["name"]: {"first_degrade": next((s for (s, _), h in zip(rs, hs) if h[i]["status"] != "green"), None),
                             "first_break": next((s for (s, _), h in zip(rs, hs) if h[i]["status"] == "red"), None)}
                 for i, p in enumerate(legend["parts"])}
        res["targets"][t] = {
            "rungs": sizes, "score": score(hs), "mean_iou": round(float(np.mean([r["iou_plain"] for h in hs for r in h])), 4),
            "first": first, "health": {str(s): h for s, h in zip(sizes, hs)},
            "score_intended": score(hsi), "health_intended": {str(s): h for s, h in zip(sizes, hsi)},
            "v1": {"score": score(hs1), "mean_iou": round(float(np.mean([r["iou_plain"] for h in hs1 for r in h])), 4),
                   "health": {str(s): h for s, h in zip(sizes, hs1)}},
            "significance": {str(s): {k: r[k] for k in ("k", "floors", "checks", "checks_v1", "changed_px_vs_v1",
                                                        "adjacency_lost_vs_v1", "adjacency_added_vs_v1")} for s, r in recs}}
    L.legend_strip(legend, out / "legend_strip.png")
    (out / "results.json").write_text(json.dumps(res, indent=1, default=str))

    judge = {}
    if use_judge:
        cb = Image.open(REG / "m1/concept_on_white.png").convert("RGB")
        cb.thumbnail((256, 256))
        cpng = _png(cb)
        for s in JUDGE_SIZES:
            r = rung(src, legend, specs, P, s, cache)
            judge[str(s)] = judge_pair(r["v1"], r["lab"], s, legend, cpng, ledger, prompt_dir)
    sig = {"version": VERSION, "prompt": str((prompt_dir / "significance.md").relative_to(ROOT)),
           "judge_prompt": str((prompt_dir / "judge_pairwise.md").relative_to(ROOT)),
           "params": {"model": MODEL, "schedule_from": SCHEDULE_FROM, "k_step": K_STEP, "judge_samples_per_order": SAMPLES,
                      "budget": budget},
           "guidance": {k: g[k] for k in ("model", "call", "cost", "cached")} | {"summary": g["raw"].get("summary"),
                                                                                 "raw": g["raw"]},
           "parts": [q | {"pieces": [{"area": pc["area"], "centroid": [round(v, 1) for v in pc["centroid"]],
                                      "anchor": [round(v, 1) for v in pc["anchor"]]} for pc in P[i]]}
                     for i, q in enumerate(specs)],
           "rungs": {str(s): {k: r[k] for k in ("k", "floors", "changed_px_vs_v1", "adjacency_lost_vs_v1")}
                     for s, r in sorted(cache.items(), reverse=True)},
           "judge": judge,
           "cost": {"spent": round(ledger.spent, 5), "calls": ledger.calls, "cached": ledger.cached}}
    (out / "significance.json").write_text(json.dumps(sig, indent=1, default=str))
    return {"results": res, "significance": sig}


if __name__ == "__main__":
    b = float(sys.argv[sys.argv.index("--budget") + 1]) if "--budget" in sys.argv else 1.5
    r = run(use_judge="--no-judge" not in sys.argv, budget=b)
    sg = r["significance"]
    print("summary:", sg["guidance"]["summary"])
    for q in sg["parts"]:
        print(f"  {q['name']:16} sig {q['significance']:.2f} {q['mode']:10} x{q['max_exaggeration']:.2f} {q['essential_shape']:13}"
              f" anchor {q['anchor']:22} bg {q['may_claim_background']!s:5} in {q['container']} fp16 {q['min_footprint']['16']}"
              f" fp32 {q['min_footprint']['32']}")
    for s, d in sg["rungs"].items():
        print(s, {k: (v["k"], v["k_schedule"], v["chosen_by"]) for k, v in d["k"].items()}, d["floors"],
              "changed", d["changed_px_vs_v1"], "lost", d["adjacency_lost_vs_v1"])
    for t, v in r["results"]["targets"].items():
        print(t, "v3", v["score"], v["score_intended"], v["mean_iou"], "| v1", v["v1"]["score"], v["v1"]["mean_iou"])
    for s, j in sg["judge"].items():
        print("judge", s, j["votes"], j["v3_win_rate"], j["cost"])
    print("cost", sg["cost"])
