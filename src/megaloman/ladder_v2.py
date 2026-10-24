"""Ladder for the v2 pipeline: source mode + exact mirror symmetry (blessed) + significance (background-only).

Built from the existing ladders, no new ideas beyond the critique:
  - ladder.py v1: each rung is computed from the full-size source (coverage + priority assign).
  - ladder_sym.py v2b: when the concept-stage spec verifies a vertical axis, every rung is exactly mirror-symmetric
    pixel for pixel. Here the whole full-size source is first centred on the axis (ladder_sym.centre_axis) and
    mirrored (ladder_sym.mirror_half); the half is chosen once, deterministically, by the summed ladder health of
    both candidates over all rungs (ladder_sym.fallback's measure), so every rung uses the same half. Each rung is
    then re-mirrored after the shrink and the floors (box filtering and templates can break ties asymmetrically)
    and checked with ladder_sym.exact_fraction == 1.
  - ladder_sig.py v3 (blessed by Johnny 2026-10-23): its significance guidance is reused verbatim (fins exaggerate
    up to x1.8 with triangle footprints, eyes floored to two 1x2 rectangles, nozzle floored 1x2 at <= 24), mapped
    from the v1 legend's part names onto v2's. Per the critique, exaggeration takes pixels from the empty BACKGROUND
    only, never from the head or body (v3 could also paint over lower-significance parts).
  - NEW: limb instances. Each piece of a limb part (each tentacle) gets its own instance label through the shrink,
    so the renderer can draw a separating line wherever two arms meet, even two arms of the same part. Mirroring
    remaps instance labels to their mirror partners.

A rung is computed at M = N - 2 (the renderer draws the outer outline around it), with footprints keyed by N.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from . import ladder as L
from . import ladder_sig as SG
from . import ladder_sym as SY


ROOT = Path(__file__).resolve().parents[2]
REG = ROOT / "workspace/experiments/regions"

SPEC = REG / "symmetry/spec.json"
MIN_INSTANCE = 0.02            # a limb piece is an instance if >= this share of its part


# ------------------------------------------------------------------ symmetric source

def square(lab: np.ndarray) -> np.ndarray:
    h, w = lab.shape
    if w > h:
        return np.vstack([lab, np.zeros((w - h, w), lab.dtype)])
    if h > w:                       # pad both sides equally (axis stays the centre), then rows if now wider
        d = (h - w + 1) // 2
        lab = np.hstack([np.zeros((h, d), lab.dtype), lab, np.zeros((h, d), lab.dtype)])
        return square(lab)
    return lab


def symmetric_source(src: np.ndarray, legend: dict, spec: dict | None) -> tuple[dict, dict]:
    """-> ({'left': map, 'right': map}, info). Axis: the spec's vertical axis refined on the map (+-1%)."""
    frac = spec["axes"]["vertical"]["axis_frac"] if spec and spec["axes"].get("vertical", {}).get("verified") else 0.5
    a2, score = SY.best_axis(src > 0, frac - 0.01, frac + 0.01)
    lab, a2c, note = SY.centre_axis(src.astype(np.int32), a2)
    cands = {}
    for keep in ("left", "right"):
        cands[keep] = square(SY.mirror_half(lab, a2c, keep))     # axis2 == width: the centre, odd or even
    info = {"axis_px": a2 / 2, "axis_frac": round(a2 / 2 / src.shape[1], 4), "mirror_iou": round(score, 4),
            "recentre": note, "size": list(cands["left"].shape),
            "exact": {k: SY.exact_fraction(v, v.shape[1]) for k, v in cands.items()}}
    return cands, info


# ------------------------------------------------------------------ limb instances

def instances(src: np.ndarray, legend: dict) -> tuple[np.ndarray, list[int], list[int]]:
    """Instance map: base parts keep labels 1..n; extra pieces of limb parts get n+1.. .
    -> (inst map, base_of (index = inst label), pair (index = inst label -> mirror partner))."""
    n = len(legend["parts"])
    inst = src.copy()
    base_of = list(range(n + 1))
    for i, p in enumerate(legend["parts"]):
        if p["role"] != "limb":
            continue
        m = src == i + 1
        cc, k = ndimage.label(m, L.EIGHT)
        if k < 2:
            continue
        areas = ndimage.sum_labels(np.ones_like(cc), cc, range(1, k + 1))
        big = [c + 1 for c in np.argsort(-areas) if areas[c] >= MIN_INSTANCE * m.sum()]
        lab_of = {big[0]: i + 1}
        for c in big[1:]:
            base_of.append(i + 1)
            lab_of[c] = len(base_of) - 1
        # every pixel of the part takes its nearest big piece's label (small crumbs join a neighbour)
        keep = np.isin(cc, big)
        _, (iy, ix) = ndimage.distance_transform_edt(~keep, return_indices=True)
        near = cc[iy, ix]
        for c, v in lab_of.items():
            inst[m & (near == c)] = v
    # mirror partners (source is exactly symmetric about its centre)
    pair = list(range(len(base_of)))
    flip = inst[:, ::-1]
    for v in range(1, len(base_of)):
        if base_of.count(base_of[v]) < 2:
            continue                       # parts without extra instances map to themselves
        vals, cnt = np.unique(flip[inst == v], return_counts=True)
        pair[v] = int(vals[cnt.argmax()])
    return inst, base_of, pair


def extend(base_map: np.ndarray, inst_src: np.ndarray, base_of: list[int]) -> np.ndarray:
    """A (possibly exaggerated) base map -> instance map: limb pixels take the nearest source instance of their part."""
    out = base_map.copy()
    for b in sorted(set(base_of) - {0}):
        ids = [v for v in range(1, len(base_of)) if base_of[v] == b]
        if len(ids) < 2:
            continue
        m = base_map == b
        srcm = np.isin(inst_src, ids)
        _, (iy, ix) = ndimage.distance_transform_edt(~srcm, return_indices=True)
        out[m] = inst_src[iy, ix][m]
    return out


def mirror_inst(lab: np.ndarray, keep: str, pair: list[int]) -> np.ndarray:
    """Exact mirror of an instance map about its centre, with the mirrored side relabelled to mirror partners."""
    w = lab.shape[1]
    out = SY.mirror_half(lab, w, keep)
    cols = np.arange(w)
    tgt = (2 * cols + 1 > w) if keep == "left" else (2 * cols + 1 < w)
    P = np.array(pair)
    out[:, tgt] = P[out[:, tgt]]
    return out


# ------------------------------------------------------------------ significance (background only)

def build_bg_only(src, specs, P, ks: dict) -> np.ndarray:
    """Like ladder_sig.build_source, but an exaggerated part only claims empty background (critique rule)."""
    out = src.copy()
    for i in sorted(ks, key=lambda j: specs[j]["significance"]):
        if ks[i] == 1.0:
            continue
        grown = np.zeros_like(src, bool)
        for pc in P[i]:
            grown |= SG.scale_mask(pc["mask"], pc["anchor"], ks[i])
        out[grown & (out == 0)] = i + 1
    return out


def choose_k(src, legend, specs, P, i, M, N, ks) -> dict:
    sp = specs[i]
    side = src.shape[0]
    k_sched = SG.schedule(sp, N)
    trials, chosen = [], None
    for k in [round(x, 3) for x in np.arange(1.0, sp["max_exaggeration"] + 1e-9, SG.K_STEP)]:
        if SG.clipped(P[i], k, side, M):
            trials.append({"k": k, "clipped_by_frame": True})
            break
        lab = SG.shrink(build_bg_only(src, specs, P, {**ks, i: k}), legend, M)
        c = _check(lab, i, sp, P[i], M, N, side)
        trials.append({"k": k, "deficit": c["deficit"]})
        if c["ok"] and k >= k_sched - 1e-9:
            chosen = k
            break
    if chosen is None:
        ok = [t for t in trials if "deficit" in t]
        chosen = min(ok, key=lambda t: (t["deficit"], abs(t["k"] - k_sched)))["k"] if ok else 1.0
        why = ("frame margin" if trials[-1].get("clipped_by_frame") else "cap") + ": best"
    else:
        why = "schedule" if chosen < k_sched + SG.K_STEP - 1e-9 else "footprint"
    return {"k": chosen, "k_schedule": round(k_sched, 3), "chosen_by": why, "trials": len(trials)}


def _check(lab, i, spec, P_i, M, N, side):
    """ladder_sig.check with the footprint of the display size N (the rung is N-2 + outline)."""
    f = SG.footprint_at(spec, N)
    s2 = dict(spec)
    s2["min_footprint"] = {k: f for k in SG.FOOT_SIZES}
    return SG.check(lab, i, s2, P_i, M, side)


def _floor(lab, i, spec, P_i, M, N, side, specs):
    f = SG.footprint_at(spec, N)
    s2 = dict(spec)
    s2["min_footprint"] = {k: f for k in SG.FOOT_SIZES}
    return SG.floor_fix(lab, i, s2, P_i, M, side, specs)


# ------------------------------------------------------------------ blessed v3 guidance, mapped onto the v2 legend

V3_SIG = REG / "ladder/v3_significance/significance.json"
V3_NAMES = {   # v1-legend part -> v2-legend parts it became
    "mantle_shell": ["cap"], "side_fins": ["cap_side_lobes"], "face_frame": ["face_housing"], "visor": ["visor"],
    "eyes": ["eyes"], "side_pods": ["side_pods"], "tentacle_joints": ["undercarriage"],
    "tentacles": ["tentacles_front_outer", "tentacles_front_inner", "tentacles_back"],
    "suction_cups": ["suction_cups_front"], "central_nozzle": ["nozzle"]}


def v3_specs(legend) -> tuple[list[dict], dict]:
    """Ladder v3 was blessed, so its significance guidance (VLM output + the eyes override) is reused verbatim, part
    by part, instead of a new guidance call. Anchors/containers are renamed to v2 parts. -> (specs, name map)."""
    sig = json.loads(V3_SIG.read_text())
    by = {q["name"]: q for q in sig["parts"]}
    inv = {n2: n1 for n1, l in V3_NAMES.items() for n2 in l}
    ren = lambda n: (V3_NAMES.get(n) or [None])[0]
    specs, smap = [], {}
    for p in legend["parts"]:
        q = by.get(inv.get(p["name"]))
        if q is None:
            specs.append({"name": p["name"], "significance": 0.3, "essential_shape": "blob", "mode": "none",
                          "max_exaggeration": 1.0, "anchor": "centroid", "may_claim_background": False, "container": None,
                          "min_footprint": {k: {"w": 0, "h": 0, "area": 0} for k in SG.FOOT_SIZES}})
            continue
        smap[p["name"]] = q["name"]
        s = {k: v for k, v in q.items() if k not in ("pieces",)}
        s["name"] = p["name"]
        if s["anchor"].startswith("attach:"):
            s["anchor"] = "attach:" + (ren(s["anchor"][7:]) or "")
        s["container"] = ren(s["container"]) if s["container"] else None
        specs.append(s)
    return specs, smap


# ------------------------------------------------------------------ run

def prepare(src_raw: np.ndarray, legend: dict, spec=None, sizes=(64, 32, 24, 20, 16)):
    """Symmetric source (half chosen by summed health), instances, significance specs. -> context dict."""
    n = len(legend["parts"])
    cands, sym = symmetric_source(src_raw, legend, spec)
    sst0 = {k: L.source_stats(v, n) for k, v in cands.items()}
    pr = np.array([p["priority"] for p in legend["parts"]], float)
    scores = {}
    for k, v in cands.items():
        tot = 0
        for N in sizes:
            lab = L.assign(L.coverage(v, n, N - 2), pr)
            tot += SY.health_score(L.health(lab, v, legend, sst0[k]))
        scores[k] = tot
    keep = max(scores, key=lambda k: (scores[k], k == "left"))
    src = cands[keep]
    sym.update(keep=keep, half_scores=scores, chosen_by="summed ladder health over rungs (deterministic)")
    inst, base_of, pair = instances(src, legend)
    names = [p["name"] for p in legend["parts"]]
    specs, smap = v3_specs(legend)
    P = [SG.pieces(src, i) for i in range(n)]
    for i, q in enumerate(specs):
        for pc in P[i]:
            pc["anchor"] = SG.anchor_of(pc, src, q, names)
    return {"src": src, "inst": inst, "base_of": base_of, "pair": pair, "specs": specs, "P": P, "sym": sym,
            "guidance": {"source": str(V3_SIG.relative_to(ROOT)), "blessed": "ladder v3 (Johnny, 2026-10-23)",
                         "name_map": smap},
            "sst": L.source_stats(src, n)}


def rung(ctx, legend, N: int) -> dict:
    src, specs, P = ctx["src"], ctx["specs"], ctx["P"]
    n, side, M = len(legend["parts"]), src.shape[0], N - 2
    ks, klog = {}, {}
    for i in sorted((i for i, q in enumerate(specs) if q["mode"] == "exaggerate" and q["max_exaggeration"] > 1),
                    key=lambda i: -specs[i]["significance"]):
        klog[specs[i]["name"]] = choose_k(src, legend, specs, P, i, M, N, ks)
        ks[i] = klog[specs[i]["name"]]["k"]
    mod = build_bg_only(src, specs, P, ks) if ks else src
    ext_src = extend(mod, ctx["inst"], ctx["base_of"])
    base_of = np.array(ctx["base_of"])
    pr = np.array([legend["parts"][b - 1]["priority"] for b in ctx["base_of"][1:]], float)
    ext = L.assign(L.coverage(ext_src, len(base_of) - 1, M), pr)
    base = base_of[ext]
    floors = {}
    for i, q in enumerate(specs):
        if q["mode"] == "floor":
            nb, notes = _floor(base, i, q, P[i], M, N, side, specs)
            if notes:
                floors[q["name"]] = notes
            ch = nb != base
            ext[ch] = nb[ch]          # floored parts are not limbs: instance label == base label
            base = nb
    ext = mirror_inst(ext, ctx["sym"]["keep"], ctx["pair"])
    base = base_of[ext]
    exact = SY.exact_fraction(base, base.shape[1])
    h = L.health(base, src, legend, ctx["sst"])
    checks = {q["name"]: _check(base, i, q, P[i], M, N, side) for i, q in enumerate(specs) if q["mode"] != "none"}
    return {"N": N, "M": M, "ext": ext, "base": base, "k": klog, "floors": floors, "exact_fraction": exact,
            "health": h, "health_score": SY.health_score(h), "checks": checks}


def summary(ctx, rungs: dict) -> dict:
    return {"symmetry": ctx["sym"], "guidance": ctx["guidance"],
            "instances": {str(v): {"base": ctx["base_of"][v], "pair": ctx["pair"][v]} for v in range(1, len(ctx["base_of"]))},
            "specs": [{k: v for k, v in q.items()} for q in ctx["specs"]],
            "rungs": {str(N): {k: r[k] for k in ("M", "k", "floors", "exact_fraction", "health_score", "checks")}
                      | {"health": [{k: x[k] for k in ("part", "status", "area", "components", "reasons")} for x in r["health"]]}
                      for N, r in rungs.items()}}
