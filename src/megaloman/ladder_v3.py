"""Ladder v3: Johnny's distilled rules after v2 (PIPELINE_RULES, 2026-10-23), on top of the v2 region map and ladder.

Reused unchanged: regions v2 (legend + labels, no new paint), ladder v2's symmetric source (exact mirror, blessed),
limb instances, and the blessed v3 significance guidance (fins exaggerate into the BACKGROUND only).

What is new, rule by rule:
 1. TARGET PALETTE FIRST. The final palette (black + 3 NES colours) and every part's colour are decided here, before
    any shrinking, and recorded (PALETTE, ASSIGN). Parts that share a colour are one colour group from the start.
 2. NO PATTERNING: one solid colour per part (nothing is sampled from the concept).
 3. REGION PRIORITY: an explicit, recorded order (PRIORITY, eyes highest) replaces the legend's model-chosen
    priorities in the shrink (ladder.assign: the higher priority wins a contested pixel).
 4. EYES: a face template applied last, so nothing can take its pixels: two boxy, filled eye rectangles, a black
    visor moat of >= 1 px on every side of each eye, and a ring of the face colour around the visor. Sizes are the
    concept's scaled sizes with a floor (EYE_FLOOR); v1's 32 px face (2x3 eyes, moat, ring) is the reference.
 5. LIKE INSTANCES STAY DISTINCT: every front arm (a tentacle instance plus the suction cups that sit on it) is an
    arm group. Wherever two arm groups touch, a 1 px gap is cut from the arm BEHIND (rule 6); where the cut reaches
    the bottom it opens into a notch, widened at the tips. Each arm keeps a cup at its tip (the curl hint: the cups
    line the curled tip in the concept).
 6. LINE SACRIFICE BY DEPTH: `behind(a, b)` (legend occluded_by, then depth back, then the v2 layer order) decides
    which region gives up the pixels for a gap here, and for every separating line in render_v3.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from . import ladder as L
from . import ladder_sig as SG
from . import ladder_sym as SY
from . import ladder_v2 as LV
from . import regions_v2 as RV

ROOT = LV.ROOT
REG = LV.REG
V3 = REG / "v3"

# ------------------------------------------------------------------ 1. target palette (decided first)

PALETTE = {    # group -> (NES index, RGB hex). One NES hue ramp (x6): dark / mid / light, plus black.
    "black": ("0F", "000000"),
    "dark": ("16", "F83800"),     # the head: the darkest of the three, so the big mass grounds the sprite
    "mid": ("26", "F87858"),      # the accents: the concept's coral
    "light": ("36", "F0D0B0"),    # the face ring, the arms and the eyes: the concept's cream
}
ASSIGN = {     # part -> group, with the reason (executive calls, Johnny's rule 1)
    "cap": ("dark", "the head is the darker colour"),
    "cap_side_lobes": ("mid", "fins share with nozzle and cups; differ from the head so the jaunty fins read "
                              "without a line"),
    "nozzle": ("mid", "fins + nozzle + cups share one colour"),
    "suction_cups_front": ("mid", "fins + nozzle + cups share one colour; the cups mark each arm's curled tip"),
    "side_pods": ("mid", "mechanical accent like the nozzle; keeps the cream face ring a clean ring and the pods "
                         "distinct ear pieces"),
    "face_housing": ("light", "face ring + tentacles share one colour"),
    "tentacles_front_outer": ("light", "face ring + tentacles share one colour"),
    "tentacles_front_inner": ("light", "face ring + tentacles share one colour"),
    "eyes": ("light", "the concept's pale glowing eyes; stay distinct by the black moat, not by colour"),
    "visor": ("black", "the moat around the eyes (concept: dark glass)"),
    "undercarriage": ("black", "dark mechanical base (concept median L 17)"),
    "tentacles_back": ("black", "occluded instances render black (depth)"),
}

# ------------------------------------------------------------------ 3. region priority (highest first)

PRIORITY = [   # (part, why)
    ("eyes", "the face is the character: eyes first"),
    ("visor", "the moat that keeps the eyes distinct"),
    ("face_housing", "the ring that frames the moat"),
    ("nozzle", "floored character feature on the axis"),
    ("cap", "the head silhouette"),
    ("cap_side_lobes", "the jaunty fins (also protected by significance growth into the background)"),
    ("tentacles_front_inner", "front arms; inner arms are in front of the outer ones"),
    ("tentacles_front_outer", "front arms, behind the inner ones"),
    ("suction_cups_front", "accents on the arms; floored per arm at the tip instead"),
    ("side_pods", "minor mechanical detail"),
    ("undercarriage", "black base: fills what is left"),
    ("tentacles_back", "hidden arms: black depth, lowest"),
]

# eye rectangle floor (w, h) by display size; above the floor the eyes scale with the concept
EYE_FLOOR = {64: (2, 3), 32: (2, 3), 24: (2, 3), 20: (2, 2), 16: (2, 2)}
MOAT = 1          # black visor margin around each eye, every size
EYE_GAP_MIN = 2   # black between the two eyes (even: exact symmetry about a between-columns axis)
RING = 1          # face-colour ring around the visor (sides and bottom), every size
RING_TB_MIN = 24  # the ring's top/bottom rows only from this size; below it the moat itself meets the dark head
                  # above and the black undercarriage below (a ring row there costs 2 rows: ring + line to the arms)
NOTCH_W = {64: 3, 32: 2, 24: 1, 20: 1, 16: 1}   # gap width at an arm tip contact (bg shows through when >= 3)
NOTCH_ROWS = {64: 3, 32: 2, 24: 1, 20: 1, 16: 1}
CUP_FLOOR = {64: 3, 32: 2, 24: 1, 20: 1, 16: 1}  # cup pixels each front arm keeps at its tip


def legend_v3(legend: dict) -> dict:
    """The v2 legend with v3's explicit priorities and colour groups."""
    order = [p for p, _ in PRIORITY]
    names = [p["name"] for p in legend["parts"]]
    assert sorted(order) == sorted(names), (order, names)
    leg = json.loads(json.dumps(legend))
    for p in leg["parts"]:
        p["priority_v2"] = p["priority"]
        p["priority"] = len(order) - order.index(p["name"])
        p["group"] = ASSIGN[p["name"]][0]
        p["colour"] = PALETTE[p["group"]][1]
    leg["version"] = "v3"
    return leg


def palette_record(legend: dict) -> dict:
    return {"palette": {g: {"nes": i, "hex": h} for g, (i, h) in PALETTE.items()},
            "assign": {n: {"group": g, "hex": PALETTE[g][1], "why": w} for n, (g, w) in ASSIGN.items()},
            "groups": {g: [n for n, (gg, _) in ASSIGN.items() if gg == g] for g in PALETTE},
            "priority": [{"rank": k + 1, "part": n, "why": w} for k, (n, w) in enumerate(PRIORITY)],
            "eye_floor": {str(k): v for k, v in EYE_FLOOR.items()}, "moat": MOAT, "ring": RING, "ring_tb_min": RING_TB_MIN,
            "eye_gap_min": EYE_GAP_MIN, "notch_w": NOTCH_W, "cup_floor": CUP_FLOOR}


# ------------------------------------------------------------------ 6. depth: who is behind

def behind_fn(legend: dict):
    """-> behind(i, j) for base part indices (1-based): True if part i is behind part j (i gives up the pixel)."""
    parts = legend["parts"]
    names = [p["name"] for p in parts]
    layer = {n: k for k, n in enumerate(RV.layer_order(legend))}

    def behind(i: int, j: int) -> bool:
        a, b = parts[i - 1], parts[j - 1]
        if b["name"] in a["occluded_by"] and a["name"] not in b["occluded_by"]:
            return True
        if a["name"] in b["occluded_by"] and b["name"] not in a["occluded_by"]:
            return False
        if a["depth"] != b["depth"]:
            return a["depth"] == "back"
        return layer[a["name"]] < layer[b["name"]]
    behind.names = names
    return behind


# ------------------------------------------------------------------ context

def prepare(src_raw, legend_v2, spec, sizes=(64, 32, 24, 20, 16)) -> dict:
    leg = legend_v3(legend_v2)
    ctx = LV.prepare(src_raw, leg, spec, sizes)
    src, names = ctx["src"], [p["name"] for p in leg["parts"]]
    idx = {n: k + 1 for k, n in enumerate(names)}
    ctx["idx"] = idx
    ctx["legend"] = leg
    # arm groups in the source: each front-arm instance + the cups nearest to it
    base_of = np.array(ctx["base_of"])
    arm_parts = {idx["tentacles_front_outer"], idx["tentacles_front_inner"]}
    arm_ids = [v for v in range(1, len(base_of)) if base_of[v] in arm_parts]
    inst = ctx["inst"]
    arm_src = np.where(np.isin(inst, arm_ids), inst, 0)
    _, (iy, ix) = ndimage.distance_transform_edt(arm_src == 0, return_indices=True)
    cups = src == idx["suction_cups_front"]
    group = arm_src.copy()
    group[cups] = arm_src[iy, ix][cups]
    ctx["arm_ids"], ctx["arm_src"] = arm_ids, group
    # per arm: the source cup centroid (the curl hint) and the source face geometry
    S = src.shape[0]
    ctx["cup_c"] = {}
    for v in arm_ids:
        m = cups & (group == v)
        if m.any():
            ys, xs = np.nonzero(m)
            ctx["cup_c"][v] = (ys.mean() / S, xs.mean() / S)
    ctx["face"] = face_geometry(src, idx)
    noz = src == idx["nozzle"]
    ys, xs = np.nonzero(noz)
    ctx["nozzle_box"] = (ys.min() / S, ys.max() / S, (xs.max() - xs.min() + 1) / S)
    return ctx


def face_geometry(src, idx) -> dict:
    """Source face (fractions of the side): eye size, gap between eyes, visor margins, visor centre row."""
    S = src.shape[0]
    eyes = src == idx["eyes"]
    lab, k = ndimage.label(eyes)
    boxes = sorted(ndimage.find_objects(lab), key=lambda b: -(b[0].stop - b[0].start) * (b[1].stop - b[1].start))[:2]
    boxes = sorted(boxes, key=lambda b: b[1].start)
    (ly, lx), (ry, rx) = boxes
    vis = (src == idx["visor"]) | eyes
    vy, vx = np.nonzero(vis)
    eh = np.mean([b[0].stop - b[0].start for b in boxes])
    ew = np.mean([b[1].stop - b[1].start for b in boxes])
    return {"eye_w": ew / S, "eye_h": eh / S, "gap": (rx.start - lx.stop) / S,
            "moat_x": (lx.start - vx.min()) / S, "moat_y": (ly.start - vy.min()) / S,
            "cy": (ly.start + ly.stop) / 2 / S}


# ------------------------------------------------------------------ rung

def rung(ctx, N: int) -> dict:
    leg, src, specs, P = ctx["legend"], ctx["src"], ctx["specs"], ctx["P"]
    idx = ctx["idx"]
    M = N - 2
    # shrink: significance (fins, background only) exactly as v2, but with the v3 priorities
    ks, klog = {}, {}
    for i in sorted((i for i, q in enumerate(specs) if q["mode"] == "exaggerate" and q["max_exaggeration"] > 1),
                    key=lambda i: -specs[i]["significance"]):
        klog[specs[i]["name"]] = LV.choose_k(src, leg, specs, P, i, M, N, ks)
        ks[i] = klog[specs[i]["name"]]["k"]
    mod = LV.build_bg_only(src, specs, P, ks) if ks else src
    ext_src = LV.extend(mod, ctx["inst"], ctx["base_of"])
    base_of = np.array(ctx["base_of"])
    pr = np.array([leg["parts"][b - 1]["priority"] for b in ctx["base_of"][1:]], float)
    ext = L.assign(L.coverage(ext_src, len(base_of) - 1, M), pr)
    log = {}
    behind = behind_fn(leg)
    # arm groups at this rung: coverage of each source group
    groups = arm_groups(ext, base_of, ctx, M)
    log["cup_floor"] = cup_floor(ext, groups, ctx, N, M)
    groups = arm_groups(ext, base_of, ctx, M)
    log["nozzle"] = nozzle_floor(ext, ctx, N, M)
    log["face"] = face_template(ext, ctx, N, M)
    groups = arm_groups(ext, base_of, ctx, M)
    log["separate"] = separate_arms(ext, groups, base_of, behind, ctx, N, M)
    ext = LV.mirror_inst(ext, ctx["sym"]["keep"], ctx["pair"])
    base = base_of[ext]
    exact = SY.exact_fraction(base, base.shape[1])
    h = L.health(base, src, leg, ctx["sst"])
    return {"N": N, "M": M, "ext": ext, "base": base, "k": klog, "log": log, "exact_fraction": exact,
            "health": h, "health_score": SY.health_score(h)}


def arm_groups(ext, base_of, ctx, M) -> np.ndarray:
    """Arm-group id per rung pixel (0 = not an arm): arm instances keep their own id; cup pixels take the group
    with the most source coverage in their cell."""
    g = np.where(np.isin(ext, ctx["arm_ids"]), ext, 0)
    cups = base_of[ext] == ctx["idx"]["suction_cups_front"]
    if cups.any():
        ids = ctx["arm_ids"]
        cov = L.coverage(_dense(ctx["arm_src"], ids), len(ids), M)
        best = np.array(ids)[cov.argmax(0)]
        g[cups] = best[cups]
    return g


def _dense(arm_src, ids):
    out = np.zeros_like(arm_src)
    for k, v in enumerate(ids):
        out[arm_src == v] = k + 1
    return out


def cup_floor(ext, groups, ctx, N, M) -> dict:
    """Each front arm keeps >= CUP_FLOOR[N] cup pixels at its tip: arm pixels on its background edge nearest the
    source cup centroid (the curled tip) are recoloured as cups."""
    cup = ctx["idx"]["suction_cups_front"]
    need = CUP_FLOOR.get(N, 1)
    out = {}
    bg = ext == 0
    edge = ndimage.binary_dilation(bg, structure=ndimage.generate_binary_structure(2, 1))
    for v, (cy, cx) in ctx["cup_c"].items():
        have = int(((groups == v) & (ext == cup)).sum())
        if have >= need:
            out[str(v)] = {"had": have}
            continue
        cand = np.argwhere((ext == v) & edge)
        if not len(cand):
            cand = np.argwhere(ext == v)
        if not len(cand):
            out[str(v)] = {"had": have, "missing_arm": True}
            continue
        d = np.hypot(cand[:, 0] - cy * M, cand[:, 1] - cx * M)
        add = cand[np.argsort(d)[:need - have]]
        for y, x in add:
            ext[y, x] = cup
        out[str(v)] = {"had": have, "added": [list(map(int, p)) for p in add]}
    return out


def nozzle_floor(ext, ctx, N, M) -> dict:
    """The nozzle keeps a 2-wide column on the axis (exact symmetry), >= 2 rows, at its scaled position."""
    nz = ctx["idx"]["nozzle"]
    y0f, y1f, wf = ctx["nozzle_box"]
    h = max(2, int(round((y1f - y0f) * M)))
    w = max(2, 2 * int(round(wf * M / 2)))
    have = ext == nz
    y0 = int(round(y0f * M))
    if have.any():
        y0 = int(np.nonzero(have.any(1))[0].min())
    c = M // 2
    ext[y0:y0 + h, c - w // 2:c + w // 2] = nz
    return {"box": [y0, h, w]}


def face_template(ext, ctx, N, M) -> dict:
    """Rule 4: ring (face colour) > visor moat (black) > two filled eye rectangles, centred on the axis at the
    scaled visor row. Applied over whatever the shrink left (eyes have the highest priority)."""
    f = ctx["face"]
    idx = ctx["idx"]
    fw, fh = EYE_FLOOR.get(N, (2, 2))
    ew = max(fw, int(round(f["eye_w"] * M)))
    eh = max(fh, int(round(f["eye_h"] * M)))
    gap = max(EYE_GAP_MIN, 2 * int(round(f["gap"] * M / 2)))
    mx = max(MOAT, int(round(f["moat_x"] * M)))
    my = max(MOAT, int(round(f["moat_y"] * M)))
    vw, vh = 2 * ew + gap + 2 * mx, eh + 2 * my
    c = M // 2
    vy0 = int(round(f["cy"] * M - vh / 2))
    vx0 = c - vw // 2
    r = RING
    # ring: the face rectangle, corners left as they are (keeps the housing's rounded corners)
    rt = r if N >= RING_TB_MIN else 0
    ring = np.zeros_like(ext, bool)
    ring[max(0, vy0 - rt):vy0 + vh + rt, max(0, vx0 - r):vx0 + vw + r] = True
    corners = [(vy0 - rt, vx0 - r), (vy0 - rt, vx0 + vw + r - 1), (vy0 + vh + rt - 1, vx0 - r),
               (vy0 + vh + rt - 1, vx0 + vw + r - 1)]
    for y, x in corners:
        if 0 <= y < M and 0 <= x < M and N >= 24:
            ring[y, x] = False
    ext[ring] = idx["face_housing"]
    ext[vy0:vy0 + vh, vx0:vx0 + vw] = idx["visor"]
    ey0 = vy0 + my
    ext[ey0:ey0 + eh, vx0 + mx:vx0 + mx + ew] = idx["eyes"]
    ext[ey0:ey0 + eh, vx0 + mx + ew + gap:vx0 + mx + 2 * ew + gap] = idx["eyes"]
    return {"eye": [ew, eh], "gap": gap, "moat": [mx, my], "visor": [vw, vh], "visor_top": vy0, "ring": r, "ring_top_bottom": rt}


def separate_arms(ext, groups, base_of, behind, ctx, N, M) -> dict:
    """Rule 5 + 6: wherever two arm groups touch (4-neighbours), the pixel of the group BEHIND becomes a gap (bg;
    rendered as ink, or as a notch where it opens to the outside). At the tips (the lowest NOTCH_ROWS rows of each
    contact) the gap widens to NOTCH_W px, taken from the behind arm. Equal depth on the axis: both sides cut."""
    cut = np.zeros_like(ext, bool)
    contacts = {}
    H, W = ext.shape
    for y in range(H):
        for x in range(W):
            a = groups[y, x]
            if not a:
                continue
            for dy, dx in ((1, 0), (0, 1)):
                yy, xx = y + dy, x + dx
                if yy >= H or xx >= W:
                    continue
                b = groups[yy, xx]
                if not b or b == a:
                    continue
                A, B = base_of[a], base_of[b]
                if A == B:      # mirror partners (left/right instances of one part): equal depth, both give a pixel
                    pa = (y, x)
                    cut[yy, xx] = True
                    cut[pa] = True
                    key = tuple(sorted((int(a), int(b))))
                else:
                    pa = (y, x) if behind(A, B) else (yy, xx)
                    cut[pa] = True
                    key = (int(a), int(b)) if behind(A, B) else (int(b), int(a))
                contacts.setdefault(key, []).append(pa)
    # tips: widen the lowest rows of each contact into a notch, from the behind arm
    notch = 0
    nw, nr = NOTCH_W.get(N, 1), NOTCH_ROWS.get(N, 1)
    for (back, front), pts in contacts.items():
        ys = sorted({p[0] for p in pts})
        low = set(ys[-nr:])
        for y, x in pts:
            if y not in low:
                continue
            # extend away from the front arm, inside the behind arm
            fx = [xx for xx in (x - 1, x + 1) if 0 <= xx < W and groups[y, xx] == front]
            d = (x - fx[0]) if fx else (1 if x >= W // 2 else -1)
            for k in range(1, nw):
                xx = x + d * k
                if 0 <= xx < W and groups[y, xx] == back:
                    cut[y, xx] = True
                    notch += 1
    ext[cut] = 0
    return {"cut_px": int(cut.sum()), "notch_px": notch,
            "contacts": {f"{a}>{b}": len(v) for (a, b), v in contacts.items()}}


def summary(ctx, rungs: dict) -> dict:
    return {"symmetry": ctx["sym"], "guidance": ctx["guidance"], "face_source": ctx["face"],
            "arm_ids": ctx["arm_ids"],
            "instances": {str(v): {"base": ctx["base_of"][v], "pair": ctx["pair"][v]} for v in range(1, len(ctx["base_of"]))},
            "rungs": {str(N): {k: r[k] for k in ("M", "k", "log", "exact_fraction", "health_score")}
                      | {"health": [{k: x[k] for k in ("part", "status", "area", "components", "reasons")} for x in r["health"]]}
                      for N, r in rungs.items()}}
