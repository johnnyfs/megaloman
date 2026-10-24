"""Ladder v4: Johnny's review of v3 (PIPELINE_RULES "finding (v3)" and the v4 spec, 2026-10-23), on top of ladder v3.

Reused unchanged from v3: the v2 region map + legend (no new paint), the exact mirror, the blessed significance
guidance (fins grow into the background only), the target palette, the explicit priority, the arm groups, the cup tip
floor, the nozzle floor, the arm gaps cut from the arm behind, and the depth rule `behind`.

What is new, point by point:
 1. SEPARATION LINES AT EVERY SIZE (blessed). The renderer now draws a 1 px line between every pair of distinct,
    non-black regions at all sizes (render_v4). The ladder makes room for those lines instead of letting them
    erase parts:
      - the face ring keeps its TOP row at every size (v3 dropped it below 24), so the head and the eye surround are
        separated by head-red | line | cream ring | black moat | eye, as at 32;
      - the face block is anchored at the concept's ring-top row and grows DOWN, so the extra rows (ring top + its
        line) are paid by the lower-priority arms, not the head (v3 priority: head > arms);
      - below 32, the fins pay for their line to the head (and to the ring/pods) from the BACKGROUND: every fin
        pixel that becomes line is re-grown one pixel further out on its row (exaggeration takes pixels from the
        background, never from the head or body). This keeps the fins >= 2 px wide where v3's alt_finline left 1 px
        stripes. A fin's 1 px top tip is left to the line.
 2. EYES: rectangular, not boxy. At <= SLIT_MAX the eyes are vertical slits (1 px wide, 2-3 px tall) inside a 1 px
    black moat, so the visor is smaller (6 px wide instead of 8 at 16 and 20). Larger sizes keep v3's eyes.
 3. INNER-ARM CURL. At 20 (v3) the inner arms read as curled because their cups form a hook INSIDE the arm: a run
    one column in from the arm's outer edge (arm colour on both sides) ending in a tip row that is all cup. v3 at 16
    put the single cup pixel on the arm's outer corner instead. v4 draws the 20 px hook explicitly at <= CURL_MAX
    (CURL_RUN px of run above the tip row), so 20 is unchanged and 16 gets the same shape.
 Also: the nozzle floor runs after the face (below the visor), and at slit sizes stray face pixels outside the new,
 narrower block go to the nearest non-face part.
"""
from __future__ import annotations

import json

import numpy as np
from scipy import ndimage

from . import ladder as L
from . import ladder_sym as SY
from . import ladder_v2 as LV
from . import ladder_v3 as L3

ROOT = LV.ROOT
REG = LV.REG
V4 = REG / "v4"

PALETTE, ASSIGN, PRIORITY = L3.PALETTE, L3.ASSIGN, L3.PRIORITY
behind_fn = L3.behind_fn

SLIT_MAX = 20                    # eyes are vertical slits at and below this size
EYE_SLIT = {20: (1, 3), 16: (1, 2)}   # (w, h); 1 px wide, 2-3 px tall
SLIT_MOAT = 1                    # the smaller surround: 1 px black on every side of each slit
RING_BOTTOM_MIN = 24             # ring bottom row only from this size (the moat meets the black base below it)
FACE_PARTS = ("eyes", "visor", "face_housing")
CURL_MAX = 20                    # draw the inner-arm hook at and below this size
CURL_RUN = {20: 2, 16: 1}        # cup pixels running up the arm above the tip row
FIN_PAY_BELOW = 32               # fins re-grow the pixels they give to lines, from the background, below this size
                                 # (at >= 32 v3 already drew the fin line and the significance shrink left room)

# E1 knobs (MODEL_IN_LOOP_PLAN). The defaults are v4 exactly; `rung(ctx, N, params)` overrides them per call.
#  fin_dk     added to the fins' chosen exaggeration k (may pass the frame margin; the hard gate decides)
#  eye_slit   (w, h) of the eye slit at this size (<= SLIT_MAX only); None = EYE_SLIT[N]
#  curl_run   cup pixels running up the inner arm above its tip row (<= CURL_MAX); None = CURL_RUN[N]
#  fin_pay    re-grow the fins' line pixels from the background (below FIN_PAY_BELOW)
#  size_yield which part keeps a contested cell between head and fins: "head" (v4 priority) or "fins"
DEFAULT_PARAMS = {"fin_dk": 0.0, "eye_slit": None, "curl_run": None, "fin_pay": True, "size_yield": "head"}


def legend_v4(legend_v2: dict) -> dict:
    leg = L3.legend_v3(legend_v2)
    leg["version"] = "v4"
    return leg


def palette_record(legend: dict) -> dict:
    rec = L3.palette_record(legend)
    rec.update(slit_max=SLIT_MAX, eye_slit={str(k): v for k, v in EYE_SLIT.items()}, slit_moat=SLIT_MOAT,
               ring_top="every size", ring_bottom_min=RING_BOTTOM_MIN, curl_max=CURL_MAX,
               curl_run={str(k): v for k, v in CURL_RUN.items()}, fin_pay_below=FIN_PAY_BELOW)
    return rec


def prepare(src_raw, legend_v2, spec, sizes=(64, 32, 24, 20, 16)) -> dict:
    ctx = L3.prepare(src_raw, legend_v2, spec, sizes)
    ctx["legend"]["version"] = "v4"
    src, idx = ctx["src"], ctx["idx"]
    S = src.shape[0]
    ys, _ = np.nonzero(src == idx["face_housing"])
    ctx["ring_top"] = ys.min() / S
    return ctx


# ------------------------------------------------------------------ rung

def rung(ctx, N: int, params: dict | None = None) -> dict:
    prm = {**DEFAULT_PARAMS, **(params or {})}
    leg, src, specs, P = ctx["legend"], ctx["src"], ctx["specs"], ctx["P"]
    M = N - 2
    ks, klog = {}, {}
    for i in sorted((i for i, q in enumerate(specs) if q["mode"] == "exaggerate" and q["max_exaggeration"] > 1),
                    key=lambda i: -specs[i]["significance"]):
        klog[specs[i]["name"]] = LV.choose_k(src, leg, specs, P, i, M, N, ks)
        ks[i] = klog[specs[i]["name"]]["k"]
        if prm["fin_dk"] and specs[i]["name"] == "cap_side_lobes":
            ks[i] = round(max(1.0, ks[i] + prm["fin_dk"]), 3)
            klog[specs[i]["name"]] = {**klog[specs[i]["name"]], "k": ks[i], "fin_dk": prm["fin_dk"]}
    mod = LV.build_bg_only(src, specs, P, ks) if ks else src
    ext_src = LV.extend(mod, ctx["inst"], ctx["base_of"])
    base_of = np.array(ctx["base_of"])
    pri = {p["name"]: p["priority"] for p in leg["parts"]}
    if prm["size_yield"] == "fins":            # the fins keep contested cells at the head's expense (addendum)
        pri["cap"], pri["cap_side_lobes"] = pri["cap_side_lobes"], pri["cap"]
    pr = np.array([pri[leg["parts"][b - 1]["name"]] for b in ctx["base_of"][1:]], float)
    ext = L.assign(L.coverage(ext_src, len(base_of) - 1, M), pr)
    log = {}
    behind = behind_fn(leg)
    groups = L3.arm_groups(ext, base_of, ctx, M)
    log["cup_floor"] = L3.cup_floor(ext, groups, ctx, N, M)
    log["face"] = face_template(ext, ctx, N, M, prm["eye_slit"])
    log["nozzle"] = nozzle_floor(ext, ctx, N, M, log["face"])
    groups = L3.arm_groups(ext, base_of, ctx, M)
    log["separate"] = L3.separate_arms(ext, groups, base_of, behind, ctx, N, M)
    log["curl"] = curl_tips(ext, base_of, ctx, N, M, prm["curl_run"])
    log["fin_pay"] = fin_pay(ext, base_of, behind, ctx, N, M) if N < FIN_PAY_BELOW and prm["fin_pay"] else {}
    ext = LV.mirror_inst(ext, ctx["sym"]["keep"], ctx["pair"])
    base = base_of[ext]
    exact = SY.exact_fraction(base, base.shape[1])
    h = L.health(base, src, leg, ctx["sst"])
    return {"N": N, "M": M, "ext": ext, "base": base, "k": klog, "log": log, "exact_fraction": exact,
            "health": h, "health_score": SY.health_score(h), "params": prm}


def face_template(ext, ctx, N, M, eye_slit=None) -> dict:
    """Ring (face colour, top + sides at every size, bottom at >= RING_BOTTOM_MIN) > black moat > two filled eye
    rectangles; vertical slits at <= SLIT_MAX. The block hangs from the concept's ring-top row. At slit sizes, face
    pixels the shrink left outside the new (narrower) block go to the nearest non-face part, so the ring stays 1 px
    and nothing is left as a hole."""
    f, idx = ctx["face"], ctx["idx"]
    if N <= SLIT_MAX:
        ew, eh = eye_slit or EYE_SLIT[N]
        mx = my = SLIT_MOAT
    else:
        fw, fh = L3.EYE_FLOOR.get(N, (2, 2))
        ew = max(fw, int(round(f["eye_w"] * M)))
        eh = max(fh, int(round(f["eye_h"] * M)))
        mx = max(L3.MOAT, int(round(f["moat_x"] * M)))
        my = max(L3.MOAT, int(round(f["moat_y"] * M)))
    gap = max(L3.EYE_GAP_MIN, 2 * int(round(f["gap"] * M / 2)))
    vw, vh = 2 * ew + gap + 2 * mx, eh + 2 * my
    c = M // 2
    ry = int(round(ctx["ring_top"] * M))           # ring top row: the concept's, so the block grows down
    vy0, vx0 = ry + 1, c - vw // 2
    rb = 1 if N >= RING_BOTTOM_MIN else 0
    ring = np.zeros_like(ext, bool)
    ring[ry:vy0 + vh + rb, max(0, vx0 - 1):vx0 + vw + 1] = True
    if N >= 24:                                    # rounded corners (v3)
        for y, x in [(ry, vx0 - 1), (ry, vx0 + vw), (vy0 + vh + rb - 1, vx0 - 1), (vy0 + vh + rb - 1, vx0 + vw)]:
            if 0 <= y < M and 0 <= x < M:
                ring[y, x] = False
    # face pixels outside the new block -> nearest non-face, non-background part
    face_ids = [idx[n] for n in FACE_PARTS]
    base_of = np.array(ctx["base_of"])
    stray = np.isin(base_of[ext], face_ids) & ~ring
    if N > SLIT_MAX:                               # above the slit sizes the block is v3's: keep the shrink's wider
        stray[:] = False                           # housing outline (its bulges are the concept's shape)
    stray[vy0:vy0 + vh, vx0:vx0 + vw] = False
    moved = int(stray.sum())
    if moved:
        donor = (ext > 0) & ~np.isin(base_of[ext], face_ids)
        if donor.any():
            _, (iy, ix) = ndimage.distance_transform_edt(~donor, return_indices=True)
            ext[stray] = ext[iy, ix][stray]
    ext[ring] = idx["face_housing"]
    ext[vy0:vy0 + vh, vx0:vx0 + vw] = idx["visor"]
    ey0 = vy0 + my
    ext[ey0:ey0 + eh, vx0 + mx:vx0 + mx + ew] = idx["eyes"]
    ext[ey0:ey0 + eh, vx0 + mx + ew + gap:vx0 + mx + 2 * ew + gap] = idx["eyes"]
    return {"eye": [ew, eh], "slit": N <= SLIT_MAX, "gap": gap, "moat": [mx, my], "visor": [vw, vh],
            "ring_top": ry, "visor_top": vy0, "ring_bottom": bool(rb), "stray_face_px_moved": moved}


def nozzle_floor(ext, ctx, N, M, face) -> dict:
    """v3's nozzle floor (2-wide column on the axis, >= 2 rows), placed AFTER the face so the taller face block
    cannot cover it: it starts at its scaled row or just below the visor, whichever is lower."""
    nz = ctx["idx"]["nozzle"]
    y0f, y1f, wf = ctx["nozzle_box"]
    h = max(2, int(round((y1f - y0f) * M)))
    w = max(2, 2 * int(round(wf * M / 2)))
    below = face["visor_top"] + face["visor"][1] + (1 if face["ring_bottom"] else 0)
    y0 = max(int(round(y0f * M)), below)
    have = np.nonzero((ext == nz).any(1))[0]
    if len(have):
        y0 = max(int(have.min()), below)
    c = M // 2
    ext[y0:y0 + h, c - w // 2:c + w // 2] = nz
    return {"box": [int(y0), h, w]}


def curl_tips(ext, base_of, ctx, N, M, run=None) -> dict:
    """The 20 px curl, carried down. At 20 (v3, praised) each inner front arm reads as curled because its cups form a
    hook inside the arm: a run of cup pixels one column in from the arm's OUTER edge (arm colour on both sides of it)
    that ends in a tip row that is all cup. Here that hook is drawn explicitly at <= CURL_MAX: the arm's own cup
    pixels are reset to arm, then tip row -> cup, and CURL_RUN rows above it get one cup pixel one column in from the
    outer edge (on the outer edge if the row is < 3 wide). Only that arm group's pixels change."""
    if N > CURL_MAX:
        return {}
    idx = ctx["idx"]
    cup, inner = idx["suction_cups_front"], idx["tentacles_front_inner"]
    run = CURL_RUN.get(N, 1) if run is None else run
    groups = L3.arm_groups(ext, base_of, ctx, M)
    out = {}
    W = ext.shape[1]
    for v in ctx["arm_ids"]:
        if base_of[v] != inner:
            continue
        g = groups == v
        if not g.any():
            continue
        left = np.nonzero(g.any(0))[0].mean() < W / 2
        ext[g & (ext == cup)] = v
        ys = np.nonzero(g.any(1))[0]
        tip = int(ys.max())
        hook = [(tip, int(x)) for x in np.nonzero(g[tip])[0]]
        for dy in range(1, run + 1):
            r = np.nonzero(g[tip - dy])[0]
            if not len(r):
                break
            outer = r.min() if left else r.max()
            step = 1 if left else -1
            x = outer + step if len(r) >= 3 else outer
            hook.append((tip - dy, int(x)))
        for y, x in hook:
            ext[y, x] = cup
        out[str(v)] = [list(p) for p in hook]
    return out


def fin_pay(ext, base_of, behind, ctx, N, M) -> dict:
    """Fins (the exaggerated, background-claiming part) pay for their separating lines from the background: for every
    fin pixel that will become line (it touches a non-black region it gives to), the fin grows one pixel outward on
    that row, into background only."""
    idx = ctx["idx"]
    fin = idx["cap_side_lobes"]
    leg = ctx["legend"]
    black = {i + 1 for i, p in enumerate(leg["parts"]) if p["group"] == "black"}
    prot = {idx[n] for n in ("eyes", "visor", "face_housing", "nozzle")}   # render_v4.PROTECTED: never give
    H, W = ext.shape
    base = base_of[ext]
    pay = {}
    for y, x in zip(*np.nonzero(base == fin)):
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            yy, xx = y + dy, x + dx
            if not (0 <= yy < H and 0 <= xx < W):
                continue
            q = base[yy, xx]
            if q == 0 or q == fin or q in black:
                continue
            if q in prot or behind(fin, q):
                pay[(y, "L" if x < W / 2 else "R")] = pay.get((y, "L" if x < W / 2 else "R"), 0) + 1
                break
    grown = 0
    tips = 0
    for (y, side), n in pay.items():
        row = np.nonzero(base[y] == fin)[0]
        row = row[row < W / 2] if side == "L" else row[row >= W / 2]
        if not len(row):
            continue
        top = all(not (base[yy] == fin)[row.min() if side == "L" else row.max()] for yy in range(y))
        if len(row) == 1 and top:
            tips += 1                  # the fin's 1 px top tip gives its only pixel: let the line take it (a re-grown
            continue                   # pixel there is a detached speck with its own outline cap)
        x = row.min() if side == "L" else row.max()
        d = -1 if side == "L" else 1
        v = ext[y, x]
        for k in range(1, n + 1):
            xx = x + d * k
            if 0 <= xx < W and ext[y, xx] == 0:
                ext[y, xx] = v
                grown += 1
            else:
                break
    return {"rows": len(pay), "grown_px": grown, "tips_left_to_line": tips}


def summary(ctx, rungs: dict) -> dict:
    return L3.summary(ctx, rungs) | {"ring_top_source": ctx["ring_top"]}
