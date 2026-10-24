"""Render v2: regions over coloration (PIPELINE_RULES, Johnny's critique of render v1, 2026-10-23).

Inputs: the v2 region map + legend (regions_v2), the ladder v2 rungs (exact symmetry + blessed v3 significance,
limb instances), and the shaded concept (for colours only). No model calls.

Rules, and how each is implemented (differences from render.py v1 marked NEW / CHANGED):
 1. CHANGED - regions are SOLID colours. Each material (legend `material`; parts sharing one share a colour) is its
    concept's majority tone (2-means in Lab over the eroded region, ink excluded), snapped to the NES palette. If
    that collides (Lab dE < 20) with a material already coloured (largest first) that comes within one 16-px-sprite
    pixel of it (parts that touch once shrunk, e.g. face frame and arms across the undercarriage), it takes its second
    tone. No patterns (spots, stripes) and no shade tone.
 2. NEW - depth: parts with depth 'back' (occluded instances: the back tentacles) are black. Dark parts (concept
    median L < 40: visor, undercarriage) are black too (v1 rule 4).
 3. NEW - cups: their own solid colour; edged (a line on the tentacle side) when N >= cup_edge_min (32; at 24 a
    cup is ~2 px and its edge would eat the arm).
 4. NEW - separating lines between front tentacles: between two limb instances (different arms, same part or
    not) a line is ALWAYS drawn, at every size; the v1 "contrasting colours need no line below 32 px" saving is
    applied only to non-limb pairs, so the line budget at small sizes goes to the arms.
 5. NEW - depth line under the side fins: flaps of the body behind it that stick out into the background
    (legend occluded_by a body part of the SAME material, role body/limb, touching background: the cap side lobes) get a 2-px line
    under their bottom edge, taken from the background (the physical gap between fin and head), never the fin.
 6. NEW - floored features (eyes, nozzle: ladder v3 'floor' parts) are always ringed in ink against any non-black
    neighbour, and their colours weigh x8 in the tile rule, so the eye floor (two distinct rectangles) survives the
    palette and the tile rule (in a first run the 16 px eyes were remapped away and the 32 px eyes merged into
    the frame).
 7. Kept from v1: thin 1-px outline drawn AROUND an (N-2) map (outset), part lines on the larger region's side,
    the no-clump clean-up (ink pixels with > 2 ink 8-neighbours removed where nothing leaks), no line next to black
    parts, exact mirror symmetry of the final pixels, NES palette and <= 3 colours per 8x8 tile (render.tile_reduce).
    Dropped from v1: concept-shadow depth thickening (replaced by black back instances), shading, patterns.

uv run python -m megaloman.render_v2
Writes workspace/experiments/regions/v2/{ladder,render}/ and v2/sheet_same_zoom.png, v2/results.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from . import imaging, ladder as L, ladder_v2 as LV, render as R1
from .vendor.pixelize import to_lab

ROOT = R1.ROOT
REG = R1.REG
V2 = REG / "v2"
SIZES = (64, 32, 24, 20, 16)
BLACK = (0, 0, 0)
DARK_L = 40
SALIENT_W = 8.0
FOUR = R1.FOUR
EIGHT = R1.EIGHT


# ------------------------------------------------------------------ materials

def _tones(px):
    lo = px[px[:, 0] <= np.percentile(px[:, 0], 15)].mean(0)
    hi = px[px[:, 0] >= np.percentile(px[:, 0], 85)].mean(0)
    cen = np.array([lo, hi])
    for _ in range(30):
        a = ((px[:, None] - cen[None]) ** 2).sum(-1).argmin(1)
        cen = np.array([px[a == k].mean(0) if (a == k).any() else cen[k] for k in range(2)])
    return a


def materials(src, concept, legend) -> list[dict]:
    rgb = concept[..., :3].astype(float)
    lab = to_lab(rgb.reshape(-1, 3)).reshape(rgb.shape)
    solid = concept[..., 3] > 200
    parts = legend["parts"]
    groups = {}
    for i, p in enumerate(parts):         # back instances never share a colour with their front siblings
        groups.setdefault(p["material"] + ("/back" if p["depth"] == "back" else ""), []).append(i)
    near_px = max(1, src.shape[0] // 16)       # "adjacent" = within one pixel at the 16 px target
    mats = [None] * len(parts)
    mat_col = {}
    # largest material first
    order = sorted(groups, key=lambda g: -sum((src == i + 1).sum() for i in groups[g]))
    for g in order:
        idx = groups[g]
        reg = np.isin(src, [i + 1 for i in idx])
        m = ndimage.binary_erosion(reg, iterations=3) & solid
        medL = float(np.median(lab[m][:, 0]))
        back = all(parts[i]["depth"] == "back" for i in idx)
        rec = {"material": g, "median_L": round(medL, 1)}
        if back:
            rec.update(mode="black", colour=BLACK, why="depth back: occluded instances render black (depth)")
        elif medL < DARK_L:
            rec.update(mode="black", colour=BLACK, why=f"dark part (concept median L={medL:.0f})")
        else:
            mm = m & (lab[..., 0] >= 32)
            a = _tones(lab[mm])
            frac = [float((a == k).mean()) for k in range(2)]
            cols = [R1.snap(np.median(rgb[mm][a == k], 0)) for k in range(2)]
            first = int(np.argmax(frac))
            ring = ndimage.binary_dilation(reg, iterations=near_px) & ~reg
            gof = {i: k for k, v in groups.items() for i in v}
            nbrs = {gof[v - 1] for v in np.unique(src[ring]) if v > 0} - {g}
            nbrs = {g2 for g2 in nbrs if g2 in mat_col}
            pick, why = first, f"majority tone ({frac[first]:.2f})"
            for g2 in nbrs:
                c2 = mat_col[g2]
                if c2 != BLACK and _de(cols[first], c2) < 20 and cols[1 - first] != cols[first]:
                    pick, why = 1 - first, f"second tone: majority collides with adjacent {g2}"
            rec.update(mode="solid", colour=cols[pick], tones=[R1.hexs(c) for c in cols], tone_frac=frac, why=why)
        mat_col[g] = rec["colour"]
        for i in idx:
            mats[i] = dict(rec, name=parts[i]["name"], role=parts[i]["role"], depth=parts[i]["depth"])
    # cups (rule 3) and fins (rule 5)
    names = [p["name"] for p in parts]
    for i, p in enumerate(parts):
        mats[i]["cup"] = "cup" in p["material"].lower() or "cup" in p["name"].lower()
        body_occ = [o for o in p["occluded_by"] if parts[names.index(o)]["role"] == "body"
                    and parts[names.index(o)]["material"] == p["material"]]      # a flap of the body, behind it
        touches_bg = bool((ndimage.binary_dilation(src == i + 1, iterations=2) & (src == 0)).any())
        mats[i]["fin"] = bool(p["role"] in ("body", "limb") and body_occ and touches_bg and p["depth"] == "front")
    return mats


def _de(a, b):
    la, lb = to_lab(np.array([a, b], float))
    return float(np.sqrt(((la - lb) ** 2).sum()))


# ------------------------------------------------------------------ render

def render(ext, base_of, mats, legend, N, keep="left", opts=None):
    o = {"contrast_below": 32, "cup_edge_min": 32, "fin_depth": 2, "clean": True}
    o.update(opts or {})
    base_of = np.array(base_of)
    pr = np.array([0] + [p["priority"] for p in legend["parts"]], float)
    lab = np.pad(ext, 1)                          # outset: the outline goes around the (N-2) map
    base = base_of[lab]
    H = W = N
    m_of = lambda e: mats[base_of[e] - 1]
    blk = np.zeros((H, W), bool)
    for i, d in enumerate(mats):
        if d["mode"] == "black":
            blk |= base == i + 1
    fg = lab > 0
    inb = lambda y, x: 0 <= y < H and 0 <= x < W

    def nl(a, b):
        """Does the boundary between instance labels a and b need a line?"""
        A, B = base_of[a], base_of[b]
        ma, mb = mats[A - 1], mats[B - 1]
        if ma["mode"] == "black" or mb["mode"] == "black":
            return False
        if A == B:
            return a != b                                   # two arms of the same part: always (rule 4)
        if ma.get("floor") or mb.get("floor"):
            return True                                     # floored features (eyes) are always ringed in ink
        if ma["cup"] or mb["cup"]:
            return N >= o["cup_edge_min"]                   # cup edge where the size allows (rule 3)
        if ma["role"] == "limb" and mb["role"] == "limb":
            return True                                     # front arms: always (rule 4)
        if ma["colour"] == mb["colour"]:
            return True                                     # same colour: the line is the only boundary
        if N < o["contrast_below"] and _de(ma["colour"], mb["colour"]) >= 30:
            return False                                    # v1 finding, non-limb pairs only
        return True

    ink = ~fg & ndimage.binary_dilation(fg, structure=ndimage.generate_binary_structure(2, 1))
    area = np.bincount(lab.ravel(), minlength=lab.max() + 1)
    for y in range(H):
        for x in range(W):
            a = lab[y, x]
            if a == 0 or ink[y, x] or blk[y, x]:
                continue
            for dy, dx in FOUR:
                yy, xx = y + dy, x + dx
                if not inb(yy, xx):
                    continue
                b = lab[yy, xx]
                if b == 0 or b == a or not nl(a, b):
                    continue
                # ink on the larger region's side; cups always keep their pixels (edge on the arm side)
                ka = (not m_of(a)["cup"], area[a], -pr[base_of[a]], a)
                kb = (not m_of(b)["cup"], area[b], -pr[base_of[b]], b)
                if ka > kb:
                    ink[y, x] = True
                    break

    def leaks(y, x):
        a = lab[y, x]
        for dy, dx in FOUR:
            yy, xx = y + dy, x + dx
            if not inb(yy, xx) or ink[yy, xx] or blk[yy, xx]:
                continue
            b = lab[yy, xx]
            if (a == 0) != (b == 0):
                return True
            if a > 0 and b > 0 and a != b and nl(a, b):
                return True
        return False

    nbs = lambda y, x, m: sum(1 for dy, dx in EIGHT if inb(y + dy, x + dx) and m[y + dy, x + dx])
    if o["clean"]:                      # rule 6: thin lines, no clumps
        changed = True
        while changed:
            changed = False
            for c, y, x in sorted(((nbs(y, x, ink | blk), y, x) for y, x in zip(*np.nonzero(ink))), reverse=True):
                if c <= 2 or nbs(y, x, ink | blk) <= 2:
                    continue
                own = lab[y, x]
                opts_ = [own]
                if own == 0:
                    nb = [lab[y + dy, x + dx] for dy, dx in FOUR if inb(y + dy, x + dx) and lab[y + dy, x + dx] > 0
                          and not ink[y + dy, x + dx] and not blk[y + dy, x + dx]]
                    opts_ += sorted(set(nb) - {own}, key=nb.count, reverse=True)
                ink[y, x] = False
                for lv in opts_:
                    lab[y, x] = lv
                    if not leaks(y, x):
                        changed = True
                        break
                else:
                    lab[y, x] = own
                    ink[y, x] = True
    base = base_of[lab]
    # rule 5: thick depth line under the fins, taken from the background
    depth = np.zeros((H, W), bool)
    for i, d in enumerate(mats):
        if not d["fin"]:
            continue
        fin = (base == i + 1) & ~ink
        for y, x in zip(*np.nonzero(fin)):
            yy = y + 1
            while yy < H and ink[yy, x]:
                yy += 1                                       # skip the existing outline under the fin
            if yy == y + 1 or yy >= H:
                continue                                      # the fin sits on another part: no gap here
            for k in range(o["fin_depth"] - (yy - y - 1)):
                if yy + k < H and lab[yy + k, x] == 0 and not ink[yy + k, x]:
                    depth[yy + k, x] = True
    out = np.zeros((H, W, 4), np.uint8)
    for i, d in enumerate(mats):
        m = (base == i + 1) & ~ink
        out[m] = (*d["colour"], 255)
    out[ink | depth] = (*BLACK, 255)
    for arr in (out, lab, ink, depth):              # exact symmetry of the final pixels
        half = arr[:, :W // 2].copy() if keep == "left" else arr[:, W - W // 2:][:, ::-1].copy()
        arr[:, :W // 2] = half
        arr[:, W - W // 2:] = half[:, ::-1]
    grid_ch = ".abcdefghijklmnopqrstuvwxyz"
    info = {"size": N, "ink_px": int(ink.sum()), "fin_depth_px": int(depth.sum()), "black_fill_px": int((blk & ~ink).sum()),
            "clumps": R1.clumps(ink, blk | depth),
            "grid": ["".join("#" if ink[y, x] or depth[y, x] else grid_ch[base_of[lab[y, x]]] for x in range(W)) for y in range(H)]}
    return Image.fromarray(out, "RGBA"), info


# ------------------------------------------------------------------ run + sheets

def run(sizes=SIZES) -> dict:
    rdir = V2 / "regions"
    legend = json.loads((rdir / "legend.json").read_text())
    src_raw = np.load(rdir / "labels.npy")
    spec = json.loads(LV.SPEC.read_text())
    ctx = LV.prepare(src_raw, legend, spec, sizes)
    src = ctx["src"]
    concept = np.array(Image.open(R1.CONCEPT).convert("RGBA"))
    # the concept, shifted/padded exactly like the source (centre_axis + square) so colours sample the same pixels
    concept = _align_concept(concept, src_raw, src, ctx)
    mats = materials(src, concept, legend)
    for i, q in enumerate(ctx["specs"]):
        mats[i]["floor"] = q["mode"] == "floor"
    ldir, odir = V2 / "ladder", V2 / "render"
    ldir.mkdir(parents=True, exist_ok=True)
    odir.mkdir(parents=True, exist_ok=True)
    L.colorize(src, legend).save(ldir / "source_symmetric.png")
    rungs = {}
    # floored features (eyes, nozzle: ladder v3 'floor' parts) must survive the tile rule: their colours, as they
    # are after consolidation, weigh SALIENT_W x in the tile damage
    floor_cols = [tuple(mats[i]["colour"]) for i, q in enumerate(ctx["specs"])
                  if q["mode"] == "floor" and mats[i]["mode"] != "black"]
    res = {"inputs": {"regions": str(rdir / "labels.npy"), "legend": str(rdir / "legend.json"), "concept": str(R1.CONCEPT),
                      "spec": str(LV.SPEC)},
           "materials": [{k: (R1.hexs(v) if k == "colour" else v) for k, v in d.items()} for d in mats], "renders": {}}
    checks = {"tile_colors": {}, "orphans": {}, "holes": {}, "outline": {}, "colors": {"max_total": 4, "gate": False}}
    for N in sizes:
        r = LV.rung(ctx, legend, N)
        rungs[N] = r
        L.colorize(r["base"], legend).save(ldir / f"rung_{N}.png")
        im, info = render(r["ext"], ctx["base_of"], mats, legend, N, ctx["sym"]["keep"])
        cons, merged = R1.consolidate(im)
        salient = [R1.imaging.rgb(merged.get(R1.hexs(c), R1.hexs(c))) for c in floor_cols]
        legal, tinfo = R1.tile_reduce(cons, salient=salient, salient_w=SALIENT_W, symmetric=True)
        tinfo["merged"] = merged
        im.save(odir / f"render_{N}_free.png")
        legal.save(odir / f"render_{N}.png")
        a = np.array(legal)
        info.update(tile=tinfo, checks=imaging.run_checks(legal, checks),
                    colours=sorted(R1.hexs(c) for c in np.unique(a[a[..., 3] > 0][:, :3], axis=0)),
                    mirror_exact=bool((a == a[:, ::-1]).all()))
        res["renders"][str(N)] = info
    res["ladder"] = LV.summary(ctx, rungs)
    (V2 / "results.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def _align_concept(concept, src_raw, src, ctx):
    a2 = int(round(2 * ctx["sym"]["axis_px"]))
    w = concept.shape[1]
    note = ctx["sym"]["recentre"] or ""
    c = concept.copy()
    for step in note.split(";"):
        if step.startswith("shift"):
            c = np.roll(c, int(step.split()[1]), axis=1)
        elif step.startswith("pad left"):
            c = np.pad(c, ((0, 0), (1, 0), (0, 0)))
        elif step.startswith("pad right"):
            c = np.pad(c, ((0, 0), (0, 1), (0, 0)))
    h, w = c.shape[:2]
    S = src.shape[0]
    if S > w:
        d = (S - w) // 2
        c = np.pad(c, ((0, 0), (d, S - w - d), (0, 0)))
    if S > h:
        c = np.pad(c, ((0, S - h), (0, 0), (0, 0)))
    return c


def sheets(sizes=SIZES, zoom=6):
    """Same-zoom comparison: v2 renders, v1 renders, authentic refs."""
    refs = [(n, Image.open(R1.REFS / n / "subject.png")) for n in ("telly", "met_walk", "pipi")]
    rows = [("v2 render", [(f"{N}", Image.open(V2 / "render" / f"render_{N}.png")) for N in sizes]),
            ("v1 render", [(f"{N}", Image.open(REG / "render" / f"render_{N}.png")) for N in sizes]),
            ("refs", refs)]
    R1.sheet(rows, V2 / "sheet_same_zoom.png", box=zoom * 64, same_zoom=zoom)


if __name__ == "__main__":
    r = run()
    sheets()
    for d in r["materials"]:
        print(f"{d['name']:24s} {d['mode']:6s} {d['colour']} cup={d['cup']} fin={d['fin']} {d['why']}")
    lad = r["ladder"]
    print("sym", lad["symmetry"])
    for N, i in r["renders"].items():
        g = lad["rungs"][N]
        print(N, {k: i[k] for k in ("ink_px", "fin_depth_px", "clumps", "mirror_exact")}, i["tile"]["changed_px"],
              i["colours"], "| exact", g["exact_fraction"], "k", {k: v["k"] for k, v in g["k"].items()}, g["floors"])
