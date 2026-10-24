"""Render v3: Johnny's distilled rules after v2 (PIPELINE_RULES, 2026-10-23). No model calls.

Inputs: the v2 region map + legend (reused, no new paint), ladder v3 rungs (ladder_v3: target palette, explicit
priority, face template, arm gaps/notches/cup tips), the NES palette. The concept is NOT sampled for colour any more:
the palette was decided up front (ladder_v3.PALETTE / ASSIGN).

Changes from render_v2, rule by rule:
 1. Palette first: every part's colour is its group's colour (black + 3 NES colours, one hue ramp: 0F/16/26/36).
    No tones, no second-tone collision logic, no consolidation needed.
 2. No patterning (as v2).
 4. Eyes: the face template parts (eyes, visor, face ring) and the floored nozzle never give up a pixel to a line;
    the line goes on the other side. Eyes vs visor need no line (the visor is the black moat).
 6. Line sacrifice by depth: where two regions need a separating line, the ink goes on the pixel of the region
    BEHIND (ladder_v3.behind_fn: legend occluded_by, then depth, then layer order), not the larger one (v2).
 Kept from v2: outline around the (N-2) map, lines between same-colour neighbours at every size, the v1 finding
 (no line between strongly contrasting colours, Lab dE >= 30, below 32 px), cup edges at >= 32, the 2 px depth line
 under the fins taken from the background, the no-clump clean-up, exact mirror symmetry, the NES tile rule
 (<= 3 colours per 8x8 tile incl. black, render.tile_reduce) with the eye colour weighted.

uv run python -m megaloman.render_v3
Writes workspace/experiments/regions/v3/{palette.json, ladder/, render/, results.json, sheet_same_zoom.png}.
"""
from __future__ import annotations

import json

import numpy as np
from PIL import Image
from scipy import ndimage

from . import imaging, ladder as L, ladder_v2 as LV, ladder_v3 as L3, render as R1
from .render_v2 import _de

ROOT = R1.ROOT
REG = R1.REG
V3 = REG / "v3"
SIZES = (64, 32, 24, 20, 16)
BLACK = (0, 0, 0)
FOUR = R1.FOUR
EIGHT = R1.EIGHT
PROTECTED = ("eyes", "visor", "face_housing", "nozzle")   # template / floored parts: never give a pixel to a line
SALIENT_W = 8.0


def materials(legend, src) -> list[dict]:
    parts = legend["parts"]
    names = [p["name"] for p in parts]
    mats = []
    for i, p in enumerate(parts):
        g = p["group"]
        col = BLACK if g == "black" else imaging.rgb(L3.PALETTE[g][1])
        body_occ = [o for o in p["occluded_by"] if parts[names.index(o)]["role"] == "body"
                    and parts[names.index(o)]["material"] == p["material"]]
        touches_bg = bool((ndimage.binary_dilation(src == i + 1, iterations=2) & (src == 0)).any())
        mats.append({"name": p["name"], "role": p["role"], "depth": p["depth"], "group": g,
                     "mode": "black" if g == "black" else "solid", "colour": tuple(col),
                     "nes": L3.PALETTE[g][0], "priority": p["priority"],
                     "cup": "cup" in p["material"].lower() or "cup" in p["name"].lower(),
                     "fin": bool(p["role"] in ("body", "limb") and body_occ and touches_bg and p["depth"] == "front"),
                     "protected": p["name"] in PROTECTED})
    return mats


def render(ext, base_of, mats, legend, N, keep="left", opts=None):
    o = {"contrast_below": 32, "cup_edge_min": 32, "fin_depth": 2, "clean": True, "always_line": set()}
    o.update(opts or {})
    base_of = np.array(base_of)
    behind = L3.behind_fn(legend)
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
            return a != b                                   # two arms of the same part
        if ma["cup"] or mb["cup"]:
            if ma["colour"] == mb["colour"]:
                return True                                 # cup next to a same-coloured accent (nozzle, pod)
            return N >= o["cup_edge_min"]
        if ma["colour"] == mb["colour"]:
            return True                                     # same colour: the line is the only boundary
        if {ma["name"], mb["name"]} & o["always_line"] and {ma["role"], mb["role"]} <= {"body"}:
            return True                                     # silhouette-carrying body pairs (head | fins)
        if N < o["contrast_below"] and _de(ma["colour"], mb["colour"]) >= 30:
            return False                                    # v1 finding
        return True

    def gives(a, b) -> bool:
        """Rule 6: does the pixel of region a become the line between a and b?"""
        A, B = base_of[a], base_of[b]
        pa, pb = mats[A - 1]["protected"], mats[B - 1]["protected"]
        if pa != pb:
            return pb                                       # protected parts keep their pixels
        if mats[A - 1]["cup"] != mats[B - 1]["cup"]:
            return mats[B - 1]["cup"]                       # cups sit on the arm: the arm gives
        if A == B:
            return area[a] > area[b] or (area[a] == area[b] and a > b)
        return behind(A, B)

    ink = ~fg & ndimage.binary_dilation(fg, structure=ndimage.generate_binary_structure(2, 1))
    area = np.bincount(lab.ravel(), minlength=lab.max() + 1)
    sides = {}
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
                if gives(a, b):
                    ink[y, x] = True
                    k = f"{mats[base_of[a] - 1]['name']}|{mats[base_of[b] - 1]['name']}"
                    sides[k] = sides.get(k, 0) + 1
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
    arm_like = lambda e: mats[base_of[e] - 1]["role"] == "limb" or mats[base_of[e] - 1]["cup"]
    if o["clean"]:                      # thin lines, no clumps (v1/v2)
        changed = True
        while changed:
            changed = False
            for c, y, x in sorted(((nbs(y, x, ink | blk), y, x) for y, x in zip(*np.nonzero(ink))), reverse=True):
                if c <= 2 or nbs(y, x, ink | blk) <= 2:
                    continue
                own = lab[y, x]
                opts_ = [own]
                if own == 0:          # rule 5: gaps cut between arms (ladder_v3) are never filled back in
                    nb = [lab[y + dy, x + dx] for dy, dx in FOUR if inb(y + dy, x + dx) and lab[y + dy, x + dx] > 0
                          and not ink[y + dy, x + dx] and not blk[y + dy, x + dx]
                          and not arm_like(lab[y + dy, x + dx])]
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
    # depth line under the fins, taken from the background (v2 rule 5)
    depth = np.zeros((H, W), bool)
    for i, d in enumerate(mats):
        if not d["fin"]:
            continue
        fin = (base == i + 1) & ~ink
        for y, x in zip(*np.nonzero(fin)):
            yy = y + 1
            while yy < H and ink[yy, x]:
                yy += 1
            if yy == y + 1 or yy >= H:
                continue
            for k in range(o["fin_depth"] - (yy - y - 1)):
                if yy + k < H and lab[yy + k, x] == 0 and not ink[yy + k, x]:
                    depth[yy + k, x] = True
    out = np.zeros((H, W, 4), np.uint8)
    for i, d in enumerate(mats):
        m = (base == i + 1) & ~ink
        out[m] = (*d["colour"], 255)
    out[ink | depth] = (*BLACK, 255)
    # background enclosed between arms (a cut gap that closed up) reads as shadow: black, not a hole
    enclosed = ndimage.binary_fill_holes(out[..., 3] > 0) & (out[..., 3] == 0)
    out[enclosed] = (*BLACK, 255)
    ink |= enclosed
    for arr in (out, lab, ink, depth):              # exact symmetry of the final pixels
        half = arr[:, :W // 2].copy() if keep == "left" else arr[:, W - W // 2:][:, ::-1].copy()
        arr[:, :W // 2] = half
        arr[:, W - W // 2:] = half[:, ::-1]
    grid_ch = ".abcdefghijklmnopqrstuvwxyz"
    info = {"size": N, "ink_px": int(ink.sum()), "fin_depth_px": int(depth.sum()), "black_fill_px": int((blk & ~ink).sum()),
            "clumps": R1.clumps(ink, blk | depth), "line_side": sides,
            "grid": ["".join("#" if ink[y, x] or depth[y, x] else grid_ch[base_of[lab[y, x]]] for x in range(W)) for y in range(H)]}
    return Image.fromarray(out, "RGBA"), info


def eye_check(im: Image.Image, grid: list[str]) -> dict:
    """Rule 4 check on the final pixels: two eye rectangles, filled, each with a black margin on every side."""
    a = np.array(im)
    g = np.array([list(r) for r in grid])
    eyes = g == "e"
    lab, k = ndimage.label(eyes)
    out = {"components": int(k), "eyes": []}
    for sl in ndimage.find_objects(lab):
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        filled = bool(eyes[y0:y1, x0:x1].all())
        ring = [(y, x) for y in range(y0 - 1, y1 + 1) for x in range(x0 - 1, x1 + 1)
                if not (y0 <= y < y1 and x0 <= x < x1)]
        moat = all(0 <= y < a.shape[0] and 0 <= x < a.shape[1] and a[y, x, 3] > 0 and not a[y, x, :3].any()
                   for y, x in ring)
        col = R1.hexs(a[y0, x0, :3])
        out["eyes"].append({"box": [int(x1 - x0), int(y1 - y0)], "filled": filled, "black_margin": moat,
                            "colour": col})
    out["ok"] = k == 2 and all(e["filled"] and e["black_margin"] for e in out["eyes"])
    return out


def run(sizes=SIZES, opts=None, out=None) -> dict:
    rdir = REG / "v2" / "regions"
    legend_v2 = json.loads((rdir / "legend.json").read_text())
    src_raw = np.load(rdir / "labels.npy")
    spec = json.loads(LV.SPEC.read_text())
    ctx = L3.prepare(src_raw, legend_v2, spec, sizes)
    leg, src = ctx["legend"], ctx["src"]
    D = out or V3
    D.mkdir(parents=True, exist_ok=True)
    pal = L3.palette_record(leg)
    behind = L3.behind_fn(leg)
    names = [p["name"] for p in leg["parts"]]
    pal["behind"] = {f"{a}|{b}": a if behind(i + 1, j + 1) else b
                     for i, a in enumerate(names) for j, b in enumerate(names) if i < j}
    (D / "palette.json").write_text(json.dumps(pal, indent=1))
    (D / "legend_v3.json").write_text(json.dumps(leg, indent=1))
    mats = materials(leg, src)
    ldir, odir = D / "ladder", D / "render"
    ldir.mkdir(parents=True, exist_ok=True)
    odir.mkdir(parents=True, exist_ok=True)
    L.colorize(src, leg).save(ldir / "source_symmetric.png")
    light = imaging.rgb(L3.PALETTE["light"][1])
    res = {"opts": {k: sorted(v) if isinstance(v, set) else v for k, v in (opts or {}).items()}, "inputs": {"regions": str(rdir / "labels.npy"), "legend": str(rdir / "legend.json"), "spec": str(LV.SPEC)},
           "palette": pal, "materials": [{k: (R1.hexs(v) if k == "colour" else v) for k, v in d.items()} for d in mats],
           "renders": {}}
    checks = {"tile_colors": {}, "orphans": {}, "holes": {}, "outline": {}, "colors": {"max_total": 4, "gate": True}}
    rungs = {}
    for N in sizes:
        r = L3.rung(ctx, N)
        rungs[N] = r
        L.colorize(r["base"], leg).save(ldir / f"rung_{N}.png")
        im, info = render(r["ext"], ctx["base_of"], mats, leg, N, ctx["sym"]["keep"], opts)
        legal, tinfo = R1.tile_reduce(im, salient=[light], salient_w=SALIENT_W, symmetric=True)
        im.save(odir / f"render_{N}_free.png")
        legal.save(odir / f"render_{N}.png")
        a = np.array(legal)
        info.update(tile=tinfo, checks=imaging.run_checks(legal, checks), eyes=eye_check(legal, info["grid"]),
                    colours=sorted(R1.hexs(c) for c in np.unique(a[a[..., 3] > 0][:, :3], axis=0)),
                    mirror_exact=bool((a == a[:, ::-1]).all()))
        res["renders"][str(N)] = info
    res["ladder"] = L3.summary(ctx, rungs)
    (D / "results.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def sheets(sizes=SIZES, zoom=6):
    refs = [(n, Image.open(R1.REFS / n / "subject.png")) for n in ("telly", "met_walk")]
    rows = [("v3 render", [(f"{N}", Image.open(V3 / "render" / f"render_{N}.png")) for N in sizes]),
            ("v2 render", [(f"{N}", Image.open(REG / "v2" / "render" / f"render_{N}.png")) for N in sizes]),
            ("v1 render", [(f"{N}", Image.open(REG / "render" / f"render_{N}.png")) for N in sizes]),
            ("refs", refs)]
    R1.sheet(rows, V3 / "sheet_same_zoom.png", box=zoom * 64, same_zoom=zoom)


if __name__ == "__main__":
    r = run()
    sheets()
    for d in r["materials"]:
        print(f"{d['name']:24s} {d['group']:6s} {d['colour']} prio={d['priority']} prot={d['protected']}")
    for N, i in r["renders"].items():
        g = r["ladder"]["rungs"][N]
        print(N, {k: i[k] for k in ("ink_px", "fin_depth_px", "clumps", "mirror_exact")}, "tile", i["tile"]["changed_px"],
              i["colours"], "eyes ok", i["eyes"]["ok"], [e["box"] for e in i["eyes"]["eyes"]], "| exact", g["exact_fraction"])
        print("\n".join(i["grid"]))
