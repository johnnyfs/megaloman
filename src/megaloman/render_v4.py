"""Render v4: Johnny's review of v3 (2026-10-23). No model calls.

Inputs: the v2 region map + legend (reused), ladder v4 rungs (ladder_v4: ring top at every size, slit eyes at
<= 20, fins paying their lines from the background, the inner-arm curl at <= 20), the v3 palette (unchanged).

The drawing machinery is render_v3.render (copied here as `render`, one change: its no-clump clean-up never grows a
protected template part); otherwise only its options differ:
 - SEPARATION LINES AT EVERY SIZE (blessed rule): `contrast_below = 0` turns off the v1 finding (no line between
   strongly contrasting colours below 32 px). Every pair of distinct non-black regions now gets a 1 px line at every
   size: fins | head, ring | head, ring | fins, ring | pods, nozzle | arms, arm | arm.
 - Where the pixels come from is unchanged from v3 (rule 6 + priority): protected template parts (eyes, visor,
   ring, nozzle) never give; otherwise the region BEHIND gives (legend occluded_by, depth, layer order).
 - Body specks (speck_below = 32): a head or fin pixel left with no 4-neighbour of its own region once the lines are
   in joins the line (the head's corners beside the ring's top line at 20).
 - Cups on the arms keep v3's edge threshold (edged at >= 32 only). Below 32 the cups are the 1-3 px curl hint inside
   the arm, which Johnny praised at 20; an edge there erases the hook (see `alt_cupline/`, which edges them too).

uv run python -m megaloman.render_v4
Writes workspace/experiments/regions/v4/{palette.json, legend_v4.json, ladder/, render/, results.json,
sheet_same_zoom.png} and the variant v4/alt_cupline/.
"""
from __future__ import annotations

import json

import numpy as np
from PIL import Image

from scipy import ndimage

from . import imaging, ladder as L, ladder_v2 as LV, ladder_v4 as L4, render as R1, render_v3 as R3

REG = R1.REG
V4 = REG / "v4"
SIZES = (64, 32, 24, 20, 16)
OPTS = {"contrast_below": 0, "speck_below": 32}     # lines between every pair of distinct regions, every size
ALT_CUPLINE = {**OPTS, "cup_edge_min": 0}
BLACK, FOUR, EIGHT = R3.BLACK, R3.FOUR, R3.EIGHT


def render(ext, base_of, mats, legend, N, keep="left", opts=None):
    """render_v3.render with one change: the no-clump clean-up never fills a background ink pixel with a protected
    template part (eyes, visor, ring, nozzle). Templates are exact; v3's clean-up could grow a 1 px bump on the ring
    at 16 (it had no ring row there in v3, so it never showed)."""
    o = {"contrast_below": 32, "cup_edge_min": 32, "fin_depth": 2, "clean": True, "always_line": set(),
         "speck_below": 0}
    o.update(opts or {})
    base_of = np.array(base_of)
    behind = L4.behind_fn(legend)
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
        if N < o["contrast_below"] and R3._de(ma["colour"], mb["colour"]) >= 30:
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
                          and not arm_like(lab[y + dy, x + dx])
                          and not mats[base_of[lab[y + dy, x + dx]] - 1]["protected"]]   # v4: templates are exact
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
    # v4: body specks. Below speck_below, a pixel of a body part (head, fins) left with no 4-neighbour of its own
    # region once the lines are in (e.g. the head's corner beside the ring's top line) is a 1 px speck of colour;
    # it joins the line. Cups, arms and protected template parts are exempt (their 1 px pieces are deliberate).
    specks = 0
    if N < o["speck_below"]:
        for y, x in zip(*np.nonzero((lab > 0) & ~ink & ~blk)):
            d = mats[base_of[lab[y, x]] - 1]
            if d["role"] != "body" or d["protected"] or d["cup"]:
                continue
            if not any(inb(y + dy, x + dx) and lab[y + dy, x + dx] == lab[y, x] and not ink[y + dy, x + dx]
                       for dy, dx in FOUR):
                ink[y, x] = True
                specks += 1
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
    info = {"size": N, "specks_to_line": specks, "ink_px": int(ink.sum()), "fin_depth_px": int(depth.sum()), "black_fill_px": int((blk & ~ink).sum()),
            "clumps": R1.clumps(ink, blk | depth), "line_side": sides,
            "grid": ["".join("#" if ink[y, x] or depth[y, x] else grid_ch[base_of[lab[y, x]]] for x in range(W)) for y in range(H)]}
    return Image.fromarray(out, "RGBA"), info


def run(sizes=SIZES, opts=None, out=None) -> dict:
    opts = OPTS if opts is None else opts
    rdir = REG / "v2" / "regions"
    legend_v2 = json.loads((rdir / "legend.json").read_text())
    src_raw = np.load(rdir / "labels.npy")
    spec = json.loads(LV.SPEC.read_text())
    ctx = L4.prepare(src_raw, legend_v2, spec, sizes)
    leg, src = ctx["legend"], ctx["src"]
    D = out or V4
    D.mkdir(parents=True, exist_ok=True)
    pal = L4.palette_record(leg)
    behind = L4.behind_fn(leg)
    names = [p["name"] for p in leg["parts"]]
    pal["behind"] = {f"{a}|{b}": a if behind(i + 1, j + 1) else b
                     for i, a in enumerate(names) for j, b in enumerate(names) if i < j}
    (D / "palette.json").write_text(json.dumps(pal, indent=1))
    (D / "legend_v4.json").write_text(json.dumps(leg, indent=1))
    mats = R3.materials(leg, src)
    ldir, odir = D / "ladder", D / "render"
    ldir.mkdir(parents=True, exist_ok=True)
    odir.mkdir(parents=True, exist_ok=True)
    L.colorize(src, leg).save(ldir / "source_symmetric.png")
    light = imaging.rgb(L4.PALETTE["light"][1])
    res = {"opts": opts, "inputs": {"regions": str(rdir / "labels.npy"), "legend": str(rdir / "legend.json"),
                                    "spec": str(LV.SPEC)},
           "palette": pal, "materials": [{k: (R1.hexs(v) if k == "colour" else v) for k, v in d.items()} for d in mats],
           "renders": {}}
    checks = {"tile_colors": {}, "orphans": {}, "holes": {}, "outline": {}, "colors": {"max_total": 4, "gate": True}}
    rungs = {}
    for N in sizes:
        r = L4.rung(ctx, N)
        rungs[N] = r
        L.colorize(r["base"], leg).save(ldir / f"rung_{N}.png")
        im, info = render(r["ext"], ctx["base_of"], mats, leg, N, ctx["sym"]["keep"], opts)
        legal, tinfo = R1.tile_reduce(im, salient=[light], salient_w=R3.SALIENT_W, symmetric=True)
        im.save(odir / f"render_{N}_free.png")
        legal.save(odir / f"render_{N}.png")
        a = np.array(legal)
        info.update(tile=tinfo, checks=imaging.run_checks(legal, checks), eyes=R3.eye_check(legal, info["grid"]),
                    colours=sorted(R1.hexs(c) for c in np.unique(a[a[..., 3] > 0][:, :3], axis=0)),
                    mirror_exact=bool((a == a[:, ::-1]).all()), separation=separation_check(info["grid"], leg))
        res["renders"][str(N)] = info
    res["ladder"] = L4.summary(ctx, rungs)
    (D / "results.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def separation_check(grid: list[str], legend: dict) -> dict:
    """The blessed rule on the final pixels: no two distinct non-black regions touch (4-neighbours) without a line,
    except the cups on their arm below 32 (the curl hint). -> {'touching': {pair: n}, 'ok': bool}."""
    parts = legend["parts"]
    ch = ".abcdefghijklmnopqrstuvwxyz"
    black = {ch[i + 1] for i, p in enumerate(parts) if p["group"] == "black"}
    name = {ch[i + 1]: p["name"] for i, p in enumerate(parts)}
    g = [list(r) for r in grid]
    H, W = len(g), len(g[0])
    touch = {}
    for y in range(H):
        for x in range(W):
            a = g[y][x]
            if a in ".#" or a in black:
                continue
            for dy, dx in ((1, 0), (0, 1)):
                yy, xx = y + dy, x + dx
                if yy >= H or xx >= W:
                    continue
                b = g[yy][xx]
                if b in ".#" or b in black or b == a:
                    continue
                k = "|".join(sorted((name[a], name[b])))
                touch[k] = touch.get(k, 0) + 1
    cupish = {k for k in touch if "suction_cups_front" in k and "tentacles_front" in k}
    return {"touching": touch, "ok": not (set(touch) - cupish), "cup_on_arm_unlined": sorted(cupish)}


def sheets(sizes=SIZES, zoom=6):
    refs = [(n, Image.open(R1.REFS / n / "subject.png")) for n in ("telly", "met_walk")]
    rows = [("v4 render", [(f"{N}", Image.open(V4 / "render" / f"render_{N}.png")) for N in sizes]),
            ("v4 alt: cups edged", [(f"{N}", Image.open(V4 / "alt_cupline" / "render" / f"render_{N}.png"))
                                    for N in sizes]),
            ("v3 render", [(f"{N}", Image.open(REG / "v3" / "render" / f"render_{N}.png")) for N in sizes]),
            ("refs", refs)]
    R1.sheet(rows, V4 / "sheet_same_zoom.png", box=zoom * 64, same_zoom=zoom)


if __name__ == "__main__":
    r = run()
    run(opts=ALT_CUPLINE, out=V4 / "alt_cupline")
    sheets()
    for N, i in r["renders"].items():
        g = r["ladder"]["rungs"][N]
        print(N, {k: i[k] for k in ("ink_px", "fin_depth_px", "clumps", "mirror_exact")}, "tile", i["tile"]["changed_px"],
              i["colours"], "eyes ok", i["eyes"]["ok"], [e["box"] for e in i["eyes"]["eyes"]], "| exact",
              g["exact_fraction"], "| sep", i["separation"]["ok"], i["separation"]["touching"])
        print(g["log"]["face"], g["log"]["fin_pay"], g["log"].get("curl"))
        print("\n".join(i["grid"]))
