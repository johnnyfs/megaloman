"""regions.render: a reduced region map + the shaded concept -> stylized NES pixel art.

No model calls. Inputs: the full-size region map (legend colours), legend.json, the shaded concept it was painted
from. The label map at each size comes from `ladder.ladder` in 'source' mode (the Milestone 2 method).

Style rules (Johnny's), and how each is implemented here:
 1. Thin black outline: 1-px lines. Part lines go on the larger part's side (one consistent line per pair); a
    clean-up pass removes ink pixels with > 2 ink neighbours (8-conn) where that opens no leak, and merges doubled
    outer lines (two limbs 2 px apart) into one shared line. 'outset' (default) draws the outer outline *around* an
    (N-2)^2 label map, so it costs no part pixels; 'inset' draws it on the N^2 map's edge pixels.
 2. Lines between parts by default, except: next to black parts (they already read as ink); between a marking and
    its host (a repeated feature/detail part whose boundary is mostly one body/limb part: suction cups on
    tentacles); patterns inside a part (mantle spots) are never outlined; below `contrast_below` px, pairs whose
    colours already contrast (Lab dE >= 30) get no line, since there a line costs a whole row of a part.
 3. Depth: limb pixels next to ink thicken to ink where the concept is dark there (occlusion shadow), if the limb
    keeps >= 2 px of colour beside it; doubled lines in shadow are kept as a depth cue.
 4. Black parts: the visor (explicit), and any part whose concept median is near-black (the tentacle joints).
 5. Colours: per part, 2-means in Lab over the concept inside the eroded region. If the two tones differ in chroma,
    the part is *patterned* (pink/cream: mantle spots, tentacle stripes, fin undersides): the tone that carries the
    part's edge is the base, and each source mark of the other tone is drawn if it covers >= 0.6 px (limb marks
    need >= 2 px). Else it is *shaded*. Snapped to the NES palette, near-duplicates merged, then a per-8x8-tile
    reduction to <= 3 colours (least perceptual damage).
 6. Shading: shaded parts get their darker tone only at sizes >= shade_min, in blobs >= 2 px.
Render-level repairs (borrowed from what the ladder's reshape step will own): small features (eyes, nozzle) keep
>= 1 px and grow to the legend's min_px height inside a black part; a separator part lost at this size (the face
frame between visor and joints) is painted back along the contact.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from . import imaging, ladder
from .vendor.pixelize import to_lab

ROOT = Path(__file__).resolve().parents[2]
REG = ROOT / "workspace/experiments/regions"
CONCEPT = ROOT / "workspace/shared/sources/squid_concept_gpt_front.png"
REFS = ROOT / "workspace/games/megaman/classes/small_enemy/keyframe/benchmarks"
BLACK_PARTS = {"visor"}          # rule 4 (explicit)
INK_L = 32                        # concept pixels darker than this (Lab L) are ink / occlusion
NES_RGB = np.array([imaging.rgb(v) for v in sorted(set(imaging.NES.values()))], float)
NES_LAB = to_lab(NES_RGB)
BLACK = (0, 0, 0)
FOUR = ((1, 0), (-1, 0), (0, 1), (0, -1))
GRID_CH = ".MFfVEptjcn"           # bg + one char per legend part (squid legend order); '#' = ink
EIGHT = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]


def snap(rgb) -> tuple[int, int, int]:
    lab = to_lab(np.asarray(rgb, float)[None])[0]
    return tuple(int(v) for v in NES_RGB[((NES_LAB - lab) ** 2).sum(-1).argmin()])


def hexs(c):
    return "%02X%02X%02X" % tuple(c)


# ------------------------------------------------------------------ materials (rule 5/6 colour analysis)

def materials(src: np.ndarray, concept: np.ndarray, legend: dict) -> list[dict]:
    rgb = concept[..., :3].astype(float)
    lab = to_lab(rgb.reshape(-1, 3)).reshape(rgb.shape)
    solid = concept[..., 3] > 200
    ink = solid & (lab[..., 0] < INK_L)
    out = []
    for i, p in enumerate(legend["parts"]):
        reg = src == i + 1
        m = ndimage.binary_erosion(reg, iterations=4) & solid
        if m.sum() < 50:
            m = reg & solid
        med = np.median(rgb[m], 0)
        medL = float(to_lab(med[None])[0, 0])
        d = {"name": p["name"], "role": p["role"], "median": hexs(snap(med)), "median_L": round(medL, 1)}
        if p["name"] in BLACK_PARTS or medL < INK_L:
            d.update(mode="black", base=BLACK, reason="explicit (rule 4)" if p["name"] in BLACK_PARTS else f"concept median L={medL:.0f}")
        mm = m & ~ink
        px = lab[mm]
        lo, hi = px[px[:, 0] <= np.percentile(px[:, 0], 15)].mean(0), px[px[:, 0] >= np.percentile(px[:, 0], 85)].mean(0)
        cen = np.array([lo, hi])
        for _ in range(30):
            a = ((px[:, None] - cen[None]) ** 2).sum(-1).argmin(1)
            cen = np.array([px[a == k].mean(0) if (a == k).any() else cen[k] for k in range(2)])
        allpx = lab[reg & solid]
        tone = np.zeros(src.shape, np.int8) - 1           # -1 ink/none, 0 dark cluster, 1 light cluster
        tone[reg & solid] = ((allpx[:, None] - cen[None]) ** 2).sum(-1).argmin(1)
        tone[ink] = -1
        cols = [snap(np.median(rgb[mm][a == k], 0)) for k in range(2)]
        frac = [float((a == k).mean()) for k in range(2)]
        chroma = [float(np.hypot(*c[1:])) for c in cen]
        if d.get("mode") == "black":
            d.update(alt=cols[1], alt_mask=tone == 1, alt_frac=frac[1])        # highlight, used only at large sizes
        elif abs(chroma[0] - chroma[1]) > 20 and min(frac) > 0.15 and cols[0] != cols[1]:
            # base = the tone that carries the part's edge band; the other tone forms the marks inside it
            band = reg & ~ndimage.binary_erosion(reg, iterations=12) & (tone >= 0)
            b = int(np.argmax([(tone[band] == k).mean() for k in range(2)]))
            d.update(mode="pattern", base=cols[b], alt=cols[1 - b], alt_mask=tone == 1 - b, alt_frac=frac[1 - b])
        elif cols[0] != cols[1] and cen[1][0] - cen[0][0] > 12 and frac[0] > 0.15:
            d.update(mode="shade", base=cols[1], alt=cols[0], alt_mask=tone == 0, alt_frac=frac[0])
        else:
            d.update(mode="flat", base=snap(med), alt=None, alt_mask=np.zeros_like(reg), alt_frac=0.0)
        out.append(d)
    # markings (rule 2): feature/detail part whose non-background boundary is >= 75% one body/limb part
    for i, d in enumerate(out):
        d["host"] = None
        if d["role"] not in ("feature", "detail"):
            continue
        reg = src == i + 1
        ring = ndimage.binary_dilation(reg, iterations=3) & ~reg
        nb = src[ring]
        nb = nb[nb > 0]
        if len(nb):
            vals, cnt = np.unique(nb, return_counts=True)
            j = vals[cnt.argmax()] - 1
            ncomp = ndimage.label(reg)[1]           # markings are repeated (cups, spots), not one-off features
            if cnt.max() / cnt.sum() >= 0.75 and out[j]["role"] in ("body", "limb") and ncomp >= 2:
                d["host"] = out[j]["name"]
    return out


# ------------------------------------------------------------------ rendering

def _cov(mask: np.ndarray, size: int) -> np.ndarray:
    return np.array(Image.fromarray(mask.astype(np.float32), "F").resize((size, size), Image.BOX))


def need_line(a: dict, b: dict, N: int = 64, contrast_below: int = 0, contrast_de: float = 30) -> bool:
    """Rule 2. Below `contrast_below` px, a pair whose base colours already contrast (Lab dE >= contrast_de)
    gets no line: at that size the line would cost a whole row of a part."""
    if a["mode"] == "black" or b["mode"] == "black":
        return False
    if a["host"] == b["name"] or b["host"] == a["name"]:
        return False
    if N < contrast_below:
        la, lb = to_lab(np.array([a["base"], b["base"]], float))
        if np.sqrt(((la - lb) ** 2).sum()) >= contrast_de:
            return False
    return True


def render(src, concept, legend, mats, N, mode="outset", opts=None):
    o = {"shade_min": 32, "highlight_min": 48, "depth_th": 0.4, "pattern_th": 0.5, "clean": True, "depth": True,
         "patterns": True, "survive": True, "separators": True, "limb_mark_min": 2, "contrast_below": 32, "mirror": True}
    o.update(opts or {})
    n = len(mats)
    M = N - 2 if mode == "outset" else N
    pr = np.array([p["priority"] for p in legend["parts"]], float)
    nl = lambda a, b: need_line(a, b, N, o["contrast_below"])
    lab = ladder.assign(ladder.coverage(src, n, M), pr)
    sym_info = None
    if o["mirror"]:           # PIPELINE_RULES: exact symmetry (source is centred on its axis, so the axis is M/2)
        lab, sym_info = symmetrize(lab, src, legend)
    solid = concept[..., 3] > 200
    L = to_lab(concept[..., :3].astype(float).reshape(-1, 3)).reshape(concept.shape[:2] + (3,))[..., 0]
    dark = _cov(solid & (L < INK_L) & (src > 0), M) / np.maximum(_cov(src > 0, M), 1e-6)
    alt = np.zeros((M, M), bool)
    k2 = (M / src.shape[0]) ** 2
    for i, d in enumerate(mats):
        if d.get("alt") is None:
            continue
        reg = src == i + 1
        am = reg & d["alt_mask"]
        if d["mode"] != "pattern":
            f = _cov(am, M) / np.maximum(_cov(reg & ~(solid & (L < INK_L)), M), 1e-6)
            alt |= (lab == i + 1) & (f >= o["pattern_th"])
            continue
        # pattern marks, one source mark (spot / stripe) at a time: kept if it covers >= 0.6 px at this size;
        # drawn where it covers a pixel, else as one pixel at its centroid
        cl, nc = ndimage.label(ndimage.binary_opening(am, iterations=2))
        for c in range(1, nc + 1):
            cm = cl == c
            if cm.sum() * k2 < 0.6:
                continue
            cells = (lab == i + 1) & (_cov(cm, M) >= o["pattern_th"])
            if not cells.any():
                cy, cx = (np.array(ndimage.center_of_mass(cm)) * M / src.shape[0]).astype(int)
                if lab[cy, cx] == i + 1:
                    cells[cy, cx] = True
            alt |= cells
    fgn = (src > 0).sum()
    # small salient features (eyes, nozzle): feature role, not a marking, not black, < 2% of the sprite's area
    small_feature = lambda i: (mats[i]["role"] == "feature" and not mats[i]["host"] and mats[i]["mode"] != "black"
                               and (src == i + 1).sum() < 0.02 * fgn)
    if o["survive"]:          # salient features (eyes, nozzle): every source component keeps >= 1 px
        for i, d in enumerate(mats):
            if not small_feature(i):
                continue
            cl, nc = ndimage.label(src == i + 1)
            for c in range(1, nc + 1):
                cm = cl == c
                if cm.sum() < 0.002 * (src > 0).sum():
                    continue
                cov = _cov(cm, M)
                if ((lab == i + 1) & (cov > 0)).any():
                    continue
                ys, xs = np.nonzero(cm)
                y0, y1 = int(ys.min() * M / src.shape[0]), int(ys.max() * M / src.shape[0])
                cx = int(round(xs.mean() * M / src.shape[0] - 0.5))
                hh = max(1, min(2, y1 - y0 + 1))          # eyes stay tall: up to 2 px
                cy = int(round(ys.mean() * M / src.shape[0] - hh / 2))
                lab[cy:cy + hh, cx] = i + 1
    sep_log = []
    if o["separators"]:       # restore a lost separator: parts that touch here but not in the source get the part
        sb = _boundary_counts(src)       # that lies between them in the source painted along the contact
        for a, b in sorted(set(_boundary_counts(lab)) - set(sb)):
            cs = [c for c in range(1, n + 1) if c not in (a, b)
                  and (min(a, c), max(a, c)) in sb and (min(b, c), max(b, c)) in sb]
            if not cs:
                continue
            c = max(cs, key=lambda c: min(sb[(min(a, c), max(a, c))], sb[(min(b, c), max(b, c))]))
            lo_, hi_ = (a, b) if pr[a - 1] < pr[b - 1] else (b, a)
            touch = ndimage.binary_dilation(lab == hi_, structure=ndimage.generate_binary_structure(2, 1))
            sep_log.append([GRID_CH[lo_], GRID_CH[hi_], GRID_CH[c], int(((lab == lo_) & touch).sum())])
            lab[(lab == lo_) & touch] = c
    if o["survive"]:
        # legend min_px as a minimum height for these features (the eyes are tall ovals): grow into the part
        # that contains them (never into the outline-side parts)
        for i, d in enumerate(mats):
            if not small_feature(i):
                continue
            mp = legend["parts"][i]["min_px"]
            cl, nc = ndimage.label(lab == i + 1)
            for c in range(1, nc + 1):
                ys, xs = np.nonzero(cl == c)
                while ys.max() - ys.min() + 1 < mp:
                    y0, y1 = ys.min(), ys.max()
                    grown = False
                    for yy in (y0 - 1, y1 + 1):           # prefer up: keeps the visor row that separates eyes from below
                        row = sorted(set(xs.tolist()))
                        nxt = yy + (yy - y1 if yy > y1 else yy - y0)          # the row beyond the grown one
                        ok = 0 <= yy < M and all(lab[yy, x] > 0 and mats[lab[yy, x] - 1]["mode"] == "black" for x in row)
                        # don't grow into contact with a coloured part (the eye would merge into the face frame)
                        ok = ok and all(not (0 <= nxt < M) or lab[nxt, x] == 0 or mats[lab[nxt, x] - 1]["mode"] == "black"
                                        or lab[nxt, x] == i + 1 for x in row)
                        if ok:
                            for x in set(xs):
                                lab[yy, x] = i + 1
                            ys = np.concatenate([ys, np.full(len(set(xs)), yy)]); xs = np.concatenate([xs, list(set(xs))])
                            grown = True
                            break
                    if not grown:
                        break
    if mode == "outset":
        lab, dark, alt = (np.pad(x, 1) for x in (lab, dark, alt))
    H = W = N
    lab_in = lab.copy()                     # the label map this render starts from (after survival/separators)
    ink = np.zeros((H, W), bool)            # thin outline ink
    blk = np.zeros((H, W), bool)            # black fills (rule 4)
    thick = np.zeros((H, W), bool)          # depth ink (rule 3)
    for i, d in enumerate(mats):
        if d["mode"] == "black":
            blk |= lab == i + 1
    fg = lab > 0
    inb = lambda y, x: 0 <= y < H and 0 <= x < W
    # outer outline
    if mode == "outset":
        ring = ~fg & ndimage.binary_dilation(fg, structure=ndimage.generate_binary_structure(2, 1))
        ink |= ring
    else:
        pad = np.pad(fg, 1)
        ink |= fg & ~(pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]) & ~blk
    # part lines: per pair, the ink goes on the larger part's side (consistent single lines, small parts keep
    # their pixels); ties -> lower priority
    area = np.bincount(lab.ravel(), minlength=n + 1)
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
                if b == 0 or b == a or not nl(mats[a - 1], mats[b - 1]):
                    continue
                if (area[a], -pr[a - 1]) > (area[b], -pr[b - 1]):
                    ink[y, x] = True
                    break
    # rule 1 clean-up: drop ink pixels with > 2 ink neighbours whose removal leaks nothing
    def leaks(y, x):
        a = lab[y, x]
        for dy, dx in FOUR:
            yy, xx = y + dy, x + dx
            if not inb(yy, xx):
                continue
            if ink[yy, xx] or blk[yy, xx]:
                continue
            b = lab[yy, xx]
            if a == 0 and b > 0:
                return True                       # outer ring pixel guarding a part pixel
            if a > 0 and b == 0:
                return True                       # inset outer outline
            if a > 0 and b > 0 and a != b and nl(mats[a - 1], mats[b - 1]):
                return True
        return False

    def nbs(y, x, m):
        return sum(1 for dy, dx in EIGHT if inb(y + dy, x + dx) and m[y + dy, x + dx])
    if o["clean"]:
        changed = True
        while changed:
            changed = False
            dark_all = ink | blk
            cand = sorted(((nbs(y, x, dark_all), y, x) for y, x in zip(*np.nonzero(ink))), reverse=True)
            for c, y, x in cand:
                if c <= 2 or nbs(y, x, ink | blk) <= 2:
                    continue
                # options: un-ink the pixel as it was; or let a neighbouring part take it over, which merges a
                # doubled line (two limbs 2 px apart) into one shared line - except in the concept's shadow,
                # where the doubled line is kept as a depth cue (rule 3)
                own = lab[y, x]
                opts_ = [own]
                if own == 0 and dark[y, x] < o["depth_th"]:
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
    fg = lab > 0
    # rule 3 depth: limb pixels touching ink thicken where the concept is in shadow, if the limb stays >= 2 px wide
    if o["depth"]:
        cands = []
        for y, x in zip(*np.nonzero(fg & ~ink & ~blk)):
            k = lab[y, x]
            if mats[k - 1]["role"] != "limb" or dark[y, x] < o["depth_th"]:
                continue
            if not any(inb(y + dy, x + dx) and (ink | blk)[y + dy, x + dx] for dy, dx in FOUR):
                continue
            cands.append((dark[y, x], y, x))
        for _, y, x in sorted(cands, reverse=True):
            k = lab[y, x]
            free = (lab == k) & ~ink & ~blk & ~thick
            ok = False
            for dy, dx in FOUR:          # ink on one side, >= 2 more limb pixels on the other: stays >= 2 px wide
                if inb(y + dy, x + dx) and (ink | blk | thick)[y + dy, x + dx]:
                    y1, x1, y2, x2 = y - dy, x - dx, y - 2 * dy, x - 2 * dx
                    if inb(y2, x2) and free[y1, x1] and free[y2, x2]:
                        ok = True
            if ok:
                thick[y, x] = True
    # colours
    out = np.zeros((H, W, 4), np.uint8)
    role = np.full((H, W), "", object)
    for i, d in enumerate(mats):
        m = (lab == i + 1) & ~ink & ~thick
        if d["mode"] == "black":
            out[m] = (*BLACK, 255)
            if N >= o["highlight_min"] and d.get("alt") is not None:
                hl = m & alt
                hl &= _keep_blobs(hl, 2)
                out[hl] = (*d["alt"], 255)
            continue
        out[m] = (*d["base"], 255)
        if d["mode"] == "pattern" and o["patterns"]:
            pm = m & alt
            if d["role"] == "limb":       # stripes on a limb must be >= 2 px marks, else they are just noise
                pm &= _keep_blobs(pm, o["limb_mark_min"])
            if pm.sum() >= 0.6 * m.sum():
                pm[:] = False
            out[pm] = (*d["alt"], 255)
        elif d["mode"] == "shade" and N >= o["shade_min"]:
            sm = m & alt & _keep_blobs(m & alt, 2)
            # don't let the shade swallow the part: keep >= half of it in the base tone
            if sm.sum() <= 0.5 * m.sum():
                out[sm] = (*d["alt"], 255)
    out[ink | thick] = (*BLACK, 255)
    if o["mirror"]:           # the pixel passes above aren't mirror-invariant; copy the chosen half exactly
        for arr in (out, lab, ink, thick, blk):
            half = arr[:, :W // 2].copy() if sym_info["keep"] == "left" else arr[:, W - W // 2:][:, ::-1].copy()
            arr[:, :W // 2] = half
            arr[:, W - W // 2:] = half[:, ::-1]
    info = {"size": N, "mode": mode, "label_size": M, "ink_px": int(ink.sum()), "depth_px": int(thick.sum()),
            "black_fill_px": int((blk & ~ink).sum()), "clumps": clumps(ink, blk | thick),
            "separators": sep_log, "symmetry": sym_info, "_labels": lab_in, "grid": ["".join("#" if ink[y, x] or thick[y, x] else GRID_CH[lab[y, x]] for x in range(W)) for y in range(H)]}
    return Image.fromarray(out, "RGBA"), info


def centre_on_axis(src: np.ndarray, concept: np.ndarray, search=40):
    """Find the vertical mirror axis of the silhouette and shift source + concept so it is the image centre;
    every downsampled size is then centred on the axis too."""
    fg = src > 0
    W = fg.shape[1]
    flip = fg[:, ::-1]
    best = None
    for d in range(-search, search + 1):                   # axis2 = W + d (pixel-edge coords, x2)
        m = np.roll(flip, d, axis=1)
        iou = (fg & m).sum() / max((fg | m).sum(), 1)
        if best is None or iou > best[1]:
            best = (d, iou)
    d, iou = best
    dx = -int(round(d / 2))
    sh = lambda a: np.roll(a, dx, axis=1)
    return sh(src), sh(concept), {"axis2_offset": d, "shift_px": dx, "mirror_iou": round(float(iou), 4)}


def symmetrize(lab: np.ndarray, src: np.ndarray, legend: dict):
    """Two candidates (left half mirrored, right half mirrored); keep the healthier (ladder.health score, then
    mean plain IoU). Deterministic; the ladder's v2 judge pick is the model-backed version of this."""
    n = len(legend["parts"])
    sst = ladder.source_stats(src, n)
    W = lab.shape[1]
    cands = {}
    for keep in ("left", "right"):
        c = lab.copy()
        if keep == "left":
            c[:, W - W // 2:] = c[:, :W // 2][:, ::-1]
        else:
            c[:, :W // 2] = c[:, W - W // 2:][:, ::-1]
        h = ladder.health(c, src, legend, sst)
        cands[keep] = (sum({"green": 2, "amber": 1, "red": 0}[r["status"]] for r in h),
                       float(np.mean([r["iou_plain"] for r in h])), c)
    keep = max(cands, key=lambda k: cands[k][:2])
    return cands[keep][2], {"keep": keep, "scores": {k: [v[0], round(v[1], 3)] for k, v in cands.items()}}


def _boundary_counts(lab):
    out = {}
    for a, b in ((lab[:, :-1], lab[:, 1:]), (lab[:-1], lab[1:])):
        m = (a != b) & (a > 0) & (b > 0)
        for x, y in zip(a[m].tolist(), b[m].tolist()):
            k = (min(x, y), max(x, y))
            out[k] = out.get(k, 0) + 1
    return out


def _keep_blobs(m, min_area):
    lab, k = ndimage.label(m, np.ones((3, 3), int))
    if not k:
        return m
    areas = ndimage.sum(m, lab, range(1, k + 1))
    return np.isin(lab, [i + 1 for i, a in enumerate(areas) if a >= min_area])


def clumps(ink, other_black):
    """Thin-outline pixels with > 2 ink 8-neighbours (ignoring pixels that touch black fills / depth ink)."""
    H, W = ink.shape
    n = 0
    for y, x in zip(*np.nonzero(ink)):
        near = [(y + dy, x + dx) for dy, dx in EIGHT if 0 <= y + dy < H and 0 <= x + dx < W]
        if any(other_black[p] for p in near):
            continue
        if sum(ink[p] for p in near) > 2:
            n += 1
    return n


# ------------------------------------------------------------------ NES tile rule

def consolidate(im: Image.Image, de=15.0):
    """Merge near-identical NES colours (Lab dE < de) into the more frequent one (palette economy)."""
    a = np.array(im)
    op = a[..., 3] > 0
    cols, cnt = np.unique(a[op][:, :3], axis=0, return_counts=True)
    order = np.argsort(-cnt)
    cols, cnt = cols[order], cnt[order]
    lab = to_lab(cols.astype(float))
    merged = {}
    for j in range(len(cols)):
        for i in range(j):
            if tuple(cols[i]) not in merged and np.sqrt(((lab[i] - lab[j]) ** 2).sum()) < de:
                merged[tuple(cols[j])] = tuple(cols[i])
                break
    for c, t in merged.items():
        a[op & (a[..., :3] == c).all(-1), :3] = t
    return Image.fromarray(a, "RGBA"), {hexs(c): hexs(t) for c, t in merged.items()}


def tile_reduce(im: Image.Image, salient=(), salient_w=3.0, symmetric=False):
    """<= 3 opaque colours per 8x8 tile, at the grid offset with the least perceptual damage (sum of Lab dE over
    remapped pixels; salient colours count x salient_w). Black (the outline) is always kept; a tile's other two
    colours are the pair that minimises the damage. Light colours are never remapped to black."""
    from itertools import combinations
    a0 = np.array(im)
    h, w = a0.shape[:2]
    best = None
    # symmetric: only grid offsets whose tiles mirror onto tiles (so remaps stay symmetric)
    oxs = sorted({(-(w // 2)) % 8, (4 - w // 2) % 8}) if symmetric else range(8)
    for oy in range(8):
        for ox in oxs:
            a = a0.copy()
            cost, changed = 0.0, 0
            for y0 in range(-oy, h, 8):
                for x0 in range(-ox, w, 8):
                    ys, xs = slice(max(0, y0), y0 + 8), slice(max(0, x0), x0 + 8)
                    t = a[ys, xs]
                    op = t[..., 3] > 0
                    if not op.any():
                        continue
                    cols, cnt = np.unique(t[op][:, :3], axis=0, return_counts=True)
                    if len(cols) <= 3:
                        continue
                    lab = to_lab(cols.astype(float))
                    isk = (cols == 0).all(1)
                    wgt = np.array([salient_w if tuple(c) in salient else 1.0 for c in cols]) * cnt
                    D = np.sqrt(((lab[:, None] - lab[None]) ** 2).sum(-1))
                    D[:, isk] = np.where(lab[:, 0:1] > 30, 1e6, D[:, isk])
                    fixed = list(np.nonzero(isk)[0])
                    rest = [j for j in range(len(cols)) if not isk[j]]
                    bc = None
                    for keep in combinations(rest, 3 - len(fixed)):
                        k = fixed + list(keep)
                        c = sum(wgt[j] * D[j, k].min() for j in range(len(cols)) if j not in k)
                        if bc is None or c < bc[0]:
                            bc = (c, k)
                    c, k = bc
                    cost += c
                    for j in range(len(cols)):
                        if j in k:
                            continue
                        tgt = cols[k[int(np.argmin(D[j, k]))]]
                        sel = op & (t[..., :3] == cols[j]).all(-1)
                        t[sel, :3] = tgt
                        changed += int(sel.sum())
            if best is None or cost < best[0]:
                best = (cost, changed, ox, oy, a)
    cost, changed, ox, oy, a = best
    return Image.fromarray(a, "RGBA"), {"changed_px": changed, "damage_dE": round(float(cost), 1), "offset": [ox, oy]}


# ------------------------------------------------------------------ sheets

def _font(sz):
    return ladder._font(sz)


def up(im: Image.Image, box: int, bg=(255, 255, 255)) -> Image.Image:
    """Nearest-neighbour upscale to fit `box` (integer factor), on a background."""
    im = im.convert("RGBA")
    f = max(1, box // max(im.size))
    big = im.resize((im.width * f, im.height * f), Image.NEAREST)
    c = Image.new("RGB", (box, box), bg)
    c.paste(big, ((box - big.width) // 2, (box - big.height) // 2), big)
    return c


def sheet(rows: list[tuple[str, list[tuple[str, Image.Image]]]], path: Path, box=192, same_zoom: int | None = None):
    lab_h, gap = 22, 10
    cols = max(len(r[1]) for r in rows)
    W = 120 + cols * (box + gap)
    Hh = len(rows) * (box + lab_h + gap) + gap
    img = Image.new("RGB", (W, Hh), (30, 30, 36))
    d = ImageDraw.Draw(img)
    f = _font(15)
    for r, (title, cells) in enumerate(rows):
        y = gap + r * (box + lab_h + gap)
        d.text((8, y + lab_h + box // 2 - 8), title, fill=(230, 230, 230), font=f)
        for c, (cap, im) in enumerate(cells):
            x = 120 + c * (box + gap)
            d.text((x, y), cap, fill=(200, 200, 200), font=f)
            if same_zoom:
                im = im.convert("RGBA")
                big = im.resize((im.width * same_zoom, im.height * same_zoom), Image.NEAREST)
                tile = Image.new("RGB", (box, box), (160, 160, 176))
                tile.paste(big, ((box - big.width) // 2, (box - big.height) // 2), big)
            else:
                tile = up(im, box, (160, 160, 176))
            img.paste(tile, (x, y + lab_h))
    img.save(path)


def run(out: Path = REG / "render", sizes=(64, 32, 24, 20, 16)) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    legend = json.loads((REG / "m1/legend.json").read_text())
    src = ladder.load_labels(REG / "m1/gpt-s1/regions.png", legend)
    concept = np.array(Image.open(CONCEPT).convert("RGBA"))
    src, concept, axis = centre_on_axis(src, concept)
    mats = materials(src, concept, legend)
    res = {"inputs": {"regions": str(REG / "m1/gpt-s1/regions.png"), "legend": str(REG / "m1/legend.json"),
                      "concept": str(CONCEPT)},
           "materials": [{k: (hexs(v) if k in ("base", "alt") and v is not None else v) for k, v in d.items()
                          if k != "alt_mask"} for d in mats],
           "axis": axis, "lines": {}, "renders": {}}
    names = [d["name"] for d in mats]
    res["lines_small"] = {f"{a}|{b}": need_line(mats[i], mats[j], 16, 32) for i, a in enumerate(names) for j, b in enumerate(names)
                          if i < j and (ladder.adjacency(src) & {(i + 1, j + 1)})}
    res["lines"] = {f"{a}|{b}": need_line(mats[i], mats[j]) for i, a in enumerate(names) for j, b in enumerate(names) if i < j
                    and (ladder.adjacency(src) & {(i + 1, j + 1)})}
    checks = {"tile_colors": {}, "orphans": {}, "holes": {}, "outline": {}, "colors": {"max_total": 4, "gate": False}}
    salient = [tuple(d["base"]) for d in mats if d["role"] == "feature" and not d["host"] and d["mode"] != "black"]
    for N in sizes:
        for tag, mode, opts in ((f"{N}", "outset", {}), (f"{N}_alllines", "outset", {"contrast_below": 0}),
                                (f"{N}_asym", "outset", {"mirror": False}), (f"{N}_inset", "inset", {})):
            im, info = render(src, concept, legend, mats, N, mode, opts)
            free = im
            cons, merged = consolidate(im)
            legal, tinfo = tile_reduce(cons, salient=salient, symmetric=opts.get("mirror", True))
            tinfo["merged"] = merged
            free.save(out / f"render_{tag}_free.png")
            legal.save(out / f"render_{tag}.png")
            info.update(tile=tinfo, checks_free=imaging.run_checks(free, checks), checks=imaging.run_checks(legal, checks),
                        colours=sorted(hexs(c) for c in np.unique(np.array(legal)[np.array(legal)[..., 3] > 0][:, :3], axis=0)))
            if tag == f"{N}":
                ladder.colorize(info["_labels"], legend).save(out / f"labels_{N}.png")
            info.pop("_labels")
            res["renders"][tag] = info
        # the Milestone 2 'source' rung for reference (same method; outset uses N-2)
        p = REG / f"m2/t{N}/source/rung_{N}.png"
        if not p.exists():
            p = REG / f"m2/t32/source/rung_{N}.png"
        if p.exists():
            Image.open(p).save(out / f"regions_{N}.png")
    (out / "results.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def sheets(out: Path = REG / "render", sizes=(64, 32, 24, 20, 16)):
    refs = [(n, Image.open(REFS / n / "subject.png")) for n in ("telly", "met_walk", "pipi")]
    rows = []
    rows.append(("M2 rung", [(f"{N}", Image.open(out / f"regions_{N}.png")) for N in sizes if (out / f"regions_{N}.png").exists()]))
    rows.append(("labels used", [(f"{N - 2}+outline", Image.open(out / f"labels_{N}.png")) for N in sizes]))
    rows.append(("render", [(f"{N}", Image.open(out / f"render_{N}.png")) for N in sizes]))
    rows.append(("no tile rule", [(f"{N}", Image.open(out / f"render_{N}_free.png")) for N in sizes]))
    rows.append(("all lines", [(f"{N}", Image.open(out / f"render_{N}_alllines.png")) for N in sizes]))
    rows.append(("asymmetric", [(f"{N}", Image.open(out / f"render_{N}_asym.png")) for N in sizes]))
    rows.append(("inset", [(f"{N}", Image.open(out / f"render_{N}_inset.png")) for N in sizes]))
    sheet(rows, out / "sheet.png", box=192)
    # same zoom (x8) as the authentic refs
    small = [s for s in sizes if s <= 32]
    sheet([("x8 render", [(f"{N}", Image.open(out / f"render_{N}.png")) for N in small] + [("", Image.new("RGBA", (1, 1)))]),
           ("x8 refs", refs + [("compiler v2", _compiler_ref())])], out / "sheet_same_zoom.png", box=8 * 32, same_zoom=8)


def _compiler_ref():
    p = ROOT / "workspace/experiments/compiler/v6_outer_thin/rank1_outer_19x20.png"     # compiler v2, outer thinning
    return Image.open(p) if p.exists() else Image.new("RGBA", (1, 1))


if __name__ == "__main__":
    r = run()
    sheets()
    for d in r["materials"]:
        print(f"{d['name']:16s} {d['mode']:8s} base={d['base']} alt={d.get('alt')} host={d['host']} {d.get('reason', '')}")
    print("no-line pairs:", [k for k, v in r["lines"].items() if not v])
    print("no-line pairs < 32 px (extra):", [k for k, v in r["lines_small"].items() if not v and r["lines"][k]])
    for t, i in r["renders"].items():
        c = i["checks"]
        print(t, {k: i[k] for k in ("ink_px", "depth_px", "clumps")}, i["tile"], c["tile_colors"]["detail"],
              "| free:", i["checks_free"]["tile_colors"]["detail"], "|", len(i["colours"]), "colours")
