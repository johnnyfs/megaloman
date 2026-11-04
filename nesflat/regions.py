"""Region renderer: vectorise-then-rasterise for flat art with black ink lines (the default method).

Thick source lines are not downsampled (they would eat whole pixels and leave grey mush); they are removed,
the flat regions they separate are rasterised at the target size, and 1 px lines are redrawn there.

  1. tones      every source pixel -> nearest flat tone (tone 0 = ink)
  2. ink        line ink (thinner than ~3 half-widths) is given to the nearest region; solid black areas
                (the cannon bore, pupils) stay as ink regions
  3. regions    connected components of the line-free label map; slivers merge into a neighbour
  4. raster     per target cell: argmax of region coverage x priority, priority = boost for small regions;
                every region big enough to matter keeps at least one cell (feature floor)
  5. lines      1 px ink between adjacent regions: always between same-colour regions, between all regions
                from `line_min` px up; the pixel comes from the larger region (the one 'behind')
  6. outline    1 px ink outside the silhouette (4-connected), exact mirror when symmetric, tile rule
"""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from .model import T, Sprite, Source, Weights, _mirror, close_outline, tile_reduce
from .palette import nearest_ramp


@dataclass
class RegionWeights(Weights):
    gamma: float = 0.6         # priority boost for small regions: w = 1 + gamma * log2(4 / cells), >= 1
    floor_cells: float = 0.3   # a region covering at least this many target cells keeps >= 1 cell
    fg_th: float = 0.45        # silhouette coverage threshold
    line_min: int = 32         # from this size up, lines separate *all* adjacent regions
    sliver: float = 0.0004     # regions smaller than this share of the subject merge into a neighbour


class RegionSource(Source):
    def __init__(self, img, w=None):
        w = w or RegionWeights()
        super().__init__(img, w)
        self._r = {}

    def regions(self, k):
        if k in self._r:
            return self._r[k]
        tl = self.tones(k)
        lab = ndi.median_filter(self.lab, size=(3, 3, 1))
        lbl = ((lab[..., None, :] - tl) ** 2).sum(-1).argmin(-1)
        lbl[~self.fg] = T
        ink = lbl == 0
        dt = ndi.distance_transform_edt(ink)
        ridge = ink & (dt >= ndi.maximum_filter(dt, 3)) & (dt > 0)
        hw = float(np.median(dt[ridge])) if ridge.any() else 1.0
        r = max(1, int(round(1.6 * hw)))
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        solid = ndi.binary_opening(ink, structure=yy ** 2 + xx ** 2 <= r * r)
        line = ink & ~solid
        tone = np.where(line, -2, lbl)                                    # -2 = to be reassigned
        # connected regions of each tone (solid ink included), on the line-free map
        reg = np.zeros(tone.shape, int)
        nreg = 0
        reg_tone = [T]
        for t in range(k):
            l, n = ndi.label(tone == t)
            reg[l > 0] = l[l > 0] + nreg
            reg_tone += [t] * n
            nreg += n
        reg_tone = np.array(reg_tone)
        area = np.bincount(reg.ravel(), minlength=nreg + 1)
        small = area < max(4, self.fg.sum() * RegionWeights.sliver)
        small[0] = False
        reg[small[reg]] = 0
        # give line ink and slivers to the nearest real region
        hole = self.fg & (reg == 0)
        if hole.any():
            _, (iy, ix) = ndi.distance_transform_edt(hole | ~self.fg & False | (reg == 0), return_indices=True)
            reg = np.where(hole, reg[iy, ix], reg)
        reg[~self.fg] = 0
        ids = np.unique(reg[reg > 0])
        remap = np.zeros(nreg + 1, int)
        remap[ids] = np.arange(1, len(ids) + 1)
        reg = remap[reg]
        reg_tone = np.concatenate([[T], reg_tone[ids]])
        area = np.bincount(reg.ravel(), minlength=len(ids) + 1).astype(float)
        self._r[k] = (reg, reg_tone, area, hw)
        return self._r[k]


def render_regions(src, N, w: RegionWeights = None, ramp=None):
    w = w or RegionWeights()
    src = src if isinstance(src, RegionSource) else RegionSource(src, w)
    k = w.colors
    tl = src.tones(k)
    reg, rtone, area, hw = src.regions(k)
    R = len(rtone)
    sym = src.iou >= w.sym_iou

    # --- fit (inner box leaves 1 px for the outline)
    inner = N - 2
    y0, x0, y1, x1 = src.box
    bh, bw = y1 - y0, x1 - x0
    s = inner / max(bh, bw)
    h, wd = max(1, round(bh * s)), max(1, round(bw * s))
    if sym and (N - wd) % 2:
        wd += 1 if wd < inner else -1
    sub = reg[y0:y1, x0:x1]
    yy = np.minimum((np.arange(bh) * h) // bh, h - 1)
    xx = np.minimum((np.arange(bw) * wd) // bw, wd - 1)
    cell = (yy[:, None] * wd + xx[None, :]).ravel()
    cov = np.bincount(cell * R + sub.ravel(), minlength=h * wd * R).reshape(h, wd, R).astype(float)
    cov /= np.maximum(cov.sum(-1, keepdims=True), 1)
    if sym:
        cov = (cov + cov[:, ::-1]) / 2
    fgcov = 1 - cov[..., 0]

    # --- raster with small-region priority
    cells = area * (h * wd) / max(1, bh * bw)                         # region size in target cells
    pri = 1 + w.gamma * np.log2(4 / np.maximum(cells, 1e-3)).clip(0, 4)
    pri[0] = 1.0
    score = cov * pri
    score[..., 0] = 0
    X = np.where(fgcov >= w.fg_th, score.argmax(-1), 0)
    # feature floor: regions that matter keep their best cell
    present = np.zeros(R, bool)
    present[np.unique(X)] = True
    for r in np.argsort(-pri):
        if r == 0 or present[r] or cells[r] < w.floor_cells:
            continue
        c = np.unravel_index(cov[..., r].argmax(), (h, wd))
        if cov[c + (r,)] > 0:
            X[c] = r
            present[r] = True
            if sym:
                X[c[0], wd - 1 - c[1]] = r

    # --- lines between regions (pixel taken from the larger region)
    L = np.zeros_like(X, bool)
    same_only = N < w.line_min
    for (a, b, axis) in ((X[:, :-1], X[:, 1:], 1), (X[:-1, :], X[1:, :], 0)):
        diff = (a != b) & (a > 0) & (b > 0)
        ta, tb = rtone[a], rtone[b]
        need = diff & (ta != 0) & (tb != 0)
        if same_only:
            need &= ta == tb
        take_a = area[a] >= area[b]
        ys, xs = np.where(need)
        for y, x, ta_ in zip(ys, xs, take_a[ys, xs]):
            if axis == 1:
                L[(y, x) if ta_ else (y, x + 1)] = True
            else:
                L[(y, x) if ta_ else (y + 1, x)] = True
    # don't let lines erase a region completely
    grid = np.where(X > 0, rtone[X], T)
    for r in np.unique(X[X > 0]):
        m = X == r
        if (m & ~L).sum() == 0:
            L[m] = False
    grid[L] = 0

    canvas = np.full((N, N), T)
    oy, ox = (N - h) // 2, (N - wd) // 2
    canvas[oy:oy + h, ox:ox + wd] = grid
    f = canvas != T
    ring = ndi.binary_dilation(f, structure=ndi.generate_binary_structure(2, 1)) & ~f
    canvas[ring] = 0
    canvas = close_outline(canvas)
    if w.tile_rule:
        canvas = tile_reduce(canvas, w.tile_rule, tl)
    if sym:
        canvas = _mirror(canvas, True)
    ramp = ramp or nearest_ramp(tl)
    return Sprite(canvas, list(ramp), dict(size=N, method='regions', regions=int(R - 1), line_halfwidth=hw,
                                           symmetric=bool(sym), tones=np.round(tl, 3).tolist()))
