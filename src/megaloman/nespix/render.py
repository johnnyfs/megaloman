"""The nespix renderer: a fixed, hand-weighted model that turns any hi-res image into NES pixel art.

Pipeline (every stage is deterministic; all weights are set by hand in `Params`):

  1. matte      foreground mask (alpha, else border flood fill in OKLab), keep the main components
  2. palette    palette FIRST: exhaustive search over NES colours for the set that best reconstructs the
                foreground (black is fixed as the ink/outline colour for sprites)
  3. classify   every hi-res pixel -> nearest palette entry; the concept's own outer outline is ignored
                (we draw our own)
  4. pool       area-pool class probabilities into the target grid (symmetric pooling when the source
                is mirror-symmetric, so the result mirrors exactly, pixel for pixel)
  5. sharpen    a fixed 3x3 unsharp kernel on each class map ("detail-preserving downsampler"): thin limbs,
                eyes and small features survive the shrink instead of being voted away
  6. decide     argmax per cell, foreground threshold
  7. clean      remove specks, fill pinholes, absorb orphans, enforce exact mirror
  8. outline    1 px black outline drawn *outside* the fill (4-connected: no clumps, rounded corners)

`render(img, size)` returns a `Sprite` (index grid + palette).
"""
from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from .palette import BLACK, CODES, NES, NES_LAB, NES_RGB, code_index, dist2, hex_rgb, oklab


@dataclass
class Params:
    colors: int = 4                 # opaque colours (sprite: incl. the black ink); transparent is extra
    tile_rule: int = 3              # NES: max opaque colours per 8x8 tile (0 = off). 4 colours + this rule =
                                    # a 2-palette metasprite like Mega Man's face; colors=3 = one sprite palette
    mode: str = 'sprite'            # 'sprite' (transparent bg + black outline) | 'tile' (opaque, no outline)
    outline: bool = True
    symmetry: str = 'auto'          # 'auto' | 'on' | 'off'
    sym_iou: float = 0.88           # source mirror IoU needed for 'auto'
    fg_th: float = 0.5              # foreground threshold after sharpening
    sharpen: float = 0.9            # unsharp amount on pooled class maps (0 = plain area vote)
    ink_weight: float = 1.0         # >1 keeps more interior black (lines) ; <1 less
    wl: float = 1.4                 # lightness weight in colour distances (pixel art reads by value)
    min_sep: float = 0.10           # minimum OKLab distance between chosen palette colours
    min_from_black: float = 0.45    # sprite fills must stand clear of the black ink (black does the shadows)
    err_cap: float = 0.15           # truncated colour error: gradients/noise don't dominate palette choice
    smooth: float = 0.6             # hi-res mode filter radius, in target cells (flattens shading speckle)
    chroma: float = 1.5             # chroma boost before palette fit (NES colours are saturated; concepts are muddy)
    ink_open: float = 0.35          # below 48 px, interior ink thinner than 2*ink_open cells is dropped
                                    # (lines vanish, blobs like eyes/visors survive); the outline is redrawn
    ink_l: float = 0.22             # black 'stands for' every deep shadow: it matches as if its lightness were this
    flat: float = 0.55              # parts (regions between the concept's ink lines) whose pixels mostly agree
                                    # (>= this share near the median) become one solid colour; 0 = off
    saliency: float = 0.5           # extra palette weight for pixels whose colour stands out from the mean
    majority: int = 2               # passes of 3x3 majority smoothing on fill colours at target res
    palette: tuple = ()             # force NES codes (e.g. ('0F','16','27')); empty = search
    bg_tol: float = 0.06            # OKLab tolerance for flood-fill background removal
    pad: int = 0                    # extra transparent margin (px) inside the canvas
    aspect: str = 'keep'            # 'keep' | 'fill' (stretch to the square)


@dataclass
class Sprite:
    grid: np.ndarray                # (N, N) int: -1 transparent, else index into codes
    codes: list                     # NES codes of the palette entries
    meta: dict = field(default_factory=dict)

    def rgba(self):
        h, w = self.grid.shape
        out = np.zeros((h, w, 4), np.uint8)
        for i, c in enumerate(self.codes):
            out[self.grid == i] = hex_rgb(NES[c]) + (255,)
        return Image.fromarray(out, 'RGBA')

    def preview(self, scale=8, bg=(64, 72, 96)):
        im = self.rgba()
        base = Image.new('RGBA', im.size, bg + (255,))
        base.alpha_composite(im)
        return base.convert('RGB').resize((im.width * scale, im.height * scale), Image.NEAREST)

    def text(self):
        keys = '.' + ''.join('KABCDEFG'[:len(self.codes)])
        return '\n'.join(''.join(keys[v + 1] for v in row) for row in self.grid)


# ---------------------------------------------------------------- 1. matte

def _load(img):
    if isinstance(img, (str, bytes)) or hasattr(img, 'read'):
        img = Image.open(img)
    if isinstance(img, Image.Image):
        img = np.array(img.convert('RGBA'))
    img = np.asarray(img)
    if img.shape[-1] == 3:
        img = np.concatenate([img, np.full(img.shape[:2] + (1,), 255, np.uint8)], -1)
    return img


def matte(rgba, tol=0.06):
    """Foreground mask. Alpha if the image has transparency; otherwise flood-fill the background from the
    border on a median-smoothed copy (paper texture / JPEG noise), with a tolerance adapted to the border's
    own variation."""
    a = rgba[..., 3]
    if (a < 128).mean() > 0.02:
        fg = a >= 128
    else:
        lab = oklab(ndi.median_filter(rgba[..., :3], size=(5, 5, 1)))
        border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
        bg = np.median(border, 0)
        bd = np.sqrt(((border - bg) ** 2).sum(-1))
        t = min(0.16, max(tol, np.percentile(bd, 90) * 1.8))
        near = np.sqrt(((lab - bg) ** 2).sum(-1)) < t
        lbl, _ = ndi.label(near)
        edge = np.unique(np.concatenate([lbl[0], lbl[-1], lbl[:, 0], lbl[:, -1]]))
        fg = ~np.isin(lbl, edge[edge > 0])
    fg = ndi.binary_opening(fg, iterations=2)
    lbl, n = ndi.label(fg)
    if n == 0:
        raise ValueError('no foreground found')
    areas = ndi.sum(fg, lbl, range(1, n + 1))
    keep = 1 + np.where(areas >= 0.05 * areas.max())[0]
    fg = np.isin(lbl, keep)
    return _small_holes(fg)


def _small_holes(fg, frac=0.002):
    """Holes enclosed by the foreground smaller than frac of the image are noise -> foreground."""
    holes = ndi.binary_fill_holes(fg) & ~fg
    lbl, n = ndi.label(holes)
    if n == 0:
        return fg
    areas = ndi.sum(holes, lbl, range(1, n + 1))
    small = 1 + np.where(areas < frac * fg.size)[0]
    return fg | np.isin(lbl, small)


# ---------------------------------------------------------------- 2. palette

def choose_palette(lab_px, n_fill, p: Params, weights=None):
    """Exhaustive NES search (n_fill <= 3): black + n_fill colours minimising weighted, truncated OKLab error.
    Fills must stand clear of black and of each other."""
    kb = code_index(BLACK)
    pal_d = np.sqrt(dist2(NES_LAB, NES_LAB, 1.0))
    cand = [i for i, c in enumerate(CODES) if c != BLACK and pal_d[i, kb] >= p.min_from_black]
    if len(lab_px) > 6000:
        sel = np.random.default_rng(1).choice(len(lab_px), 6000, replace=False)
        lab_px = lab_px[sel]
        weights = None if weights is None else weights[sel]
    D = np.minimum(dist2(lab_px, _ink_lab(p), p.wl), p.err_cap ** 2)   # (P, K) truncated
    w = np.ones(len(lab_px)) if weights is None else weights
    w = w / w.sum()
    base = D[:, kb]
    C = np.array(cand)
    ok = pal_d[np.ix_(C, C)] >= p.min_sep                          # allowed pairs
    best, best_err = None, np.inf
    if n_fill == 1:
        errs = w @ np.minimum(base[:, None], D[:, C])
        j = int(errs.argmin())
        best, best_err = (C[j],), errs[j]
    elif n_fill == 2:
        for i in range(len(C)):
            ma = np.minimum(base, D[:, C[i]])
            errs = w @ np.minimum(ma[:, None], D[:, C])
            errs[~ok[i]] = np.inf
            errs[:i + 1] = np.inf
            j = int(errs.argmin())
            if errs[j] < best_err:
                best, best_err = (C[i], C[j]), errs[j]
    elif n_fill == 3:
        for i in range(len(C)):
            ma = np.minimum(base, D[:, C[i]])
            for j in range(i + 1, len(C)):
                if not ok[i, j]:
                    continue
                mb = np.minimum(ma, D[:, C[j]])
                errs = w @ np.minimum(mb[:, None], D[:, C])
                errs[~(ok[i] & ok[j])] = np.inf
                errs[:j + 1] = np.inf
                k = int(errs.argmin())
                if errs[k] < best_err:
                    best, best_err = (C[i], C[j], C[k]), errs[k]
    else:  # greedy + coordinate descent beyond 3 fills
        sel = []
        for _ in range(n_fill):
            cur = np.minimum.reduce([base] + [D[:, s] for s in sel])
            errs = [w @ np.minimum(cur, D[:, a]) if a not in sel else np.inf for a in cand]
            sel.append(cand[int(np.argmin(errs))])
        best = tuple(sel)
        best_err = w @ np.minimum.reduce([base] + [D[:, s] for s in sel])
    best = sorted(best, key=lambda i: NES_LAB[i, 0])               # fills dark -> light
    return [BLACK] + [CODES[i] for i in best], float(best_err)


# ---------------------------------------------------------------- helpers

def _ink_lab(p, lab=None):
    """Palette Lab table with black moved up to the ink lightness (used for matching only)."""
    t = (NES_LAB if lab is None else lab).copy()
    blk = np.abs(t).sum(-1) < 1e-6
    t[blk, 0] = p.ink_l
    return t


_K = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], float) / 16


def _blur(m):
    return ndi.convolve(m, _K, mode='constant')


def _dilate4(m):
    return ndi.binary_dilation(m, structure=ndi.generate_binary_structure(2, 1))


def mirror_iou(fg):
    ys, xs = np.where(fg)
    m = fg[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    f = m[:, ::-1]
    return (m & f).sum() / max(1, (m | f).sum())


# ---------------------------------------------------------------- 3-6. classify, pool, sharpen, decide

def _pool(cls, fg, n_cls, box, h_in, w_in):
    """Area-pool per-class coverage of hi-res pixels into an (h_in, w_in) grid. cls=-2 = 'fg, no vote'."""
    y0, x0, y1, x1 = box
    H, W = y1 - y0, x1 - x0
    yy = np.minimum((np.arange(H) * h_in) // H, h_in - 1)
    xx = np.minimum((np.arange(W) * w_in) // W, w_in - 1)
    cell = (yy[:, None] * w_in + xx[None, :]).ravel()
    c = cls[y0:y1, x0:x1].ravel()
    f = fg[y0:y1, x0:x1].ravel()
    total = np.bincount(cell, minlength=h_in * w_in).astype(float)
    cover = np.bincount(cell, weights=f.astype(float), minlength=h_in * w_in) / np.maximum(total, 1)
    votes = np.zeros((n_cls, h_in * w_in))
    ok = c >= 0
    np.add.at(votes, (c[ok], cell[ok]), 1.0)
    vs = votes.sum(0)
    probs = votes / np.maximum(vs, 1)
    # cells with fg but no votes (only ignored outline pixels) inherit from neighbours later
    return cover.reshape(h_in, w_in), probs.reshape(n_cls, h_in, w_in), (vs > 0).reshape(h_in, w_in)


class Concept:
    """A source image plus every analysis that does not depend on the target size (cached across variants).

    Work resolution is capped (`work`), so the per-variant cost is a few tens of ms."""

    def __init__(self, img, work=768, bg_tol=0.06, mode='sprite'):
        rgba = _load(img)
        H, W = rgba.shape[:2]
        if max(H, W) > work:
            s = work / max(H, W)
            rgba = np.array(Image.fromarray(rgba, 'RGBA').resize((round(W * s), round(H * s)), Image.LANCZOS))
        self.rgba = rgba
        self.tile = mode == 'tile'
        self.fg = np.ones(rgba.shape[:2], bool) if self.tile else matte(rgba, bg_tol)
        ys, xs = np.where(self.fg)
        self.box = (ys.min(), xs.min(), ys.max() + 1, xs.max() + 1)
        self.lab0 = oklab(rgba[..., :3])
        self.iou = mirror_iou(self.fg)
        dist = ndi.distance_transform_edt(self.fg)
        dark = self.lab0[..., 0] < 0.33
        ow = 0
        if not self.tile:
            for d in range(1, 40):
                ring = self.fg & (dist > d - 1) & (dist <= d)
                if ring.sum() == 0 or dark[ring].mean() < 0.5:
                    break
                ow = d
        self.outline_px = ow
        self.outer_ink = self.fg & dark & (dist <= ow + 1) if ow else np.zeros_like(self.fg)
        self._cache = {}

    def lab(self, chroma, flat=0.0):
        k = ('lab', chroma, flat)
        if k not in self._cache:
            lab = self.lab0.copy()
            lab[..., 1:] *= chroma
            if flat:
                lab = self._flatten(lab, flat)
            self._cache[k] = lab
        return self._cache[k]

    def parts(self):
        """Connected regions of the foreground between the concept's dark ink lines."""
        if 'parts' not in self._cache:
            ink = self.lab0[..., 0] < 0.30
            lbl, n = ndi.label(self.fg & ~ink)
            self._cache['parts'] = (lbl, n)
        return self._cache['parts']

    def _flatten(self, lab, share, tau=0.09):
        lbl, n = self.parts()
        if n == 0:
            return lab
        out = lab.copy()
        flat = lbl.ravel()
        order = np.argsort(flat, kind='stable')
        bounds = np.searchsorted(flat[order], np.arange(n + 2))
        L = lab.reshape(-1, 3)
        O = out.reshape(-1, 3)
        minarea = max(12, int(self.fg.sum() * 0.0005))
        for i in range(1, n + 1):
            idx = order[bounds[i]:bounds[i + 1]]
            if len(idx) < minarea:
                continue
            px = L[idx]
            # 2-means split first: a small distinct feature inside a part (an eye in a visor) survives
            c = np.stack([px[px[:, 0].argmin()], px[px[:, 0].argmax()]])
            for _ in range(6):
                asg = ((px[:, None] - c) ** 2).sum(-1).argmin(1)
                c = np.stack([np.median(px[asg == j], 0) if (asg == j).any() else c[j] for j in (0, 1)])
            minor = min((asg == 0).sum(), (asg == 1).sum())
            if np.sqrt(((c[0] - c[1]) ** 2).sum()) > 2 * tau and minor >= max(minarea, 0.03 * len(idx)):
                O[idx] = c[asg]
                continue
            med = np.median(px, 0)
            near = np.sqrt(((px - med) ** 2).sum(-1)) < tau
            if near.mean() >= share:
                O[idx] = med
        return out

    def palette(self, p: 'Params'):
        k = ('pal', p.colors, tuple(p.palette), p.chroma, p.wl, p.min_sep, p.min_from_black, p.err_cap,
             p.saliency, p.ink_l, p.flat)
        if k not in self._cache:
            if p.palette:
                codes = list(p.palette)
                if not self.tile and BLACK not in codes:
                    codes = [BLACK] + codes[:p.colors - 1]
                self._cache[k] = (codes, None)
            else:
                lab = self.lab(p.chroma, p.flat)
                idx = np.flatnonzero((self.fg & ~self.outer_ink).ravel())
                if len(idx) > 40000:
                    idx = np.random.default_rng(0).choice(idx, 40000, replace=False)
                px = lab.reshape(-1, 3)[idx]
                sal = np.sqrt(((px - np.median(px, 0)) ** 2).sum(-1))
                wts = 1 + p.saliency * sal / max(1e-6, sal.mean())
                self._cache[k] = choose_palette(px, p.colors - 1, p, wts)
        return self._cache[k]

    def classes(self, p: 'Params'):
        codes, _ = self.palette(p)
        k = ('cls', tuple(codes), p.chroma, p.wl, p.ink_weight, p.ink_l, p.flat)
        if k not in self._cache:
            pal_lab = NES_LAB[[code_index(c) for c in codes]]
            D = dist2(self.lab(p.chroma, p.flat), _ink_lab(p, pal_lab), p.wl)
            if codes[0] == BLACK:
                D[..., 0] /= p.ink_weight
            cls = D.argmin(-1)
            cls = np.where(self.fg, cls, -1)
            self._cache[k] = np.where(self.outer_ink, -2, cls)
        return self._cache[k]


def render(img, size, params: Params = None, **kw):
    p = Params(**{**(params.__dict__ if params else {}), **kw})
    c = img if isinstance(img, Concept) else Concept(img, bg_tol=p.bg_tol, mode=p.mode)
    N = int(size)
    tile = c.tile
    fg = c.fg
    y0, x0, y1, x1 = c.box
    bh, bw = y1 - y0, x1 - x0
    codes, pal_err = c.palette(p)
    pal_lab = NES_LAB[[code_index(x) for x in codes]]
    cls = c.classes(p)
    has_ink = codes[0] == BLACK and not tile
    sym = p.symmetry == 'on' or (p.symmetry == 'auto' and c.iou >= p.sym_iou and not tile)

    # --- fit into the canvas (inner box leaves 1 px for the outline)
    inner = N - 2 * p.pad - (2 if (p.outline and not tile) else 0)
    if tile:
        y0, x0, y1, x1 = 0, 0, fg.shape[0], fg.shape[1]
        bh, bw = y1, x1
    if p.aspect == 'fill' or tile:
        h_in = w_in = inner
    else:
        s = inner / max(bh, bw)
        h_in, w_in = max(1, round(bh * s)), max(1, round(bw * s))
        if sym and (N - w_in) % 2:      # keep the mirror axis on the canvas centre
            w_in += 1 if w_in < inner else -1
    box = (y0, x0, y1, x1)
    cellpx = max(bh / h_in, bw / w_in)

    # work on the bbox crop only
    sub = (slice(y0, y1), slice(x0, x1))
    cls = cls[sub]
    fgc = fg[sub]
    if has_ink and p.ink_open and N < 48:
        r = p.ink_open * cellpx
        if r >= 1:
            ink = cls == 0
            f = max(1, int(r // 3))                     # open at reduced resolution (fast, same result)
            small = ink[::f, ::f]
            rr = r / f
            yy, xx = np.mgrid[-int(rr):int(rr) + 1, -int(rr):int(rr) + 1]
            keep = ndi.binary_opening(small, structure=yy ** 2 + xx ** 2 <= rr * rr)
            keep = np.repeat(np.repeat(keep, f, 0), f, 1)[:ink.shape[0], :ink.shape[1]]
            cls = np.where(ink & ~keep, -3, cls)        # -3: thin ink -> takes the surrounding fill
    if p.smooth > 0:
        r = p.smooth * cellpx
        if r >= 1.5:
            first = 1 if has_ink else 0   # ink is never smoothed away
            sc = np.stack([ndi.uniform_filter((cls == k).astype(np.float32), size=int(2 * r + 1))
                           for k in range(first, len(codes))])
            sm = sc.argmax(0) + first
            cls = np.where((cls >= first) | (cls == -3), sm, cls)
    cls = np.where(cls == -3, -2, cls)
    cover, probs, voted = _pool(cls, fgc, len(codes), (0, 0, bh, bw), h_in, w_in)

    if sym:
        cover = (cover + cover[:, ::-1]) / 2
        probs = (probs + probs[:, :, ::-1]) / 2
        voted = voted | voted[:, ::-1]

    # --- 5. sharpen (fixed 3x3 unsharp kernel) and 6. decide
    if p.sharpen:
        cover_s = cover + p.sharpen * (cover - _blur(cover))
        probs_s = probs + p.sharpen * (probs - np.stack([_blur(q) for q in probs]))
    else:
        cover_s, probs_s = cover, probs
    is_fg = np.ones((h_in, w_in), bool) if tile else cover_s >= p.fg_th
    lab_grid = probs_s.argmax(0)
    if (~voted & is_fg).any() and voted.any():       # cells with no vote: nearest voted neighbour
        _, (iy, ix) = ndi.distance_transform_edt(~voted, return_indices=True)
        lab_grid = np.where(voted, lab_grid, lab_grid[iy, ix])
    grid = np.where(is_fg, lab_grid, -1)

    # --- 7. clean
    if not tile:
        grid = _clean(grid, N, sym)
        for _ in range(p.majority):
            grid = _majority(grid)

    # --- place on canvas
    canvas = np.full((N, N), -1, int)
    oy, ox = (N - h_in) // 2, (N - w_in) // 2
    canvas[oy:oy + h_in, ox:ox + w_in] = grid

    # --- 8. outline (outside the fill)
    if p.outline and not tile:
        f = canvas >= 0
        canvas[_dilate4(f) & ~f] = 0
        canvas = _post(canvas)
    if p.tile_rule and not tile:
        canvas = tile_reduce(canvas, p.tile_rule, pal_lab)
    if sym:
        canvas = _mirror(canvas)

    meta = dict(size=N, codes=codes, palette_err=pal_err, mirror_iou=round(float(c.iou), 3), symmetric=bool(sym),
                concept_outline_px=int(c.outline_px), inner=(int(h_in), int(w_in)))
    return Sprite(canvas, codes, meta)


def tile_colours(g, ox=0, oy=0):
    worst = 0
    H, W = g.shape
    for y0 in range(-oy, H, 8):
        for x0 in range(-ox, W, 8):
            t = g[max(0, y0):y0 + 8, max(0, x0):x0 + 8]
            worst = max(worst, len(np.unique(t[t >= 0])))
    return worst


def tile_reduce(g, maxc, pal_lab):
    """Enforce <= maxc opaque colours per 8x8 tile (NES sprite rule) on the 8x8 grid aligned to the canvas
    (a metasprite's tiles). In an over-full tile the least-used fill colour is remapped to the nearest
    colour already in the tile; black (ink) is always kept."""
    g = g.copy()
    H, W = g.shape
    for y0 in range(0, H, 8):
        for x0 in range(0, W, 8):
            t = g[y0:y0 + 8, x0:x0 + 8]
            while True:
                vals, cnt = np.unique(t[t >= 0], return_counts=True)
                if len(vals) <= maxc:
                    break
                fills = [(c, v) for v, c in zip(vals, cnt) if v != 0]
                _, drop = min(fills)
                rest = [v for v in vals if v != drop]
                d = [((pal_lab[v] - pal_lab[drop]) ** 2).sum() for v in rest]
                t[t == drop] = rest[int(np.argmin(d))]
    return g


def _mirror(g):
    W = g.shape[1]
    g = g.copy()
    g[:, W - W // 2:] = g[:, :W // 2][:, ::-1]
    return g


def _clean(g, N, sym):
    g = g.copy()
    fg = g >= 0
    # specks: tiny fg components
    lbl, n = ndi.label(fg, structure=np.ones((3, 3)))
    if n > 1:
        areas = ndi.sum(fg, lbl, range(1, n + 1))
        minarea = 1 if N < 16 else 2 if N < 48 else 4
        for i, a in enumerate(areas, 1):
            if a <= minarea and a < areas.max():
                g[lbl == i] = -1
    # pinholes: small enclosed transparent areas
    holes = ~(g >= 0)
    lbl, n = ndi.label(holes)
    border = set(np.unique(np.concatenate([lbl[0], lbl[-1], lbl[:, 0], lbl[:, -1]])))
    maxhole = max(1, (N * N) // 600)
    for i in range(1, n + 1):
        if i in border:
            continue
        m = lbl == i
        if m.sum() <= maxhole:
            nb = g[_dilate4(m) & ~m]
            nb = nb[nb >= 0]
            if len(nb):
                g[m] = np.bincount(nb).argmax()
    # orphans: a fill pixel with no same-colour 4-neighbour, mostly surrounded by one other fill colour
    for _ in range(2):
        changed = False
        P = np.pad(g, 1, constant_values=-1)
        for y, x in zip(*np.where(g > 0)):
            v = g[y, x]
            nb = [P[y, x + 1], P[y + 2, x + 1], P[y + 1, x], P[y + 1, x + 2]]
            if v in nb:
                continue
            nbf = [u for u in nb if u > 0]
            if len(nbf) >= 3:
                vals, cnt = np.unique(nbf, return_counts=True)
                if cnt.max() >= 3:
                    g[y, x] = vals[cnt.argmax()]
                    changed = True
        if not changed:
            break
    return g


def _majority(g):
    """Fill pixel takes the colour held by >= 5 of its 8 neighbours (region boundaries straighten)."""
    P = np.pad(g, 1, constant_values=-1)
    H, W = g.shape
    out = g.copy()
    shifts = [(dy, dx) for dy in (0, 1, 2) for dx in (0, 1, 2) if (dy, dx) != (1, 1)]
    nb = np.stack([P[dy:dy + H, dx:dx + W] for dy, dx in shifts])
    for k in np.unique(g[g > 0]):
        cnt = (nb == k).sum(0)
        out[(g > 0) & (g != k) & (cnt >= 5)] = k
    return out


def _post(c):
    """Post-outline tidy: an interior black pixel that only thickens the outline (touches transparent
    diagonally and has outline on two orthogonal sides) is turned back into its fill neighbour."""
    c = c.copy()
    P = np.pad(c, 1, constant_values=-1)
    H, W = c.shape
    for y in range(H):
        for x in range(W):
            if c[y, x] != 0:
                continue
            n4 = [P[y, x + 1], P[y + 2, x + 1], P[y + 1, x], P[y + 1, x + 2]]
            if -1 in n4:
                continue  # it is the outline itself
            fills = [u for u in n4 if u > 0]
            blacks = sum(u == 0 for u in n4)
            n8 = P[y:y + 3, x:x + 3]
            if (n8 == -1).any() and blacks == 2 and len(fills) == 2 and fills[0] == fills[1]:
                c[y, x] = fills[0]
    return c
