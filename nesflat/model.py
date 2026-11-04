"""nesflat: flat-colour (or grayscale) art with black outlines on a solid background -> NES pixel art.

Input constraint: high-contrast art (flat fills, dark ink lines) on one solid background colour.
Output: N x N (8..256), transparent background, black ink + up to 3 tones from the NES palette (4 colours),
<= 3 opaque colours per 8x8 tile.

The model is an energy over discrete pixel labels, with hand-set weights, minimised directly:

    labels  T (transparent), 0 (black ink), 1..k-1 (the source's flat tones)

    E(X) = sum_s w_s sum_blocks n_s ( wa (alpha_b - alpha*_b)^2 + |c_b - c*_b|^2 )      fidelity, scales 1, 2, 4
         + wh * sum_px | (v - mean3x3 v) - beta (v* - mean3x3 v*) |^2                   local contrast (detail)
         + lam_potts   * #4-pairs with different labels                                  flat regions
         + lam_orphan  * #opaque pixels with no 8-neighbour of the same label            no specks
         + lam_outline * #4-pairs tone|transparent (canvas edge counts as transparent)  closed black outline
         + lam_clump   * #2x2 all-black squares                                          thin ink

c is premultiplied OKLab colour, v the colour composited on the background, starred = source area means on
the same grid. Fidelity at scales 2 and 4 lets the optimiser *place* a line or an eye as whole pixels where
the source has it, even when no single cell is mostly ink; the local-contrast term rewards detail where the
source has detail (slightly exaggerated). Minimisation: iterated conditional modes on independent pixel
subsets (stride-4 lattice, vectorised) from a nearest-label start; exact mirror when the source is symmetric.
"""
from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from .palette import nearest_ramp, oklab, rgb

T = -1


@dataclass
class Weights:
    """Every hand-set weight of the model."""
    colors: int = 4            # opaque labels incl. black ink (4 = black + 3 tones; 3 = one NES sprite palette)
    tile_rule: int = 3         # max opaque colours per 8x8 tile (NES sprite hardware); 0 = off
    scales: tuple = ((1, 1.0), (2, 1.0), (4, 0.6))
    wa: float = 0.5            # silhouette (alpha) vs colour in fidelity (colour distances are ~0.1-0.8)
    wh: float = 0.6            # local-contrast fidelity
    beta: float = 1.4          # exaggeration of source detail
    lam_potts: float = 0.004
    lam_orphan: float = 0.02
    lam_outline: float = 0.3
    lam_clump: float = 0.004
    sweeps: int = 12
    sym_iou: float = 0.9       # mirror IoU of the source silhouette needed to enforce exact symmetry
    bg_tol: float = 0.10       # OKLab distance from the background colour that still counts as background
    work: int = 768            # source resized to at most this many px a side
    grey: bool = False         # treat the input as grayscale (tones = grey levels)


@dataclass
class Sprite:
    grid: np.ndarray           # (N, N) int; -1 transparent, 0 black, 1.. tones
    ramp: list                 # NES code per label
    meta: dict = field(default_factory=dict)

    def rgba(self, ramp=None):
        ramp = ramp or self.ramp
        out = np.zeros(self.grid.shape + (4,), np.uint8)
        for i, c in enumerate(ramp):
            out[self.grid == i] = rgb(c) + (255,)
        return Image.fromarray(out, 'RGBA')

    def preview(self, scale=8, bg=(52, 44, 72), ramp=None):
        im = self.rgba(ramp)
        base = Image.new('RGBA', im.size, bg + (255,))
        base.alpha_composite(im)
        return base.convert('RGB').resize((im.width * scale, im.height * scale), Image.NEAREST)

    def text(self):
        return '\n'.join(''.join('.' if v < 0 else str(v) for v in row) for row in self.grid)


# ------------------------------------------------------------------ source analysis

def load(img, work=768, grey=False):
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    img = img.convert('RGBA')
    if (np.asarray(img)[..., 3] < 128).mean() > 0.01:     # transparent PNG -> solid white background
        bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
        bg.alpha_composite(img)
        img = bg
    img = img.convert('L').convert('RGB') if grey else img.convert('RGB')
    if max(img.size) > work:
        s = work / max(img.size)
        img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    return oklab(np.asarray(img, float))


def matte(lab, tol=0.10):
    """Solid background: flood fill from the border over pixels close to the border's median colour."""
    border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    bgc = np.median(border, 0)
    sm = ndi.median_filter(lab, size=(3, 3, 1))
    near = np.sqrt(((sm - bgc) ** 2).sum(-1)) < tol
    lbl, _ = ndi.label(near)
    edge = np.unique(np.concatenate([lbl[0], lbl[-1], lbl[:, 0], lbl[:, -1]]))
    fg = ~np.isin(lbl, edge[edge > 0])
    fg = ndi.binary_opening(fg, iterations=1)
    lbl, n = ndi.label(fg)
    if n == 0:
        raise ValueError('no subject found on the background')
    areas = ndi.sum(fg, lbl, range(1, n + 1))
    return np.isin(lbl, 1 + np.where(areas >= 0.08 * areas.max())[0]), bgc


def tones(lab, fg, k):
    """k flat tones by k-means in OKLab over the foreground. Tone 0 is the ink and is pinned to black."""
    px = lab[fg]
    if len(px) > 60000:
        px = px[np.random.default_rng(0).choice(len(px), 60000, replace=False)]
    ink = px[:, 0] < 0.35
    rest = px[~ink]
    # farthest-point init on the non-ink pixels, then Lloyd
    c = [np.median(rest, 0)]
    for _ in range(k - 2):
        d = np.min([((rest - ci) ** 2).sum(-1) for ci in c], 0)
        c.append(rest[d.argmax()])
    c = np.array(c)
    for _ in range(25):
        a = ((rest[:, None] - c) ** 2).sum(-1).argmin(1)
        c = np.array([np.median(rest[a == j], 0) if (a == j).sum() else c[j] for j in range(len(c))])
    c = c[np.argsort(c[:, 0])]
    return np.vstack([[0.0, 0.0, 0.0], c])


def mirror_iou(fg):
    ys, xs = np.where(fg)
    m = fg[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return float((m & m[:, ::-1]).sum() / max(1, (m | m[:, ::-1]).sum()))


class Source:
    """Size-independent analysis of one input (reused across sizes)."""

    def __init__(self, img, w: 'Weights' = None):
        w = w or Weights()
        self.lab = load(img, w.work, w.grey)
        self.fg, self.bg = matte(self.lab, w.bg_tol)
        ys, xs = np.where(self.fg)
        self.box = (ys.min(), xs.min(), ys.max() + 1, xs.max() + 1)
        self.iou = mirror_iou(self.fg)
        self._t = {}

    def tones(self, k):
        if k not in self._t:
            self._t[k] = tones(self.lab, self.fg, k)
        return self._t[k]

    def targets(self, N, sym):
        """Per-cell alpha and premultiplied colour (area means) on the N x N canvas."""
        y0, x0, y1, x1 = self.box
        bh, bw = y1 - y0, x1 - x0
        s = N / max(bh, bw)
        h, wd = max(1, round(bh * s)), max(1, round(bw * s))
        if sym and (N - wd) % 2:
            wd += 1 if wd < N else -1
        a = self.fg[y0:y1, x0:x1].astype(np.float32)
        tA = np.zeros((N, N))
        tC = np.zeros((N, N, 3))
        oy, ox = (N - h) // 2, (N - wd) // 2
        tA[oy:oy + h, ox:ox + wd] = np.asarray(Image.fromarray(a).resize((wd, h), Image.BOX), float)
        for ch in range(3):
            pc = (self.lab[y0:y1, x0:x1, ch] * a).astype(np.float32)
            tC[oy:oy + h, ox:ox + wd, ch] = np.asarray(Image.fromarray(pc).resize((wd, h), Image.BOX), float)
        if sym:
            tA, tC = (tA + tA[:, ::-1]) / 2, (tC + tC[:, ::-1]) / 2
        return tA, tC


# ------------------------------------------------------------------ energy + optimiser

def _blocks(M, s):
    N = M.shape[0]
    nb = -(-N // s)
    P = np.zeros((nb * s, nb * s) + M.shape[2:])
    P[:N, :N] = M
    return P.reshape((nb, s, nb, s) + M.shape[2:]).sum((1, 3))


def _mean3(M):
    return ndi.uniform_filter(M, size=(3, 3) + (1,) * (M.ndim - 2), mode='nearest')


OFF8 = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]
OFF9 = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]


class Solver:
    def __init__(self, tA, tC, tone_lab, bg, w: Weights):
        self.N = N = tA.shape[0]
        self.w = w
        self.k = len(tone_lab)
        self.av = np.array([0.0] + [1.0] * self.k)                  # label+1 -> alpha
        self.cv = np.vstack([np.zeros(3), tone_lab])                # label+1 -> premultiplied colour
        self.vv = np.vstack([bg, tone_lab])                         # label+1 -> composited colour
        self.S = []
        for s, ws in w.scales:
            n = _blocks(np.ones((N, N)), s)
            self.S.append(dict(s=s, ws=ws, n=n, tA=_blocks(tA, s) / n, tC=_blocks(tC, s) / n[..., None]))
        self.tA, self.tC = tA, tC
        tV = tC + (1 - tA)[..., None] * bg
        self.tH = w.beta * (tV - _mean3(tV))

    def init(self):
        X = np.full((self.N, self.N), T)
        op = self.tA >= 0.5
        col = self.tC[op] / self.tA[op][:, None]
        X[op] = ((col[:, None] - self.cv[1:]) ** 2).sum(-1).argmin(1)
        return X

    def _sums(self, X):
        self.V = self.vv[X + 1].astype(float)
        a, c = self.av[X + 1], self.cv[X + 1]
        for d in self.S:
            d['A'] = _blocks(a, d['s'])
            d['C'] = _blocks(c, d['s'])

    @staticmethod
    def _same8(X):
        N = X.shape[0]
        P = np.pad(X, 1, constant_values=-9)
        c = np.zeros_like(X)
        for dy, dx in OFF8:
            c += P[1 + dy:1 + dy + N, 1 + dx:1 + dx + N] == X
        return c

    def energy(self, X):
        self._sums(X)
        w = self.w
        e = 0.0
        for d in self.S:
            e += d['ws'] * (d['n'] * (w.wa * (d['A'] / d['n'] - d['tA']) ** 2
                                      + ((d['C'] / d['n'][..., None] - d['tC']) ** 2).sum(-1))).sum()
        e += w.wh * ((self.V - _mean3(self.V) - self.tH) ** 2).sum()
        P = np.pad(X, 1, constant_values=T)
        for A, B in ((P[1:-1, 1:], P[1:-1, :-1]), (P[1:, 1:-1], P[:-1, 1:-1])):
            inner = (A != T) | (B != T)
            e += w.lam_potts * ((A != B) & inner).sum()
            e += w.lam_outline * (((A >= 1) & (B == T)) | ((B >= 1) & (A == T))).sum()
        e += w.lam_orphan * (self._same8(X)[X != T] == 0).sum()
        b = X == 0
        e += w.lam_clump * (b[1:, 1:] & b[:-1, 1:] & b[1:, :-1] & b[:-1, :-1]).sum()
        return float(e)

    def step(self, X, oy, ox, allowed=None):
        """One ICM update of the independent subset {y % 4 == oy, x % 4 == ox}."""
        N, w = self.N, self.w
        ys, xs = np.meshgrid(np.arange(oy, N, 4), np.arange(ox, N, 4), indexing='ij')
        ys, xs = ys.ravel(), xs.ravel()
        if len(ys) == 0:
            return 0
        old = X[ys, xs]
        P = np.pad(X, 1, constant_values=T)
        n4 = np.stack([P[ys + 1, xs + 2], P[ys + 2, xs + 1], P[ys + 1, xs], P[ys, xs + 1]], 1)
        e4 = np.stack([xs + 1 >= N, ys + 1 >= N, xs - 1 < 0, ys - 1 < 0], 1)
        n8 = np.stack([P[ys + 1 + dy, xs + 1 + dx] for dy, dx in OFF8], 1)
        C = np.pad(self._same8(X), 1, constant_values=99)
        cq = np.stack([C[ys + 1 + dy, xs + 1 + dx] for dy, dx in OFF8], 1)
        n8in = np.stack([(ys + dy >= 0) & (ys + dy < N) & (xs + dx >= 0) & (xs + dx < N) for dy, dx in OFF8], 1)
        H = np.pad(self.V - _mean3(self.V), ((1, 1), (1, 1), (0, 0)))
        tH = np.pad(self.tH, ((1, 1), (1, 1), (0, 0)))
        in9 = np.stack([(ys + dy >= 0) & (ys + dy < N) & (xs + dx >= 0) & (xs + dx < N) for dy, dx in OFF9], 1)
        H9 = np.stack([H[ys + 1 + dy, xs + 1 + dx] for dy, dx in OFF9], 1)      # (P, 9, 3)
        T9 = np.stack([tH[ys + 1 + dy, xs + 1 + dx] for dy, dx in OFF9], 1)
        self9 = np.array([o == (0, 0) for o in OFF9])
        # black 2x2 squares containing the pixel: the other three cells of each of its 4 squares
        Pb = np.pad(X == 0, 1)
        sq = []
        for dy, dx in ((-1, -1), (-1, 0), (0, -1), (0, 0)):
            cells = [(dy + a, dx + b) for a in (0, 1) for b in (0, 1) if (dy + a, dx + b) != (0, 0)]
            sq.append(np.all([Pb[ys + 1 + cy, xs + 1 + cx] for cy, cx in cells], 0))
        sq = np.sum(sq, 0)

        def local(l):
            lv = np.full(len(old), l)
            e = np.zeros(len(old))
            dA = self.av[lv + 1] - self.av[old + 1]
            dC = self.cv[lv + 1] - self.cv[old + 1]
            for d in self.S:
                s = d['s']
                by, bx = ys // s, xs // s
                n = d['n'][by, bx]
                A = (d['A'][by, bx] + dA) / n
                Cc = (d['C'][by, bx] + dC) / n[:, None]
                e += d['ws'] * n * (w.wa * (A - d['tA'][by, bx]) ** 2 + ((Cc - d['tC'][by, bx]) ** 2).sum(-1))
            dv = (self.vv[lv + 1] - self.vv[old + 1])[:, None, :]
            dH = np.where(self9[None, :, None], dv * (8 / 9), -dv / 9)
            e += w.wh * ((((H9 + dH - T9) ** 2).sum(-1)) * in9).sum(1)
            inner = (lv[:, None] != T) | (n4 != T)
            e += w.lam_potts * ((n4 != lv[:, None]) & inner).sum(1)
            out = ((lv[:, None] >= 1) & ((n4 == T) | e4)) | ((lv[:, None] == T) & (n4 >= 1))
            e += w.lam_outline * out.sum(1)
            own = (lv != T) & ((n8 == lv[:, None]).sum(1) == 0)
            cqn = cq - (n8 == old[:, None]) + (n8 == lv[:, None])
            e += w.lam_orphan * (own + ((n8 != T) & n8in & (cqn == 0)).sum(1))
            e += w.lam_clump * (lv == 0) * sq
            return e

        cand = np.arange(-1, self.k) if allowed is None else np.asarray(allowed)
        E = np.stack([local(l) for l in cand], 1)
        best = cand[E.argmin(1)]
        ch = best != old
        if ch.any():
            yc, xc, bo, oo = ys[ch], xs[ch], best[ch], old[ch]
            X[yc, xc] = bo
            for d in self.S:
                s = d['s']
                np.add.at(d['A'], (yc // s, xc // s), self.av[bo + 1] - self.av[oo + 1])
                np.add.at(d['C'], (yc // s, xc // s), self.cv[bo + 1] - self.cv[oo + 1])
            self.V[yc, xc] = self.vv[bo + 1]
        return int(ch.sum())

    def solve(self, X=None, seed=0):
        X = self.init() if X is None else X.copy()
        self._sums(X)
        rng = np.random.default_rng(seed)
        offs = [(a, b) for a in range(4) for b in range(4)]
        hist = []
        for _ in range(self.w.sweeps):
            ch = sum(self.step(X, *offs[i]) for i in rng.permutation(16))
            hist.append(ch)
            if ch == 0:
                break
        return X, hist


def _mirror(X, left=True):
    N = X.shape[1]
    X = X.copy()
    if left:
        X[:, N - N // 2:] = X[:, :N // 2][:, ::-1]
    else:
        X[:, :N // 2] = X[:, N - N // 2:][:, ::-1]
    return X


def close_outline(X):
    """Hard guarantee: a tone pixel never 4-touches transparency or the canvas edge (it becomes ink)."""
    X = X.copy()
    P = np.pad(X, 1, constant_values=T)
    bad = (X >= 1) & ((P[1:-1, 2:] == T) | (P[1:-1, :-2] == T) | (P[2:, 1:-1] == T) | (P[:-2, 1:-1] == T))
    X[bad] = 0
    return X


def tile_reduce(X, maxc, tone_lab):
    """<= maxc opaque labels per 8x8 tile: in an over-full tile the least-used tone becomes the nearest other."""
    X = X.copy()
    N = X.shape[0]
    for y in range(0, N, 8):
        for x in range(0, N, 8):
            t = X[y:y + 8, x:x + 8]
            while True:
                v, c = np.unique(t[t >= 0], return_counts=True)
                if len(v) <= maxc:
                    break
                drop = min((ci, vi) for vi, ci in zip(v, c) if vi != 0)[1]
                rest = [vi for vi in v if vi != drop]
                t[t == drop] = min(rest, key=lambda r: ((tone_lab[r] - tone_lab[drop]) ** 2).sum())
    return X


def render(src, N, w: Weights = None, ramp=None):
    """Source (or image path / PIL image) -> Sprite at N x N."""
    w = w or Weights()
    src = src if isinstance(src, Source) else Source(src, w)
    sym = src.iou >= w.sym_iou
    tA, tC = src.targets(N, sym)
    tl = src.tones(w.colors)
    sol = Solver(tA, tC, tl, src.bg, w)
    X, hist = sol.solve()
    if sym:
        X = min([_mirror(X, True), _mirror(X, False)], key=sol.energy)
        X, h2 = sol.solve(X)
        X = min([_mirror(X, True), _mirror(X, False)], key=sol.energy)
        hist += h2
    X = close_outline(X)
    if w.tile_rule:
        X = tile_reduce(X, w.tile_rule, tl)
        if sym:
            X = _mirror(X, True)
    ramp = ramp or nearest_ramp(tl)
    return Sprite(X, list(ramp), dict(size=N, tones=np.round(tl, 3).tolist(), symmetric=bool(sym),
                                      mirror_iou=round(src.iou, 3), energy=round(sol.energy(X), 3), sweeps=hist))
