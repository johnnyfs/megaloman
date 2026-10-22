"""Vendored from johnnyfs/fusion-man src/gridsnap.py; changes: package-relative import, pitch range params."""
"""Recover a true 1x sprite from image-model 'pixel art' (upscaled, slightly irregular grid).

Detects the pixel pitch and phase per axis from edge periodicity, samples each cell's centre region,
and maps it to an exact palette. Background (flood-filled from the borders) becomes transparent.
usage: gridsnap.py in.png out_prefix [--pitch P] [--palette KEY=HEX ...] [--keep-bg]
"""
import argparse, sys, os
import numpy as np
from PIL import Image
from .pixelize import background_mask, to_lab

def fft_pitch(edge1d, lo=4.0, hi=20.0):
    f = np.abs(np.fft.rfft(edge1d - edge1d.mean())); fr = np.fft.rfftfreq(len(edge1d))
    ok = (fr > 1 / hi) & (fr < 1 / lo)
    return 1 / fr[ok][f[ok].argmax()]

def pitch_phase(edge1d, lo=5.0, hi=16.0):
    if hi - lo > 0.05:                      # spectrum picks the fundamental; avoids harmonics
        p0 = fft_pitch(edge1d, lo, hi); lo, hi = p0 - 0.08, p0 + 0.08
    best = (0, None, None)
    n = len(edge1d)
    for p in np.arange(lo, hi, 0.02):
        # score = mean edge energy sampled at a lattice of period p, best over phase
        for ph in np.arange(0, p, 0.25):
            idx = np.round(np.arange(ph, n, p)).astype(int)
            idx = idx[idx < n]
            sc = edge1d[idx].mean() - edge1d.mean()
            if sc > best[0]: best = (sc, p, ph)
    return best[1], best[2]

def snap(path, pal, pitch=None, keep_bg=False, lo=5.0, hi=16.0):
    img = np.array(Image.open(path).convert('RGB')).astype(float)
    g = np.abs(np.diff(img, axis=1)).sum(-1); gx = g.sum(0)
    g = np.abs(np.diff(img, axis=0)).sum(-1); gy = g.sum(1)
    px, ox = pitch_phase(gx, lo, hi) if pitch is None else (pitch, pitch_phase(gx, pitch, pitch + 0.01)[1])
    py, oy = pitch_phase(gy, lo, hi) if pitch is None else (pitch, pitch_phase(gy, pitch, pitch + 0.01)[1])
    bgm = np.zeros(img.shape[:2], bool) if keep_bg else background_mask(img.astype(np.uint8))
    bgc = np.median(np.array([img[0, 0], img[0, -1], img[-1, 0], img[-1, -1]]), 0)
    keys = list(pal); cols = np.array([[int(pal[k][i:i + 2], 16) for i in (0, 2, 4)] for k in keys], float)
    plab = to_lab(cols)
    xs = np.arange(ox + 1, img.shape[1] - 1, px); ys = np.arange(oy + 1, img.shape[0] - 1, py)
    rows = []
    for y0 in ys:
        r = ''
        for x0 in xs:
            a, b = int(y0 + py * 0.3), int(y0 + py * 0.7) + 1
            c, d = int(x0 + px * 0.3), int(x0 + px * 0.7) + 1
            if bgm[a:b, c:d].mean() > 0.5: r += '.'; continue
            cell = img[a:b, c:d].reshape(-1, 3)
            # enclosed background (ring hole, between legs) is not reached by the border flood fill
            if not keep_bg and np.linalg.norm(cell.mean(0) - bgc) < 22 and cell.std(0).max() < 12: r += '.'; continue
            lab = to_lab(cell)
            k = ((lab[:, None, :] - plab[None]) ** 2).sum(-1).argmin(1)
            r += keys[np.bincount(k, minlength=len(keys)).argmax()]
        rows.append(r)
    if keep_bg:
        return rows, (px, py)
    # trim empty border rows/cols
    while rows and set(rows[0]) == {'.'}: rows.pop(0)
    while rows and set(rows[-1]) == {'.'}: rows.pop()
    L = min(len(r) - len(r.lstrip('.')) for r in rows); R = max(len(r.rstrip('.')) for r in rows)
    return [r[L:R] for r in rows], (px, py)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('inp'); ap.add_argument('out'); ap.add_argument('--pitch', type=float)
    ap.add_argument('--palette', nargs='+', metavar='KEY=HEX', help='palette to snap to (default: the toad4 set)')
    ap.add_argument('--keep-bg', action='store_true', help='full-screen backgrounds: no transparency, no trimming')
    a = ap.parse_args()
    pal = dict(kv.split('=') for kv in a.palette) if a.palette else {'k': '000000', 'p': '6844FC', 'c': '00E8D8', 'w': 'FCFCFC'}
    rows, (px, py) = snap(a.inp, pal, a.pitch, a.keep_bg)
    print('pitch', round(px, 2), round(py, 2), 'size', len(rows[0]), 'x', len(rows))
    from pixelize import save
    import pixelize
    pixelize.PALETTES['snap'] = pal
    save(rows, a.out, 'snap')
