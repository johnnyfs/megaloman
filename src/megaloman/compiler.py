"""Pixel-art compiler: re-render a sprite at a target size, keeping its lines and salient features.

Layers, each scaled the way it survives best:
  silhouette + fills   area-resampled (low-frequency shapes survive resampling)
  outer outline        regenerated as a clean 1-px line on the new silhouette (Capcom: continuous black outline)
  interior lines       dark pixels thinned to centre lines, the lines moved to the new scale and redrawn 1-px
  salient features     survival rule: every small source feature (eye white, pupil, highlight) keeps >= 1 pixel
Everything is deterministic, so frames compiled from the same description share every unchanged pixel.
"""
import numpy as np
from PIL import Image

from .vendor.pixelize import zhang_suen

DARK = 90   # sum of RGB below this = ink


def _arr(im):
    return np.array(im.convert('RGBA'))


def _samples(sw, sh, tw, th, n=8):
    """For every target cell, n*n sample coordinates in source pixel space -> (th, tw, n*n, 2) int indices."""
    sx, sy = sw / tw, sh / th
    off = (np.arange(n) + 0.5) / n
    ys = (np.arange(th)[:, None] + off[None, :]) * sy          # th x n
    xs = (np.arange(tw)[:, None] + off[None, :]) * sx          # tw x n
    yy = np.clip(np.floor(ys).astype(int), 0, sh - 1)
    xx = np.clip(np.floor(xs).astype(int), 0, sw - 1)
    Y = np.broadcast_to(yy[:, None, :, None], (th, tw, n, n)).reshape(th, tw, n * n)
    X = np.broadcast_to(xx[None, :, None, :], (th, tw, n, n)).reshape(th, tw, n * n)
    return Y, X


def _components(mask):
    """4-connected components -> list of (ys, xs) arrays."""
    h, w = mask.shape
    seen = np.zeros_like(mask)
    out = []
    for y0, x0 in zip(*np.nonzero(mask)):
        if seen[y0, x0]:
            continue
        st, ys, xs = [(y0, x0)], [], []
        seen[y0, x0] = True
        while st:
            y, x = st.pop()
            ys.append(y); xs.append(x)
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    st.append((ny, nx))
        out.append((np.array(ys), np.array(xs)))
    return out


def _line(img_mask, x0, y0, x1, y1):
    """Bresenham onto a boolean mask."""
    x0, y0, x1, y1 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err = dx + dy
    h, w = img_mask.shape
    while True:
        if 0 <= y0 < h and 0 <= x0 < w:
            img_mask[y0, x0] = True
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy; x0 += sx
        if e2 <= dx:
            err += dx; y0 += sy


def _pixel_perfect(mask):
    """Remove L-corner pixels so lines are clean 1-px 8-connected strokes."""
    m = mask.copy()
    h, w = m.shape
    changed = True
    while changed:
        changed = False
        for y in range(h):
            for x in range(w):
                if not m[y, x]:
                    continue
                n = lambda dy, dx: 0 <= y + dy < h and 0 <= x + dx < w and m[y + dy, x + dx]
                # an L corner: two orthogonal neighbours that are diagonal to each other, and not a junction
                for (a, b) in (((0, 1), (1, 0)), ((0, 1), (-1, 0)), ((0, -1), (1, 0)), ((0, -1), (-1, 0))):
                    if n(*a) and n(*b) and not n(a[0] + b[0], a[1] + b[1]):
                        others = sum(n(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy or dx)) - 2
                        if others == 0:
                            m[y, x] = False
                            changed = True
                            break
    return m


def compile_sprite(im, tw, th, opts=None):
    """Re-render RGBA sprite `im` at tw x th. Returns (RGBA image, info)."""
    o = {'coverage': 0.5, 'outer_band': 1.5, 'features': True, 'interior_lines': True, 'ink_color': None}
    o.update(opts or {})
    a = _arr(im)
    sh, sw = a.shape[:2]
    op = a[..., 3] > 0
    dark = op & (a[..., :3].astype(int).sum(-1) < DARK)
    ink = tuple(int(v) for v in (o['ink_color'] or (a[dark][0, :3] if dark.any() else (0, 0, 0))))
    Y, X = _samples(sw, sh, tw, th)
    # 1. silhouette by coverage
    cov = op[Y, X].mean(-1)
    sil = cov >= o['coverage']
    # 2. fills: majority non-ink colour among samples (ink excluded so lines don't swallow fills)
    key = a[..., 0].astype(np.int64) * 65536 + a[..., 1].astype(np.int64) * 256 + a[..., 2]
    key = np.where(op & ~dark, key, -1)
    sk = key[Y, X]
    out = np.zeros((th, tw, 4), np.uint8)
    fill_key = np.full((th, tw), -1, np.int64)
    for y in range(th):
        for x in range(tw):
            if not sil[y, x]:
                continue
            v = sk[y, x][sk[y, x] >= 0]
            if len(v):
                vals, cnt = np.unique(v, return_counts=True)
                fill_key[y, x] = vals[cnt.argmax()]
    # cells inside the silhouette with no fill sample take the nearest filled neighbour (or ink)
    for y, x in zip(*np.nonzero(sil & (fill_key < 0))):
        best, bd = -1, 1e9
        for yy, xx in zip(*np.nonzero(fill_key >= 0)):
            d = (yy - y) ** 2 + (xx - x) ** 2
            if d < bd:
                bd, best = d, fill_key[yy, xx]
        fill_key[y, x] = best
    def kc(k):
        return (int(k // 65536), int(k // 256 % 256), int(k % 256))
    for y, x in zip(*np.nonzero(sil)):
        out[y, x] = (*(kc(fill_key[y, x]) if fill_key[y, x] >= 0 else ink), 255)
    # 3. interior lines: skeleton of ink away from the silhouette edge, moved to target scale, redrawn 1-px
    lines = np.zeros((th, tw), bool)
    if o['interior_lines'] and dark.any():
        from scipy.ndimage import distance_transform_edt
        dist = distance_transform_edt(np.pad(op, 1))[1:-1, 1:-1]
        skel = zhang_suen(dark.astype(np.uint8)).astype(bool)
        inner = skel & (dist > o['outer_band'] * max(1.0, sw / tw))
        sx, sy = tw / sw, th / sh
        pts = list(zip(*np.nonzero(inner)))
        S = set(pts)
        for y, x in pts:
            tx, ty = (x + 0.5) * sx - 0.5, (y + 0.5) * sy - 0.5
            lines[int(round(np.clip(ty, 0, th - 1))), int(round(np.clip(tx, 0, tw - 1)))] = True
            for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
                if (y + dy, x + dx) in S:
                    _line(lines, tx, ty, (x + dx + 0.5) * sx - 0.5, (y + dy + 0.5) * sy - 0.5)
        lines = _pixel_perfect(lines & sil)
    # 4. salient features: every small source component (fill or ink blob) keeps >= 1 pixel at its scaled centroid
    if o['features']:
        sx, sy = tw / sw, th / sh
        for k in np.unique(key[key >= 0]):
            for ys, xs in _components(key == k):
                if len(ys) < 2:
                    continue
                ty = np.clip(((ys + 0.5) * sy - 0.5).round().astype(int), 0, th - 1)
                tx = np.clip(((xs + 0.5) * sx - 0.5).round().astype(int), 0, tw - 1)
                if not (fill_key[ty, tx] == k).any():
                    cy, cx = int(round(ty.mean())), int(round(tx.mean()))
                    if sil[cy, cx]:
                        out[cy, cx] = (*kc(k), 255); fill_key[cy, cx] = k
                        lines[cy, cx] = False
    # 5. outer outline: 1-px border of the silhouette
    pad = np.pad(sil, 1)
    border = sil & ~(pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])
    for m in (lines, border):
        out[m] = (*ink, 255)
    return Image.fromarray(out, 'RGBA'), {'size': [tw, th], 'opts': o}


# ------------------------------------------------------------------ v2
# Lesson from v1: Capcom uses black as a FILL too (Telly's eye band, the Met's face shadow). Ink is split by
# thickness: thin ink = lines (ink priority per cell, then thinned to 1 px), thick ink = a fill colour like any other.

def _offset_samples(sw, sh, tw, th, ox, oy, n=8):
    sx, sy = sw / tw, sh / th
    off = (np.arange(n) + 0.5) / n
    ys = (np.arange(th)[:, None] + off[None, :] + oy) * sy
    xs = (np.arange(tw)[:, None] + off[None, :] + ox) * sx
    yy = np.clip(np.floor(ys).astype(int), 0, sh - 1)
    xx = np.clip(np.floor(xs).astype(int), 0, sw - 1)
    Y = np.broadcast_to(yy[:, None, :, None], (th, tw, n, n)).reshape(th, tw, n * n)
    X = np.broadcast_to(xx[None, :, None, :], (th, tw, n, n)).reshape(th, tw, n * n)
    valid = np.broadcast_to(((ys >= 0) & (ys < sh))[:, None, :, None] & ((xs >= 0) & (xs < sw))[None, :, None, :],
                            (th, tw, n, n)).reshape(th, tw, n * n)
    return Y, X, valid


def _render_v2(a, tw, th, ox, oy, o):
    from scipy.ndimage import binary_opening
    sh, sw = a.shape[:2]
    s = max(sw / tw, sh / th)
    op = a[..., 3] > 0
    ink = op & (a[..., :3].astype(int).sum(-1) < DARK)
    r = max(1, int(np.ceil(s * o['fill_ink_scale'])))
    fill_ink = binary_opening(ink, structure=np.ones((2 * r + 1, 2 * r + 1), bool)) if ink.any() else ink
    line_ink = ink & ~fill_ink
    # dot features (pupils, bolts): small ink blobs that do not touch the outline network -> never thinned
    dots = np.zeros_like(ink)
    if o.get('dots', True) and ink.any():
        from scipy.ndimage import label, binary_dilation
        edge_ink = ink & binary_dilation(~op, structure=np.ones((3, 3), bool))       # ink on the silhouette edge
        lab, n = label(ink, structure=np.ones((3, 3), int))
        for c in range(1, n + 1):
            comp = lab == c
            if not (comp & edge_ink).any() and comp.sum() <= o.get('dot_max_area', 16) * max(1.0, s) ** 2:
                dots |= comp
        line_ink &= ~dots
        fill_ink &= ~dots
    key = a[..., 0].astype(np.int64) * 65536 + a[..., 1].astype(np.int64) * 256 + a[..., 2]
    key = np.where(op & ~line_ink, key, -1)        # fills incl. fill-ink; transparent and line ink excluded
    Y, X, valid = _offset_samples(sw, sh, tw, th, ox, oy)
    cov = (op[Y, X] & valid).sum(-1) / valid.shape[-1]
    lcov = (line_ink[Y, X] & valid).sum(-1) / valid.shape[-1]
    sil = cov >= o['coverage']
    lines = sil & (lcov >= o['ink_th'])
    sk = np.where(valid, key[Y, X], -1)
    fill = np.full((th, tw), -1, np.int64)
    for y, x in zip(*np.nonzero(sil)):
        v = sk[y, x][sk[y, x] >= 0]
        if len(v):
            vals, cnt = np.unique(v, return_counts=True)
            fill[y, x] = vals[cnt.argmax()]
    # thin the line mask to 1 px (never thinning away the silhouette edge: keep border ink)
    if lines.any() and o.get('thin', 'skeleton') == 'outer':
        # Capcom outline rule: the outer outline is exactly 1 px; interior ink (pupils, mouths) is left alone.
        # Peel ink cells that sit just inside the border ink and whose fill neighbour would take their place.
        pad = np.pad(sil, 1)
        border = sil & ~(pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])
        inner_ring = sil & ~border & np.pad(border, 1)[1:-1, 1:-1]
        nb_border = np.zeros_like(sil)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb_border |= np.roll(np.roll(border & lines, dy, 0), dx, 1)
        peel = lines & ~border & nb_border
        # only peel ink that has a non-ink fill neighbour to inherit (otherwise it's a real interior line)
        has_fill = np.zeros_like(sil)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            has_fill |= np.roll(np.roll(sil & ~lines & (fill >= 0), dy, 0), dx, 1)
        peel &= has_fill
        for y, x in zip(*np.nonzero(peel)):
            cand = [(y + dy, x + dx) for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))
                    if 0 <= y + dy < th and 0 <= x + dx < tw and sil[y + dy, x + dx] and not lines[y + dy, x + dx] and fill[y + dy, x + dx] >= 0]
            if cand:
                fill[y, x] = fill[cand[0]]
        lines = lines & ~peel
    elif lines.any():
        thin = zhang_suen(lines.astype(np.uint8)).astype(bool)
        pad = np.pad(sil, 1)
        border = sil & ~(pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])
        lines = thin | (lines & border)
    # outline closure: silhouette edge cells whose source footprint touched line ink become ink
    pad = np.pad(sil, 1)
    border = sil & ~(pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])
    lines |= border & (lcov > 0)
    # cells with no fill sample and no line: nearest filled neighbour
    for y, x in zip(*np.nonzero(sil & ~lines & (fill < 0))):
        cand = np.argwhere(fill >= 0)
        if len(cand):
            d = ((cand - [y, x]) ** 2).sum(1)
            fill[y, x] = fill[tuple(cand[d.argmin()])]
    # place dots: each blob covers its scaled footprint, at least one cell
    dot_cells = np.zeros((th, tw), bool)
    if dots.any():
        from scipy.ndimage import label
        lab, n = label(dots, structure=np.ones((3, 3), int))
        dcov = np.zeros((th, tw))
        for c in range(1, n + 1):
            comp = lab == c
            ccov = (comp[Y, X] & valid).sum(-1) / valid.shape[-1]
            cells = sil & (ccov >= 0.35)
            if not cells.any():
                cells = np.zeros_like(sil); cells.flat[int(np.argmax(ccov))] = True
            dot_cells |= cells
        lines = lines | dot_cells
    return sil, lines, fill, key


def compile_v2(im, tw, th, opts=None):
    o = {'coverage': 0.4, 'ink_th': 0.5, 'fill_ink_scale': 0.5, 'phase_search': True, 'features': True,
         'thin': 'outer', 'dots': True}   # tuned: experiments/compiler/v2d_sweep2, v6_thin_sweep, v6_outer_thin
    o.update(opts or {})
    a = _arr(im)
    sh, sw = a.shape[:2]
    op = a[..., 3] > 0
    dark = op & (a[..., :3].astype(int).sum(-1) < DARK)
    inkc = tuple(int(v) for v in (a[dark][0, :3] if dark.any() else (0, 0, 0)))
    def kc(k):
        return (int(k // 65536), int(k // 256 % 256), int(k % 256))
    def build(ox, oy):
        sil, lines, fill, key = _render_v2(a, tw, th, ox, oy, o)
        out = np.zeros((th, tw, 4), np.uint8)
        for y, x in zip(*np.nonzero(sil)):
            out[y, x] = (*(inkc if (lines[y, x] or fill[y, x] < 0) else kc(fill[y, x])), 255)
        return out, sil, lines, fill, key
    best = None
    offs = ([tuple(o['phase'])] if o.get('phase') is not None else
            [(-0.5 + i / 4, -0.5 + j / 4) for i in range(4) for j in range(4)] if o['phase_search'] else [(0.0, 0.0)])
    for ox, oy in offs:
        out, sil, lines, fill, key = build(ox, oy)
        # self-consistency: re-enlarge (nearest, same phase) and compare with the input
        jx = np.floor((np.arange(sw) + 0.5) * tw / sw - ox).astype(int)     # source pixel -> target cell (same phase)
        jy = np.floor((np.arange(sh) + 0.5) * th / sh - oy).astype(int)
        okx, oky = (jx >= 0) & (jx < tw), (jy >= 0) & (jy < th)
        up = out[np.clip(jy, 0, th - 1)][:, np.clip(jx, 0, tw - 1)].copy()
        up[~oky] = 0; up[:, ~okx] = 0
        err = ((up[..., 3] > 0) != op).sum() + (op & (up[..., 3] > 0) & (up[..., :3] != a[..., :3]).any(-1)).sum()
        if best is None or err < best[0]:
            best = (err, ox, oy, out, sil, lines, fill, key)
    err, ox, oy, out, sil, lines, fill, key = best
    if o['features']:
        sx, sy = tw / sw, th / sh
        for k in np.unique(key[key >= 0]):
            for ys, xs in _components(key == k):
                if len(ys) < 2:
                    continue
                ty = np.clip(((ys + 0.5) * sy - 0.5 - oy).round().astype(int), 0, th - 1)
                tx = np.clip(((xs + 0.5) * sx - 0.5 - ox).round().astype(int), 0, tw - 1)
                if not (fill[ty, tx] == k).any():
                    cy, cx = int(round(ty.mean())), int(round(tx.mean()))
                    if sil[cy, cx]:
                        out[cy, cx] = (*kc(k), 255)
    return Image.fromarray(out, 'RGBA'), {'size': [tw, th], 'phase': [ox, oy], 'recon_err': int(err), 'opts': o}
