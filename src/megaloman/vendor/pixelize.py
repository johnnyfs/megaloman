"""Vendored from johnnyfs/fusion-man src/pixelize.py (unchanged)."""
"""Turn a painted reference (e.g. image-model key art) into a clean NES pixel underpainting.

Plain downscaling smears ink lines into grey mush and averages highlights away. This instead:
  1. removes the flat background (flood fill from the edges, so interior greys survive)
  2. maps every hi-res pixel to the asset's NES palette in Lab space (ink = dark pixels)
  3. downsamples per target cell with priorities: ink if enough ink falls in the cell, white gloss if a
     strong highlight does, else the majority colour
  4. thins interior ink to 1-px lines (Zhang-Suen) and adds a 1-px silhouette outline
  5. removes orphan pixels
usage: pixelize.py in.png out_prefix --width 104 [--palette toad] [--ink 0.22] [--gloss 0.30]
writes <out_prefix>.png (1x), <out_prefix>_x6.png, <out_prefix>.txt (key grid)
"""
import argparse
import numpy as np
from PIL import Image

PALETTES = {
    'toad': {'k': '000000', 'd': '4428BC', 'p': '6844FC', 'l': '9878F8', 'w': 'FCFCFC',
             'e': '004058', 't': '008888', 'c': '00E8D8', 'M': 'F878F8'},
}

def to_lab(rgb):
    rgb = rgb / 255.0
    rgb = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = rgb @ m.T / np.array([0.9505, 1.0, 1.089])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)

def background_mask(rgb, tol=28):
    H, W, _ = rgb.shape
    corners = np.array([rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]], float)
    bg = np.median(corners, 0)
    near = np.linalg.norm(rgb.astype(float) - bg, axis=-1) < tol
    out = np.zeros((H, W), bool)
    stack = [(0, x) for x in range(W)] + [(H - 1, x) for x in range(W)] + [(y, 0) for y in range(H)] + [(y, W - 1) for y in range(H)]
    while stack:
        y, x = stack.pop()
        if out[y, x] or not near[y, x]: continue
        out[y, x] = True
        if y > 0: stack.append((y - 1, x))
        if y < H - 1: stack.append((y + 1, x))
        if x > 0: stack.append((y, x - 1))
        if x < W - 1: stack.append((y, x + 1))
    return out

def zhang_suen(mask):
    m = mask.copy().astype(np.uint8)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            P = np.pad(m, 1)
            p2, p3, p4, p5 = P[:-2, 1:-1], P[:-2, 2:], P[1:-1, 2:], P[2:, 2:]
            p6, p7, p8, p9 = P[2:, 1:-1], P[2:, :-2], P[1:-1, :-2], P[:-2, :-2]
            nb = [p2, p3, p4, p5, p6, p7, p8, p9]
            B = sum(n.astype(int) for n in nb)
            A = sum(((nb[i] == 0) & (nb[(i + 1) % 8] == 1)).astype(int) for i in range(8))
            c1 = (p2 * p4 * p6 == 0) if step == 0 else (p2 * p4 * p8 == 0)
            c2 = (p4 * p6 * p8 == 0) if step == 0 else (p2 * p6 * p8 == 0)
            rem = (m == 1) & (B >= 2) & (B <= 6) & (A == 1) & c1 & c2
            if rem.any():
                m[rem] = 0; changed = True
    return m.astype(bool)

def pixelize(path, width, palette='toad', ink_th=0.22, gloss_th=0.30, height=None):
    img = np.array(Image.open(path).convert('RGB'))
    bgm = background_mask(img)
    ys, xs = np.where(~bgm)
    img, bgm = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1], bgm[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    H0, W0 = bgm.shape
    if height is None: height = round(width * H0 / W0)
    keys = list(PALETTES[palette])
    cols = np.array([[int(PALETTES[palette][k][i:i + 2], 16) for i in (0, 2, 4)] for k in keys], float)
    lab, plab = to_lab(img.astype(float)), to_lab(cols)
    dist = ((lab[:, :, None, :] - plab[None, None]) ** 2).sum(-1)
    cls = dist.argmin(-1)
    kidx, widx = keys.index('k'), keys.index('w')
    out = np.full((height, width), '.', object)
    for ty in range(height):
        y0, y1 = int(ty * H0 / height), max(int((ty + 1) * H0 / height), int(ty * H0 / height) + 1)
        for tx in range(width):
            x0, x1 = int(tx * W0 / width), max(int((tx + 1) * W0 / width), int(tx * W0 / width) + 1)
            cell_bg = bgm[y0:y1, x0:x1]
            if cell_bg.mean() > 0.5: continue
            c = cls[y0:y1, x0:x1][~cell_bg]
            n = len(c)
            counts = np.bincount(c, minlength=len(keys))
            if counts[kidx] / n >= ink_th: out[ty, tx] = 'k'; continue
            if counts[widx] / n >= gloss_th: out[ty, tx] = 'w'; continue
            counts[kidx] = 0
            out[ty, tx] = keys[int(counts.argmax())]
    opaque = out != '.'
    # silhouette edge, then 1-px interior lines
    edge = opaque & ~(np.pad(opaque, 1)[:-2, 1:-1] & np.pad(opaque, 1)[2:, 1:-1] & np.pad(opaque, 1)[1:-1, :-2] & np.pad(opaque, 1)[1:-1, 2:])
    ink = (out == 'k') & ~edge
    # solid ink (pupils, mouth interior): any pixel inside a full 3x3 ink block, grown by 1, is kept as fill
    P = np.pad(ink, 1)
    core = np.ones_like(ink)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            core &= P[dy:dy + ink.shape[0], dx:dx + ink.shape[1]]
    Pc = np.pad(core, 1); solid = core.copy()
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            solid |= Pc[dy:dy + ink.shape[0], dx:dx + ink.shape[1]]
    solid &= ink
    thin = zhang_suen(ink & ~solid) | solid
    removed = ink & ~thin
    def neighbour_mode(y, x):
        vals = [out[yy, xx] for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1))
                if 0 <= yy < height and 0 <= xx < width and out[yy, xx] not in ('.', 'k')]
        return max(set(vals), key=vals.count) if vals else 'k'
    for y, x in zip(*np.where(removed)):
        out[y, x] = neighbour_mode(y, x)
    out[edge] = 'k'
    # drop isolated ink specks (1-2 px fragments of faint seams), keep anything touching other ink/edge
    inkm = out == 'k'
    seen = np.zeros_like(inkm)
    for y0, x0 in zip(*np.where(inkm & ~edge)):
        if seen[y0, x0]: continue
        comp, st = [], [(y0, x0)]; seen[y0, x0] = True
        while st:
            y, x = st.pop(); comp.append((y, x))
            for yy in (y - 1, y, y + 1):
                for xx in (x - 1, x, x + 1):
                    if 0 <= yy < height and 0 <= xx < width and inkm[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True; st.append((yy, xx))
        if len(comp) <= 2 and not any(edge[y, x] for y, x in comp):
            for y, x in comp: out[y, x] = neighbour_mode(y, x)
    # orphan cleanup (non-ink pixel with all 4 neighbours one other colour)
    for _ in range(2):
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                v = out[y, x]
                if v in ('.', 'k'): continue
                nb = [out[y - 1, x], out[y + 1, x], out[y, x - 1], out[y, x + 1]]
                if len(set(nb)) == 1 and nb[0] not in (v, '.'):
                    out[y, x] = nb[0]
    return [''.join(r) for r in out]

def save(rows, prefix, palette='toad'):
    pal = PALETTES[palette]
    H, W = len(rows), len(rows[0])
    im = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    for y, r in enumerate(rows):
        for x, ch in enumerate(r):
            if ch != '.': im.putpixel((x, y), tuple(int(pal[ch][i:i + 2], 16) for i in (0, 2, 4)) + (255,))
    im.save(prefix + '.png')
    bg = Image.new('RGBA', im.size, (96, 96, 120, 255)); bg.alpha_composite(im)
    bg.resize((W * 6, H * 6), Image.NEAREST).convert('RGB').save(prefix + '_x6.png')
    open(prefix + '.txt', 'w').write('\n'.join(rows))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('inp'); ap.add_argument('out')
    ap.add_argument('--width', type=int, default=104); ap.add_argument('--palette', default='toad')
    ap.add_argument('--ink', type=float, default=0.16); ap.add_argument('--gloss', type=float, default=0.30)
    a = ap.parse_args()
    save(pixelize(a.inp, a.width, a.palette, a.ink, a.gloss), a.out, a.palette)
