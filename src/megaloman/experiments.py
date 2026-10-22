"""Compiler experiments: recover authentic sprites from mixed-pixel-size enlargements (ground truth known)."""
import json, os

import numpy as np
from PIL import Image

from . import compiler, imaging, layout

BENCH = ['telly', 'met_walk', 'pipi']


def bench_sprites():
    d = os.path.join(layout.type_dir('megaman', 'small_enemy', 'keyframe'), 'benchmarks')
    return {n: Image.open(os.path.join(d, n, 'subject.png')).convert('RGBA') for n in BENCH}


def enlarge(im, f, mode='painted', phase=0.35):
    """Simulate an image model's 'pixel art' at f x the true size.
    nearest: exact non-integer nearest enlargement (trivially invertible; kept as a sanity case)
    painted: smooth (bilinear) enlargement with a sub-pixel phase offset, then snapped back to the sprite's own
             palette with alpha thresholded: uneven 1-3 px lines, misaligned grid, like our stills"""
    W, H = round(im.width * f), round(im.height * f)
    if mode == 'nearest':
        return im.resize((W, H), Image.NEAREST)
    a = np.array(im).astype(float)
    # premultiply so transparent pixels don't bleed black into edges
    pm = a.copy(); pm[..., :3] *= pm[..., 3:4] / 255
    pm = np.pad(pm, ((1, 1), (1, 1), (0, 0)))                     # room for the sub-pixel offset
    box = (1 - phase, 1 - phase, 1 - phase + im.width, 1 - phase + im.height)
    up = np.stack([np.array(Image.fromarray(pm[..., c].astype(np.uint8)).resize((W, H), Image.BILINEAR, box=box)).astype(float)
                   for c in range(4)], -1)
    alpha = up[..., 3]
    rgb = up[..., :3] / np.maximum(alpha[..., None], 1) * 255
    cols = np.unique(a[a[..., 3] > 0][:, :3], axis=0)
    from .imaging import _lab
    idx = ((_lab(rgb.reshape(-1, 3))[:, None] - _lab(cols)[None]) ** 2).sum(-1).argmin(1)
    out = np.zeros((H, W, 4), np.uint8)
    out[..., :3] = cols[idx].reshape(H, W, 3)
    out[..., 3] = np.where(alpha >= 128, 255, 0)
    return Image.fromarray(out, 'RGBA')


def m_nearest(src, tw, th):
    return src.resize((tw, th), Image.NEAREST)


def m_majority(src, tw, th):
    a = np.array(src); sh, sw = a.shape[:2]
    Y, X = compiler._samples(sw, sh, tw, th)
    k = a[..., 0].astype(np.int64) * 2 ** 24 + a[..., 1].astype(np.int64) * 65536 + a[..., 2].astype(np.int64) * 256 + a[..., 3]
    k = np.where(a[..., 3] > 0, k, 0)
    s = k[Y, X]
    out = np.zeros((th, tw, 4), np.uint8)
    for y in range(th):
        for x in range(tw):
            v, c = np.unique(s[y, x], return_counts=True)
            m = int(v[c.argmax()])
            out[y, x] = (m >> 24 & 255, m >> 16 & 255, m >> 8 & 255, m & 255)
    return Image.fromarray(out, 'RGBA')


def m_compiler(src, tw, th, **opts):
    return compiler.compile_sprite(src, tw, th, opts)[0]


def m_compiler_v2(src, tw, th, **opts):
    return compiler.compile_v2(src, tw, th, opts)[0]


def m_pixelize(src, tw, th):
    """fusion-man's pixelize (ink priority, Zhang-Suen thinning, silhouette outline, orphan removal)."""
    import tempfile
    from .vendor import pixelize
    a = np.array(src)
    cols = np.unique(a[a[..., 3] > 0][:, :3], axis=0)
    order = sorted(range(len(cols)), key=lambda i: cols[i].sum())       # darkest = 'k' (ink), lightest = 'w' (gloss)
    keys = ['k'] + list('abcdefghij')[:max(0, len(order) - 2)] + ['w']
    pal = {keys[n]: '%02X%02X%02X' % tuple(cols[i]) for n, i in enumerate(order)}
    pixelize.PALETTES['_exp'] = pal
    bg = np.zeros_like(a); bg[..., :3] = (96, 96, 120); bg[..., 3] = 255
    comp = np.where(a[..., 3:4] > 0, a, bg)
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        Image.fromarray(comp.astype(np.uint8), 'RGBA').convert('RGB').save(f.name)
        rows = pixelize.pixelize(f.name, tw, '_exp', height=th)
    os.unlink(f.name)
    return imaging.rows_to_image_hex(rows, pal)


def metrics(out, gt):
    """Best over +-1 px translations: where the sprite sits in its canvas does not matter (it gets cropped)."""
    best = None
    a0 = np.array(out)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            a = np.zeros_like(a0)
            ys, yd = (slice(max(0, -dy), a0.shape[0] - max(0, dy)), slice(max(0, dy), a0.shape[0] - max(0, -dy)))
            xs, xd = (slice(max(0, -dx), a0.shape[1] - max(0, dx)), slice(max(0, dx), a0.shape[1] - max(0, -dx)))
            a[yd, xd] = a0[ys, xs]
            m = _metrics(Image.fromarray(a, 'RGBA'), gt)
            if best is None or m['pixel_acc'] > best['pixel_acc']:
                best = dict(m, shift=[dx, dy])
    return best


def _metrics(out, gt):
    a, b = np.array(out), np.array(gt)
    oa, ob = a[..., 3] > 0, b[..., 3] > 0
    same = (oa == ob) & (~oa | (a[..., :3] == b[..., :3]).all(-1))
    union = oa | ob
    da = oa & (a[..., :3].astype(int).sum(-1) < compiler.DARK)
    db = ob & (b[..., :3].astype(int).sum(-1) < compiler.DARK)
    ink_iou = (da & db).sum() / max(1, (da | db).sum())
    acc = (same & union).sum() / max(1, union.sum())
    sil_iou = (oa & ob).sum() / max(1, union.sum())
    return {'pixel_acc': round(float(acc), 3), 'ink_iou': round(float(ink_iou), 3), 'silhouette_iou': round(float(sil_iou), 3),
            'orphans': int(imaging.check_orphans(out)[1].split()[0])}


def run(factors=(1.4, 1.7, 2.0), variants=None, tag='v1', mode='painted'):
    variants = variants or {'nearest': m_nearest, 'majority': m_majority, 'pixelize(fusion-man)': m_pixelize,
                            'compiler v1': m_compiler, 'compiler v2': m_compiler_v2}
    sprites = bench_sprites()
    res, rows = {}, []
    for name, gt in sprites.items():
        for f in factors:
            src = enlarge(gt, f, mode)
            cells = [(gt, f'{name} truth', None), (src, f'x{f} input', None)]
            for vn, fn in variants.items():
                out = fn(src, gt.width, gt.height)
                m = metrics(out, gt)
                res.setdefault(vn, []).append({'sprite': name, 'factor': f, **m})
                cells.append((out, f"{vn}\n{m['pixel_acc']:.0%} ink {m['ink_iou']:.0%}", None))
            rows.append((f'{name} x{f}', cells))
    summary = {vn: {k: round(float(np.mean([r[k] for r in rs])), 3) for k in ('pixel_acc', 'ink_iou', 'silhouette_iou', 'orphans')}
               for vn, rs in res.items()}
    d = os.path.join(layout.W, 'experiments', 'compiler', tag)
    os.makedirs(d, exist_ok=True)
    layout.write_json(os.path.join(d, 'results.json'), {'created': layout.now(), 'code': layout.code_version(),
        'question': f'recover authentic sprites (ground truth) from {mode} enlargements', 'mode': mode,
        'factors': list(factors), 'summary': summary, 'per_case': res})
    imaging.contact_sheet(rows, box=110, title=f'compiler {tag}: recover truth from xF enlargement').save(os.path.join(d, 'sheet.png'))
    layout.event('experiment', name=f'compiler/{tag}', summary=summary, outputs=[layout.rel(d)])
    for vn, s in summary.items():
        print(f'{vn:22s} {s}')
    return summary
