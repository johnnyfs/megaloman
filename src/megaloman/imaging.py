"""Pixel tools: NES palette, grid snap, palette lock, deterministic checks, presentation renders, contact sheets."""
import io, string

import numpy as np
from PIL import Image, ImageDraw

from .vendor import gridsnap

# 2C02 palette (from fusion-man src/assets.py)
NES = {
 '00':'7C7C7C','01':'0000FC','02':'0000BC','03':'4428BC','04':'940084','05':'A80020','06':'A81000','07':'881400','08':'503000','09':'007800','0A':'006800','0B':'005800','0C':'004058','0F':'000000',
 '10':'BCBCBC','11':'0078F8','12':'0058F8','13':'6844FC','14':'D800CC','15':'E40058','16':'F83800','17':'E45C10','18':'AC7C00','19':'00B800','1A':'00A800','1B':'00A844','1C':'008888',
 '20':'F8F8F8','21':'3CBCFC','22':'6888FC','23':'9878F8','24':'F878F8','25':'F85898','26':'F87858','27':'FCA044','28':'F8B800','29':'B8F818','2A':'58D854','2B':'58F898','2C':'00E8D8','2D':'787878',
 '30':'FCFCFC','31':'A4E4FC','32':'B8B8F8','33':'D8B8F8','34':'F8B8F8','35':'F8A4C0','36':'F0D0B0','37':'FCE0A8','38':'F8D878','39':'D8F878','3A':'B8F8B8','3B':'B8F8D8','3C':'00FCFC','3D':'F8D8F8',
}
NEUTRAL = (96, 96, 120)
_CHARS = string.ascii_letters + string.digits + '!#$%&*+-/:;<=>?@^_~'
NES_KEYED = {_CHARS[i]: hexv for i, hexv in enumerate(sorted(set(NES.values())))}


def rgb(hexv):
    return tuple(int(hexv[i:i + 2], 16) for i in (0, 2, 4))


def to_png(img: Image.Image) -> bytes:
    b = io.BytesIO()
    img.save(b, 'PNG', optimize=False)
    return b.getvalue()


def from_png(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert('RGBA')


def rows_to_image(rows, pal_codes):
    """fusion-man grid (rows of keys, '.' transparent) + {key: NES code} -> RGBA 1x."""
    h, w = len(rows), max(len(r) for r in rows)
    im = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    for y, r in enumerate(rows):
        for x, c in enumerate(r):
            if c in pal_codes:
                im.putpixel((x, y), rgb(NES[pal_codes[c]]) + (255,))
    return im


def rows_to_image_hex(rows, pal_hex):
    """Grid of keys + {key: 'RRGGBB'} -> RGBA."""
    h, w = len(rows), max(len(r) for r in rows)
    im = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    for y, r in enumerate(rows):
        for x, c in enumerate(r):
            if c in pal_hex:
                im.putpixel((x, y), rgb(pal_hex[c]) + (255,))
    return im


# ------------------------------------------------------------------ snap + palette

def snap(png: bytes, lo=4.0, hi=96.0):
    """Image-model 'pixel art' -> true 1x RGBA sprite in exact NES colours (background flood-filled to transparent)."""
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        f.write(png)
        path = f.name
    try:
        # Pixels are square: estimate the pitch per axis, then use one shared pitch. When one axis locked onto a
        # harmonic (an integer fraction of the other), the larger estimate is the fundamental.
        a = np.array(Image.open(path).convert('RGB')).astype(float)
        gx = np.abs(np.diff(a, axis=1)).sum(-1).sum(0)
        gy = np.abs(np.diff(a, axis=0)).sum(-1).sum(1)
        px0, py0 = gridsnap.pitch_phase(gx, lo, hi)[0], gridsnap.pitch_phase(gy, lo, hi)[0]
        big, small = max(px0, py0), min(px0, py0)
        ratio = big / small
        pitch = big if ratio > 1.5 and abs(ratio - round(ratio)) < 0.12 else (px0 + py0) / 2
        rows, (px, py) = gridsnap.snap(path, NES_KEYED, pitch=float(pitch))
    finally:
        os.unlink(path)
    h, w = len(rows), max((len(r) for r in rows), default=0)
    im = Image.new('RGBA', (max(w, 1), max(h, 1)), (0, 0, 0, 0))
    for y, r in enumerate(rows):
        for x, c in enumerate(r):
            if c != '.':
                im.putpixel((x, y), rgb(NES_KEYED[c]) + (255,))
    return im, {'pitch': [round(float(px), 2), round(float(py), 2)], 'size': [w, h]}


def _lab(cols):
    from .vendor.pixelize import to_lab
    return to_lab(np.array(cols, float))


def palette_lock(im: Image.Image, max_colors=4, keep_black=True):
    """Keep the max_colors most frequent opaque colours (black always kept if present); remap the rest by Lab distance."""
    a = np.array(im)
    op = a[..., 3] > 0
    cols, counts = np.unique(a[op][:, :3], axis=0, return_counts=True)
    if len(cols) <= max_colors:
        return im, {'colors_before': int(len(cols)), 'colors_after': int(len(cols))}
    order = list(np.argsort(-counts))
    keep = []
    if keep_black:
        dark = [i for i in order if cols[i].sum() < 30]
        keep += dark[:1]
    for i in order:
        if len(keep) >= max_colors:
            break
        if i not in keep:
            keep.append(i)
    kc = cols[keep]
    klab = _lab(kc)
    px = a[op][:, :3]
    nearest = ((_lab(px)[:, None, :] - klab[None]) ** 2).sum(-1).argmin(1)
    out = a.copy()
    out[op, :3] = kc[nearest]
    return Image.fromarray(out, 'RGBA'), {'colors_before': int(len(cols)), 'colors_after': int(len(kc))}


# ------------------------------------------------------------------ deterministic checks

def _opaque(im):
    return np.array(im)[..., 3] > 0


def check_size(im, min_px=10, max_px=24):
    w, h = im.size
    ok = min_px <= w <= max_px and min_px <= h <= max_px
    return ok, f'{w}x{h} (allowed {min_px}-{max_px})'


def check_colors(im, max_total=4):
    a = np.array(im)
    n = len(np.unique(a[a[..., 3] > 0][:, :3], axis=0))
    return n <= max_total, f'{n} opaque colours (max {max_total})'


def check_tile_colors(im, max_per_tile=3):
    """NES sprite rule: <= 3 opaque colours per 8x8 tile. Best alignment over the 64 grid offsets."""
    a = np.array(im)
    h, w = a.shape[:2]
    best = None
    for oy in range(8):
        for ox in range(8):
            worst = 0
            for y0 in range(-oy, h, 8):
                for x0 in range(-ox, w, 8):
                    t = a[max(0, y0):y0 + 8, max(0, x0):x0 + 8]
                    t = t[t[..., 3] > 0][:, :3]
                    if len(t):
                        worst = max(worst, len(np.unique(t, axis=0)))
            if best is None or worst < best[0]:
                best = (worst, ox, oy)
    return best[0] <= max_per_tile, f'max {best[0]} colours/tile at offset ({best[1]},{best[2]}) (max {max_per_tile})'


def check_orphans(im):
    op = _opaque(im)
    h, w = op.shape
    pad = np.pad(op, 1)
    nb = sum(pad[1 + dy:h + 1 + dy, 1 + dx:w + 1 + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)
    n = int((op & (nb == 0)).sum())
    return n == 0, f'{n} orphan pixels'


def check_holes(im):
    op = _opaque(im)
    h, w = op.shape
    seen = np.zeros_like(op)
    st = [(x, y) for x in range(w) for y in (0, h - 1)] + [(x, y) for y in range(h) for x in (0, w - 1)]
    while st:
        x, y = st.pop()
        if not (0 <= x < w and 0 <= y < h) or seen[y, x] or op[y, x]:
            continue
        seen[y, x] = True
        st += [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    enclosed = int((~op & ~seen).sum())
    return enclosed == 0, f'{enclosed} enclosed transparent pixels'


def check_outline(im, min_dark=0.75):
    """Share of silhouette-edge pixels that are dark (Capcom black outline)."""
    a = np.array(im).astype(int)
    op = a[..., 3] > 0
    h, w = op.shape
    pad = np.pad(op, 1)
    edge = op & ~(pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])
    if not edge.any():
        return False, 'empty sprite'
    dark = (a[..., :3].sum(-1) < 90) & edge
    r = dark.sum() / edge.sum()
    return r >= min_dark, f'{r:.0%} of silhouette edge is dark (min {min_dark:.0%})'


CHECKS = {'size': check_size, 'colors': check_colors, 'tile_colors': check_tile_colors,
          'orphans': check_orphans, 'holes': check_holes, 'outline': check_outline}


def run_checks(im, spec):
    """spec: {name: {params..., gate: bool}} -> {name: {pass, detail, gate}}"""
    out = {}
    for name, p in spec.items():
        p = dict(p or {})
        gate = p.pop('gate', True)
        ok, detail = CHECKS[name](im, **p)
        out[name] = {'pass': bool(ok), 'detail': detail, 'gate': gate}
    return out


# ------------------------------------------------------------------ presentation

def present(im: Image.Image, scale=6, pad=8, with_1x=True):
    """Judge presentation: the sprite at 1x and at `scale`x, nearest-neighbour, on the neutral background."""
    w, h = im.size
    W = pad * 3 + w + w * scale if with_1x else pad * 2 + w * scale
    H = pad * 2 + h * scale
    out = Image.new('RGBA', (W, H), NEUTRAL + (255,))
    x = pad
    if with_1x:
        out.alpha_composite(im, (x, pad + (h * scale - h) // 2))
        x += w + pad
    out.alpha_composite(im.resize((w * scale, h * scale), Image.NEAREST), (x, pad))
    return out.convert('RGB')


def thumb(im: Image.Image, box=96):
    """Contact-sheet cell: small sprites scaled up by an integer (nearest); big images scaled down."""
    w, h = im.size
    if max(w, h) <= box:
        s = max(1, box // max(w, h))
        im = im.resize((w * s, h * s), Image.NEAREST)
    else:
        im = im.copy()
        im.thumbnail((box, box), Image.LANCZOS)
    cell = Image.new('RGBA', (box, box), NEUTRAL + (255,))
    cell.alpha_composite(im.convert('RGBA'), ((box - im.width) // 2, (box - im.height) // 2))
    return cell


def contact_sheet(rows, box=96, title=None):
    """rows: [(row_label, [(Image, caption, mark)])], mark in {None, 'best', 'reject'}."""
    lab_w, cap_h, gap = 120, 26, 6
    ncol = max((len(c) for _, c in rows), default=1)
    W = lab_w + ncol * (box + gap) + gap
    top = 22 if title else 0
    H = top + len(rows) * (box + cap_h + gap) + gap
    sheet = Image.new('RGB', (W, H), (24, 24, 32))
    d = ImageDraw.Draw(sheet)
    if title:
        d.text((gap, 5), title, fill=(230, 230, 230))
    for r, (label, cells) in enumerate(rows):
        y = top + gap + r * (box + cap_h + gap)
        d.text((gap, y + box // 2 - 6), label[:18], fill=(200, 200, 210))
        for c, (im, cap, mark) in enumerate(cells):
            x = lab_w + gap + c * (box + gap)
            sheet.paste(thumb(im, box).convert('RGB'), (x, y))
            if mark == 'best':
                d.rectangle([x - 3, y - 3, x + box + 2, y + box + 2], outline=(250, 210, 40), width=3)
            elif mark == 'reject':
                d.line([x, y, x + box, y + box], fill=(220, 50, 50), width=2)
                d.line([x + box, y, x, y + box], fill=(220, 50, 50), width=2)
            for i, line in enumerate(str(cap).split('\n')[:2]):
                d.text((x, y + box + 2 + i * 11), line[:17], fill=(210, 210, 210))
    return sheet


# ------------------------------------------------------------------ animation helpers

def colors_of(im):
    a = np.array(im)
    return [tuple(int(v) for v in c) for c in np.unique(a[a[..., 3] > 0][:, :3], axis=0)]


def recolor_to(im, colors):
    """Map every opaque pixel to the nearest of `colors` (Lab)."""
    a = np.array(im.convert('RGBA'))
    op = a[..., 3] > 0
    if not op.any():
        return im
    cols = np.array(colors, float)
    idx = ((_lab(a[op][:, :3])[:, None, :] - _lab(cols)[None]) ** 2).sum(-1).argmin(1)
    a[op, :3] = cols[idx].astype(np.uint8)
    return Image.fromarray(a, 'RGBA')


def strip(frames, gap=2):
    """Frames side by side (bottom-aligned, transparent gap): the image a judge sees for an animation."""
    W = sum(f.width for f in frames) + gap * (len(frames) - 1)
    H = max(f.height for f in frames)
    out = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    x = 0
    for f in frames:
        out.alpha_composite(f.convert('RGBA'), (x, H - f.height))
        x += f.width + gap
    return out


def save_gif(frames, path, rate=8, scale=6):
    """Looping preview at 60 fps game time: `rate` = game frames per frame (int) or one duration per frame (list)."""
    W = max(f.width for f in frames); H = max(f.height for f in frames)
    out = []
    for f in frames:
        c = Image.new('RGBA', (W, H), NEUTRAL + (255,))
        c.alpha_composite(f.convert('RGBA'), ((W - f.width) // 2, H - f.height))
        out.append(c.convert('RGB').resize((W * scale, H * scale), Image.NEAREST))
    durs = rate if isinstance(rate, (list, tuple)) else [rate] * len(frames)
    out[0].save(path, save_all=True, append_images=out[1:], duration=[max(20, round(d * 1000 / 60)) for d in durs], loop=0)


def diff_strip(frames, gap=2):
    """For each loop step (1->2, ..., last->1): pixels that change in magenta, unchanged pixels dimmed grey."""
    W = max(f.width for f in frames); H = max(f.height for f in frames)
    def canvas(f):
        c = Image.new('RGBA', (W, H), (0, 0, 0, 0)); c.alpha_composite(f.convert('RGBA'), (0, H - f.height)); return np.array(c)
    arr = [canvas(f) for f in frames]
    panels = []
    for i in range(len(arr)):
        a, b = arr[i], arr[(i + 1) % len(arr)]
        out = np.zeros_like(a)
        oa, ob = a[..., 3] > 0, b[..., 3] > 0
        changed = (oa != ob) | (oa & ob & (a[..., :3] != b[..., :3]).any(-1))
        out[(oa | ob) & ~changed] = (70, 70, 80, 255)
        out[changed] = (255, 0, 255, 255)
        panels.append(Image.fromarray(out, 'RGBA'))
    return strip(panels, gap)


def timeline_strip(frames, durations, step=4, gap=1):
    """The loop sampled every `step` game frames (60 fps): rhythm made visible as repeated frames."""
    seq = []
    for f, d in zip(frames, durations):
        seq += [f] * max(1, round(d / step))
    return strip(seq, gap)


def describe_durations(durations):
    return ', '.join(f'frame {i + 1}: {d} game frames ({round(d * 1000 / 60)} ms)' for i, d in enumerate(durations)) + \
           f'; loop {sum(durations)} game frames ({round(sum(durations) * 1000 / 60)} ms)'


def present_anim(frames, scale=6, pad=8, gap=2, durations=None, seq=None):
    """Frames row (1x + enlarged), the difference row, and with timing a timeline row (sampled every 4 frames).
    `seq` (a preview sequence) takes precedence over plain per-frame `durations`."""
    rows = [present(strip(frames, gap), scale, pad), present(diff_strip(frames, gap), scale, pad)]
    if seq:
        rows.append(present(timeline_seq(frames, seq), max(2, scale // 2), pad))
    elif durations:
        rows.append(present(timeline_strip(frames, durations), max(2, scale // 2), pad))
    out = Image.new('RGB', (max(r.width for r in rows), sum(r.height for r in rows) + pad * (len(rows) - 1)), (40, 40, 48))
    y = 0
    for r in rows:
        out.paste(r, (0, y)); y += r.height + pad
    return out


# ------------------------------------------------------------------ state animations: loops and one-off actions
# A sequence is [[frame_number (1-based), game_frames], ...]. A loop repeats its sequence; a one-off action plays
# once and returns to the state's still (frame 1). Previews of one-offs: still for `hold`, the action at its real
# timing, then a `pause` on the still, so a viewer can read it.

def preview_sequence(mode, sequence, hold=30, pause=30):
    if mode == 'oneshot':
        return [[1, hold]] + [list(x) for x in sequence] + [[1, pause]]
    return [list(x) for x in sequence]


def describe_sequence(mode, sequence):
    parts = ', '.join(f'frame {f} for {d} game frames ({round(d * 1000 / 60)} ms)' for f, d in sequence)
    total = sum(d for _, d in sequence)
    if mode == 'oneshot':
        return (f'one-off action, played once and then back to the still (frame 1): {parts}; '
                f'action length {total} game frames ({round(total * 1000 / 60)} ms)')
    return f'loop: {parts}; loop length {total} game frames ({round(total * 1000 / 60)} ms)'


def save_preview(frames, path, mode, sequence, hold=30, pause=30, scale=6):
    seq = preview_sequence(mode, sequence, hold, pause)
    save_gif([frames[f - 1] for f, _ in seq], path, [d for _, d in seq], scale)


def timeline_seq(frames, seq, step=4, gap=1):
    out = []
    for f, d in seq:
        out += [frames[f - 1]] * max(1, round(d / step))
    return strip(out, gap)


def motion_measures(f1, f2, facing='left'):
    """Deterministic measures of a derived frame vs the still (report-only; quantify human critiques).
    bands: share of each horizontal third's pixels that change (top band ~ head: should stay ~0 if it doesn't move)
    shift: centroid of added pixels minus centroid of removed pixels (+x right, +y down), and 'forward' along facing
    symmetry: share of frame 2's changed pixels whose mirror (about the still's centre column) is also changed"""
    a, b = np.array(f1.convert('RGBA')), np.array(f2.convert('RGBA'))
    oa, ob = a[..., 3] > 0, b[..., 3] > 0
    changed = (oa != ob) | (oa & ob & (a[..., :3] != b[..., :3]).any(-1))
    occ = oa | ob
    ys = np.where(occ.any(1))[0]
    y0, y1 = (ys.min(), ys.max() + 1) if len(ys) else (0, a.shape[0])
    edges = np.linspace(y0, y1, 4).round().astype(int)
    bands = {}
    for name, lo, hi in zip(('top', 'middle', 'bottom'), edges[:-1], edges[1:]):
        region = occ[lo:hi]
        bands[name] = round(float(changed[lo:hi].sum() / max(1, region.sum())), 3)
    added, removed = ob & ~oa, oa & ~ob
    def cen(m):
        yy, xx = np.nonzero(m)
        return (float(xx.mean()), float(yy.mean())) if len(xx) else None
    ca, cr = cen(added), cen(removed)
    shift = None
    if ca and cr:
        dx, dy = ca[0] - cr[0], ca[1] - cr[1]
        shift = {'dx': round(dx, 2), 'dy': round(dy, 2), 'forward': round(-dx if facing == 'left' else dx, 2)}
    xs = np.nonzero(oa.any(0))[0]
    cx = (xs.min() + xs.max()) / 2 if len(xs) else a.shape[1] / 2
    yy, xx = np.nonzero(changed & ob)
    mir = np.round(2 * cx - xx).astype(int)
    ok = (mir >= 0) & (mir < a.shape[1])
    sym = float((changed & ob)[yy[ok], mir[ok]].sum() / max(1, len(xx)))
    return {'bands_changed': bands, 'shift': shift, 'symmetry': round(sym, 3)}
