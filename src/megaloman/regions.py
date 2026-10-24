"""Region maps (REGIONS_PLAN.md, milestone 1): concept art -> legend of named parts -> flat region map, checked.

  legend()   a vision model names 4-10 parts (role, priority, colour, min_px at a 32 px rung)
  paint()    an image-edit model repaints the concept with one flat legend colour per part
  snap()     deterministic: nearest legend colour, drop specks, register to the concept, alignment check
  segment()  fallback: colour clustering + connected components on the concept, the VLM names the segments

A region map is an int array: 0 = background, i + 1 = legend part i. Every model call is cached by request content.
"""
import json, re

import numpy as np
from PIL import Image
from scipy import ndimage

from . import imaging, models, store

LEGEND_MODEL = 'openai/gpt-5.6-sol'   # the judge's vision model
ROLES = ('body', 'feature', 'limb', 'detail')


def _hex(c):
    return '%02X%02X%02X' % tuple(int(v) for v in c)


# ------------------------------------------------------------------ the concept

def concept_masks(im, shadow_band=0.2):
    """-> (silhouette, shadow). silhouette = alpha >= 128 minus the ground shadow.
    The shadow is a flat, opaque, low-saturation, blue-tinted (lavender/grey) blob in the bottom band of the art;
    it is the part of that colour class connected to the bottom of the bounding box."""
    a = np.array(im.convert('RGBA')).astype(int)
    opaque = a[..., 3] >= 128
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx, mn = a[..., :3].max(-1), a[..., :3].min(-1)
    ys = np.nonzero(opaque.any(1))[0]
    top, bot = ys.min(), ys.max()
    band = np.zeros_like(opaque)
    band[int(bot - shadow_band * (bot - top)):] = True
    cand = opaque & band & (mx - mn < 45) & (b >= r + 4) & (b >= g + 8) & (mn > 120)
    cand = ndimage.binary_opening(cand, iterations=1)
    lab, n = ndimage.label(cand)
    shadow = np.zeros_like(opaque)
    for i in range(1, n + 1):
        m = lab == i
        if m.sum() > 200 and np.nonzero(m)[0].max() >= bot - 6:   # a sizeable blob touching the bottom
            shadow |= m
    # absorb its anti-aliased rim (pixels next to the shadow that are not squid-coloured enough to matter)
    rim = ndimage.binary_dilation(shadow, iterations=2) & opaque & ~shadow & (mx - mn < 60) & (mn > 90)
    shadow |= rim
    return opaque & ~shadow, shadow


def concept_on_white(im, silhouette):
    """The concept as a model sees it best: the squid only (shadow removed), on white."""
    a = np.array(im.convert('RGBA'))
    a[..., 3] = np.where(silhouette, 255, 0)
    out = Image.new('RGBA', im.size, (255, 255, 255, 255))
    out.alpha_composite(Image.fromarray(a, 'RGBA'))
    return out.convert('RGB')


# ------------------------------------------------------------------ legend

LEGEND_PROMPT = """You are preparing concept art for pixel-art reduction. Split this creature into its PARTS: the regions a
pixel artist would keep distinct when drawing it as a tiny 16-32 px sprite.

Return JSON only: {"parts": [ {"name", "role", "priority", "color", "min_px", "where"} ... ]}
- 4 to 10 parts. Every visible pixel of the creature must belong to exactly one part (no background part; ignore any
  ground shadow).
- name: short snake_case (e.g. "head", "eye_left", "visor").
- role: one of body | feature | limb | detail.
- priority: integer 1-10; higher wins a contested pixel when shrinking (small important features like eyes high,
  big body masses low).
- color: a "#RRGGBB" flat fill colour for this part in a region map. Colours must be highly saturated, far apart from
  each other in RGB, and far from white and from light grey (the background is white). Do NOT use the part's real colour.
- min_px: smallest size in pixels (width or height) this part must keep at a 32 px sprite height to stay readable.
- where: one sentence locating the part in this image, precise enough that someone could paint it.
Repeated identical parts (e.g. several tentacles) should be ONE part unless they must stay distinguishable (e.g. a
left and a right eye that blink separately can still be one part). Surface markings (spots, stripes) that are just
paint on a bigger part belong to that part unless they are a key identifying feature."""


def _parse(content):
    m = re.search(r'\{.*\}', content or '', re.S)
    return json.loads(m.group(0)) if m else {'parse_error': (content or '')[:500]}


def _rgbdist(a, b):
    return float(np.sqrt(((np.array(a, float) - np.array(b, float)) ** 2).sum()))


def fix_colors(parts, min_pair=110, min_white=150):
    """Enforce distinct, saturated, far-from-white colours; replace offenders from SAFE. Records what changed."""
    white = (255, 255, 255)
    kept = {}
    for i, p in enumerate(parts):      # pass 1: keep the model's colours that are valid against the ones kept so far
        s = str(p.get('color', ''))
        if re.fullmatch(r'#?[0-9A-Fa-f]{6}', s):
            c = imaging.rgb(s.lstrip('#').upper())
            if _rgbdist(c, white) >= min_white and all(_rgbdist(c, u) >= min_pair for u in kept.values()):
                kept[i] = c
    grid = [(r, g, b) for r in range(0, 256, 51) for g in range(0, 256, 51) for b in range(0, 256, 51)
            if _rgbdist((r, g, b), white) >= min_white and max(r, g, b) - min(r, g, b) >= 100]
    fixes = []
    for i, p in enumerate(parts):      # pass 2: replace the rest with the grid colour farthest from everything used
        if i in kept:
            continue
        used = list(kept.values())
        c = max(grid, key=lambda g: (min((_rgbdist(g, u) for u in used), default=999), -grid.index(g)))
        fixes.append({'part': p.get('name'), 'from': p.get('color'), 'to': '#' + _hex(c)})
        kept[i] = c
    for i, p in enumerate(parts):
        p['color'] = '#' + _hex(kept[i])
    return fixes


def legend(concept_rgb, ledger, model=LEGEND_MODEL, seed=0):
    """Vision model -> {'parts': [...], 'color_fixes': [...], 'call', 'cost', 'cached'}."""
    png = imaging.to_png(concept_rgb)
    body = {'model': model, 'response_format': {'type': 'json_object'}, 'seed': seed,
            'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': LEGEND_PROMPT},
                                                      {'type': 'image_url', 'image_url': {'url': models.data_uri(png)}}]}]}
    ck = store.key({'step': 'regions.legend', 'model': model, 'prompt': LEGEND_PROMPT, 'image': store.sha(png), 'seed': seed})
    resp, _, cost, cached = models.call(body, ledger, ck)
    out = _parse(resp['choices'][0]['message'].get('content'))
    parts = out.get('parts') or []
    for p in parts:
        p['role'] = p.get('role') if p.get('role') in ROLES else 'detail'
        p['priority'] = int(p.get('priority', 1))
        p['min_px'] = int(p.get('min_px', 1))
    fixes = fix_colors(parts)
    return {'model': model, 'parts': parts, 'color_fixes': fixes, 'call': ck, 'cost': cost, 'cached': cached}


def colors(leg):
    return [imaging.rgb(p['color'].lstrip('#')) for p in leg['parts']]


# ------------------------------------------------------------------ paint

PAINT_PROMPT = """Repaint this image as a flat COLOUR-CODED REGION MAP for a segmentation mask.

Keep the exact same creature, pose, size, position and framing: every part must stay exactly where it is, pixel for
pixel. Change only the colours: fill each part completely with its one flat colour from this table:

{table}

Rules:
- Each part is ONE solid flat colour, exactly the hex value given. No shading, no gradients, no highlights, no texture.
- No outlines, no black lines, no borders between parts: colour regions touch directly.
- Background (everything that is not the creature, including any ground shadow): pure white #FFFFFF.
- No text, no labels, no legend, nothing added.
- Do not move, resize, redraw or restyle anything; only recolour.
- Pale spots, stripes, highlights and markings are paint on the part they sit on: give them that part's colour.
  Nothing inside the creature's outline may stay white."""


def paint_prompt(leg):
    table = '\n'.join(f"- {p['color']}  {p['name']}: {p.get('where', '')}" for p in leg['parts'])
    return PAINT_PROMPT.format(table=table)


RETRY_429 = (10, 30, 90)


def paint(concept_rgb, leg, model, seed, ledger, on_retry=None):
    """Image-edit call -> {'png': bytes | None, 'call', 'cost', 'cached', 'text', 'error', 'retries'}.
    Upstream rate limits (429) get extra backoff (RETRY_429 seconds) on top of models.call's own retries."""
    png = imaging.to_png(concept_rgb)
    prompt = paint_prompt(leg)
    body = {'model': model, 'modalities': ['image', 'text'], 'seed': seed,
            'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': prompt},
                                                      {'type': 'image_url', 'image_url': {'url': models.data_uri(png)}}]}]}
    ck = store.key({'step': 'regions.paint', 'model': model, 'prompt': prompt, 'images': [store.sha(png)], 'seed': seed})
    import time
    retries = 0
    while True:
        try:
            resp, imgs, cost, cached = models.call(body, ledger, ck)
            break
        except models.BudgetExceeded:
            raise
        except Exception as e:   # a failed call is a failed candidate, not a crashed experiment
            if 'HTTP 429' in str(e) and retries < len(RETRY_429):
                if on_retry:
                    on_retry(retries + 1, RETRY_429[retries])
                time.sleep(RETRY_429[retries])
                retries += 1
                continue
            return {'png': None, 'call': ck, 'cost': 0.0, 'cached': False, 'error': str(e)[:300], 'retries': retries}
    return {'png': imgs[0] if imgs else None, 'call': ck, 'cost': cost, 'cached': cached, 'retries': retries,
            'text': (resp['choices'][0]['message'].get('content') or '')[:300],
            'error': None if imgs else 'no image returned'}


# ------------------------------------------------------------------ snap + align

def _fill_unknown(labels, known):
    """Unknown pixels take the label of the nearest known pixel."""
    if known.all():
        return labels
    _, (iy, ix) = ndimage.distance_transform_edt(~known, return_indices=True)
    return labels[iy, ix]


def snap(painted, leg, size, min_island_frac=0.0005, max_dist=90):
    """Painted image -> region map at `size`. Nearest legend colour or background (white / transparent); pixels far
    from every colour (anti-aliasing between regions) and islands smaller than min_island_frac of the area take
    their nearest confident neighbour's label."""
    im = painted.convert('RGBA')
    if im.size != tuple(size):
        im = im.resize(size, Image.LANCZOS)
    a = np.array(im).astype(float)
    pal = np.array([(255, 255, 255)] + colors(leg), float)
    rgb = a[..., :3]
    d = np.sqrt(((rgb[:, :, None, :] - pal[None, None]) ** 2).sum(-1))
    labels = d.argmin(-1)
    known = d.min(-1) <= max_dist
    transparent = a[..., 3] < 128
    labels[transparent], known[transparent] = 0, True
    labels = _fill_unknown(labels, known)
    min_area = max(1, int(min_island_frac * labels.size))
    for _ in range(3):   # removing an island can expose another; a few passes settle it
        known = np.ones(labels.shape, bool)
        for k in range(len(pal)):
            lab, n = ndimage.label(labels == k)
            if n == 0:
                continue
            sizes = ndimage.sum_labels(np.ones_like(lab), lab, index=np.arange(1, n + 1))
            small = np.isin(lab, np.nonzero(sizes < min_area)[0] + 1)
            known &= ~small
        if known.all():
            break
        labels = _fill_unknown(labels, known)
    return labels.astype(np.uint8)


def _warp(a, sx, sy, tx, ty):
    """Nearest-neighbour axis-aligned scale + translate: out(x, y) = a((x - tx) / sx, (y - ty) / sy)."""
    h, w = a.shape
    xs = np.round((np.arange(w) - tx) / sx).astype(int)
    ys = np.round((np.arange(h) - ty) / sy).astype(int)
    vx, vy = (xs >= 0) & (xs < w), (ys >= 0) & (ys < h)
    out = a[np.clip(ys, 0, h - 1)][:, np.clip(xs, 0, w - 1)]
    return np.where(vy[:, None] & vx[None, :], out, 0).astype(a.dtype)


def _bbox(m):
    ys, xs = np.nonzero(m)
    return xs.min(), ys.min(), xs.max(), ys.max()


def register(labels, silhouette, max_scale_dev=0.15):
    """Models redraw slightly shifted or squashed. Fit an axis-aligned scale + translation of the map onto the
    concept silhouette: start from matching bounding boxes, then refine by IoU. Larger fixes than max_scale_dev are
    refused (the painting is then judged as drawn). -> (labels, transform, iou)."""
    fg = labels > 0
    ident = {'sx': 1.0, 'sy': 1.0, 'tx': 0.0, 'ty': 0.0}
    best = (iou(fg, silhouette), ident)
    if not fg.any():
        return labels, ident, best[0]
    px0, py0, px1, py1 = _bbox(fg)
    cx0, cy0, cx1, cy1 = _bbox(silhouette)
    sx0, sy0 = (cx1 - cx0) / max(1, px1 - px0), (cy1 - cy0) / max(1, py1 - py0)
    for ds in (-0.02, -0.01, 0, 0.01, 0.02):
        for dt in (-0.02, -0.01, 0, 0.01, 0.02):
            sx, sy = sx0 * (1 + ds), sy0 * (1 + dt)
            if abs(sx - 1) > max_scale_dev or abs(sy - 1) > max_scale_dev:
                continue
            for ox in (-4, 0, 4):
                for oy in (-4, 0, 4):
                    tx = (cx0 + cx1) / 2 - sx * (px0 + px1) / 2 + ox
                    ty = (cy0 + cy1) / 2 - sy * (py0 + py1) / 2 + oy
                    v = iou(_warp(fg, sx, sy, tx, ty), silhouette)
                    if v > best[0]:
                        best = (v, {'sx': round(sx, 4), 'sy': round(sy, 4), 'tx': round(tx, 1), 'ty': round(ty, 1)})
    v, t = best
    return _warp(labels, t['sx'], t['sy'], t['tx'], t['ty']), t, v


def fill_holes(labels, silhouette, min_inside=0.8):
    """Enclosed background inside the map (a model leaving a pale marking white) where the concept is opaque: give it
    the surrounding part. Background that the concept also shows as background is kept. -> (labels, filled px)."""
    bg = labels == 0
    lab, n = ndimage.label(bg)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    unknown = np.zeros_like(bg)
    for i in range(1, n + 1):
        if i in border:
            continue
        m = lab == i
        if silhouette[m].mean() >= min_inside:
            unknown |= m
    return _fill_unknown(labels, ~unknown).astype(labels.dtype), int(unknown.sum())


def _shift(a, dx, dy):
    out = np.zeros_like(a)
    h, w = a.shape[:2]
    out[max(0, dy):h + min(0, dy), max(0, dx):w + min(0, dx)] = a[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
    return out


def iou(a, b):
    return float((a & b).sum() / max(1, (a | b).sum()))


def align(labels, silhouette, leg, min_iou=0.9, min_part_frac=0.0005):
    """Alignment check: silhouette IoU >= min_iou and every legend part present (>= min_part_frac of the image)."""
    fg = labels > 0
    min_area = max(1, int(min_part_frac * labels.size))
    areas = {p['name']: int((labels == i + 1).sum()) for i, p in enumerate(leg['parts'])}
    present = {n: a >= min_area for n, a in areas.items()}
    v = iou(fg, silhouette)
    return {'iou': round(v, 4), 'iou_ok': v >= min_iou, 'parts_present': present,
            'all_present': all(present.values()), 'part_areas': areas,
            'outside_silhouette_px': int((fg & ~silhouette).sum()), 'missed_silhouette_px': int((~fg & silhouette).sum()),
            'pass': bool(v >= min_iou and all(present.values()))}


def clip_to_silhouette(labels, silhouette):
    """The concept's alpha is the ground truth for the outline: background outside it; silhouette pixels the paint
    left as background take the nearest part. Used only after the alignment check passes."""
    lab = np.where(silhouette, labels, 0).astype(np.uint8)
    known = ~silhouette | (lab > 0)
    return np.where(silhouette, _fill_unknown(lab, known), 0).astype(np.uint8)


# ------------------------------------------------------------------ renders

def render_map(labels, leg, bg=None):
    """Region map -> RGBA (background transparent unless bg is given)."""
    pal = np.array([(255, 255, 255)] + colors(leg), np.uint8)
    out = np.zeros(labels.shape + (4,), np.uint8)
    out[..., :3] = pal[labels]
    out[..., 3] = np.where(labels > 0, 255, 0)
    im = Image.fromarray(out, 'RGBA')
    if bg is not None:
        base = Image.new('RGBA', im.size, bg + (255,))
        base.alpha_composite(im)
        im = base
    return im


def overlay(concept, labels, leg, alpha=0.5):
    """Region map at `alpha` over the concept (on white)."""
    base = Image.new('RGBA', concept.size, (255, 255, 255, 255))
    base.alpha_composite(concept.convert('RGBA'))
    m = np.array(render_map(labels, leg))
    m[..., 3] = (m[..., 3] * alpha).astype(np.uint8)
    base.alpha_composite(Image.fromarray(m, 'RGBA'))
    return base


# ------------------------------------------------------------------ fallback: deterministic segmentation

def segment(concept, silhouette, k=12, min_frac=0.002, seed=0):
    """Colour clustering (k-means in Lab) + connected components inside the silhouette. Thin dark ink lines are not
    segments of their own: they go to the nearest segment (big dark areas like a visor stay). -> (int map, n)."""
    a = np.array(concept.convert('RGB')).astype(float)
    L = imaging._lab(a.reshape(-1, 3)).reshape(a.shape)
    dark = silhouette & (L[..., 0] < 35)
    ink = dark & ~ndimage.binary_dilation(ndimage.binary_opening(dark, iterations=4), iterations=4)
    body = silhouette & ~ink
    pts = L[body]
    rng = np.random.default_rng(seed)
    cent = pts[rng.choice(len(pts), k, replace=False)]
    for _ in range(25):
        idx = ((pts[:, None] - cent[None]) ** 2).sum(-1).argmin(1)
        cent = np.array([pts[idx == j].mean(0) if (idx == j).any() else cent[j] for j in range(k)])
    cl = np.zeros(silhouette.shape, int)
    cl[body] = idx + 1
    cl = _fill_unknown(cl, ~silhouette | body) * silhouette
    cl = ndimage.median_filter(cl, size=5) * silhouette
    seg = np.zeros(silhouette.shape, int)
    nid, min_area = 0, int(min_frac * silhouette.size)
    for j in range(1, k + 1):
        cc, n = ndimage.label(cl == j)
        for c in range(1, n + 1):
            m = cc == c
            if m.sum() >= min_area:
                nid += 1
                seg[m] = nid
    known = ~silhouette | (seg > 0)
    return np.where(silhouette, _fill_unknown(seg, known), 0), nid


def segments_sheet(concept, seg, n, scale=2):
    """The concept with segment borders in magenta and each segment's number at its most interior point."""
    from PIL import ImageDraw, ImageFont
    a = np.array(concept_on_white(concept, seg > 0))
    edge = (seg != np.roll(seg, 1, 0)) | (seg != np.roll(seg, 1, 1))
    a[edge] = (255, 0, 255)
    im = Image.fromarray(a).resize((concept.width * scale, concept.height * scale), Image.NEAREST)
    d = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=13 * scale // 2 + 6)
    for i in range(1, n + 1):
        dt = ndimage.distance_transform_edt(seg == i)
        y, x = np.unravel_index(dt.argmax(), dt.shape)
        d.text((x * scale, y * scale), str(i), fill=(0, 0, 0), font=font, anchor='mm', stroke_width=3,
               stroke_fill=(255, 255, 255))
    return im.convert('RGB')


NAME_PROMPT = """This image is a creature with its pixels split into numbered segments (magenta borders, black numbers).
Assign every segment number 1..{n} to exactly one of these parts (or "background" if it is not part of the creature,
e.g. a ground shadow):

{table}

Return JSON only: {{"assign": {{"1": "<part name>", "2": "<part name>", ...}}}}"""


def name_segments(concept, seg, n, leg, ledger, model=LEGEND_MODEL, seed=0):
    """VLM maps each segment to a legend part. -> (region map, record)."""
    png = imaging.to_png(segments_sheet(concept, seg, n))
    table = '\n'.join(f"- {p['name']}: {p.get('where', '')}" for p in leg['parts'])
    prompt = NAME_PROMPT.format(n=n, table=table)
    body = {'model': model, 'response_format': {'type': 'json_object'}, 'seed': seed,
            'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': prompt},
                                                      {'type': 'image_url', 'image_url': {'url': models.data_uri(png)}}]}]}
    ck = store.key({'step': 'regions.name_segments', 'model': model, 'prompt': prompt, 'image': store.sha(png), 'seed': seed})
    resp, _, cost, cached = models.call(body, ledger, ck)
    assign = _parse(resp['choices'][0]['message'].get('content')).get('assign') or {}
    idx = {p['name']: i + 1 for i, p in enumerate(leg['parts'])}
    lut = np.zeros(n + 1, np.uint8)
    for k, v in assign.items():
        if str(k).isdigit() and int(k) <= n:
            lut[int(k)] = idx.get(v, 0)
    return lut[seg], {'call': ck, 'cost': cost, 'cached': cached, 'assign': assign, 'sheet_png': png}
