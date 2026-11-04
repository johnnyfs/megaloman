"""nespix model: text and/or image -> NES pixel art at any size from 8 to 256.

    imagine (text)  ->  concepts  ->  render K hand-set variants per size  ->  judge  ->  best sprite

* imagine: pretrained SDXL-Turbo (CPU, bf16), only when no image is given (or to restyle one)
* render:  `render.py`, the hand-weighted NES renderer; VARIANTS is a small fixed grid of its weights
* judge:   CLIP image similarity to the concept (does it still read as the same thing at this size?),
           CLIP text similarity (does it read as the description?), plus hand-weighted structural checks
"""
import itertools
import json
import os

import numpy as np
from PIL import Image

from .render import Concept, Params, render
from . import score as S

# The variant grid: each axis is a known trade-off at small sizes.
AXES = dict(
    colors=(3, 4),            # one sprite palette vs. 2-palette metasprite (tile rule still enforced)
    sharpen=(0.4, 1.1),       # keep thin features vs. calm shapes
    ink_open=(0.0, 0.35),     # keep concept lines vs. drop them (outline redrawn anyway)
    smooth=(0.3, 0.8),        # shading detail vs. flat regions
    fg_th=(0.42, 0.55),       # fat vs. tight silhouette
)
W_IMG, W_TXT, W_STRUCT = 25.0, 15.0, 1.0       # judge weights (CLIP cosines differ by ~0.01-0.05)


def variants(axes=AXES):
    keys = list(axes)
    for vals in itertools.product(*axes.values()):
        yield dict(zip(keys, vals))


def _crop_square(im, fg):
    ys, xs = np.where(fg)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    s = max(y1 - y0, x1 - x0)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    box = (cx - s // 2, cy - s // 2, cx - s // 2 + s, cy - s // 2 + s)
    return im.crop(box)


def concept_view(c: Concept):
    im = Image.fromarray(c.rgba, 'RGBA')
    m = Image.fromarray((c.fg * 255).astype(np.uint8))
    im.putalpha(m)
    return S.on_white(_crop_square(im, c.fg), smooth=True)


def pixelate(image, size, text=None, base: Params = None, axes=AXES, judge=True, top=1, **kw):
    """Image -> best Sprite at `size` (or the top-k list). With judge=False, renders the default Params."""
    c = image if isinstance(image, Concept) else Concept(image)
    base = base or Params(**kw)
    if not judge:
        return render(c, size, base)
    return judge_sprites(c, list(_variants(c, size, base.__dict__, axes)), text, top)


def judge_sprites(c, sprites, text=None, top=1, txt_emb=None):
    st = [S.structure(sp) for sp in sprites]
    views = [S.on_white(sp.rgba()) for sp in sprites]
    e = S.embed_images([concept_view(c)] + views)
    if e is not None:
        img_sim = e[1:] @ e[0]
        if text and txt_emb is None:
            txt_emb = S.embed_text([f'pixel art of {text}'])
        txt_sim = (e[1:] @ txt_emb[0]) if txt_emb is not None else np.zeros(len(sprites))
    else:
        img_sim = txt_sim = np.zeros(len(sprites))
    for sp, (s, terms), a, b in zip(sprites, st, img_sim, txt_sim):
        sp.meta['judge'] = dict(total=round(float(W_IMG * a + W_TXT * b + W_STRUCT * s), 3),
                                img=round(float(a), 4), txt=round(float(b), 4), struct=terms)
    ranked = sorted(sprites, key=lambda sp: -sp.meta['judge']['total'])
    return ranked[0] if top == 1 else ranked[:top]


def choose_concept(images, text, keep=2):
    """Rank candidate concept images for a text prompt: CLIP text match + structural sanity."""
    concepts, rows = [], []
    t = S.embed_text([text])
    for i, im in enumerate(images):
        try:
            c = Concept(im)
        except ValueError:
            continue
        st, terms = S.structure_concept(c)
        concepts.append(c)
        rows.append((i, st, terms))
    if not concepts:
        raise RuntimeError('no usable concept')
    e = S.embed_images([concept_view(c) for c in concepts])
    sims = (e @ t[0]) if e is not None and t is not None else np.zeros(len(concepts))
    scored = []
    for c, (i, st, terms), sim in zip(concepts, rows, sims):
        total = 30.0 * float(sim) + st
        c.meta = dict(seed_index=i, clip_txt=round(float(sim), 4), struct=terms, total=round(total, 3))
        scored.append((total, c))
    scored.sort(key=lambda x: -x[0])
    return [c for _, c in scored[:keep]]


def generate(text=None, image=None, sizes=(16, 32, 64), seeds=6, restyle=None, out=None, keep_concepts=2, **kw):
    """The full model. Returns {'concept': Concept, 'sprites': {size: Sprite}, 'candidates': [...]}

    text only        -> imagine `seeds` concepts with SDXL-Turbo, keep the best `keep_concepts`
    image (+ text)   -> use the image as the concept (text, if given, also steers the judge)
    restyle=float    -> with an image: first flatten it toward the NES concept style (img2img strength)
    """
    from . import imagine as I
    if image is None and not text:
        raise ValueError('need text and/or image')
    if image is None:
        raw = I.imagine(text, seeds=range(seeds))
        cands = choose_concept(raw, text, keep=keep_concepts)
    elif restyle:
        raw = I.restyle(image, text or '', seeds=range(max(1, seeds // 2)), strength=restyle)
        cands = choose_concept(raw, text or 'a video game character', keep=keep_concepts)
        raw = [image] + raw
    else:
        raw = [image]
        cands = [Concept(image)]
    txt = S.embed_text([f'pixel art of {text}']) if text else None
    best = None
    for c in cands:
        sprites = {n: judge_sprites(c, list(_variants(c, n, kw)), text, 1, txt) for n in sizes}
        tot = np.mean([sp.meta['judge']['total'] for sp in sprites.values()]) + getattr(c, 'meta', {}).get('total', 0) / 30
        if best is None or tot > best[0]:
            best = (tot, c, sprites)
    _, c, sprites = best
    res = dict(concept=c, sprites=sprites, raw=raw, text=text)
    if out:
        save(res, out)
    return res


def _variants(c, n, kw, axes=AXES):
    seen = set()
    base = Params(**kw)
    for v in variants(axes):
        sp = render(c, n, Params(**{**base.__dict__, **v}))
        key = sp.grid.tobytes() + str(sp.codes).encode()
        if key in seen:
            continue
        seen.add(key)
        sp.meta['variant'] = v
        yield sp


def save(res, out):
    os.makedirs(out, exist_ok=True)
    c = res['concept']
    Image.fromarray(c.rgba, 'RGBA').save(os.path.join(out, 'concept.png'))
    meta = dict(text=res.get('text'), concept=getattr(c, 'meta', None), sprites={})
    for n, sp in res['sprites'].items():
        sp.rgba().save(os.path.join(out, f'sprite_{n}.png'))
        sp.preview(scale=max(1, 256 // n)).save(os.path.join(out, f'sprite_{n}_preview.png'))
        meta['sprites'][n] = _jsonable(sp.meta)
    sheet(res).save(os.path.join(out, 'sheet.png'))
    with open(os.path.join(out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


def sheet(res, cell=192, bg=(64, 72, 96)):
    """Concept + every size side by side (nearest-upscaled to the same display size)."""
    sp = res['sprites']
    tiles = [concept_view(res['concept']).resize((cell, cell))]
    for n in sorted(sp):
        s = sp[n]
        k = max(1, cell // n)
        im = s.preview(scale=k, bg=bg)
        t = Image.new('RGB', (cell, cell), bg)
        t.paste(im, ((cell - im.width) // 2, (cell - im.height) // 2))
        tiles.append(t)
    out = Image.new('RGB', (len(tiles) * (cell + 4), cell), (24, 24, 28))
    for i, t in enumerate(tiles):
        out.paste(t, (i * (cell + 4), 0))
    return out
