"""Candidate scoring: hand-weighted structural checks + (optional) CLIP text alignment.

Used to pick the best of K concepts for a text prompt. Every term is cheap and explainable.
"""
import numpy as np
from scipy import ndimage as ndi

from .render import matte, _load

_clip = {}


def structure(sprite, concept=None):
    """Return (score, terms). Higher is better; 0 is 'nothing wrong'."""
    g = sprite.grid
    N = g.shape[0]
    fg = g >= 0
    t = {}
    frac = fg.mean()
    t['fill'] = -abs(np.clip(frac, 0.3, 0.75) - frac) * 4                 # too sparse or a full square
    _, n = ndi.label(fg, structure=np.ones((3, 3)))
    t['parts'] = -0.5 * max(0, n - 1)                                     # fragments
    fills = g > 0
    same = np.zeros_like(fills)
    P = np.pad(g, 1, constant_values=-9)
    for dy, dx in ((0, 1), (2, 1), (1, 0), (1, 2)):
        same |= P[dy:dy + N, dx:dx + N] == g
    t['noise'] = -3.0 * (fills & ~same).sum() / max(1, fills.sum())       # orphan pixels
    used = len(np.unique(g[g > 0]))
    t['colours'] = -0.5 * max(0, len(sprite.codes) - 1 - used)            # a palette colour went unused
    ink = (g == 0).sum() / max(1, fg.sum())
    t['ink'] = -2.0 * max(0, ink - 0.45)                                  # mostly black blob
    if concept is not None:
        rgba = _load(concept)
        try:
            m = matte(rgba)
        except ValueError:
            return -9.0, {'matte': -9.0}
        border = np.concatenate([m[0], m[-1], m[:, 0], m[:, -1]])
        t['cropped'] = -4.0 * border.mean()                               # subject runs off the canvas
        t['coverage'] = -abs(np.clip(m.mean(), 0.12, 0.7) - m.mean()) * 3
    return float(sum(t.values())), {k: round(float(v), 3) for k, v in t.items()}


def structure_concept(c):
    """Sanity of a concept image (a render.Concept): one subject, not cropped, sensible size."""
    m = c.fg
    t = {}
    border = np.concatenate([m[0], m[-1], m[:, 0], m[:, -1]])
    t['cropped'] = -6.0 * border.mean()
    t['coverage'] = -abs(np.clip(m.mean(), 0.1, 0.6) - m.mean()) * 4
    lbl, n = ndi.label(m)
    if n > 1:
        areas = np.sort(ndi.sum(m, lbl, range(1, n + 1)))[::-1]
        t['clutter'] = -3.0 * areas[1:].sum() / areas.sum()          # other objects around the subject
    return float(sum(t.values())), {k: round(float(v), 3) for k, v in t.items()}


def _clip_model():
    if 'm' not in _clip:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        torch.set_grad_enabled(False)
        name = 'openai/clip-vit-base-patch32'
        _clip['m'] = CLIPModel.from_pretrained(name).eval()
        _clip['p'] = CLIPProcessor.from_pretrained(name)
    return _clip['m'], _clip['p']


def on_white(im, size=224, smooth=False):
    """Composite RGBA on white and fit to size x size (nearest for sprites, Lanczos for concepts)."""
    from PIL import Image
    im = im.convert('RGBA')
    bg = Image.new('RGBA', im.size, (255, 255, 255, 255))
    bg.alpha_composite(im)
    return bg.convert('RGB').resize((size, size), Image.LANCZOS if smooth else Image.NEAREST)


def embed_images(images):
    """Unit CLIP embeddings (n, 512) for PIL images, or None when CLIP is unavailable."""
    try:
        m, p = _clip_model()
    except Exception:
        return None
    out = []
    for i in range(0, len(images), 32):
        inp = p(images=[im.convert('RGB') for im in images[i:i + 32]], return_tensors='pt')
        e = m.get_image_features(**inp)
        e = getattr(e, 'pooler_output', e)
        out.append((e / e.norm(dim=-1, keepdim=True)).numpy())
    return np.concatenate(out)


def embed_text(texts):
    try:
        m, p = _clip_model()
    except Exception:
        return None
    inp = p(text=list(texts), return_tensors='pt', padding=True)
    e = m.get_text_features(**inp)
    e = getattr(e, 'pooler_output', e)
    return (e / e.norm(dim=-1, keepdim=True)).numpy()
