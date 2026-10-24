"""OpenRouter Decisions API client: typed answers (choice / noul / score / box) with probabilities.

POST https://openrouter.ai/api/alpha/decisions  body {model, state, questions: {name: {type, instructions, criteria}}}
Images go in `state` as image_url parts; each side must be 256-1568 px, so sprites are nearest-upscaled (>= 280 px).
Calls are cached by content key (`store`) and budgeted by the shared `models.Ledger`.
"""
from __future__ import annotations

import io
import json
import time
import urllib.error
import urllib.request

from PIL import Image

from . import models, store

URL = 'https://openrouter.ai/api/alpha/decisions'
JEV = 'typesafe/jev-1.14'
LUNA = 'openai/gpt-6-luna-decisions'
models.EST.setdefault(JEV, 0.001)
models.EST.setdefault(LUNA, 0.001)

MIN_SIDE, MAX_SIDE = 280, 1568


def fit(im: Image.Image, target=320) -> Image.Image:
    """Nearest-neighbour upscale (integer factor) so the short side is >= MIN_SIDE; downscale if over MAX_SIDE."""
    im = im.convert('RGBA')
    bg = Image.new('RGBA', im.size, (255, 255, 255, 255))
    bg.alpha_composite(im)
    im = bg.convert('RGB')
    w, h = im.size
    if max(w, h) > MAX_SIDE:
        im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
        w, h = im.size
    if min(w, h) < MIN_SIDE:
        f = -(-max(target, MIN_SIDE) // min(w, h))
        if max(w, h) * f > MAX_SIDE:
            f = MAX_SIDE // max(w, h)
        im = im.resize((w * f, h * f), Image.NEAREST)
    return im


def image_part(im) -> tuple[dict, str]:
    """(state part, blob hash). Accepts a PIL image, PNG bytes or a path."""
    if isinstance(im, (bytes, bytearray)):
        im = Image.open(io.BytesIO(im))
    elif not isinstance(im, Image.Image):
        im = Image.open(im)
    b = io.BytesIO()
    fit(im).save(b, 'PNG')
    png = b.getvalue()
    return {'type': 'image_url', 'image_url': {'url': models.data_uri(png)}}, store.put_blob(png)


def decide(model: str, state: list, questions: dict, ledger: models.Ledger, sample: int = 0, tag: str = 'd1'):
    """state: list of str | PIL image | path(.png) | PNG bytes. Returns {answers, cost, secs, cached, call}."""
    parts, redacted = [], []
    for s in state:
        if isinstance(s, str) and not s.endswith('.png'):
            parts.append(s); redacted.append(s)
        else:
            p, h = image_part(s)
            parts.append(p); redacted.append({'blob': h})
    ck = store.key({'decisions': tag, 'model': model, 'state': redacted, 'questions': questions, 'sample': sample})
    hit = store.get_json('calls', ck)
    if hit is not None:
        ledger.cached += 1
        return {'answers': hit['response']['answers'], 'cost': hit['cost'], 'secs': hit['secs'], 'cached': True, 'call': ck}
    ledger.check(model)
    body = {'model': model, 'state': parts, 'questions': questions}
    data = json.dumps(body).encode()
    t0 = time.time()
    for attempt in range(5):
        try:
            req = urllib.request.Request(URL, data, {'Authorization': 'Bearer ' + models._key(),
                                                     'Content-Type': 'application/json'})
            resp = json.load(urllib.request.urlopen(req, timeout=120))
            if 'error' in resp:
                raise RuntimeError(f'{model}: {resp["error"]}')
            break
        except urllib.error.HTTPError as e:
            raw = e.read()
            models._record_http_error(model, attempt, e, raw)
            if e.code in (429, 500, 502, 503, 504) and attempt < 4:
                time.sleep(3 * 2 ** attempt)
                continue
            raise RuntimeError(f'{model} HTTP {e.code}: {raw[:300]!r}') from None
    secs = round(time.time() - t0, 3)
    cost = (resp.get('usage') or {}).get('cost', 0.0) or 0.0
    ledger.add(cost)
    store.put_json('calls', ck, {'request': {'model': model, 'state': redacted, 'questions': questions},
                                 'response': resp, 'cost': cost, 'secs': secs})
    return {'answers': resp['answers'], 'cost': cost, 'secs': secs, 'cached': False, 'call': ck}


# ---- symmetry classifier ------------------------------------------------------------------
SYMMETRY_Q = {'symmetry': {
    'type': 'choice',
    'instructions': 'What overall symmetry does the depicted design have? Ignore small drawing imperfections, '
                    'shading and lighting; judge the intended shape.',
    'criteria': {
        'none': 'No mirror or rotational symmetry',
        'vertical': 'Mirror symmetric across a vertical axis (left half mirrors right half), but not top/bottom',
        'horizontal': 'Mirror symmetric across a horizontal axis (top half mirrors bottom half), but not left/right',
        'both': 'Mirror symmetric across both a vertical and a horizontal axis',
        'radial': 'Rotational / radial symmetry around a centre (e.g. a starfish, a wheel)',
    }}}


def classify_symmetry(img, ledger: models.Ledger | None = None, model: str = JEV, sample: int = 0) -> dict:
    """Choice none/vertical/horizontal/both/radial -> {type, probabilities, confidence, cost, secs, cached, model}."""
    ledger = ledger or models.Ledger(0.05)
    r = decide(model, ['Concept image of a game character / object:', img], SYMMETRY_Q, ledger, sample, tag='sym1')
    a = r['answers']['symmetry']
    return {'type': a['choice'], 'probabilities': a.get('probabilities'), 'confidence': a.get('confidence'),
            'model': model, 'cost': r['cost'], 'secs': r['secs'], 'cached': r['cached'], 'call': r['call']}


if __name__ == '__main__':
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else str(store.ROOT) + '/workspace/shared/sources/squid_concept_gpt_front.png'
    led = models.Ledger(0.05)
    for m in (JEV, LUNA):
        print(json.dumps(classify_symmetry(p, led, m), indent=1))
