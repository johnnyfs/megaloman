"""Retro Diffusion API client (gen.pixel). Native low-res, grid-aligned pixel art.

API (v2, https://github.com/Retro-Diffusion/api-examples):
  POST https://api.retrodiffusion.ai/v2/inferences      header X-RD-Token: rdpk-...
    {prompt, prompt_style, width, height, num_images, seed?, input_image? (raw b64 RGB), strength?,
     reference_images? (rd_pro only, <=9), input_palette? (b64 png), remove_bg?, check_cost?}
  -> either the result inline or {"status": "accepted", "task_id"}; poll GET /v2/inferences/tasks/{task_id}.
  result: {base64_images: [...], balance_cost, remaining_balance, model}
Pricing per image: rd_fast $0.03, rd_plus/rd_mini $0.06, rd_pro $0.18. check_cost=true is a free dry run.
Keys: https://www.retrodiffusion.ai/app/devtools . Env var: RETRO_DIFFUSION_API_KEY (env or .env).

Calls are cached in cache/calls by a content key (same as models.call): a hit makes no network request.
"""
from __future__ import annotations

import base64, json, os, time, urllib.error, urllib.request

from . import store
from .models import BudgetExceeded

BASE = 'https://api.retrodiffusion.ai/v2'
ENV = 'RETRO_DIFFUSION_API_KEY'
PRICE = {'rd_fast': 0.03, 'rd_plus': 0.06, 'rd_mini': 0.06, 'rd_tile': 0.06, 'rd_pro': 0.18}


class MissingKey(RuntimeError):
    pass


def price(style: str, n: int = 1) -> float:
    if style == 'rd_pro__fps_weapon':
        return 0.25 * n
    return PRICE.get(style.split('__')[0], 0.18) * n


def api_key() -> str:
    if os.environ.get(ENV):
        return os.environ[ENV]
    p = os.path.join(store.ROOT, '.env')
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                if line.startswith(ENV + '='):
                    v = line.split('=', 1)[1].strip().strip('\'"')
                    if v:
                        return v
    raise MissingKey(f'set {ENV} (env or .env); get a key at https://www.retrodiffusion.ai/app/devtools')


def b64(png: bytes) -> str:
    return base64.b64encode(png).decode()


class Ledger:
    def __init__(self, budget=None):
        self.budget, self.spent, self.calls, self.cached = budget, 0.0, 0, 0

    def check(self, est):
        if self.budget is not None and self.spent + est > self.budget:
            raise BudgetExceeded(f'budget ${self.budget:.2f}: spent ${self.spent:.3f}, next call ~${est:.2f}')

    def add(self, cost):
        self.spent += cost or 0.0
        self.calls += 1


def _http(method, path, body=None, key=None, timeout=300):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data, method=method,
                                 headers={'X-RD-Token': key, 'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def _redact(body):
    b = dict(body)
    for k in ('input_image', 'input_palette'):
        if b.get(k):
            b[k] = {'blob': store.put_blob(base64.b64decode(b[k]))}
    if b.get('reference_images'):
        b['reference_images'] = [{'blob': store.put_blob(base64.b64decode(x))} for x in b['reference_images']]
    return b


def _result(resp, key, poll_secs=2.0, max_wait=600):
    """Inline result, or poll a task until it has one."""
    t0 = time.time()
    while True:
        r = resp.get('result') if isinstance(resp.get('result'), dict) else resp
        if r.get('base64_images'):
            return r
        st = str(resp.get('status', '')).lower()
        if st in ('failed', 'error', 'cancelled'):
            raise RuntimeError(f'retrodiffusion task failed: {json.dumps(resp)[:300]}')
        tid = resp.get('task_id')
        if not tid:
            raise RuntimeError(f'retrodiffusion: no images and no task_id: {json.dumps(resp)[:300]}')
        if time.time() - t0 > max_wait:
            raise RuntimeError(f'retrodiffusion task {tid} timed out')
        time.sleep(poll_secs)
        resp = _http('GET', f'/inferences/tasks/{tid}', key=key)
        resp.setdefault('task_id', tid)


def generate(ledger: Ledger, *, prompt: str, style: str, width: int, height: int, num_images: int = 1,
             seed: int | None = None, input_image: bytes | None = None, strength: float | None = None,
             reference_images=(), input_palette: bytes | None = None, remove_bg: bool = True,
             check_cost: bool = False, step: str = 'retrodiffusion.generate'):
    """Returns (images[bytes], cost, cached). check_cost=True is a free dry run returning ([], quoted_cost, False)."""
    body = {'prompt': prompt, 'prompt_style': style, 'width': width, 'height': height, 'num_images': num_images,
            'remove_bg': remove_bg}
    if seed is not None:
        body['seed'] = seed
    if input_image is not None:
        body['input_image'] = b64(input_image)
        if strength is not None:
            body['strength'] = strength
    if reference_images:
        body['reference_images'] = [b64(x) for x in reference_images]
    if input_palette is not None:
        body['input_palette'] = b64(input_palette)
    ck = store.key({'step': step, 'body': _redact(body)})
    if not check_cost:
        hit = store.get_json('calls', ck)
        if hit is not None:
            ledger.cached += 1
            return [store.get_blob(h) for h in hit['image_blobs']], hit['cost'], True
    key = api_key()
    if check_cost:
        resp = _http('POST', '/inferences', dict(body, check_cost=True), key=key)
        r = resp.get('result') if isinstance(resp.get('result'), dict) else resp
        return [], r.get('balance_cost', r.get('cost', r.get('credit_cost'))), False
    ledger.check(price(style, num_images))
    t0 = time.time()
    for attempt in range(5):
        try:
            resp = _result(_http('POST', '/inferences', body, key=key), key)
            break
        except urllib.error.HTTPError as e:
            raw = e.read()[:300]
            if e.code in (429, 500, 502, 503, 504) and attempt < 4:
                time.sleep(min(60, 5 * 2 ** attempt))
                continue
            raise RuntimeError(f'retrodiffusion {style} HTTP {e.code}: {raw!r}') from None
    imgs = [base64.b64decode(x) for x in resp['base64_images']]
    blobs = [store.put_blob(x) for x in imgs]
    cost = resp.get('balance_cost')
    cost = float(cost) if cost is not None else price(style, num_images)
    ledger.add(cost)
    meta = {k: v for k, v in resp.items() if k != 'base64_images'}
    store.put_json('calls', ck, {'request': _redact(body), 'response': meta, 'image_blobs': blobs, 'cost': cost,
                                 'secs': round(time.time() - t0, 1)})
    return imgs, cost, False
