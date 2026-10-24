"""OpenRouter client: the only network egress for model calls. Budget is enforced before every uncached call."""
import base64, json, os, threading, time, urllib.error, urllib.request

from . import store

URL = 'https://openrouter.ai/api/v1/chat/completions'

# Pre-call cost estimates (USD), measured 2026-10-21; used only for budget enforcement.
EST = {
    'openai/gpt-5.4-image-2': 0.25,
    'google/gemini-3.1-flash-image': 0.08,
    'google/gemini-nano-banana-2.1': 0.08,
    'openai/gpt-5.6-sol': 0.03,
}


# Every HTTP error seen (full body + rate-limit headers), for diagnosis; listeners get each record as it happens.
HTTP_ERRORS, ERROR_LISTENERS = [], []


def _record_http_error(model, attempt, e, raw):
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = None
    hdrs = {k: v for k, v in (e.headers or {}).items()
            if k.lower() in ('retry-after', 'date') or 'ratelimit' in k.lower() or k.lower().startswith('x-')}
    rec = {'ts': time.strftime('%Y-%m-%dT%H:%M:%S'), 'model': model, 'attempt': attempt, 'code': e.code,
           'body': parsed if parsed is not None else raw.decode('utf-8', 'replace'), 'headers': hdrs}
    HTTP_ERRORS.append(rec)
    for fn in list(ERROR_LISTENERS):
        fn(rec)


class BudgetExceeded(RuntimeError):
    pass


class Ledger:
    def __init__(self, budget=None):
        self.budget, self.spent, self.calls, self.cached = budget, 0.0, 0, 0
        self.lock = threading.Lock()

    def check(self, model):
        est = EST.get(model, 0.05)
        if self.budget is not None and self.spent + est > self.budget:
            raise BudgetExceeded(f'budget ${self.budget:.2f}: spent ${self.spent:.3f}, next {model} call ~${est:.2f}')

    def add(self, cost):
        with self.lock:
            self.spent += cost or 0.0
            self.calls += 1


def _key():
    if os.environ.get('OPENROUTER_API_KEY'):
        return os.environ['OPENROUTER_API_KEY']
    with open(os.path.join(store.ROOT, '.env')) as f:
        for line in f:
            if line.startswith('OPENROUTER_API_KEY='):
                return line.split('=', 1)[1].strip().strip('\'"')
    raise RuntimeError('OPENROUTER_API_KEY not found (env or .env)')


def data_uri(png: bytes) -> str:
    return 'data:image/png;base64,' + base64.b64encode(png).decode()


def _redact(body):
    """Transcript copy of a request with image data replaced by blob hashes."""
    b = json.loads(json.dumps(body))
    for m in b.get('messages', []):
        if isinstance(m.get('content'), list):
            for part in m['content']:
                if part.get('type') == 'image_url':
                    raw = base64.b64decode(part['image_url']['url'].split(',', 1)[1])
                    part['image_url'] = {'blob': store.put_blob(raw)}
    return b


def call(body, ledger: Ledger, cache_key: str):
    """POST a chat completion. Cached by cache_key: a hit returns the stored response with no network call.
    Returns (response, images[bytes], cost, cached)."""
    hit = store.get_json('calls', cache_key)
    if hit is not None:
        ledger.cached += 1
        imgs = [store.get_blob(h) for h in hit['image_blobs']]
        return hit['response'], imgs, hit['cost'], True
    ledger.check(body['model'])
    body = dict(body, usage={'include': True})
    data = json.dumps(body).encode()
    t0 = time.time()
    for attempt in range(6):
        try:
            req = urllib.request.Request(URL, data, {'Authorization': 'Bearer ' + _key(), 'Content-Type': 'application/json'})
            resp = json.load(urllib.request.urlopen(req, timeout=400))
            if 'error' in resp:
                raise RuntimeError(f"{body['model']}: {resp['error']}")
            break
        except urllib.error.HTTPError as e:
            raw = e.read()
            _record_http_error(body['model'], attempt, e, raw)
            msg = raw[:300]
            if e.code in (429, 500, 502, 503, 504) and attempt < 5:
                time.sleep(min(60, 5 * 2 ** attempt))
                continue
            raise RuntimeError(f"{body['model']} HTTP {e.code}: {msg!r}") from None
    imgs, blobs = [], []
    msg = resp['choices'][0]['message']
    for im in msg.pop('images', None) or []:
        raw = base64.b64decode(im['image_url']['url'].split(',', 1)[1])
        imgs.append(raw)
        blobs.append(store.put_blob(raw))
    cost = (resp.get('usage') or {}).get('cost', 0.0) or 0.0
    ledger.add(cost)
    store.put_json('calls', cache_key, {'request': _redact(body), 'response': resp, 'image_blobs': blobs,
                                        'cost': cost, 'secs': round(time.time() - t0, 1)})
    return resp, imgs, cost, False
