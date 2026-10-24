"""Bake-off d1: current chat judge (gpt-5.6-sol) vs decision models (Jev, Luna) on a small labelled sprite set.

Items:
  synthetic pairs (known answer): clean sprite vs noise / broken eye / merged parts / unmirrored (copied) half
  rubric checks (Noul, known answer): "left-right mirror symmetric?" on v2 candidates, "stray noise?" on clean/noisy
  ladder preferences (reference = existing gpt-5.6-sol judge, cached): v2 original/left/right, v3 vs v1 at 32/16
  render variants (no reference; agreement only)
Each item x judge x 3 repeats. uv run python -m megaloman.bakeoff_decisions [--budget 2]
"""
from __future__ import annotations

import json
import random
import re
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from . import decisions as D, models, store

ROOT = Path(store.ROOT)
REG = ROOT / 'workspace/experiments/regions'
OUT = ROOT / 'workspace/experiments/decisions'
CONCEPT = ROOT / 'workspace/shared/sources/squid_concept_gpt_front.png'
GPT = 'openai/gpt-5.6-sol'
JUDGES = [GPT, D.JEV, D.LUNA]
REPEATS = 3
EYE = (0xFC, 0xE0, 0xA8)


# ---- synthetic corruptions ----------------------------------------------------------------
def load(p) -> np.ndarray:
    im = Image.open(p).convert('RGBA')
    bg = Image.new('RGBA', im.size, (255, 255, 255, 255))
    bg.alpha_composite(im)
    return np.asarray(bg.convert('RGB')).copy()


def colours(a):
    c = Counter(map(tuple, a.reshape(-1, 3)))
    return [k for k, _ in c.most_common()]


def noise(a, rng, frac=0.08):
    a = a.copy()
    h, w, _ = a.shape
    pal = colours(a)
    for _ in range(int(frac * h * w)):
        a[rng.randrange(h), rng.randrange(w)] = pal[rng.randrange(len(pal))]
    return a


def broken_eye(a, rng):
    """Remove the right eye: eye-coloured pixels right of centre become the visor colour (black)."""
    a = a.copy()
    m = (a == EYE).all(-1)
    m[:, : a.shape[1] // 2] = False
    a[m] = (0, 0, 0)
    return a if m.any() else None


def merged(a, rng):
    """Merge the 2nd and 3rd most common non-white, non-black colours into one (two parts lose their boundary)."""
    pal = [c for c in colours(a) if c not in ((255, 255, 255), (0, 0, 0))]
    a = a.copy()
    a[(a == pal[1]).all(-1)] = pal[0]
    return a


def copied_half(a, rng):
    """Right half replaced by a copy (not a mirror) of the left half: a broken, shifted symmetry."""
    a = a.copy()
    w = a.shape[1]
    a[:, w - w // 2:] = a[:, : w // 2]
    return a


def png(a) -> Image.Image:
    return Image.fromarray(a.astype(np.uint8))


# ---- item set -----------------------------------------------------------------------------
PAIR_Q = 'Two candidate pixel-art sprites of the same game enemy (a cartoon robotic squid) are shown as A and B. ' \
         'Which is the better, cleaner sprite: coherent shapes, distinct parts, sensible left-right balance, no stray pixels?'


def build_items():
    rng = random.Random(7)
    items = []
    bases = {'render32': REG / 'render/render_32.png', 'render24': REG / 'render/render_24.png',
             'render64': REG / 'render/render_64.png', 'v3map32': REG / 'ladder/v3_significance/t32/rung_32.png'}
    for bn, bp in bases.items():
        a = load(bp)
        for cn, fn in (('noise', noise), ('eye', broken_eye), ('merged', merged), ('copiedhalf', copied_half)):
            b = fn(a, rng)
            if b is None:
                continue
            clean_first = rng.random() < 0.5
            imgs = [png(a), png(b)] if clean_first else [png(b), png(a)]
            items.append({'id': f'syn_{bn}_{cn}', 'group': 'synthetic', 'kind': 'choice', 'images': imgs,
                          'question': PAIR_Q, 'options': {'A': 'Sprite A is better', 'B': 'Sprite B is better'},
                          'known': 'A' if clean_first else 'B'})
    # rubric: symmetry of v2 candidates (measured: mirrored = exact, original < 0.9 on some parts)
    for s in (64, 24):
        for k in ('original', 'left'):
            items.append({'id': f'rub_sym_{s}_{k}', 'group': 'rubric', 'kind': 'noul',
                          'images': [REG / f'ladder/v2_symmetry/candidates/rung_{s}_{k}.png'],
                          'question': 'This is a pixel-art region map (each flat colour is one part). Is it EXACTLY '
                                      'left-right mirror symmetric, pixel for pixel?',
                          'known': k != 'original'})
    for bn in ('render32', 'render24'):
        a = load(bases[bn])
        for noisy in (False, True):
            items.append({'id': f'rub_noise_{bn}_{int(noisy)}', 'group': 'rubric', 'kind': 'noul',
                          'images': [png(noise(a, random.Random(3)) if noisy else a)],
                          'question': 'Does this pixel-art sprite contain stray, scattered noise pixels that do '
                                      'not belong to any shape?', 'known': noisy})
    # ladder v2: original / left / right as the existing judge saw them (concept + panel)
    v2 = json.loads((REG / 'ladder/v2_symmetry/symmetry.json').read_text())['rungs']
    for s, r in v2.items():
        j = r.get('judge')
        if not j:
            continue
        letters = 'ABC'
        ref_votes = Counter(x['letter'] for x in j['samples'] if x.get('letter'))
        items.append({'id': f'v2_{s}', 'group': 'ladder_v2', 'kind': 'choice',
                      'images': [CONCEPT, Image.open(store.blob_path(j['panel_blob']))],
                      'question': 'The first image is concept art of a robotic squid enemy. The second shows three '
                                  f'candidate {s}x{s} pixel region maps labelled A, B, C left to right. Which candidate '
                                  'reads best as this creature: clear paired eyes, balanced fins and pods, distinct tentacles?',
                      'options': {L: f'Candidate {L}' for L in letters[:len(j['order_shown'])]},
                      'reference': ref_votes.most_common(1)[0][0],
                      'cached_gpt': [x.get('letter') for x in j['samples']]})
    v3 = json.loads((REG / 'ladder/v3_significance/significance.json').read_text())['judge']
    for s, r in v3.items():
        smp = [x for x in r['samples'] if x['order'] == 'v1/v3']
        votes = Counter(x['letter'] for x in r['samples'] if x['order'] == 'v1/v3')
        ref = 'A' if r['votes'].get('v1', 0) > r['votes'].get('v3', 0) else ('B' if r['votes'].get('v3', 0) > r['votes'].get('v1', 0) else None)
        items.append({'id': f'v3_vs_v1_{s}', 'group': 'ladder_v3', 'kind': 'choice',
                      'images': [CONCEPT, Image.open(store.blob_path(smp[0]['panel_blob']))],
                      'question': 'The first image is concept art of a robotic squid enemy. The second shows two candidate '
                                  f'{s}x{s} pixel region maps, A (left) and B (right). Which keeps the squid\'s CHARACTER '
                                  'better: the shapes and proportions that make it recognisable and appealing?',
                      'options': {'A': 'Candidate A', 'B': 'Candidate B'}, 'reference': ref,
                      'cached_gpt': [x['letter'] for x in smp]})
    for s in (32, 16):
        for var in ('alllines', 'inset'):
            items.append({'id': f'render_{s}_{var}', 'group': 'render', 'kind': 'choice',
                          'images': [CONCEPT, REG / f'render/render_{s}.png', REG / f'render/render_{s}_{var}.png'],
                          'question': f'Concept art of a robotic squid enemy, then two {s}x{s} pixel-art renders A and B. '
                                      'Which render reads better as this creature at game size?',
                          'options': {'A': 'Render A', 'B': 'Render B'}, 'reference': None})
    return items


# ---- judges -------------------------------------------------------------------------------
def state_for(it):
    st = []
    labels = 'AB' if it['group'] == 'synthetic' else None
    for i, im in enumerate(it['images']):
        if labels:
            st.append(f'Sprite {labels[i]}:')
        elif it['group'] == 'render' and i:
            st.append(f'Render {"AB"[i - 1]}:')
        st.append(im)
    return st


def run_decision(model, it, ledger, rep):
    if it['kind'] == 'choice':
        q = {'q': {'type': 'choice', 'instructions': it['question'], 'criteria': it['options']}}
    else:
        q = {'q': {'type': 'noul', 'instructions': it['question'], 'criteria': {'true': 'yes', 'false': 'no'}}}
    r = D.decide(model, state_for(it), q, ledger, rep, tag='bake1')
    a = r['answers']['q']
    ans = a['choice'] if it['kind'] == 'choice' else a['noul'] >= 0.5
    p = (a.get('probabilities') or {}).get(a.get('choice')) if it['kind'] == 'choice' else a['noul']
    return {'answer': ans, 'p': p, 'cost': r['cost'], 'secs': r['secs'], 'cached': r['cached']}


def run_gpt(it, ledger, rep):
    if it['kind'] == 'choice':
        fmt = 'Reply with JSON only: {"answer": "' + '" | "'.join(it['options']) + '", "reason": "<one short sentence>"}'
    else:
        fmt = 'Reply with JSON only: {"answer": true | false, "reason": "<one short sentence>"}'
    content, red = [{'type': 'text', 'text': it['question'] + '\n' + fmt}], []
    for s in state_for(it):
        if isinstance(s, str):
            content.append({'type': 'text', 'text': s}); red.append(s)
        else:
            p, h = D.image_part(s)
            content.append(p); red.append(h)
    body = {'model': GPT, 'response_format': {'type': 'json_object'}, 'messages': [{'role': 'user', 'content': content}]}
    ck = store.key({'bake1_gpt': it['question'], 'state': red, 'fmt': fmt, 'sample': rep})
    t0 = time.time()
    resp, _, cost, cached = models.call(body, ledger, ck)
    rec = store.get_json('calls', ck)
    txt = resp['choices'][0]['message'].get('content') or ''
    m = re.search(r'\{.*\}', txt, re.S)
    ans = json.loads(m.group(0)).get('answer') if m else None
    return {'answer': ans, 'p': None, 'cost': cost, 'secs': rec.get('secs', round(time.time() - t0, 1)), 'cached': cached}


def judge_all(items, budget):
    led = models.Ledger(budget)
    jobs = []
    for it in items:
        for j in JUDGES:
            for rep in range(REPEATS):
                jobs.append((it, j, rep))

    def one(job):
        it, j, rep = job
        if j == GPT and 'cached_gpt' in it:
            v = it['cached_gpt'][rep] if rep < len(it['cached_gpt']) else None
            return it['id'], j, rep, {'answer': v, 'p': None, 'cost': 0.0, 'secs': None, 'cached': True, 'from': 'existing judge'}
        try:
            r = run_gpt(it, led, rep) if j == GPT else run_decision(j, it, led, rep)
        except Exception as e:      # noqa: BLE001 - record and continue
            r = {'answer': None, 'error': str(e)[:200], 'cost': 0.0, 'secs': None, 'cached': False}
        return it['id'], j, rep, r

    with ThreadPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(one, jobs))
    out = {}
    for iid, j, rep, r in res:
        out.setdefault(iid, {}).setdefault(j, [None] * REPEATS)[rep] = r
    return out, led


def majority(rs):
    c = Counter(r['answer'] for r in rs if r and r.get('answer') is not None)
    return c.most_common(1)[0][0] if c else None


def summarise(items, res):
    S = {}
    for j in JUDGES:
        known = [it for it in items if 'known' in it]
        refd = [it for it in items if it.get('reference')]
        acc = [majority(res[it['id']][j]) == it['known'] for it in known]
        acc_runs = [r['answer'] == it['known'] for it in known for r in res[it['id']][j]]
        ref = [majority(res[it['id']][j]) == it['reference'] for it in refd]
        stable = [len({r['answer'] for r in res[it['id']][j]}) == 1 for it in items]
        lat = [r['secs'] for it in items for r in res[it['id']][j] if r.get('secs') is not None]
        cost = sum(r['cost'] for it in items for r in res[it['id']][j])
        n_calls = sum(1 for it in items for r in res[it['id']][j] if not r.get('from'))
        by_group = {}
        for it in known:
            by_group.setdefault(it['group'], []).append(majority(res[it['id']][j]) == it['known'])
        S[j] = {'acc_known': round(statistics.mean(acc), 3), 'n_known': len(acc),
                'acc_known_per_run': round(statistics.mean(acc_runs), 3),
                'acc_by_group': {g: f'{sum(v)}/{len(v)}' for g, v in by_group.items()},
                'agree_existing_judge': f'{sum(ref)}/{len(ref)}',
                'unstable_items': sum(not s for s in stable), 'stable_frac': round(statistics.mean(stable), 3),
                'latency_median_s': round(statistics.median(lat), 2) if lat else None,
                'cost_total': round(cost, 5), 'cost_per_call': round(cost / max(1, n_calls), 6),
                'errors': sum(1 for it in items for r in res[it['id']][j] if r.get('error'))}
    pair = {}
    for a in JUDGES:
        for b in JUDGES:
            if a < b:
                ag = [majority(res[it['id']][a]) == majority(res[it['id']][b]) for it in items]
                pair[f'{a} ~ {b}'] = f'{sum(ag)}/{len(ag)}'
    return S, pair


def main():
    budget = float(sys.argv[sys.argv.index('--budget') + 1]) if '--budget' in sys.argv else 2.0
    OUT.mkdir(parents=True, exist_ok=True)
    items = build_items()
    (OUT / 'items').mkdir(exist_ok=True)
    for it in items:   # save item images for inspection
        ims = [Image.open(i) if isinstance(i, Path) else i for i in it['images']]
        fits = [D.fit(i) for i in ims]
        h = max(f.height for f in fits)
        sheet = Image.new('RGB', (sum(f.width for f in fits) + 10 * (len(fits) - 1), h), 'white')
        x = 0
        for f in fits:
            sheet.paste(f, (x, 0)); x += f.width + 10
        sheet.thumbnail((900, 300))
        sheet.save(OUT / 'items' / f'{it["id"]}.png')
    res, led = judge_all(items, budget)
    S, pair = summarise(items, res)
    rows = [{k: v for k, v in it.items() if k not in ('images',)} | {
        'majority': {j: majority(res[it['id']][j]) for j in JUDGES},
        'runs': {j: [{k: r.get(k) for k in ('answer', 'p', 'secs', 'cost', 'cached', 'error') if k in r} for r in res[it['id']][j]]
                 for j in JUDGES}} for it in items]
    report = {'version': 'd1', 'repeats': REPEATS, 'n_items': len(items), 'summary': S, 'pairwise_agreement': pair,
              'ledger': {'spent': round(led.spent, 4), 'calls': led.calls, 'cached': led.cached}, 'items': rows}
    (OUT / 'bakeoff.json').write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({k: report[k] for k in ('summary', 'pairwise_agreement', 'ledger')}, indent=1))


if __name__ == '__main__':
    main()
