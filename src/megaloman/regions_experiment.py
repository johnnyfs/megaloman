"""Region-ladder milestone 1 (REGIONS_PLAN.md): concept -> legend -> painted region maps -> snapped + alignment-checked.

uv run python -m megaloman.regions_experiment [--tag m1] [--budget 3]
Writes workspace/experiments/regions/<tag>/: concept.png, legend.json, <cand>/{painted,regions,regions_clipped,overlay}.png,
results.json, sheet.png. Model calls are cached, so a rerun costs $0.

Candidates are painted in parallel. The deterministic-segmentation fallback ('segment': colour clustering + connected
components on the concept, segments named by the VLM) runs alongside as its own candidate, so both paths are compared;
`path` in results.json says which path produced the best passing map (paint preferred).
"""
import argparse, os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from . import imaging, layout, models, regions

CONCEPT = os.path.join(layout.W, 'shared', 'sources', 'squid_concept_gpt_front.png')
CANDIDATES = [   # (id, model, seed); model None = the segmentation fallback
    ('nb2-s1', 'google/gemini-nano-banana-2.1', 1),
    ('nb2-s2', 'google/gemini-nano-banana-2.1', 2),
    ('gpt-s1', 'openai/gpt-5.4-image-2', 1),
    ('segment', None, 0),
]


def out_dir(tag):
    return os.path.join(layout.W, 'experiments', 'regions', tag)


def _finish(rec, cdir, labels, concept, sil, leg):
    """Shared tail: alignment check, renders, file list."""
    rec['align'] = regions.align(labels, sil, leg)
    regions.render_map(labels, leg).save(os.path.join(cdir, 'regions.png'))
    regions.overlay(concept, labels, leg).save(os.path.join(cdir, 'overlay.png'))
    if rec['align']['pass']:
        regions.render_map(regions.clip_to_silhouette(labels, sil), leg).save(os.path.join(cdir, 'regions_clipped.png'))
    rec['files'] = {k: layout.rel(os.path.join(cdir, f'{k}.png')) for k in ('painted', 'regions', 'overlay')}
    return rec


def _paint_candidate(d, cid, model, seed, concept, concept_rgb, sil, leg, ledger, status):
    rec = {'id': cid, 'model': model, 'seed': seed, 'path': 'paint'}
    p = regions.paint(concept_rgb, leg, model, seed, ledger,
                      on_retry=lambda n, secs: status(cid, f'rate-limited, retrying ({n}/{len(regions.RETRY_429)})'))
    rec.update(call=p['call'], cost=round(p['cost'], 4), cached=p['cached'], error=p['error'], retries=p.get('retries', 0),
               model_text=p.get('text'))
    if p['png'] is None:
        rec['align'] = {'pass': False, 'iou': 0.0}
        return rec
    status(cid, 'snapping')
    cdir = os.path.join(d, cid)
    os.makedirs(cdir, exist_ok=True)
    painted = imaging.from_png(p['png'])
    painted.save(os.path.join(cdir, 'painted.png'))
    rec['painted_size'] = list(painted.size)
    labels = regions.snap(painted, leg, concept.size)
    rec['iou_raw'] = round(regions.iou(labels > 0, sil), 4)                 # as painted, only resized
    labels, rec['transform'], v = regions.register(labels, sil)
    rec['iou_registered'] = round(v, 4)                                     # after scale + shift fit
    labels, rec['holes_filled_px'] = regions.fill_holes(labels, sil)        # final: enclosed white -> its part
    return _finish(rec, cdir, labels, concept, sil, leg)


def _segment_candidate(d, concept, sil, leg, ledger, status):
    rec = {'id': 'segment', 'model': regions.LEGEND_MODEL, 'seed': 0, 'path': 'segment'}
    cdir = os.path.join(d, 'segment')
    os.makedirs(cdir, exist_ok=True)
    seg, n = regions.segment(concept, sil)
    labels, r = regions.name_segments(concept, seg, n, leg, ledger)
    with open(os.path.join(cdir, 'painted.png'), 'wb') as f:      # the numbered segments the VLM named
        f.write(r.pop('sheet_png'))
    rec.update(segments=n, call=r['call'], cost=round(r['cost'], 4), cached=r['cached'], assign=r['assign'],
               note='built from the concept alpha, so IoU is ~1 by construction; parts present is the real test')
    unassigned = (labels == 0) & sil
    rec['unassigned_px'] = int(unassigned.sum())
    return _finish(rec, cdir, labels, concept, sil, leg)


def run(tag='m1', budget=3.0, candidates=CANDIDATES, on_update=None):
    d = out_dir(tag)
    os.makedirs(d, exist_ok=True)
    ledger = models.Ledger(budget)
    concept = Image.open(CONCEPT).convert('RGBA')
    sil, shadow = regions.concept_masks(concept)
    concept_rgb = regions.concept_on_white(concept, sil)
    cs = np.array(concept); cs[..., 3] = np.where(sil, 255, 0)
    Image.fromarray(cs, 'RGBA').save(os.path.join(d, 'concept.png'))      # the squid, shadow removed: the IoU target
    concept_rgb.save(os.path.join(d, 'concept_on_white.png'))              # what the models are shown
    state = {'created': layout.now(), 'code': layout.code_version(), 'tag': tag,
             'concept': layout.rel(CONCEPT), 'concept_size': list(concept.size),
             'silhouette_px': int(sil.sum()), 'shadow_px': int(shadow.sum()),
             'legend': None, 'candidates': [], 'status': {c[0]: 'waiting for legend' for c in candidates}, 'path': None}

    def update():
        if on_update:
            on_update(d, state)

    def status(cid, text):
        state['status'][cid] = text
        update()

    state['http_errors'] = []      # full bodies + rate-limit headers of every HTTP error (429s from upstream etc.)

    def on_err(rec):
        state['http_errors'].append(rec)
        if rec['code'] == 429:
            n = sum(e['code'] == 429 and e['model'] == rec['model'] for e in state['http_errors'])
            for c in candidates:
                if c[1] == rec['model'] and c[0] in state['status']:
                    state['status'][c[0]] = f'rate-limited, retrying ({n} x 429 so far)'
        update()

    models.ERROR_LISTENERS.append(on_err)
    update()
    leg = regions.legend(concept_rgb, ledger)
    layout.write_json(os.path.join(d, 'legend.json'), leg)
    state['legend'] = leg
    for c in candidates:
        state['status'][c[0]] = 'painting' if c[1] else 'segmenting'
    update()

    def one(c):
        cid, model, seed = c
        try:
            rec = (_paint_candidate(d, cid, model, seed, concept, concept_rgb, sil, leg, ledger, status) if model
                   else _segment_candidate(d, concept, sil, leg, ledger, status))
        except models.BudgetExceeded as e:
            rec = {'id': cid, 'model': model, 'seed': seed, 'error': str(e), 'align': {'pass': False, 'iou': 0.0}}
        state['candidates'].append(rec)
        state['candidates'].sort(key=lambda r: [x[0] for x in candidates].index(r['id']))
        state['status'].pop(cid, None)
        update()
        return rec

    try:
        with ThreadPoolExecutor(max_workers=len(candidates)) as ex:
            list(ex.map(one, candidates))
    finally:
        models.ERROR_LISTENERS.remove(on_err)

    passed = [r for r in state['candidates'] if r['align']['pass']]
    painted_ok = [r for r in passed if r.get('path') == 'paint']
    best = max(painted_ok or passed, key=lambda r: r['align']['iou'], default=None)
    state['best'] = best and best['id']
    state['path'] = best and best['path']
    state['cost'] = {'spent': round(ledger.spent, 4), 'calls': ledger.calls, 'cached_calls': ledger.cached,
                     'budget': budget}
    layout.write_json(os.path.join(d, 'results.json'), state)
    _sheet(d, concept, state, leg)
    layout.event('experiment', name=f'regions/{tag}', path=state['path'], best=state['best'],
                 iou={r['id']: r['align'].get('iou') for r in state['candidates']},
                 passed=[r['id'] for r in passed], cost=state['cost']['spent'], outputs=[layout.rel(d)])
    update()
    return state


def _sheet(d, concept, state, leg):
    rows = [('legend', [(Image.new('RGBA', (32, 32), imaging.rgb(p['color'].lstrip('#')) + (255,)), p['name'][:17], None)
                        for p in leg['parts']])]
    for r in state['candidates']:
        cells = [(concept, 'concept', None)]
        for k in ('painted', 'regions', 'overlay'):
            if r.get('files'):
                cells.append((Image.open(os.path.join(layout.W, r['files'][k])).convert('RGBA'), k, None))
        a = r['align']
        verdict = f"IoU {a.get('iou', 0):.3f} {'PASS' if a['pass'] else 'FAIL'}"
        cells[-1] = (cells[-1][0], f"{cells[-1][1]}\n{verdict}", None if a['pass'] else 'reject')
        rows.append((f"{r['id']}", cells))
    imaging.contact_sheet(rows, box=200, title=f"regions/{state['tag']}: concept -> painted -> snapped -> overlay").save(
        os.path.join(d, 'sheet.png'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='m1'); ap.add_argument('--budget', type=float, default=3.0)
    a = ap.parse_args()
    s = run(a.tag, a.budget)
    for r in s['candidates']:
        print(r['id'], r['model'], r['align'].get('iou'), r['align']['pass'], r.get('cost'), r.get('error'))
    print('path', s['path'], 'best', s['best'], 'cost', s['cost'])


if __name__ == '__main__':
    main()
