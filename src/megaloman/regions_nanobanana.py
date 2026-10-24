"""Regions v2 painted by Nano Banana (google/gemini-nano-banana-2.1, BYOK Google AI Studio key), vs GPT's v2 map.

Same legend (cached), same v2 paint prompt, same snap / register / fill / align / clip / island-fold as
regions_v2._candidate; only the paint model changes. BYOK: OpenRouter reports usage.cost 0, so the real (Google)
cost is read from usage.cost_details.upstream_inference_cost.

uv run python -m megaloman.regions_nanobanana [--seeds 1 2]
Writes workspace/experiments/regions/nanobanana/: nb2-s<seed>/{painted,regions,overlay}.png, results.json,
compare.png (GPT best | NB best).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from PIL import Image

from . import imaging, layout, models, regions, regions_v2, store

OUT = os.path.join(layout.W, 'experiments', 'regions', 'nanobanana')
NB = 'google/gemini-nano-banana-2.1'
GPT_DIR = regions_v2.OUT


def _candidate(seed, concept, concept_rgb, sil, leg, ledger):
    cid = f'nb2-s{seed}'
    rec = {'id': cid, 'model': NB, 'seed': seed}
    p = regions_v2.paint(concept_rgb, leg, seed, ledger, model=NB)
    hit = store.get_json('calls', p['call']) or {}
    resp = hit.get('response') or {}
    u = resp.get('usage') or {}
    rec.update(cached=p['cached'], error=p['error'], model_text=p.get('text'), provider=resp.get('provider'),
               is_byok=u.get('is_byok'), openrouter_cost=u.get('cost'),
               upstream_cost=(u.get('cost_details') or {}).get('upstream_inference_cost'),
               finish=(resp.get('choices') or [{}])[0].get('native_finish_reason'))
    if p['png'] is None:
        rec['align'] = {'pass': False, 'iou': 0.0}
        return rec
    cdir = os.path.join(OUT, cid)
    os.makedirs(cdir, exist_ok=True)
    painted = imaging.from_png(p['png'])
    painted.save(os.path.join(cdir, 'painted.png'))
    rec['painted_size'] = list(painted.size)
    labels = regions.snap(painted, leg, concept.size)
    rec['iou_raw'] = round(regions.iou(labels > 0, sil), 4)
    labels, rec['transform'], v = regions.register(labels, sil)
    rec['iou_registered'] = round(v, 4)
    labels, rec['holes_filled_px'] = regions.fill_holes(labels, sil)
    rec['align'] = regions.align(labels, sil, leg)
    if rec['align']['iou_ok']:
        labels = regions.clip_to_silhouette(labels, sil)
    labels, rec['islands_folded'] = regions_v2.clean_islands(labels, leg)
    np.save(os.path.join(cdir, 'labels.npy'), labels)
    regions.render_map(labels, leg).save(os.path.join(cdir, 'regions.png'))
    regions.overlay(concept, labels, leg).save(os.path.join(cdir, 'overlay.png'))
    rec['occlusion'] = regions_v2.occlusion_check(labels, leg)
    return rec


def run(seeds=(1, 2), budget=1.0):
    os.makedirs(OUT, exist_ok=True)
    ledger = models.Ledger(budget)
    concept = Image.open(regions_v2.CONCEPT).convert('RGBA')
    sil, _ = regions.concept_masks(concept)
    concept_rgb = regions.concept_on_white(concept, sil)
    leg = regions_v2.legend(concept_rgb, ledger)
    cands = [_candidate(s, concept, concept_rgb, sil, leg, ledger) for s in seeds]
    with open(os.path.join(GPT_DIR, 'results.json')) as f:
        gpt = json.load(f)
    gbest = next(c for c in gpt['candidates'] if c['id'] == gpt['best'])
    ok = [c for c in cands if c['align'].get('iou')]
    best = max(ok, key=lambda c: c['align']['iou'], default=None)
    # part-level agreement with GPT's chosen map (per-part IoU on the same legend)
    agree = None
    if best:
        a = np.load(os.path.join(OUT, best['id'], 'labels.npy'))
        b = np.load(os.path.join(GPT_DIR, 'labels.npy'))
        agree = {p['name']: round(regions.iou(a == i + 1, b == i + 1), 3) for i, p in enumerate(leg['parts'])}
        agree['_pixel_agreement'] = round(float(((a == b) & ((a > 0) | (b > 0))).sum() / max(1, ((a > 0) | (b > 0)).sum())), 3)
    state = {'created': layout.now(), 'code': layout.code_version(), 'candidates': cands, 'best': best and best['id'],
             'gpt_best': {k: gbest.get(k) for k in ('id', 'iou_raw', 'iou_registered', 'transform')} |
                         {'iou': gbest['align']['iou'], 'all_present': gbest['align']['all_present']},
             'part_iou_vs_gpt': agree,
             'upstream_cost_total': round(sum(c.get('upstream_cost') or 0 for c in cands), 4)}
    layout.write_json(os.path.join(OUT, 'results.json'), state)
    if best:
        g = Image.open(os.path.join(GPT_DIR, 'regions.png')).convert('RGBA')
        n = Image.open(os.path.join(OUT, best['id'], 'regions.png')).convert('RGBA')
        rows = [('maps', [(concept, 'concept', None), (g, f"GPT {gbest['id']}\nIoU {gbest['align']['iou']:.3f}", None),
                          (n, f"NB {best['id']}\nIoU {best['align']['iou']:.3f}", None)])]
        for c in cands:
            cells = [(Image.open(os.path.join(OUT, c['id'], f'{k}.png')).convert('RGBA'), k, None)
                     for k in ('painted', 'regions', 'overlay') if os.path.exists(os.path.join(OUT, c['id'], f'{k}.png'))]
            if cells:
                rows.append((c['id'], cells))
        imaging.contact_sheet(rows, box=240, title='regions v2: GPT vs Nano Banana').save(os.path.join(OUT, 'compare.png'))
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2])
    ap.add_argument('--budget', type=float, default=1.0)
    a = ap.parse_args()
    s = run(tuple(a.seeds), a.budget)
    for c in s['candidates']:
        print(c['id'], c.get('provider'), c.get('is_byok'), c.get('upstream_cost'), c.get('finish'), c.get('error'),
              c.get('iou_raw'), c.get('iou_registered'), c.get('transform'), c['align'].get('iou'),
              c['align'].get('all_present'), c['align'].get('parts_present'))
    print('gpt', s['gpt_best'])
    print('part_iou_vs_gpt', s['part_iou_vs_gpt'], 'upstream', s['upstream_cost_total'])


if __name__ == '__main__':
    main()
