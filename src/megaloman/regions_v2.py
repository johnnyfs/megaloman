"""Regions v2: Johnny's critique of render v1 (PIPELINE_RULES 2026-10-23, "regions over coloration").

What changes from regions.py (v1, milestone 1), which is reused for snap / register / align:
  - legend: OCCLUSION MAKES A SUBCLASS. Instances of a repeated part that are partly hidden behind other instances
    are their own part (<part>_back, depth 'back'). Front instances that touch each other are split into separate
    parts with the same `material`, so the boundary between them survives (it becomes a separating line at render).
    Suction cups are their own part. Pattern-only parts (spots, stripes) are not parts. Small mechanical connectors
    (tentacle joints) fold into a dark undercarriage or their parent part.
    New fields per part: material (render colour group), depth (front | back), occluded_by (part names).
  - paint: the prompt asks for the occlusion layering explicitly (paint back to front; a front part keeps its whole
    outline; a back part shows only where nothing covers it).

uv run python -m megaloman.regions_v2 [--budget 1.5] [--seeds 1 2]
Writes workspace/experiments/regions/v2/regions/: legend.json, <cand>/{painted,regions,overlay}.png, regions.png
(the chosen map, clipped to the concept silhouette), results.json, sheet.png. Model calls are cached.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from . import imaging, layout, models, regions, store

OUT = os.path.join(layout.W, 'experiments', 'regions', 'v2', 'regions')
CONCEPT = os.path.join(layout.W, 'shared', 'sources', 'squid_concept_gpt_front.png')
PAINT_MODEL = 'openai/gpt-5.4-image-2'
DEPTHS = ('front', 'back')

LEGEND_PROMPT = """You are preparing concept art for pixel-art reduction. Split this creature into its PARTS: the regions a
pixel artist keeps distinct when drawing it as a tiny 16-32 px sprite. At that size REGIONS matter far more than
colouring: each part becomes one solid colour, and the boundaries between parts are what carry the sprite.

Return JSON only: {"parts": [ {"name", "role", "priority", "color", "min_px", "material", "depth", "occluded_by",
"where"} ... ]}

Rules for choosing parts:
1. OCCLUSION MAKES A SUBCLASS. When one instance of a repeated part hides another instance of the same kind (a
   tentacle behind a tentacle, a leg behind a leg), the hidden instances are their OWN part, named <part>_back with
   depth "back". Do this for every kind of part that has hidden instances. Visible front instances have depth
   "front". depth "back" is ONLY for hidden instances of a repeated kind (there must also be front instances of
   that kind). A part that is merely behind a DIFFERENT part (fins behind a head, a visor behind eyes) keeps depth
   "front" and lists what covers it in occluded_by.
   Decide who is in front from the outlines: at an overlap, the part whose outline runs unbroken across the overlap
   is in front; the part whose outline disappears under it is behind.
2. If front instances of the same kind touch or overlap each other (e.g. an inner arm overlapping the base of an
   outer arm), split them into separate parts (e.g. arms_front_inner / arms_front_outer) so every touching pair has
   a different region colour. Give them the same "material" so they are drawn in the same final colour.
3. Suction cups (and similar discs on a limb) are their own part, separate from the limb they sit on. Cups on
   hidden/back instances belong to that back part.
4. NO pattern-only parts: spots, stripes, two-tone colouring, highlights and decals are paint on the part they sit
   on, not parts.
5. Small mechanical connectors, collars and joints are not separate parts: fold them into one dark "undercarriage"
   part (the dark mechanical base the limbs attach to) or into the part they belong to. Distinct character features
   (side pods, a nozzle or jet, a horn) stay their own parts. Small lights, glows and lenses belong to the part they
   sit on (a lens on a pod is the pod; a glow on a joint collar is the undercarriage).
6. Eyes, a visor or window, and other face features stay their own parts. 6 to 14 parts in total.
7. Every visible pixel of the creature belongs to exactly one part (no background part; ignore any ground shadow).

Fields:
- name: short snake_case.
- role: body | feature | limb | detail.
- priority: integer 1-10; higher wins a contested pixel when shrinking (small important features high, big masses
  low; a front part higher than the back part it covers).
- color: "#RRGGBB" flat REGION-MAP colour: highly saturated, far apart from every other part's colour, far from
  white and light grey (the background is white). Do NOT use the part's real colour.
- min_px: smallest size in pixels this part must keep at a 32 px sprite height to stay readable.
- material: the part's real-world surface (e.g. "shell", "arm", "cup", "glass", "metal_dark"); parts sharing a
  material are drawn in the same final colour.
- depth: "front" or "back".
- occluded_by: list of part names that cover parts of this one ([] if none).
- where: one precise sentence locating the part in this image, enough to paint it (say which instances: left/right,
  inner/outer, how many)."""

PAINT_PROMPT = """Repaint this image as a flat COLOUR-CODED REGION MAP for a segmentation mask.

Keep the exact same creature, pose, size, position and framing: every part must stay exactly where it is, pixel for
pixel. Change only the colours: fill each part completely with its one flat colour from this table:

{table}

OCCLUSION LAYERING (important): paint the parts in depth order, back to front:
{layers}
- A part painted later covers the earlier ones where they overlap. Every FRONT part keeps its complete outline,
  including the curve of each arm where it passes in front of another arm.
- A BACK part shows only where nothing in front covers it; never extend it over a front part.
- Where two front parts of the same material touch (an inner arm over an outer arm), the boundary between their two
  colours must follow the edge of the part in front.

Rules:
- Each part is ONE solid flat colour, exactly the hex value given. No shading, no gradients, no highlights, no
  texture, no spots or stripes (paint pale spots and stripes with the colour of the part they sit on).
- Each suction cup is a solid disc of the cup colour on its arm.
- No outlines, no black lines, no borders between parts: colour regions touch directly.
- Background (everything that is not the creature, including any ground shadow): pure white #FFFFFF.
- No text, no labels, no legend, nothing added. Do not move, resize, redraw or restyle anything; only recolour.
  Nothing inside the creature's outline may stay white."""


def legend(concept_rgb, ledger, model=regions.LEGEND_MODEL, seed=0) -> dict:
    """Vision model -> legend v2 (v1 fields + material, depth, occluded_by)."""
    png = imaging.to_png(concept_rgb)
    body = {'model': model, 'response_format': {'type': 'json_object'}, 'seed': seed,
            'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': LEGEND_PROMPT},
                                                      {'type': 'image_url', 'image_url': {'url': models.data_uri(png)}}]}]}
    ck = store.key({'step': 'regions_v2.legend', 'model': model, 'prompt': LEGEND_PROMPT, 'image': store.sha(png), 'seed': seed})
    resp, _, cost, cached = models.call(body, ledger, ck)
    out = regions._parse(resp['choices'][0]['message'].get('content'))
    parts = out.get('parts') or []
    names = [p.get('name') for p in parts]
    for p in parts:
        p['role'] = p.get('role') if p.get('role') in regions.ROLES else 'detail'
        p['priority'] = int(p.get('priority', 1))
        p['min_px'] = int(p.get('min_px', 1))
        p['material'] = str(p.get('material') or p['name'])
        p['depth'] = p.get('depth') if p.get('depth') in DEPTHS else 'front'
        p['occluded_by'] = [n for n in (p.get('occluded_by') or []) if n in names]
    fixes = regions.fix_colors(parts)
    return {'version': 'v2', 'model': model, 'parts': parts, 'color_fixes': fixes, 'call': ck, 'cost': cost,
            'cached': cached, 'prompt': LEGEND_PROMPT}


def layer_order(leg) -> list[str]:
    """Back to front: back parts first, then parts that something covers, then the rest (each part after everything
    it is occluded by is impossible by definition, so: a part comes BEFORE the parts in its occluded_by)."""
    names = [p['name'] for p in leg['parts']]
    occ = {p['name']: set(p['occluded_by']) for p in leg['parts']}
    order, placed = [], set()
    rank = {n: (0 if p['depth'] == 'back' else 1) for n, p in zip(names, leg['parts'])}
    remaining = sorted(names, key=lambda n: rank[n])
    while remaining:     # topological: n can be placed once nothing placed after it would be covered by... i.e. once
        for n in remaining:   # every part that n covers (n in their occluded_by) is placed
            covers = {m for m in names if n in occ[m]}
            if covers <= placed | {n}:
                order.append(n); placed.add(n); remaining.remove(n)
                break
        else:                 # a cycle: fall back to depth rank
            order += remaining
            break
    return order


def paint_prompt(leg) -> str:
    by = {p['name']: p for p in leg['parts']}
    table = '\n'.join(f"- {p['color']}  {p['name']} ({p['depth']}): {p.get('where', '')}" for p in leg['parts'])
    layers = '\n'.join(f"  {k + 1}. {n} ({by[n]['color']})" + (f" - behind {', '.join(by[n]['occluded_by'])}"
                                                              if by[n]['occluded_by'] else '')
                       for k, n in enumerate(layer_order(leg)))
    return PAINT_PROMPT.format(table=table, layers=layers)


def paint(concept_rgb, leg, seed, ledger, model=PAINT_MODEL) -> dict:
    """Same call shape as regions.paint (v1), with the v2 prompt."""
    png = imaging.to_png(concept_rgb)
    prompt = paint_prompt(leg)
    body = {'model': model, 'modalities': ['image', 'text'], 'seed': seed,
            'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': prompt},
                                                      {'type': 'image_url', 'image_url': {'url': models.data_uri(png)}}]}]}
    ck = store.key({'step': 'regions_v2.paint', 'model': model, 'prompt': prompt, 'images': [store.sha(png)], 'seed': seed})
    try:
        resp, imgs, cost, cached = models.call(body, ledger, ck)
    except models.BudgetExceeded:
        raise
    except Exception as e:
        return {'png': None, 'call': ck, 'cost': 0.0, 'cached': False, 'error': str(e)[:300], 'prompt': prompt}
    return {'png': imgs[0] if imgs else None, 'call': ck, 'cost': cost, 'cached': cached, 'prompt': prompt,
            'text': (resp['choices'][0]['message'].get('content') or '')[:300],
            'error': None if imgs else 'no image returned'}


def occlusion_check(labels, leg) -> dict:
    """Deterministic read of the layering: per part, components (instances) and area; for each back part, the share
    of its boundary that touches a front part of the same material family (it should sit behind them)."""
    out = {}
    from scipy import ndimage
    for i, p in enumerate(leg['parts']):
        m = labels == i + 1
        lab, k = ndimage.label(m)
        sizes = sorted(ndimage.sum_labels(np.ones_like(lab), lab, range(1, k + 1)).tolist(), reverse=True) if k else []
        big = [int(s) for s in sizes if s >= 0.02 * max(1, m.sum())]
        ring = ndimage.binary_dilation(m, iterations=2) & ~m
        nb = labels[ring]
        nb = nb[nb > 0]
        touch = {}
        for v, c in zip(*np.unique(nb, return_counts=True)):
            touch[leg['parts'][v - 1]['name']] = round(float(c / max(1, len(nb))), 3)
        out[p['name']] = {'area': int(m.sum()), 'instances': len(big), 'instance_areas': big, 'touches': touch}
    return out


def clean_islands(labels, leg, minor=0.15, passes=3):
    """Fold stray pieces into their surroundings (rule: small connectors and lights belong to the part they sit on).
    A component of a part is minor if it is < `minor` x that part's largest component; minor components merge into
    their dominant non-background neighbour. Parts with role 'detail' (cups) are exempt: they are many small pieces
    by design. Equal-sized instances (two eyes, two arms) are never minor. -> (labels, log)."""
    from scipy import ndimage
    lab = labels.copy()
    log = []
    for _ in range(passes):
        changed = False
        for i, p in enumerate(leg['parts']):
            if p['role'] == 'detail':
                continue
            cc, k = ndimage.label(lab == i + 1)
            if k < 2:
                continue
            areas = ndimage.sum_labels(np.ones_like(cc), cc, range(1, k + 1))
            for c in range(1, k + 1):
                if areas[c - 1] >= minor * areas.max():
                    continue
                m = cc == c
                ring = ndimage.binary_dilation(m, iterations=2) & ~m
                nb = lab[ring]
                nb = nb[(nb > 0) & (nb != i + 1)]
                if not len(nb):
                    continue
                v = np.bincount(nb).argmax()
                lab[m] = v
                log.append({'part': p['name'], 'px': int(areas[c - 1]), 'to': leg['parts'][v - 1]['name']})
                changed = True
        if not changed:
            break
    return lab, log


def _candidate(d, seed, concept, concept_rgb, sil, leg, ledger) -> dict:
    cid = f'gpt-s{seed}'
    rec = {'id': cid, 'model': PAINT_MODEL, 'seed': seed}
    p = paint(concept_rgb, leg, seed, ledger)
    rec.update(call=p['call'], cost=round(p['cost'], 4), cached=p['cached'], error=p['error'], model_text=p.get('text'))
    if p['png'] is None:
        rec['align'] = {'pass': False, 'iou': 0.0}
        return rec
    cdir = os.path.join(d, cid)
    os.makedirs(cdir, exist_ok=True)
    painted = imaging.from_png(p['png'])
    painted.save(os.path.join(cdir, 'painted.png'))
    labels = regions.snap(painted, leg, concept.size)
    rec['iou_raw'] = round(regions.iou(labels > 0, sil), 4)
    labels, rec['transform'], v = regions.register(labels, sil)
    rec['iou_registered'] = round(v, 4)
    labels, rec['holes_filled_px'] = regions.fill_holes(labels, sil)
    rec['align'] = regions.align(labels, sil, leg)
    if rec['align']['iou_ok']:
        labels = regions.clip_to_silhouette(labels, sil)
    labels, rec['islands_folded'] = clean_islands(labels, leg)
    np.save(os.path.join(cdir, 'labels.npy'), labels)
    regions.render_map(labels, leg).save(os.path.join(cdir, 'regions.png'))
    regions.overlay(concept, labels, leg).save(os.path.join(cdir, 'overlay.png'))
    rec['occlusion'] = occlusion_check(labels, leg)
    return rec


def run(budget=1.5, seeds=(1, 2), out=OUT) -> dict:
    os.makedirs(out, exist_ok=True)
    ledger = models.Ledger(budget)
    concept = Image.open(CONCEPT).convert('RGBA')
    sil, _ = regions.concept_masks(concept)
    concept_rgb = regions.concept_on_white(concept, sil)
    leg = legend(concept_rgb, ledger)
    layout.write_json(os.path.join(out, 'legend.json'), leg)
    with open(os.path.join(out, 'paint_prompt.md'), 'w') as f:
        f.write(paint_prompt(leg))
    with ThreadPoolExecutor(max_workers=len(seeds)) as ex:
        cands = list(ex.map(lambda s: _candidate(out, s, concept, concept_rgb, sil, leg, ledger), seeds))
    ok = [c for c in cands if c['align'].get('pass')]
    best = max(ok or [c for c in cands if c['align'].get('iou')], key=lambda c: c['align']['iou'], default=None)
    state = {'created': layout.now(), 'code': layout.code_version(), 'concept': layout.rel(CONCEPT),
             'legend': leg, 'layer_order': layer_order(leg), 'candidates': cands, 'best': best and best['id'],
             'cost': {'spent': round(ledger.spent, 4), 'calls': ledger.calls, 'cached': ledger.cached, 'budget': budget}}
    if best:
        cdir = os.path.join(out, best['id'])
        Image.open(os.path.join(cdir, 'regions.png')).save(os.path.join(out, 'regions.png'))
        np.save(os.path.join(out, 'labels.npy'), np.load(os.path.join(cdir, 'labels.npy')))
    layout.write_json(os.path.join(out, 'results.json'), state)
    _sheet(out, concept, state, leg)
    return state


def _sheet(d, concept, state, leg):
    rows = [('legend v2', [(Image.new('RGBA', (32, 32), imaging.rgb(p['color'].lstrip('#')) + (255,)),
                            f"{p['name'][:16]}\n{p['material'][:9]}/{p['depth']}", None) for p in leg['parts']])]
    for r in state['candidates']:
        cells = [(concept, 'concept', None)]
        for k in ('painted', 'regions', 'overlay'):
            p = os.path.join(d, r['id'], f'{k}.png')
            if os.path.exists(p):
                cells.append((Image.open(p).convert('RGBA'), k, None))
        a = r['align']
        cells[-1] = (cells[-1][0], f"{cells[-1][1]}\nIoU {a.get('iou', 0):.3f} {'PASS' if a.get('pass') else 'FAIL'}",
                     None if a.get('pass') else 'reject')
        rows.append((r['id'], cells))
    imaging.contact_sheet(rows, box=200, title='regions v2: occlusion subclasses, cups, no patterns').save(
        os.path.join(d, 'sheet.png'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget', type=float, default=1.5)
    ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2])
    a = ap.parse_args()
    s = run(a.budget, tuple(a.seeds))
    for p in s['legend']['parts']:
        print(f"  {p['name']:24} {p['role']:8} pr{p['priority']:2} {p['color']} {p['material']:12} {p['depth']:5} "
              f"occ_by={p['occluded_by']}")
    print('layers:', s['layer_order'])
    for c in s['candidates']:
        print(c['id'], c['align'].get('iou'), c['align'].get('pass'), c.get('cost'), c.get('error'),
              {k: (v['instances'], v['area']) for k, v in (c.get('occlusion') or {}).items()})
    print('best', s['best'], 'cost', s['cost'])


if __name__ == '__main__':
    main()
