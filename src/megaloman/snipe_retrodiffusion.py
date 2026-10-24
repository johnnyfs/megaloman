"""Snipe bot via Retro Diffusion: a size ladder 64 -> 32 -> 24 -> 16 from the compact concept.

Two routes per size, both palette-locked to an 8-colour palette taken from the concept:
  ref      rd_pro__default + the concept as a reference image + text description, seeds 1, 2
  pixelate rd_pro__pixelate with the concept as input_image (img2img-style conversion), seed 1
12 images x $0.18 = ~$2.16. Cached: re-running costs nothing.

  uv run python -m megaloman.snipe_retrodiffusion [--budget 2.5] [--quote]
--quote asks the API for the price of each call (free, no images) instead of generating.
"""
from __future__ import annotations

import argparse, io, json, sys
from pathlib import Path

from PIL import Image

from . import pipeline_manifest as pm, retrodiffusion as rd, store

COMPACT = Path(store.ROOT) / 'workspace/experiments/regions/snipe/compact'
CONCEPT = COMPACT / 'concept/chosen.png'
OUT = COMPACT / 'retrodiffusion'
SIZES = (64, 32, 24, 16)
SEEDS = (1, 2)
N_COLOURS = 8
PROMPT = ('SNIPE BOT, a small NES Mega Man style robot enemy, side view facing left: a round plump compact robot bird '
          'body, a short stubby snub-nosed gun barrel for a beak, a small scope on top of its head, one big expressive '
          'eye, a folded side wing, two short legs. Bold dark outline, few colours, chunky readable silhouette.')


def png(im: Image.Image) -> bytes:
    b = io.BytesIO()
    im.save(b, 'PNG')
    return b.getvalue()


def concept_rgb() -> Image.Image:
    im = Image.open(CONCEPT).convert('RGBA')
    bg = Image.new('RGB', im.size, 'white')
    bg.paste(im, mask=im.split()[3])
    return bg


def palette_png(im: Image.Image, n=N_COLOURS) -> tuple[bytes, list[str]]:
    q = im.resize((128, 128), Image.LANCZOS).quantize(colors=n, method=Image.Quantize.MEDIANCUT)
    pal = q.getpalette()[:n * 3]
    cols = [tuple(pal[i:i + 3]) for i in range(0, len(pal), 3)]
    strip = Image.new('RGB', (len(cols), 1))
    strip.putdata(cols)
    return png(strip), ['#%02x%02x%02x' % c for c in cols]


def jobs():
    for s in SIZES:
        for seed in SEEDS:
            yield {'id': f'ref-{s}-s{seed}', 'route': 'ref', 'size': s, 'seed': seed, 'style': 'rd_pro__default'}
        yield {'id': f'pixelate-{s}-s1', 'route': 'pixelate', 'size': s, 'seed': 1, 'style': 'rd_pro__pixelate'}


def run(budget=2.5, quote=False):
    try:
        rd.api_key()
    except rd.MissingKey as e:
        sys.exit(str(e))
    OUT.mkdir(parents=True, exist_ok=True)
    concept = concept_rgb()
    cpng = png(concept.resize((256, 256), Image.LANCZOS))
    pal_png, pal_hex = palette_png(concept)
    (OUT / 'palette.png').write_bytes(pal_png)
    led, recs = rd.Ledger(budget), []
    for j in jobs():
        kw = dict(prompt=PROMPT, style=j['style'], width=j['size'], height=j['size'], seed=j['seed'],
                  input_palette=pal_png, check_cost=quote)
        if j['route'] == 'ref':
            kw['reference_images'] = [cpng]
        else:
            kw['input_image'] = cpng
        try:
            imgs, cost, cached = rd.generate(led, **kw)
        except rd.BudgetExceeded as e:
            print('budget stop:', e)
            break
        rec = dict(j, cost=cost, cached=cached)
        if imgs:
            p = OUT / f"{j['id']}.png"
            p.write_bytes(imgs[0])
            Image.open(io.BytesIO(imgs[0])).resize((256, 256), Image.NEAREST).save(OUT / f"{j['id']}_x.png")
            rec['path'] = str(p)
        recs.append(rec)
        print(j['id'], 'quote' if quote else ('cached' if cached else 'new'), cost)
    if quote:
        print('quoted total', sum(r['cost'] or 0 for r in recs))
        return
    (OUT / 'results.json').write_text(json.dumps({'prompt': PROMPT, 'palette': pal_hex, 'spent': led.spent,
                                                  'calls': led.calls, 'cached': led.cached, 'results': recs}, indent=1))
    stages = [pm.stage('concept', '1 concept', images=[{'path': str(CONCEPT), 'caption': 'compact concept'}]),
              pm.stage('palette', '2 palette lock', kind='legend',
                       data={'items': [{'name': h, 'color': h} for h in pal_hex]})]
    for route, title in (('ref', '3 rd_pro__default + reference'), ('pixelate', '4 rd_pro__pixelate (img2img)')):
        ims = [{'path': r['path'], 'caption': f"{r['size']}x{r['size']} s{r['seed']}", 'pixel': True}
               for r in recs if r['route'] == route and r.get('path')]
        stages.append(pm.stage(f'ladder-{route}', title, kind='images', images=ims,
                               notes=[f'sizes {"/".join(map(str, SIZES))}, palette {N_COLOURS} colours']))
    pm.write(OUT, 'Snipe bot: Retro Diffusion ladder', stages, cost_usd=round(led.spent, 3),
             verdict=f'{len([r for r in recs if r.get("path")])} sprites, ${led.spent:.2f} spent')
    print('wrote', OUT / 'pipeline.json', 'spent', round(led.spent, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget', type=float, default=2.5)
    ap.add_argument('--quote', action='store_true')
    a = ap.parse_args()
    run(a.budget, a.quote)


if __name__ == '__main__':
    main()
