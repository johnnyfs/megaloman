"""Reproducible example gallery: python -m megaloman.nespix.gallery OUT_DIR

Text route for a fixed prompt list, image route for the repo's own concepts. Writes one folder per
example (concept, sprites, previews, meta.json) and a combined `gallery.png`.
"""
import os
import sys
import time

from PIL import Image

from . import model

PROMPTS = [
    'a robot squid enemy',
    'a blue slime monster',
    'a red mushroom with white spots',
    'a wooden treasure chest',
    'a purple bat with spread wings',
    'a red heart',
    'a knight with a sword and shield',
    'a green frog',
    'a yellow star',
    'a skull',
    'a ghost',
    'a small dragon breathing fire',
]
IMAGES = [
    ('squid_concept', 'workspace/experiments/regions/m1/concept_on_white.png', 'a robot squid'),
    ('snipe_concept', 'workspace/experiments/regions/snipe/compact/boxy/concept.png', 'a round robot with a rifle'),
]
SIZES = (8, 16, 24, 32, 64, 128)


def slug(s):
    return ''.join(ch if ch.isalnum() else '_' for ch in s.lower()).strip('_')[:40]


def main(out='workspace/experiments/nespix/gallery', seeds=6):
    os.makedirs(out, exist_ok=True)
    rows = []
    jobs = [(slug(p), dict(text=p)) for p in PROMPTS] + [(n, dict(image=i, text=t)) for n, i, t in IMAGES]
    for name, kw in jobs:
        if kw.get('image') and not os.path.exists(kw['image']):
            continue
        t = time.time()
        res = model.generate(sizes=SIZES, seeds=seeds, out=os.path.join(out, name), **kw)
        print(f'{name}: {time.time() - t:.0f}s', flush=True)
        rows.append(model.sheet(res, cell=128))
    W = max(r.width for r in rows)
    sheet = Image.new('RGB', (W, sum(r.height + 4 for r in rows)), (24, 24, 28))
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height + 4
    sheet.save(os.path.join(out, 'gallery.png'))


if __name__ == '__main__':
    main(*sys.argv[1:])
