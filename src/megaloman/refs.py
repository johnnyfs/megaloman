"""Build the authentic baseline sprites into workspace/refs/<name>/ (1x RGBA PNG + 6x preview).

Telly and the walking Met come from fusion-man's ripped grids; Pipi is cropped (bird only) from a ripped sheet.
"""
import json, os, subprocess

import numpy as np
from PIL import Image

from . import imaging, store

FUSION = os.path.join(store.ROOT, 'fusion-man')


def _nearest_nes(im):
    from .vendor.pixelize import to_lab
    cols = np.array([imaging.rgb(v) for v in imaging.NES.values()], float)
    lab = to_lab(cols)
    a = np.array(im.convert('RGBA'))
    op = a[..., 3] > 0
    px = a[op][:, :3].astype(float)
    a[op, :3] = cols[((to_lab(px)[:, None] - lab[None]) ** 2).sum(-1).argmin(1)].astype(np.uint8)
    return Image.fromarray(a, 'RGBA')


def _fusion_frames(ref='main', path='src/enemy_frames.json'):
    out = subprocess.run(['git', '-C', FUSION, 'show', f'{ref}:{path}'], capture_output=True, check=True, text=True).stdout
    return json.loads(out)


def build(pipi_sheet=None, outdir=None):
    outdir = outdir or os.path.join(store.ROOT, 'workspace', 'refs')
    frames = _fusion_frames()
    made = {}
    for name, sprite in [('telly', 'telly1'), ('met_walk', 'met_walk')]:
        s = frames['sprites'][sprite]
        made[name] = imaging.rows_to_image(s['rows'], frames['palettes'][s['pal']])
    if pipi_sheet:
        sheet = Image.open(pipi_sheet).convert('RGBA')
        made['pipi'] = _nearest_nes(sheet.crop((0, 0, 18, 16)))   # first bird, egg excluded
    for name, im in made.items():
        d = os.path.join(outdir, name)
        os.makedirs(d, exist_ok=True)
        im.save(os.path.join(d, f'{name}.png'))
        imaging.present(im).save(os.path.join(d, f'{name}_present.png'))
        # recolour control: same pixels, hue-rotated palette (measures how much the judge rewards recognition)
        if name == 'telly':
            a = np.array(im)
            a[..., :3] = a[..., [1, 2, 0]]
            _nearest_nes(Image.fromarray(a, 'RGBA')).save(os.path.join(d, 'telly_recolor.png'))
        print(name, im.size)
    return made
