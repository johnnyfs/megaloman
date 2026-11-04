"""nespix command line.

  nespix "a robot squid enemy" --sizes 16 24 32 -o out/squid          # text -> sprites
  nespix --image concept.png --sizes 8 16 32 64 -o out/x              # image -> sprites
  nespix "fire knight" --image ref.jpg --restyle 0.5 -o out/k         # image + text, flattened first
  nespix --image concept.png --sizes 16 --fast                        # default weights, no judge/CLIP
"""
import argparse
import os
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog='nespix', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('text', nargs='?', help='description of the sprite')
    ap.add_argument('--image', help='hi-res example / concept image')
    ap.add_argument('--sizes', type=int, nargs='+', default=[16, 32, 64])
    ap.add_argument('-o', '--out', default='nespix_out')
    ap.add_argument('--seeds', type=int, default=6, help='concepts to imagine (text only)')
    ap.add_argument('--restyle', type=float, default=None, help='img2img strength to flatten a photo/painting first')
    ap.add_argument('--colors', type=int, default=None, help='opaque colours incl. black (default: judge picks 3 or 4)')
    ap.add_argument('--palette', nargs='+', help='force NES codes, e.g. 0F 16 27 30')
    ap.add_argument('--mode', choices=['sprite', 'tile'], default='sprite')
    ap.add_argument('--fast', action='store_true', help='no variant search / CLIP judge (pure renderer)')
    a = ap.parse_args(argv)
    if not a.text and not a.image:
        ap.error('give a description and/or --image')
    for s in a.sizes:
        if not 8 <= s <= 256:
            ap.error('sizes must be within 8..256')
    kw = dict(mode=a.mode)
    if a.palette:
        kw['palette'] = tuple(a.palette)
        kw['colors'] = len(a.palette)
    from . import model
    from .render import Params
    if a.colors:
        model.AXES = {**model.AXES, 'colors': (a.colors,)}
    if a.palette:
        model.AXES = {k: v for k, v in model.AXES.items() if k != 'colors'}
    if a.fast:
        if not a.image:
            ap.error('--fast needs --image')
        from .render import Concept, render
        c = Concept(a.image, mode=a.mode)
        sprites = {n: render(c, n, Params(**kw, **({'colors': a.colors} if a.colors else {}))) for n in a.sizes}
        res = dict(concept=c, sprites=sprites, text=a.text)
        model.save(res, a.out)
    else:
        res = model.generate(text=a.text, image=a.image, sizes=a.sizes, seeds=a.seeds, restyle=a.restyle,
                             out=a.out, **kw)
    for n, sp in sorted(res['sprites'].items()):
        print(f'{n:>3}px  palette {" ".join(sp.codes)}  -> {os.path.join(a.out, f"sprite_{n}.png")}')
    print(f'sheet -> {os.path.join(a.out, "sheet.png")}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
