"""`ml status`: a text view materialized purely from the workspace files (the first slice of the future UI)."""
import glob, json, os

import yaml

from . import layout


def _j(p):
    with open(p) as f:
        return json.load(f)


def _lines(p):
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def show(game='megaman'):
    gdir = os.path.join(layout.W, 'games', game)
    print(f'game {game}')
    for cls in sorted(os.listdir(os.path.join(gdir, 'classes'))):
        cdir = layout.class_dir(game, cls)
        print(f'  class {cls}')
        for t in sorted(d for d in os.listdir(cdir) if os.path.isdir(os.path.join(cdir, d))):
            tdir = os.path.join(cdir, t)
            print(f'    {t}')
            for name in sorted(os.listdir(os.path.join(tdir, 'evals'))):
                vroot = os.path.join(tdir, 'evals', name, 'versions')
                if not os.path.isdir(vroot):
                    continue
                for v in sorted(os.listdir(vroot), key=lambda v: int(v[1:])):
                    vj = _j(os.path.join(vroot, v, 'version.json'))
                    sp = os.path.join(tdir, 'benchmarks', f'summary@{name}@{v}.json')
                    cal = ''
                    if os.path.exists(sp):
                        s = _j(sp)
                        bs = ', '.join(f"{k} {b['mean']}" for k, b in s['benchmarks'].items())
                        cal = f"bar {s['pass_bar']} | {bs}"
                    print(f'      eval {name}@{v}: {vj["change"]["what"][:70]}')
                    if cal:
                        print(f'        {cal}')
            for name in sorted(os.listdir(os.path.join(tdir, 'pipelines'))):
                vroot = os.path.join(tdir, 'pipelines', name, 'versions')
                for v in sorted(os.listdir(vroot), key=lambda v: int(v[1:])):
                    vd = os.path.join(vroot, v)
                    vj = _j(os.path.join(vd, 'version.json'))
                    runs = sorted(glob.glob(os.path.join(vd, 'runs', '*', 's*', 'run.json')))
                    best = []
                    for r in runs:
                        rec = _j(r)
                        b = next((it for s in rec['steps'] for it in s['items'] if it['id'] == rec.get('best')), None)
                        sc = b and b['meta'].get('score')
                        best.append(f"{rec['asset']}/s{rec['seed']} {sc}")
                    fb = sum(len(_lines(os.path.join(os.path.dirname(r), 'feedback.jsonl'))) for r in runs)
                    parent = f' <- {vj["parent"]}' if vj.get('parent') else ''
                    print(f'      pipeline {name}@{v}{parent}: {vj["change"]["what"][:70]}')
                    print(f'        runs: {", ".join(best) or "none"}' + (f' | feedback {fb}' if fb else ''))
    for asset in sorted(os.listdir(os.path.join(gdir, 'assets'))):
        adir = os.path.join(gdir, 'assets', asset)
        a = yaml.safe_load(open(os.path.join(adir, 'asset.yaml')))
        print(f"  asset {asset} ({a.get('class')}, theme {a.get('theme')})")
        for track in sorted(os.listdir(os.path.join(adir, 'tracks'))) if os.path.isdir(os.path.join(adir, 'tracks')) else []:
            bp = os.path.join(adir, 'tracks', track, 'blessed.json')
            if os.path.exists(bp):
                b = _j(bp)
                items = ', '.join(f"#{i['rank']} {i['candidate']} ({i['pipeline']} s{i['seed']}, {i['size'][0]}x{i['size'][1]})" for i in b['items'])
                print(f'    track {track}: blessed {items}')
    ideas = _lines(os.path.join(gdir, 'ideas.jsonl'))
    if ideas:
        print(f'  ideas: {len(ideas)} ({ideas[-1]["text"][:60]})')
    ev = _lines(os.path.join(layout.W, 'events.jsonl'))
    print(f'events: {len(ev)}; last: {ev[-1]["cmd"]} {ev[-1]["ts"]}' if ev else 'events: 0')
    total = 0.0
    for p in glob.glob(os.path.join(os.path.dirname(layout.W), 'cache', 'calls', '*', '*.json')):
        total += _j(p).get('cost') or 0
    print(f'model spend to date: ${total:.2f}')
