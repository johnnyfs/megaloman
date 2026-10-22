"""Execute a declarative pipeline version for one asset + seed; bench across seeds; promote a result to an asset version.

A pipeline is a list of steps. Each step consumes the items of an earlier step (`input: $step_id`) and maps over
them; `fanout: n` multiplies a generating step by n seeds. Only model calls are cached (by request content), so
re-running is free and byte-identical, and deterministic steps are simply recomputed.
"""
import json, os, re, statistics

import yaml
from PIL import Image

from . import imaging, judge, layout, models, store


# ------------------------------------------------------------------ loading

def _load(path):
    with open(path) as f:
        return yaml.safe_load(f)


def parse_spec(spec):
    name, _, ver = spec.partition('@')
    return name, (ver or 'v1')


def pipeline_dirs(game, cls, spec):
    """-> (asset-type dir, version dir, name, version)"""
    name, ver = parse_spec(spec)
    kdir = layout.type_dir(game, cls, layout.find_pipeline(game, cls, name))
    return kdir, os.path.join(kdir, 'pipelines', name, 'versions', ver), name, ver


def definition_hash(vdir):
    parts = {}
    for root, _, files in os.walk(vdir):
        if os.path.relpath(root, vdir).split(os.sep)[0] in ('runs', 'bench'):
            continue
        for fn in sorted(files):
            if fn in ('pipeline.yaml',) or root.endswith('prompts'):
                p = os.path.join(root, fn)
                parts[os.path.relpath(p, vdir)] = store.file_hash(p)
    return store.key(parts)[:10]


def load_asset(game, asset):
    adir = layout.asset_dir(game, asset)
    a = _load(os.path.join(adir, 'asset.yaml'))
    a['_dir'] = adir
    a['_brief'] = _load(os.path.join(adir, a['brief']))
    return a


# ------------------------------------------------------------------ steps

class Ctx:
    def __init__(self, kdir, vdir, asset, seed, ledger, run_dir):
        self.kdir, self.vdir, self.asset, self.seed, self.ledger, self.run_dir = kdir, vdir, asset, seed, ledger, run_dir
        self.brief = asset['_brief']
        self.outputs = {}


def _fill(text, ctx, params):
    def sub(m):
        scope, key = m.group(1), m.group(2)
        src = ctx.brief if scope == 'brief' else params
        return str(src.get(key, m.group(0))).strip()
    return re.sub(r'\{(brief|params)\.(\w+)\}', sub, text)


def _ref_image(ctx, ref):
    """bench:<name> -> that benchmark's subject, enlarged on the neutral background (as an image-model reference)."""
    if ref.startswith('bench:'):
        im = Image.open(os.path.join(ctx.kdir, 'benchmarks', ref[6:], 'subject.png')).convert('RGBA')
        return imaging.to_png(imaging.present(im, scale=8, with_1x=False))
    with open(os.path.join(ctx.vdir, ref), 'rb') as f:
        return f.read()


def _item(step_id, idx, img, kind, parent=None, root=None, **meta):
    png = imaging.to_png(img) if isinstance(img, Image.Image) else img
    blob = store.put_blob(png)
    w, h = (img.size if isinstance(img, Image.Image) else imaging.from_png(png).size)
    return {'id': f'{step_id}.{idx}', 'parent': parent, 'root': root, 'kind': kind, 'blob': blob, 'size': [w, h],
            'status': 'ok', 'meta': meta}


def _img(item):
    return imaging.from_png(store.get_blob(item['blob']))


def step_gen_image(ctx, st, inputs):
    with open(os.path.join(ctx.vdir, st['prompt'])) as f:
        template = f.read()
    refs = [_ref_image(ctx, r) for r in st.get('images', [])]
    sources = inputs if inputs is not None else [None]
    out = []
    for parent in sources:
        params = dict(st.get('params', {}))
        lead = []
        if parent:
            w, h = parent['size']
            params.update(input_w=w, input_h=h, input_scale=st.get('input_scale', 1))
            if st.get('input_scale'):   # show the sprite enlarged on the neutral background, as a model sees art best
                lead = [imaging.to_png(imaging.present(_img(parent), scale=st['input_scale'], with_1x=False))]
            else:
                lead = [store.get_blob(parent['blob'])]
        prompt = _fill(template, ctx, params)
        from . import guidance
        prompt, used_units = guidance.apply(prompt, st.get('guidance'))
        imgs = lead + refs
        for k in range(st.get('fanout', 1)):
            seed = ctx.seed * 100 + k
            content = [{'type': 'text', 'text': prompt}] + [{'type': 'image_url', 'image_url': {'url': models.data_uri(b)}} for b in imgs]
            body = {'model': st['model'], 'modalities': ['image', 'text'], 'seed': seed,
                    'messages': [{'role': 'user', 'content': content}]}
            ck = store.key({'model': st['model'], 'prompt': prompt, 'images': [store.sha(b) for b in imgs], 'seed': seed})
            idx = len(out)
            try:
                resp, images, cost, cached = models.call(body, ctx.ledger, ck)
            except models.BudgetExceeded:
                raise
            except Exception as e:   # a failed call is a rejected candidate, not a crashed run
                out.append({'id': f"{st['id']}.{idx}", 'parent': parent and parent['id'], 'kind': 'image', 'blob': None,
                            'size': None, 'status': 'rejected', 'meta': {'reason': f'call failed: {e}'[:300], 'seed': seed}})
                continue
            if not images:
                out.append({'id': f"{st['id']}.{idx}", 'parent': parent and parent['id'], 'kind': 'image', 'blob': None,
                            'size': None, 'status': 'rejected',
                            'meta': {'reason': 'no image returned', 'seed': seed, 'call': ck, 'cost': cost}})
                continue
            out.append(_item(st['id'], idx, images[0], 'image', parent and parent['id'], parent and parent.get('root'),
                             seed=seed, call=ck, cost=cost, cached=cached, guidance=used_units,
                             text=(resp['choices'][0]['message'].get('content') or '')[:300]))
    return out


def step_gridsnap(ctx, st, inputs):
    out = []
    for i, it in enumerate(inputs):
        try:
            im, info = imaging.snap(store.get_blob(it['blob']), lo=st.get('min_pitch', 4.0), hi=st.get('max_pitch', 96.0))
            out.append(_item(st['id'], i, im, 'sprite', it['id'], it.get('root'), **info))
        except Exception as e:
            out.append({'id': f"{st['id']}.{i}", 'parent': it['id'], 'kind': 'sprite', 'blob': None, 'size': None,
                        'status': 'rejected', 'meta': {'reason': f'snap failed: {e}'[:300]}})
    return out


def step_palette_lock(ctx, st, inputs):
    out = []
    for i, it in enumerate(inputs):
        im, info = imaging.palette_lock(_img(it), st.get('max_colors', 4))
        out.append(_item(st['id'], i, im, 'sprite', it['id'], it.get('root'), **info))
    return out


def step_checks(ctx, st, inputs):
    out = []
    for i, it in enumerate(inputs):
        res = imaging.run_checks(_img(it), st['checks'])
        failed = [k for k, v in res.items() if v['gate'] and not v['pass']]
        new = dict(it, id=f"{st['id']}.{i}", parent=it['id'], meta={'checks': res})
        if failed:
            new['status'], new['meta']['reason'] = 'rejected', 'failed gates: ' + ', '.join(failed)
        out.append(new)
    return out


def step_judge(ctx, st, inputs):
    cfg = judge.load_eval(ctx.kdir, st['eval'])
    out = []
    for i, it in enumerate(inputs):
        durs, m = it['meta'].get('durations'), it['meta']
        seq = imaging.preview_sequence(m['mode'], m['sequence']) if m.get('sequence') and m.get('mode') == 'oneshot' else None
        state = (ctx.brief.get('states') or {}).get(m.get('state') or '', {})
        brief = dict(ctx.brief, n_frames=len(m.get('frames', [None])), state=m.get('state') or 'not specified',
                     motion=state.get('motion', ctx.brief.get('motion')),
                     timing=imaging.describe_sequence(m['mode'], m['sequence']) if m.get('sequence') else 'not specified',
                     durations=imaging.describe_durations(durs) if durs else 'not specified')
        frames = [imaging.from_png(store.get_blob(b)) for b in it['meta']['frames']] if it['kind'] == 'anim' else None
        res = judge.judge(_img(it), brief, cfg, ctx.ledger, samples=st.get('samples'), frames=frames, durations=durs, seq=seq)
        out.append(dict(it, id=f"{st['id']}.{i}", parent=it['id'], meta={**it['meta'], 'judge': res}))
    return out


def step_select(ctx, st, inputs):
    key = st.get('by', 'judge.mean')
    def score(it):
        if key == 'judge.mean':
            return it['meta']['judge']['summary']['mean'] or -1
        raise ValueError(f'unknown select key {key}')
    groups = {}
    for it in inputs:
        groups.setdefault(it.get('root') if st.get('group_by') == 'root' else None, []).append(it)
    out = []
    for g, items in groups.items():
        for r, it in enumerate(sorted(items, key=score, reverse=True)[:st.get('top', 1)]):
            out.append(dict(it, id=f"{st['id']}.{len(out)}", parent=it['id'],
                            meta={**it['meta'], 'rank': r + 1, 'score': score(it), 'group': g}))
    return out


def step_source_blessed(ctx, st, inputs):
    """The top-N blessed items of an asset track (the dependency on an upstream pipeline's reviewed output)."""
    path = os.path.join(ctx.asset['_dir'], 'tracks', st.get('track', 'primary'), 'blessed.json')
    with open(path) as f:
        blessed = json.load(f)
    out = []
    for b in sorted(blessed['items'], key=lambda x: x['rank'])[:st.get('top', 1)]:
        png = store.get_blob(b['blob'])
        it = _item(st['id'], len(out), png, 'sprite', None, b['blob'], blessed_rank=b['rank'],
                   upstream={'run': b['run'], 'candidate': b['candidate'], 'blob': b['blob']})
        out.append(it)
    ctx.upstream = [{'track': st.get('track', 'primary'), 'blessed_file': layout.rel(path),
                     'items': [{'rank': b['rank'], 'blob': b['blob'], 'run': b['run'], 'candidate': b['candidate']}
                               for b in blessed['items']][:st.get('top', 1)]}]
    return out


def step_align(ctx, st, inputs):
    """Put a derived frame onto its root frame's canvas: map to the root's exact colours, then find the (dx, dy)
    that maximises matching pixels (fusion-man's transplant trick)."""
    import numpy as np
    out = []
    for i, it in enumerate(inputs):
        src = np.array(imaging.from_png(store.get_blob(it['root'])))
        new = np.array(imaging.recolor_to(_img(it), imaging.colors_of(imaging.from_png(store.get_blob(it['root'])))))
        H, W = src.shape[:2]
        h, w = new.shape[:2]
        best = None
        r = st.get('max_shift', 8)
        for dy in range(-r - max(0, h - H), r + 1):
            for dx in range(-r - max(0, w - W), r + 1):
                canvas = np.zeros_like(src)
                ys, xs = max(0, dy), max(0, dx)
                ye, xe = min(H, dy + h), min(W, dx + w)
                if ye <= ys or xe <= xs:
                    continue
                canvas[ys:ye, xs:xe] = new[ys - dy:ye - dy, xs - dx:xe - dx]
                a, b = src[..., 3] > 0, canvas[..., 3] > 0
                same = (a & b & (src[..., :3] == canvas[..., :3]).all(-1)).sum()
                score = same - 0.5 * (a ^ b).sum()
                if best is None or score > best[0]:
                    kept = (canvas[..., 3] > 0).sum()
                    best = (score, dx, dy, canvas, kept)
        _, dx, dy, canvas, kept = best
        total = (new[..., 3] > 0).sum()
        a, b = src[..., 3] > 0, canvas[..., 3] > 0
        changed = ((a != b) | (a & b & (src[..., :3] != canvas[..., :3]).any(-1))).sum() / max(1, (a | b).sum())
        out.append(_item(st['id'], i, Image.fromarray(canvas, 'RGBA'), 'sprite', it['id'], it.get('root'),
                         offset=[int(dx), int(dy)], clipped=round(1 - kept / max(1, total), 3),
                         changed=round(float(changed), 3), source_size=[w, h]))
    return out


def step_anim_checks(ctx, st, inputs):
    """Gates for a derived frame vs its root: it must change something, but not be a redraw; nothing clipped."""
    out = []
    lo, hi = st.get('changed', {}).get('min', 0.03), st.get('changed', {}).get('max', 0.5)
    for i, it in enumerate(inputs):
        res = imaging.run_checks(_img(it), st.get('checks', {}))
        m = it['meta']
        res['changed'] = {'pass': lo <= m['changed'] <= hi, 'gate': True, 'detail': f"{m['changed']:.0%} of pixels changed ({lo:.0%}-{hi:.0%})"}
        res['clipped'] = {'pass': m['clipped'] <= st.get('max_clipped', 0.05), 'gate': True,
                          'detail': f"{m['clipped']:.0%} of the new frame fell outside the canvas"}
        failed = [k for k, v in res.items() if v['gate'] and not v['pass']]
        facing = 'right' if 'facing right' in str(ctx.brief.get('deliver', '')) else 'left'
        measures = imaging.motion_measures(imaging.from_png(store.get_blob(it['root'])), _img(it), facing)
        new = dict(it, id=f"{st['id']}.{i}", parent=it['id'], meta={**m, 'checks': res, 'measures': measures})
        if failed:
            new['status'], new['meta']['reason'] = 'rejected', 'failed gates: ' + ', '.join(failed)
        out.append(new)
    return out


def step_timing_design(ctx, st, inputs):
    """A model decides per-frame durations (game frames at 60 fps) for each candidate animation, with a rationale."""
    with open(os.path.join(ctx.vdir, st['prompt'])) as f:
        template = f.read()
    out = []
    for i, it in enumerate(inputs):
        frames = [imaging.from_png(store.get_blob(it['root'])), _img(it)]
        pres = imaging.to_png(imaging.present(imaging.strip(frames), scale=st.get('input_scale', 8)))
        state = (ctx.brief.get('states') or {}).get(st.get('state'), {})
        prompt = _fill(template, ctx, {'n_frames': len(frames), 'state': st.get('state', ''),
                                       'state_motion': state.get('motion', ''), 'mode': state.get('animation', 'loop')})
        body = {'model': st['model'], 'response_format': {'type': 'json_object'},
                'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': prompt},
                                                          {'type': 'image_url', 'image_url': {'url': models.data_uri(pres)}}]}]}
        ck = store.key({'model': st['model'], 'prompt': prompt, 'image': store.sha(pres), 'seed': ctx.seed})
        resp, _, cost, cached = models.call(body, ctx.ledger, ck)
        import json as _json, re as _re
        txt = resp['choices'][0]['message'].get('content') or ''
        m = _re.search(r'\{.*\}', txt, _re.S)
        t = _json.loads(m.group(0)) if m else {}
        mode = state.get('animation', 'loop')
        if 'sequence' in t:
            seq = [[int(f), max(1, min(120, int(d)))] for f, d in (
                (x['frame'], x['duration']) if isinstance(x, dict) else x for x in t['sequence'])]
            ok = bool(seq) and all(1 <= f <= len(frames) for f, _ in seq)
        else:
            seq = [[k + 1, max(1, min(120, int(d)))] for k, d in enumerate(t.get('durations', []))]
            ok = len(seq) == len(frames)
        new = dict(it, id=f"{st['id']}.{i}", parent=it['id'],
                   meta={**it['meta'], 'timing': {'mode': mode, 'state': st.get('state'), 'sequence': seq,
                                                  'durations': [d for _, d in seq] if mode == 'loop' else None,
                                                  'rationale': t.get('rationale'), 'call': ck, 'cost': cost,
                                                  'model': st['model']}})
        if not ok:
            new['status'], new['meta']['reason'] = 'rejected', f'timing: bad sequence {t}'
        out.append(new)
    return out


def step_assemble(ctx, st, inputs):
    """[root frame, derived frame] -> an animation item: the strip is the item's image, frames listed in meta.
    Timing is per frame, in game frames at 60 fps: `durations: [glide, stroke]`, or `timings: {name: [...]}` to fan
    out over several (deterministic, free: the judge sees the same strip, so its calls are shared)."""
    out = []
    for it in inputs:
        tm = it['meta'].get('timing') if isinstance(it['meta'].get('timing'), dict) else None
        designed = tm and (tm.get('durations') or [d for _, d in tm['sequence']])
        timings = ({'designed': designed} if designed else
                   st.get('timings') or {'default': st.get('durations') or [st.get('rate', 8)] * 2})
        frames = [it['root'], it['blob']]
        strip = imaging.strip([imaging.from_png(store.get_blob(b)) for b in frames])
        for tname, durs in timings.items():
            extra = ({'timing_rationale': tm.get('rationale'), 'mode': tm.get('mode', 'loop'), 'state': tm.get('state'),
                      'sequence': tm.get('sequence')} if designed else {})
            out.append(_item(st['id'], len(out), strip, 'anim', it['id'], it.get('root'), frames=frames,
                             durations=list(durs), timing=tname, **extra))
    return out


STEPS = {'gen.image': step_gen_image, 'gridsnap': step_gridsnap, 'palette_lock': step_palette_lock,
         'checks': step_checks, 'judge': step_judge, 'select': step_select,
         'source.blessed': step_source_blessed, 'align': step_align, 'anim_checks': step_anim_checks,
         'assemble': step_assemble, 'timing.design': step_timing_design}


# ------------------------------------------------------------------ run

def run(spec, asset_name, seed, game='megaman', budget=2.0, ledger=None, quiet=False):
    asset = load_asset(game, asset_name)
    kdir, vdir, pname, ver = pipeline_dirs(game, asset['class'], spec)
    pipe = _load(os.path.join(vdir, 'pipeline.yaml'))
    ledger = ledger or models.Ledger(budget)
    run_dir = os.path.join(vdir, 'runs', asset_name, f's{seed}')
    ctx = Ctx(kdir, vdir, asset, seed, ledger, run_dir)
    spent0, calls0, cached0 = ledger.spent, ledger.calls, ledger.cached
    record = {'pipeline': {'name': pname, 'version': ver, 'hash': definition_hash(vdir)}, 'asset': asset_name,
              'game': game, 'class': asset['class'], 'asset_type': os.path.basename(kdir), 'seed': seed,
              'code': layout.code_version(), 'created': layout.now(), 'steps': [], 'status': 'ok'}
    try:
        for st in pipe['steps']:
            src = st.get('input')
            inputs = None
            if src:
                inputs = [it for it in ctx.outputs[src.lstrip('$')] if it['status'] == 'ok']
            items = STEPS[st['step']](ctx, st, inputs)
            ctx.outputs[st['id']] = items
            params = {k: v for k, v in st.items() if k not in ('id', 'step', 'input')}
            record['steps'].append({'id': st['id'], 'step': st['step'], 'input': src, 'params': params, 'items': items})
            if not quiet:
                ok = sum(i['status'] == 'ok' for i in items)
                print(f"  [{st['id']}] {st['step']}: {ok}/{len(items)} ok  (spent ${ledger.spent - spent0:.3f})")
    except models.BudgetExceeded as e:
        record['status'], record['error'] = 'budget_exceeded', str(e)
    record['depends_on'] = pipe.get('depends_on')
    record['upstream'] = getattr(ctx, 'upstream', None)
    final = ctx.outputs.get(pipe.get('output', '$best').lstrip('$')) or []
    record['best'] = final[0]['id'] if final else None
    record['cost'] = {'spent': round(ledger.spent - spent0, 4), 'calls': ledger.calls - calls0, 'cached': ledger.cached - cached0}
    _materialize(record, run_dir, kdir)
    layout.write_json(os.path.join(run_dir, 'run.json'), record)
    layout.event('pipeline.run', pipeline=f'{pname}@{ver}', asset=asset_name, seed=seed, status=record['status'],
                 best=record['best'], spent=record['cost']['spent'], outputs=[layout.rel(os.path.join(run_dir, 'run.json'))])
    if not quiet:
        b = _find(record, record['best'])
        sc = b and b['meta'].get('score')
        print(f"  best {record['best']} score {sc}  spent ${record['cost']['spent']:.3f} "
              f"({record['cost']['calls']} calls, {record['cost']['cached']} cached)  -> {layout.rel(run_dir)}")
    return record


def _find(record, item_id):
    for s in record['steps']:
        for it in s['items']:
            if it['id'] == item_id:
                return it
    return None


def _materialize(record, run_dir, kdir):
    """Copy every step output to steps/<id>/<n>.png and draw the contact sheet."""
    rows = []
    best_chain = set()
    it = _find(record, record.get('best'))
    while it:
        best_chain.add(it['id'])
        it = _find(record, it['parent']) if it['parent'] else None
    for s in record['steps']:
        cells = []
        for it in s['items']:
            if it['blob']:
                p = os.path.join(run_dir, 'steps', s['id'], it['id'].split('.')[-1] + '.png')
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, 'wb') as f:
                    f.write(store.get_blob(it['blob']))
                it['file'] = os.path.relpath(p, run_dir)
                im = _img(it)
                if it['kind'] == 'anim':
                    g = p[:-4] + '.gif'
                    fr = [imaging.from_png(store.get_blob(b)) for b in it['meta']['frames']]
                    if it['meta'].get('mode') == 'oneshot':
                        imaging.save_preview(fr, g, 'oneshot', it['meta']['sequence'])
                    else:
                        imaging.save_gif(fr, g, it['meta'].get('durations') or it['meta'].get('rate', 8))
                    it['gif'] = os.path.relpath(g, run_dir)
            else:
                im = Image.new('RGBA', (16, 16), (0, 0, 0, 0))
            cap = it['id'] + (f" {it['meta']['timing']}" if it['meta'].get('timing') else '')
            j = it['meta'].get('judge')
            if j:
                cap += f"\n{j['summary']['mean']}±{j['summary']['sd']}"
            elif it.get('size'):
                cap += f"\n{it['size'][0]}x{it['size'][1]}"
            mark = 'reject' if it['status'] == 'rejected' else ('best' if it['id'] in best_chain else None)
            cells.append((im, cap, mark))
        if s['step'] not in ('select',):
            rows.append((f"{s['id']} ({s['step']})", cells))
    bench_cells = []
    ev = next((s['params']['eval'] for s in record['steps'] if s['step'] == 'judge'), None)
    bs = (_bench_summary(kdir, ev) if ev else None) or {'benchmarks': {}}
    from .benchmarks import list_benchmarks, subject_image
    bmap = {b['name']: b for b in list_benchmarks(kdir)}
    for name, v in bs['benchmarks'].items():
        if v.get('kind') == 'authentic':
            im = subject_image(bmap[name])
            bench_cells.append((im, f"{name}\n{v['mean']}±{v['sd']}", None))
    if bench_cells:
        rows.append((f"benchmarks (bar {bs.get('pass_bar')})", bench_cells))
    title = f"{record['pipeline']['name']}@{record['pipeline']['version']}  {record['asset']} seed {record['seed']}  ${record['cost']['spent']:.3f}"
    imaging.contact_sheet(rows, box=128, title=title).save(os.path.join(run_dir, 'contact.png'))


# ------------------------------------------------------------------ bench + promote

def _bench_summary(kdir, eval_ref):
    """Calibration summary for an eval ref (any form judge.resolve_ref accepts)."""
    p = os.path.join(kdir, 'benchmarks', f"summary@{judge.load_eval(kdir, eval_ref)['_ref']}.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def bench(spec, asset_name, seeds=4, first_seed=1, game='megaman', budget=6.0):
    asset = load_asset(game, asset_name)
    kdir, vdir, pname, ver = pipeline_dirs(game, asset['class'], spec)
    pipe = _load(os.path.join(vdir, 'pipeline.yaml'))
    ev = next((s['eval'] for s in pipe['steps'] if s['step'] == 'judge'), None)
    bs = _bench_summary(kdir, ev) if ev else {}
    bar = bs.get('pass_bar')
    ledger = models.Ledger(budget)
    per_seed, judged = [], []
    for seed in range(first_seed, first_seed + seeds):
        print(f'seed {seed}')
        rec = run(spec, asset_name, seed, game, ledger=ledger)
        for s in rec['steps']:
            if s['step'] == 'judge':
                for it in s['items']:
                    sm = it['meta']['judge']['summary']
                    judged.append({'seed': seed, 'id': it['id'], 'mean': sm['mean'], 'sd': sm['sd'],
                                   'convincing_rate': sm['convincing_rate'], 'blob': it['blob']})
        gen = [it for s in rec['steps'] if s['step'] == 'gen.image' for it in s['items']]
        gated = [it for s in rec['steps'] if s['step'] == 'checks' for it in s['items']]
        b = _find(rec, rec['best'])
        bj = _find(rec, b['parent']) if b else None
        bsum = bj['meta']['judge']['summary'] if bj else None
        per_seed.append({'seed': seed, 'run': layout.rel(os.path.join(vdir, 'runs', asset_name, f's{seed}', 'run.json')),
                         'status': rec['status'], 'generated': len(gen),
                         'gate_pass': sum(i['status'] == 'ok' for i in gated), 'gated': len(gated),
                         'best': rec['best'], 'best_mean': bsum and bsum['mean'], 'best_convincing_rate': bsum and bsum['convincing_rate'],
                         'passes_bar': bool(bsum and bar is not None and bsum['mean'] >= bar and
                                            (bsum['convincing_rate'] >= 0.5 or not bs.get('verdict_usable', True))),
                         'cost': rec['cost']['spent']})
        if rec['status'] == 'budget_exceeded':
            print('  budget exceeded; stopping bench')
            break
    means = [j['mean'] for j in judged if j['mean'] is not None]
    bests = [p['best_mean'] for p in per_seed if p['best_mean'] is not None]
    top = max(judged, key=lambda j: j['mean'] or -1) if judged else None
    out = {'pipeline': {'name': pname, 'version': ver, 'hash': definition_hash(vdir)}, 'asset': asset_name,
           'created': layout.now(), 'code': layout.code_version(), 'benchmark_ref': {'pass_bar': bar, 'authentic_mean': bs.get('authentic_mean'),
                                                     'noise_sd': bs.get('noise_sd'), 'eval': bs.get('eval'),
                                                     'verdict_usable': bs.get('verdict_usable')},
           'seeds': per_seed,
           'candidates': {'judged': len(means), 'mean': round(statistics.mean(means), 2) if means else None,
                          'sd': round(statistics.stdev(means), 2) if len(means) > 1 else None},
           'best_per_seed': {'mean': round(statistics.mean(bests), 2) if bests else None, 'min': min(bests, default=None),
                             'max': max(bests, default=None)},
           'pass_rate': round(sum(p['passes_bar'] for p in per_seed) / max(1, len(per_seed)), 2),
           'gate_pass_rate': round(sum(p['gate_pass'] for p in per_seed) / max(1, sum(p['gated'] for p in per_seed)), 2),
           'top_candidate': top, 'cost': round(ledger.spent, 4),
           'cost_per_passing_seed': round(ledger.spent / max(1, sum(p['passes_bar'] for p in per_seed)), 3)}
    bdir = os.path.join(vdir, 'bench')
    layout.write_json(os.path.join(bdir, f'{asset_name}.json'), out)
    _bench_md(out, os.path.join(bdir, f'{asset_name}.md'))
    _bench_contact(out, vdir, asset_name, kdir)
    layout.event('pipeline.bench', pipeline=f'{pname}@{ver}', asset=asset_name, seeds=len(per_seed),
                 pass_rate=out['pass_rate'], spent=out['cost'], outputs=[layout.rel(os.path.join(bdir, f'{asset_name}.json'))])
    print(json.dumps({k: out[k] for k in ('candidates', 'best_per_seed', 'pass_rate', 'gate_pass_rate', 'cost')}, indent=1))
    return out


def _bench_md(o, path):
    r = o['benchmark_ref']
    L = [f"# Bench: {o['pipeline']['name']}@{o['pipeline']['version']} on {o['asset']}", '',
         f"Benchmarks: authentic mean {r['authentic_mean']}, noise sd {r['noise_sd']}, pass bar **{r['pass_bar']}**.", '',
         '| seed | generated | gates passed | best | best mean | convincing | passes bar | cost |', '|---|---|---|---|---|---|---|---|']
    for p in o['seeds']:
        L.append(f"| {p['seed']} | {p['generated']} | {p['gate_pass']}/{p['gated']} | {p['best']} | {p['best_mean']} | "
                 f"{p['best_convincing_rate']} | {'yes' if p['passes_bar'] else 'no'} | ${p['cost']:.3f} |")
    L += ['', f"- all judged candidates: mean {o['candidates']['mean']} (sd {o['candidates']['sd']}, n {o['candidates']['judged']})",
          f"- best per seed: mean {o['best_per_seed']['mean']} [{o['best_per_seed']['min']}-{o['best_per_seed']['max']}]",
          f"- pass rate {o['pass_rate']}, gate pass rate {o['gate_pass_rate']}, cost ${o['cost']:.3f}"]
    with open(path, 'w') as f:
        f.write('\n'.join(L) + '\n')


def _bench_contact(o, vdir, asset_name, kdir):
    rows = []
    for p in o['seeds']:
        rec = json.load(open(os.path.join(layout.W, p['run'])))
        cells = []
        for s in rec['steps']:
            if s['step'] == 'judge':
                for it in s['items']:
                    sm = it['meta']['judge']['summary']
                    mark = 'best' if _find(rec, rec['best']) and _find(rec, rec['best'])['parent'] == it['id'] else None
                    cells.append((_img(it), f"{it['id']}\n{sm['mean']}±{sm['sd']}", mark))
        rows.append((f"seed {p['seed']}", cells or [(Image.new('RGBA', (16, 16)), 'none passed', 'reject')]))
    from .benchmarks import list_benchmarks, subject_image
    bmap = {b['name']: b for b in list_benchmarks(kdir)}
    bs = _bench_summary(kdir, o['benchmark_ref']['eval']['ref']) if (o['benchmark_ref'].get('eval') or {}).get('ref') else {}
    cells = [(subject_image(bmap[n]), f"{n}\n{v['mean']}±{v['sd']}", None)
             for n, v in bs.get('benchmarks', {}).items() if v.get('kind') == 'authentic']
    if cells:
        rows.append((f"benchmarks bar {bs.get('pass_bar')}", cells))
    imaging.contact_sheet(rows, title=f"{o['pipeline']['name']}@{o['pipeline']['version']} bench on {asset_name}: "
                                     f"pass rate {o['pass_rate']}  ${o['cost']:.2f}").save(os.path.join(vdir, 'bench', f'{asset_name}.png'))


def promote(asset_name, spec, game='megaman', note=''):
    """The bench's top candidate becomes the asset's next version (asset version = a reviewed, recoverable pointer)."""
    asset = load_asset(game, asset_name)
    kdir, vdir, pname, ver = pipeline_dirs(game, asset['class'], spec)
    o = json.load(open(os.path.join(vdir, 'bench', f'{asset_name}.json')))
    top = o['top_candidate']
    vroot = os.path.join(asset['_dir'], 'versions')
    n = 1 + max([int(d[1:]) for d in os.listdir(vroot)] if os.path.isdir(vroot) else [0])
    vd = os.path.join(vroot, f'v{n}')
    im = imaging.from_png(store.get_blob(top['blob']))
    os.makedirs(vd, exist_ok=True)
    im.save(os.path.join(vd, 'best.png'))
    imaging.present(im).save(os.path.join(vd, 'best_present.png'))
    bar = o['benchmark_ref']['pass_bar']
    rec = {'version': n, 'created': layout.now(), 'parent': f'v{n - 1}' if n > 1 else None,
           'source': {'pipeline': f'{pname}@{ver}', 'pipeline_hash': o['pipeline']['hash'], 'seed': top['seed'],
                      'candidate': top['id'], 'blob': top['blob'],
                      'run': layout.rel(os.path.join(vdir, 'runs', asset_name, f"s{top['seed']}", 'run.json'))},
           'score': {'mean': top['mean'], 'sd': top['sd'], 'convincing_rate': top['convincing_rate'], 'pass_bar': bar,
                     'passes_bar': bool(bar is not None and top['mean'] >= bar and
                                        (top['convincing_rate'] >= 0.5 or not o['benchmark_ref'].get('verdict_usable', True)))},
           'status': 'candidate', 'note': note}
    layout.write_json(os.path.join(vd, 'version.json'), rec)
    layout.event('asset.promote', asset=asset_name, version=n, source=rec['source'], outputs=[layout.rel(vd)])
    print(json.dumps(rec, indent=1))
    return rec


# ------------------------------------------------------------------ versions

def new_version(game, cls, asset_type, name, parent, what, why, feedback=None, author='coding-agent'):
    """Copy the parent version's definition (pipeline.yaml + prompts/) into the next v<N>, recording what/why.
    The edit itself is then made in the new version; the parent stays frozen."""
    import shutil
    try:
        asset_type = layout.find_pipeline(game, cls, name)   # an existing pipeline keeps its asset type
    except FileNotFoundError:
        pass
    kdir = layout.type_dir(game, cls, asset_type)
    root = os.path.join(kdir, 'pipelines', name, 'versions')
    n = 1 + max([int(d[1:]) for d in os.listdir(root)] if os.path.isdir(root) else [0])
    vd = os.path.join(root, f'v{n}')
    if parent:
        pd = os.path.join(root, parent)
        os.makedirs(vd)
        shutil.copy(os.path.join(pd, 'pipeline.yaml'), vd)
        if os.path.isdir(os.path.join(pd, 'prompts')):
            shutil.copytree(os.path.join(pd, 'prompts'), os.path.join(vd, 'prompts'))
    rec = {'pipeline': name, 'version': f'v{n}', 'created': layout.now(), 'author': author, 'parent': parent,
           'change': {'what': what, 'why': why, 'prompted_by': feedback}}
    layout.write_json(os.path.join(vd, 'version.json'), rec)
    layout.event('pipeline.version', author=author, pipeline=f'{name}@v{n}', parent=parent, what=what,
                 outputs=[layout.rel(vd)])
    print(layout.rel(vd))
    return vd


# ------------------------------------------------------------------ blessing

def bless(game, asset_name, track, run_rel, candidates, note='', author='human'):
    """Mark run candidates as the blessed outputs of an asset track, ranked in the given order. Downstream pipelines
    depend on a track's blessed list (source.blessed), never on a raw run."""
    asset = load_asset(game, asset_name)
    rec = json.load(open(os.path.join(layout.W, run_rel)))
    tdir = os.path.join(asset['_dir'], 'tracks', track)
    path = os.path.join(tdir, 'blessed.json')
    prev = json.load(open(path)) if os.path.exists(path) else None
    items = []
    for rank, cid in enumerate(candidates, 1):
        it = _find(rec, cid)
        im = imaging.from_png(store.get_blob(it['blob']))
        f = os.path.join(tdir, f'rank{rank}.png')
        os.makedirs(tdir, exist_ok=True)
        im.save(f)
        imaging.present(im).save(os.path.join(tdir, f'rank{rank}_present.png'))
        items.append({'rank': rank, 'candidate': cid, 'run': run_rel, 'blob': it['blob'], 'size': it['size'],
                      'file': os.path.basename(f), 'pipeline': f"{rec['pipeline']['name']}@{rec['pipeline']['version']}",
                      'seed': rec['seed']})
    out = {'asset': asset_name, 'track': track, 'updated': layout.now(), 'author': author, 'note': note,
           'previous': prev and {'updated': prev['updated'], 'items': [(i['rank'], i['blob']) for i in prev['items']]},
           'items': items}
    layout.write_json(path, out)
    layout.event('asset.bless', author=author, asset=asset_name, track=track, candidates=candidates, run=run_rel,
                 outputs=[layout.rel(path)])
    print(json.dumps(out, indent=1))
    return out
