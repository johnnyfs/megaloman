"""Judge every benchmark of a kind, write per-benchmark structured results and a kind-level calibration summary."""
import os, statistics

import yaml
from PIL import Image

from . import imaging, judge, layout, models


def _load(path):
    with open(path) as f:
        return yaml.safe_load(f)


def list_benchmarks(kdir):
    bdir = os.path.join(kdir, 'benchmarks')
    out = []
    for name in sorted(os.listdir(bdir)):
        p = os.path.join(bdir, name, 'benchmark.yaml')
        if os.path.exists(p):
            b = _load(p)
            b['_dir'] = os.path.dirname(p)
            b['_brief'] = _load(os.path.join(b['_dir'], b['brief']))
            b.setdefault('type', 'animation' if 'frames' in b else 'still')
            out.append(b)
    return out


def subject_image(b):
    """A still's subject, or an animation's frames as a strip (what the judge sees)."""
    if b['type'] == 'animation':
        return imaging.strip([Image.open(os.path.join(b['_dir'], f)).convert('RGBA') for f in b['frames']])
    return Image.open(os.path.join(b['_dir'], b['subject'])).convert('RGBA')


def summary_path(kdir, eval_name, ext='json'):
    return os.path.join(kdir, 'benchmarks', f'summary@{eval_name}.{ext}')


def result_path(bdir, cfg):
    return os.path.join(bdir, 'results', f"{cfg['name']}@{cfg['_hash']}.json")


def load_result(kdir, bench_name, cfg):
    p = result_path(os.path.join(kdir, 'benchmarks', bench_name), cfg)
    return layout_load(p)


def layout_load(p):
    import json
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def run(game, cls, asset_type, eval_rel=None, samples=None, budget=None, only=None):
    kdir = layout.type_dir(game, cls, asset_type)
    ccfg = _load(os.path.join(layout.class_dir(game, cls), 'class.yaml'))
    tcfg = ccfg['asset_types'][asset_type]
    cfg = judge.load_eval(kdir, eval_rel or tcfg['default_eval'])
    n = samples or ccfg.get('benchmark_samples', 5)
    ledger = models.Ledger(budget)
    results = {}
    applies = cfg.get('applies_to', 'still')
    for b in list_benchmarks(kdir):
        if (only and b['name'] not in only) or b['type'] != applies:
            continue
        if b.get('applies_to_evals') and cfg['name'] not in b['applies_to_evals']:
            continue
        if not b.get('applies_to_evals') and b.get('mode') == 'oneshot':
            continue
        sprite = subject_image(b)
        durs = b.get('durations') or ([b['rate']] * len(b['frames']) if b.get('rate') else None)
        seq = imaging.preview_sequence('oneshot', b['sequence']) if b.get('mode') == 'oneshot' else None
        brief = dict(b['_brief'], n_frames=len(b.get('frames', [None])),
                     durations=imaging.describe_durations(durs) if durs else 'not specified',
                     timing=imaging.describe_sequence(b['mode'], b['sequence']) if b.get('sequence') else 'not specified')
        frames = ([Image.open(os.path.join(b['_dir'], f)).convert('RGBA') for f in b['frames']]
                  if b['type'] == 'animation' else None)
        res = judge.judge(sprite, brief, cfg, ledger, samples=n, frames=frames, durations=durs, seq=seq)
        if frames:   # deterministic motion measure, free: share of pixels that change per loop step
            import numpy as np
            d = np.array(imaging.diff_strip(frames))
            res['motion'] = {'changed_pixels': int((d[..., 0] == 255).sum()), 'frames': len(frames)}
        res.update({'benchmark': b['name'], 'kind': b['kind'], 'control': b.get('control'), 'type': b['type'],
                    'reference': b.get('reference'),
                    'checks': imaging.run_checks(sprite, {k: {} for k in imaging.CHECKS}), 'created': layout.now()})
        layout.write_json(result_path(b['_dir'], cfg), res)
        results[b['name']] = res
        s = res['summary']
        print(f"{b['name']:15s} mean {s['mean']:5.1f} sd {s['sd']:4.1f} [{s['min']}-{s['max']}] "
              f"fid {s['subscores_mean'].get('description_fidelity')} conv {s['convincing_rate']}  ${s['cost']:.3f}")
    summary = summarize(kdir, cfg, results)
    layout.event('benchmarks.run', game=game, cls=cls, asset_type=asset_type, eval=cfg['name'], eval_hash=cfg['_hash'], samples=n,
                 spent=round(ledger.spent, 4), cached_calls=ledger.cached, outputs=[layout.rel(summary_path(kdir, cfg['name']))])
    print(f'spent ${ledger.spent:.3f} ({ledger.calls} calls, {ledger.cached} cached)')
    return summary


def summarize(kdir, cfg, results):
    auth = {k: v for k, v in results.items() if v['kind'] == 'authentic'}
    means = [v['summary']['mean'] for v in auth.values()]
    sds = [v['summary']['sd'] for v in auth.values()]
    noise = round(statistics.mean(sds), 2) if sds else None
    base = round(statistics.mean(means), 2) if means else None
    ctrl = {}
    for k, v in results.items():
        t = results.get(v.get('reference') or 'telly', {}).get('summary')
        if v['kind'] == 'control' and t:
            s = v['summary']
            ctrl[k] = {'control': v['control'], 'mean': s['mean'], 'delta_vs_telly': round(s['mean'] - t['mean'], 2),
                       'fidelity': s['subscores_mean'].get('description_fidelity'),
                       'fidelity_delta_vs_telly': round((s['subscores_mean'].get('description_fidelity') or 0) -
                                                        (t['subscores_mean'].get('description_fidelity') or 0), 2)}
    # Pass bar for a fresh design: the authentic mean minus one noise band, less any recognition bonus measured
    # by the recolour control (a fresh design cannot be recognised).
    recog = max([0.0] + [-c['delta_vs_telly'] for c in ctrl.values() if c['control'] == 'recolor'])
    bar = round(base - noise - recog, 1) if base is not None else None
    # The verdict only counts if authentic art earns it; otherwise it is not a usable signal for this judge.
    verdict_ok = bool(auth) and min(v['summary']['convincing_rate'] for v in auth.values()) >= 0.5
    summary = {'eval': {'name': cfg['name'], 'hash': cfg['_hash'], 'model': cfg['model']}, 'created': layout.now(),
               'benchmarks': {k: {'kind': v['kind'], 'control': v.get('control'), **v['summary']} for k, v in results.items()},
               'authentic_mean': base, 'noise_sd': noise, 'recognition_bonus': round(recog, 2), 'controls': ctrl,
               'pass_bar': bar, 'verdict_usable': verdict_ok,
               'pass_rule': ('mean >= pass_bar AND convincing_rate >= 0.5 AND all gate checks pass' if verdict_ok else
                             'mean >= pass_bar AND all gate checks pass (verdict ignored: authentic benchmarks fail it)')}
    layout.write_json(summary_path(kdir, cfg['name']), summary)
    _write_md(kdir, summary)
    _write_contact(kdir, results, cfg['name'])
    return summary


def _write_md(kdir, s):
    L = [f"# Benchmarks: {os.path.basename(kdir)} (eval `{s['eval']['name']}@{s['eval']['hash']}`, judge `{s['eval']['model']}`)", '',
         '| benchmark | kind | mean | sd | range | fidelity | convincing | cost |', '|---|---|---|---|---|---|---|---|']
    for k, v in s['benchmarks'].items():
        L.append(f"| {k} | {v['control'] or v['kind']} | {v['mean']} | {v['sd']} | {v['min']}-{v['max']} | "
                 f"{v['subscores_mean'].get('description_fidelity')} | {v['convincing_rate']} | ${v['cost']:.3f} |")
    L += ['', f"- authentic mean **{s['authentic_mean']}**, noise (mean sd) **{s['noise_sd']}**, recognition bonus "
          f"**{s['recognition_bonus']}**", f"- pass bar for a fresh design: **{s['pass_bar']}** ({s['pass_rule']})"]
    for k, c in s['controls'].items():
        L.append(f"- control `{k}` ({c['control']}): score {c['delta_vs_telly']:+} vs its reference, fidelity {c['fidelity_delta_vs_telly']:+}")
    with open(summary_path(kdir, s['eval']['name'], 'md'), 'w') as f:
        f.write('\n'.join(L) + '\n')


def _write_contact(kdir, results, eval_name):
    cells = []
    bmap = {b['name']: b for b in list_benchmarks(kdir)}
    for k, v in results.items():
        im = subject_image(bmap[k])
        s = v['summary']
        cells.append((im, f"{k}\n{s['mean']}±{s['sd']}", None))
    imaging.contact_sheet([('benchmarks', cells)], box=160, title=f'benchmarks @{eval_name}').save(summary_path(kdir, eval_name, 'png'))
