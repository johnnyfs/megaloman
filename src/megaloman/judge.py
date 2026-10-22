"""Vision judge: one eval config (model + rubric + presentation) applied N times to one sprite + brief."""
import json, os, re, statistics

import yaml

from . import imaging, models, store


def load_eval(kind_dir, rel):
    path = os.path.join(kind_dir, rel)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    rubric_path = os.path.join(kind_dir, cfg['rubric'])
    with open(rubric_path) as f:
        cfg['_rubric_text'] = f.read()
    cfg['_hash'] = store.key({k: v for k, v in cfg.items() if not k.startswith('_')} | {'rubric': cfg['_rubric_text']})[:10]
    cfg['_path'] = os.path.relpath(path, kind_dir)
    return cfg


def _fill(text, brief):
    return re.sub(r'\{(\w+)\}', lambda m: str(brief.get(m.group(1), 'not specified')).strip(), text)


def _parse(content):
    m = re.search(r'\{.*\}', content, re.S)
    return json.loads(m.group(0)) if m else {'parse_error': content[:500]}


def judge(sprite, brief, cfg, ledger, samples=None, frames=None, durations=None, seq=None):
    """Returns a structured result: per-sample records and a summary. Every sample is its own cached call.
    For animations pass `frames`; with presentation.diff the judge also sees where pixels change."""
    n = samples or cfg.get('samples', 3)
    pp = dict(cfg.get('presentation', {}))
    timeline = pp.pop('timeline', False)
    if frames and pp.pop('diff', False):
        pres = imaging.present_anim(frames, durations=durations if timeline else None, seq=seq if timeline else None, **pp)
    else:
        pp.pop('diff', None)
        pres = imaging.present(sprite, **pp)
    if timeline and not (durations or seq):
        raise ValueError('eval wants a timeline but no durations were given')
    png = imaging.to_png(pres)
    prompt = _fill(cfg['_rubric_text'], brief)
    body = {'model': cfg['model'], 'response_format': {'type': 'json_object'},
            'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': prompt},
                                                      {'type': 'image_url', 'image_url': {'url': models.data_uri(png)}}]}]}
    sprite_blob = store.put_blob(imaging.to_png(sprite))
    pres_blob = store.put_blob(png)
    def one(i):
        ck = store.key({'body': {**body, 'messages': [{'prompt': prompt, 'image': pres_blob}]}, 'sample': i})
        resp, _, cost, cached = models.call(body, ledger, ck)
        out = _parse(resp['choices'][0]['message'].get('content') or '')
        return {'sample': i, 'call': ck, 'cached': cached, 'cost': cost, **out}

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=n) as ex:   # samples are independent cached calls; order is restored below
        recs = list(ex.map(one, range(n)))
    scores = [r['score'] for r in recs if isinstance(r.get('score'), (int, float))]
    subs = {}
    for r in recs:
        for k, v in (r.get('subscores') or {}).items():
            subs.setdefault(k, []).append(v)
    summary = {
        'mean': round(statistics.mean(scores), 2) if scores else None,
        'sd': round(statistics.stdev(scores), 2) if len(scores) > 1 else 0.0,
        'min': min(scores, default=None), 'max': max(scores, default=None), 'n': len(scores),
        'subscores_mean': {k: round(statistics.mean(v), 2) for k, v in subs.items()},
        'convincing_rate': round(sum(r.get('verdict') == 'convincing' for r in recs) / max(1, len(recs)), 2),
        'cost': round(sum(r['cost'] for r in recs), 4),
    }
    return {'eval': {'name': cfg['name'], 'path': cfg['_path'], 'hash': cfg['_hash'], 'model': cfg['model']},
            'sprite_blob': sprite_blob, 'presentation_blob': pres_blob, 'samples': recs, 'summary': summary}
