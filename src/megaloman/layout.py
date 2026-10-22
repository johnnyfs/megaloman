"""On-disk layout of the workspace. Every artifact a future UI needs is a file here (JSON is primary, MD/PNG derived).

workspace/
  events.jsonl                                   append-only log of every command run (who, what, outputs)
  shared/sources/                                raw downloaded reference sheets
  games/<game>/
    ideas.jsonl                                  parked ideas (e.g. new assets suggested in feedback)
    classes/<class>/                             category axis: small_enemy, big_enemy, robot_master, ...
      class.yaml
      <asset_type>/                              asset-type axis: keyframe, animation, bg_tile, bg_image, ...
        evals/<eval>.yaml   rubrics/<rubric>.md
        benchmarks/<bench>/  benchmark.yaml brief.yaml subject.png|f<i>.png  results/<eval>@<hash>.json
        benchmarks/summary@<eval>.json .md .png  calibration per eval (noise band, controls, pass bar)
        pipelines/<pipeline>/versions/v<N>/
          version.json                           parent + what changed + why (+ the feedback that prompted it)
          pipeline.yaml  prompts/*.md            the definition (frozen once runs exist)
          runs/<asset>/s<seed>/run.json          every step, every candidate, checks, scores, cost, call keys
          runs/<asset>/s<seed>/contact.png  steps/<step>/<i>.png|gif  feedback.jsonl  alignment.json
          bench/<asset>.json .md .png            distribution across seeds
    assets/<asset>/
      asset.yaml (class, theme)  brief.yaml
      tracks/<track>/blessed.json  rank<N>.png   reviewed outputs; downstream pipelines depend on these
cache/ (repo root)  content-addressed blobs and raw model-call transcripts, keyed from the JSON above
"""
import datetime, json, os

from . import store

W = os.path.join(store.ROOT, 'workspace')


def class_dir(game, cls):
    return os.path.join(W, 'games', game, 'classes', cls)


def type_dir(game, cls, asset_type):
    """Category axis (game > class) x asset-type axis (keyframe, animation, bg_tile, ...)."""
    return os.path.join(class_dir(game, cls), asset_type)


def find_pipeline(game, cls, name):
    """Pipelines live under the asset type they produce; find which one owns `name`."""
    cdir = class_dir(game, cls)
    for t in sorted(os.listdir(cdir)):
        if os.path.isdir(os.path.join(cdir, t, 'pipelines', name)):
            return t
    raise FileNotFoundError(f'no pipeline {name!r} under {rel(cdir)}')


def asset_dir(game, asset):
    return os.path.join(W, 'games', game, 'assets', asset)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1, default=str)
        f.write('\n')
    os.replace(tmp, path)


def append_jsonl(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a') as f:
        f.write(json.dumps(obj, default=str) + '\n')


def event(cmd, **kw):
    append_jsonl(os.path.join(W, 'events.jsonl'), {'ts': now(), 'author': kw.pop('author', 'coding-agent'), 'cmd': cmd, **kw})


def rel(path):
    return os.path.relpath(path, W)


def feedback(entity_dir, text, author='human', target=None, **kw):
    """Feedback lives next to the entity it is about. target narrows it (a step, a candidate, a region)."""
    rec = {'ts': now(), 'author': author, 'target': target or {'entity': rel(entity_dir)}, 'text': text, **kw}
    append_jsonl(os.path.join(entity_dir, 'feedback.jsonl'), rec)
    event('feedback', author=author, entity=rel(entity_dir), text=text)
    return rec


def code_version():
    """The code that produced a record: git commit + whether the working tree had uncommitted code changes."""
    import subprocess
    root = os.path.dirname(W)
    try:
        commit = subprocess.run(['git', '-C', root, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(['git', '-C', root, 'status', '--porcelain', '--', 'src', 'pyproject.toml'],
                                    capture_output=True, text=True).stdout.strip())
        return {'commit': commit, 'dirty': dirty}
    except Exception:
        return {'commit': None, 'dirty': None}
