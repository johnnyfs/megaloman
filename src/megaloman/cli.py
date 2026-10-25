"""`ml` command line."""
import argparse


def main():
    ap = argparse.ArgumentParser(prog='ml')
    sub = ap.add_subparsers(dest='cmd', required=True)
    from .pixel_model import cli as pixel_cli
    pixel_cli.register(sub)

    b = sub.add_parser('benchmarks', help='judge every benchmark of a kind; write results + calibration summary')
    b.add_argument('--game', default='megaman'); b.add_argument('--class', dest='cls', default='small_enemy')
    b.add_argument('--type', dest='asset_type', default='keyframe', help='asset type: keyframe, animation, ...')
    b.add_argument('--eval'); b.add_argument('--samples', type=int); b.add_argument('--budget', type=float, default=3.0)
    b.add_argument('--only', nargs='*')

    r = sub.add_parser('run', help='run one pipeline version for one asset and seed')
    r.add_argument('pipeline', help='<name>@v<N>, e.g. one_shot@v1')
    r.add_argument('asset'); r.add_argument('--seed', type=int, default=1)
    r.add_argument('--game', default='megaman'); r.add_argument('--budget', type=float, default=2.0)

    s = sub.add_parser('bench', help='run a pipeline version over several seeds and summarise the distribution')
    s.add_argument('pipeline'); s.add_argument('asset'); s.add_argument('--seeds', type=int, default=4)
    s.add_argument('--first-seed', type=int, default=1)
    s.add_argument('--game', default='megaman'); s.add_argument('--budget', type=float, default=6.0)

    p = sub.add_parser('promote', help='make a run candidate the next asset version')
    p.add_argument('asset'); p.add_argument('--from-bench', required=True, metavar='PIPELINE@vN')
    p.add_argument('--game', default='megaman'); p.add_argument('--note', default='')

    f = sub.add_parser('feedback', help='attach feedback to an entity directory (asset version, pipeline version, run)')
    f.add_argument('entity', help='path relative to workspace/'); f.add_argument('text')
    f.add_argument('--author', default='human'); f.add_argument('--step'); f.add_argument('--candidate')

    v = sub.add_parser('version', help='start a new pipeline version from a parent, recording what changed and why')
    v.add_argument('pipeline'); v.add_argument('--from', dest='parent'); v.add_argument('--what', required=True)
    v.add_argument('--why', required=True); v.add_argument('--feedback', help='ref to the feedback that prompted it')
    v.add_argument('--game', default='megaman'); v.add_argument('--class', dest='cls', default='small_enemy')
    v.add_argument('--type', dest='asset_type', default='keyframe', help='asset type a NEW pipeline produces')
    v.add_argument('--author', default='coding-agent')

    bl = sub.add_parser('bless', help='bless run candidates (ranked) as an asset track\'s output')
    bl.add_argument('asset'); bl.add_argument('--track', default='primary'); bl.add_argument('--run', required=True)
    bl.add_argument('--candidates', nargs='+', required=True); bl.add_argument('--note', default='')
    bl.add_argument('--author', default='human'); bl.add_argument('--game', default='megaman')

    st = sub.add_parser('status', help='summary of everything, materialized from the workspace files')
    st.add_argument('--game', default='megaman')

    a = ap.parse_args()
    if a.cmd == 'pixel':
        pixel_cli.run(a)
    elif a.cmd == 'benchmarks':
        from . import benchmarks
        benchmarks.run(a.game, a.cls, a.asset_type, a.eval, a.samples, a.budget, a.only)
    elif a.cmd == 'run':
        from . import runner
        runner.run(a.pipeline, a.asset, a.seed, a.game, a.budget)
    elif a.cmd == 'bench':
        from . import runner
        runner.bench(a.pipeline, a.asset, a.seeds, a.first_seed, a.game, a.budget)
    elif a.cmd == 'promote':
        from . import runner
        runner.promote(a.asset, a.from_bench, a.game, a.note)
    elif a.cmd == 'status':
        from . import status
        status.show(a.game)
    elif a.cmd == 'bless':
        from . import runner
        runner.bless(a.game, a.asset, a.track, a.run, a.candidates, a.note, a.author)
    elif a.cmd == 'version':
        from . import runner
        runner.new_version(a.game, a.cls, a.asset_type, a.pipeline, a.parent, a.what, a.why, a.feedback, a.author)
    elif a.cmd == 'feedback':
        import os
        from . import layout
        target = {'entity': a.entity}
        if a.step: target['step'] = a.step
        if a.candidate: target['candidate'] = a.candidate
        print(layout.feedback(os.path.join(layout.W, a.entity), a.text, a.author, target))


if __name__ == '__main__':
    main()
