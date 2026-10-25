"""Sidecar: mirror a pixel-model run dir onto a cockpit tab (stdlib only).

Polls RUN_DIR/metrics.jsonl, samples/ and result.json; re-sends three panels via
POST /api/panel: loss chart + status, latest samples, collapsible history.

usage: python scripts/cockpit_training_dashboard.py RUN_DIR [--tab training] [--every 10]
"""
import argparse
import html
import json
import os
from pathlib import Path
import time
import urllib.parse
import urllib.request

COCKPIT = os.environ.get("COCKPIT_URL", "http://127.0.0.1:8765")


def post(path, body):
    req = urllib.request.Request(COCKPIT + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read() or b"{}")


def img(path, step, px):
    q = urllib.parse.quote(str(path))
    return (f'<img class="pixel" src="/api/file?path={q}&v={step}" '
            f'style="width:{px}px;height:{px}px;image-rendering:pixelated;background:#2a2a33;border-radius:4px">')


def chart(rows, width=1400, height=330):
    train = [(r["step"], r["loss"]) for r in rows]
    val = [(r["step"], r["val_loss"]) for r in rows if "val_loss" in r]
    if not train:
        return '<div class="muted" style="font-size:28px">waiting for the first step…</div>'
    # Smooth train loss (EMA) so the trend reads from across the room.
    ema, sm = None, []
    for s, v in train:
        ema = v if ema is None else .95 * ema + .05 * v
        sm.append((s, ema))
    xs = max(s for s, _ in train)
    vals = [v for _, v in sm] + [v for _, v in val]
    lo, hi = min(vals), max(vals)
    hi, lo = hi + (hi - lo) * .05 + 1e-6, max(0, lo - (hi - lo) * .05)
    L, R, T, B = 80, 20, 16, 44
    X = lambda s: L + (width - L - R) * s / max(1, xs)
    Y = lambda v: T + (height - T - B) * (1 - (v - lo) / (hi - lo))
    out = [f'<svg viewBox="0 0 {width} {height}" style="width:100%;height:auto;font-family:inherit">']
    for i in range(5):
        v = lo + (hi - lo) * i / 4
        out.append(f'<line x1="{L}" x2="{width-R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#333" />'
                   f'<text x="{L-10}" y="{Y(v)+8:.1f}" fill="#999" font-size="22" text-anchor="end">{v:.2f}</text>')
    for i in range(6):
        s = xs * i / 5
        out.append(f'<text x="{X(s):.1f}" y="{height-10}" fill="#999" font-size="22" text-anchor="middle">{int(s)}</text>')
    pts = lambda seq: " ".join(f"{X(s):.1f},{Y(v):.1f}" for s, v in seq)
    out.append(f'<polyline points="{pts(train)}" fill="none" stroke="#5b8def" stroke-opacity=".25" stroke-width="1.5"/>')
    out.append(f'<polyline points="{pts(sm)}" fill="none" stroke="#5b8def" stroke-width="4"/>')
    if val:
        out.append(f'<polyline points="{pts(val)}" fill="none" stroke="#f0a04b" stroke-width="4"/>')
        out += [f'<circle cx="{X(s):.1f}" cy="{Y(v):.1f}" r="5" fill="#f0a04b"/>' for s, v in val]
    out.append("</svg>")
    legend = ('<div style="font-size:24px;display:flex;gap:28px;margin-top:4px">'
              '<span><b style="color:#5b8def">━</b> train loss (masked NLL, smoothed)</span>'
              '<span><b style="color:#f0a04b">━</b> val loss</span><span class="muted">x: optimizer step</span></div>')
    return "".join(out) + legend


def sample_sets(run):
    """{step: {size: dir}} from samples/step-XXXXXXXX-SIZE."""
    sets = {}
    for d in (run / "samples").glob("step-*"):
        if not (d / "pixels.json").exists():
            continue
        _, step, size = d.name.split("-")
        sets.setdefault(int(step), {})[int(size)] = d
    return sets


def row_html(dirs, step, px, gap=10):
    cells = []
    for size in sorted(dirs):
        imgs = "".join(img(dirs[size] / f"{i:04d}-large.png", step, px) for i in range(4))
        cells.append(f'<div style="display:flex;flex-direction:column;gap:4px"><div class="muted" '
                     f'style="font-size:{max(16, px // 6)}px">{size}×{size}</div>'
                     f'<div style="display:flex;gap:{gap // 2}px">{imgs}</div></div>')
    return f'<div style="display:flex;gap:{gap * 3}px;flex-wrap:wrap">{"".join(cells)}</div>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--tab", default="training")
    ap.add_argument("--every", type=float, default=10)
    ap.add_argument("--stop-note", default="")
    a = ap.parse_args()
    run = Path(a.run).resolve()
    name = run.name
    post("/api/tab", {"id": a.tab, "title": "Training", "switch": False})
    start, last = time.time(), None
    while True:
        rows = []
        if (run / "metrics.jsonl").exists():
            for line in (run / "metrics.jsonl").read_text().splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        result = json.loads((run / "result.json").read_text()) if (run / "result.json").exists() else None
        cfg = json.loads((run / "config.json").read_text()) if (run / "config.json").exists() else {}
        t = cfg.get("train", {})
        sets = sample_sets(run)
        key = (len(rows), tuple(sorted(sets)), bool(result))
        if key != last:
            last = key
            step = rows[-1]["step"] if rows else 0
            epoch = rows[-1]["effective_dataset_passes"] if rows else 0
            best = [r for r in rows if "val_loss" in r]
            bv = min(best, key=lambda r: r["val_loss"]) if best else None
            stop = (f'stop: val plateau {t.get("patience")} evals × {t.get("eval_every")} steps '
                    f'(Δ&lt;{t.get("min_delta")}) or {t.get("max_steps")} steps'
                    f' · samples every {t.get("sample_every")} steps (~2 min), val every {t.get("eval_every")}')
            if result:
                state = (f'<b class="ok">finished</b> — {html.escape(result["reason"])} at step {result["step"]}, '
                         f'{result["elapsed_seconds"] / 60:.0f} min')
            else:
                recent = rows[-50:]
                per = sum(r["seconds"] for r in recent) / max(1, len(recent))
                left = (t.get("max_steps", 0) - step) * per * 1.15
                state = f'<b class="warn">training</b> · ETA ≤ {left / 60:.0f} min (step cap)'
            best_txt = f'best val {bv["val_loss"]:.3f} @ {bv["step"]}' if bv else "best val —"
            status = (f'<div style="font-size:26px;display:flex;gap:26px;flex-wrap:wrap;margin-bottom:6px">'
                      f'<span><b>{html.escape(name)}</b></span><span>step {step}/{t.get("max_steps", "?")}</span>'
                      f'<span>epoch {epoch:.1f}</span><span>{best_txt}</span><span>{state}</span></div>'
                      f'<div class="muted" style="font-size:20px">{stop}</div>')
            post("/api/panel", {"id": "train-loss", "tab": a.tab, "title": "Loss", "size": "full",
                                "html": status + chart(rows)})
            if sets:
                s = max(sets)
                ep = next((r["effective_dataset_passes"] for r in rows if r["step"] == s), 0)
                post("/api/panel", {"id": "train-samples", "tab": a.tab, "size": "full",
                    "title": f"Samples · step {s} · epoch {ep:.1f} · 4 fixed seeds per size",
                    "html": row_html(sets[s], s, 150)})
                hist = []
                for st in sorted(sets, reverse=True):
                    ep = next((r["effective_dataset_passes"] for r in rows if r["step"] == st), 0)
                    hist.append(f'<div style="display:flex;align-items:center;gap:18px;padding:6px 0;'
                                f'border-bottom:1px solid #333"><div style="width:150px;font-size:22px">'
                                f'step {st}<br><span class="muted">ep {ep:.1f}</span></div>'
                                f'{row_html(sets[st], st, 56, 6)}</div>')
                post("/api/panel", {"id": "train-history", "tab": a.tab, "size": "full",
                    "title": f"History · {len(sets)} checkpoints (newest on top)", "collapsed": True,
                    "html": f'<div style="max-height:600px;overflow:auto">{"".join(hist)}</div>'})
        if result:
            return
        time.sleep(a.every)


if __name__ == "__main__":
    main()
