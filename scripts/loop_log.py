"""Re-send the cockpit `loop-log` panel from runs/loop-results.json.

results: {"best": {"label": str, "grid": path}, "rows": [{"n", "change", "bal", "nll50", "score", "verdict"}]}
"""
import html
import json
import os
from pathlib import Path
import time
import urllib.parse
import urllib.request

RESULTS = Path.home() / "Projects/megaloman-ptv0/workspace/pixel_model/runs/loop-results.json"
COCKPIT = os.environ.get("COCKPIT_URL", "http://127.0.0.1:8765")


def fmt(v):
    return "–" if v is None else f"{v:.3f}" if isinstance(v, float) else html.escape(str(v))


def main():
    data = json.loads(RESULTS.read_text())
    cls = {"kept": "ok", "reverted": "bad", "baseline": "muted", "noise": "muted"}
    rows = "".join(
        f"<tr><td>{fmt(r['n'])}</td><td>{fmt(r['change'])}</td><td>{fmt(r.get('bal'))}</td>"
        f"<td>{fmt(r.get('nll50'))}</td><td>{fmt(r.get('score'))}</td>"
        f"<td class='{cls.get(r['verdict'].split()[0], '')}'>{fmt(r['verdict'])}</td></tr>" for r in data["rows"])
    best = data["best"]
    spend = RESULTS.parent / "modal-spend.jsonl"
    usd = sum(json.loads(l)["usd"] for l in spend.read_text().splitlines() if l.strip()) if spend.exists() else 0
    data["note"] = f"Modal spent: ${usd:.2f} / $20 (L4 time estimate ×1.15)" + (" · " + data["extra"] if data.get("extra") else "")
    src = "/api/file?path=" + urllib.parse.quote(best["grid"]) + f"&v={int(time.time())}"
    body = (f"<table class='table' style='font-size:18px'><tr><th>#</th><th>change</th><th>val bal NLL</th>"
            f"<th>50% NLL</th><th>sample score</th><th></th></tr>{rows}</table>"
            f"<div style='display:flex;gap:16px;align-items:center;margin-top:8px'>"
            f"<img class='pixel' src='{src}' style='height:300px;image-rendering:pixelated'>"
            f"<div><b>best so far:</b> {html.escape(best['label'])}<br>"
            f"<span class='muted'>rows 16/24/32 px · 4 seeds · score = role L1 + fg gap (lower better)</span>"
            f"{'<br>' + html.escape(data.get('note', '')) if data.get('note') else ''}</div></div>")
    req = urllib.request.Request(COCKPIT + "/api/panel", json.dumps(
        {"id": "loop-log", "tab": "training", "title": "Pixel model loop", "html": body, "size": "l"}).encode(),
        {"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=10).read()


if __name__ == "__main__":
    main()
