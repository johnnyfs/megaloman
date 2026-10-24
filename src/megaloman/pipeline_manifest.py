"""Pipeline manifest: a generic, machine-readable record of one run's steps, so any viewer (the cockpit strip, a
contact sheet) can render any run from disk. Not tied to an enemy or a pipeline.

<run_dir>/pipeline.json
{
  "format": "megaloman.pipeline/1",
  "title": "<run title>",
  "run_dir": "<abs path>",
  "verdict": "<one line, optional>",
  "cost_usd": <float, optional>,
  "stages": [                         # ordered left -> right
    {"id": "concept",                 # short stable id
     "title": "1 concept",            # label shown above the stage
     "kind": "image" | "images" | "legend" | "list" | "text",
     "images": [{"path": "<abs png>", "caption": "16x16", "pixel": true}],   # pixel: scale nearest-neighbour
     "notes": ["<short line>", ...],
     "data": {...}}                   # free-form; kind "legend": {"items": [{"name", "color"}]}, "list": {"items": [str]}
  ]
}
"""
from __future__ import annotations

import json
from pathlib import Path

FORMAT = "megaloman.pipeline/1"


def stage(id, title, kind="image", images=(), notes=(), data=None) -> dict:
    ims = [i if isinstance(i, dict) else {"path": str(Path(i).resolve())} for i in images]
    for i in ims:
        i["path"] = str(Path(i["path"]).resolve())
    return {"id": id, "title": title, "kind": kind, "images": ims, "notes": list(notes), "data": data or {}}


def write(run_dir, title, stages, verdict=None, cost_usd=None) -> Path:
    run_dir = Path(run_dir).resolve()
    doc = {"format": FORMAT, "title": title, "run_dir": str(run_dir), "verdict": verdict, "cost_usd": cost_usd,
           "stages": stages}
    p = run_dir / "pipeline.json"
    p.write_text(json.dumps(doc, indent=1))
    return p


def read(path) -> dict:
    doc = json.loads(Path(path).read_text())
    assert doc.get("format") == FORMAT, doc.get("format")
    return doc


def to_html(doc: dict, img_h: int = 200, file_url="/api/file?path=") -> str:
    """One horizontal strip (flex row, wraps on narrow screens) for the cockpit / any browser."""
    from html import escape as e
    from urllib.parse import quote
    cells = []
    for s in doc["stages"]:
        body = []
        for im in s["images"]:
            cls = ' class="pixel"' if im.get("pixel") else ""
            body.append(f'<img{cls} src="{file_url}{quote(im["path"])}" style="height:{img_h}px;width:auto;'
                        f'background:#fff;image-rendering:{"pixelated" if im.get("pixel") else "auto"}">'
                        + (f'<div class="muted">{e(im["caption"])}</div>' if im.get("caption") else ""))
        if s["kind"] == "legend":
            body.append("<div style='font-size:15px;line-height:1.25'>" + "".join(
                f'<div><span style="display:inline-block;width:12px;height:12px;background:{e(i["color"])}"></span> '
                f'{k + 1}. {e(i["name"])}</div>' for k, i in enumerate(s["data"].get("items", []))) + "</div>")
        if s["kind"] == "list":
            body.append("<ol style='margin:0;padding-left:22px;font-size:16px;line-height:1.3'>" + "".join(
                f"<li>{e(str(i))}</li>" for i in s["data"].get("items", [])) + "</ol>")
        notes = "".join(f'<div class="muted" style="font-size:14px">{e(n)}</div>' for n in s["notes"])
        cells.append(f'<div style="flex:0 0 auto;max-width:{img_h * 2}px"><div style="font-weight:700;'
                     f'border-bottom:2px solid #888;margin-bottom:4px">{e(s["title"])}</div>{"".join(body)}{notes}</div>')
    head = f'<div style="font-size:18px;margin:4px 0"><b>{e(doc["title"])}</b>' + (
        f' &mdash; {e(doc["verdict"])}' if doc.get("verdict") else "") + "</div>"
    return head + '<div style="display:flex;gap:14px;align-items:flex-start;flex-wrap:wrap">' + "".join(cells) + "</div>"
