"""Concept-stage symmetry assessment (PIPELINE_RULES: "assess symmetry at the concept stage").

1. A vision model classifies the concept's symmetry (none / vertical / horizontal / both / radial n / rotational n)
   plus per-part notes when a legend is given. Prompt: workspace/experiments/regions/symmetry/prompts/symmetry.md.
   One cached call, ledger-budgeted.
2. Deterministic verification on the concept's alpha silhouette: mirror IoU about the best vertical and horizontal
   axis (half-pixel search), rotation IoU for orders 2..8 about the best centre. Every symmetry, claimed or not, is
   measured; the verified type comes from the measurement (>= VERIFY). Disagreements with the model are recorded.
   If a region map is given, each part is also measured about the verified axes.

Output spec (consumed by ladder_sym.run(spec=...)):
  {type, axes: {vertical: {axis_px, score, verified}, horizontal: {...}}, rotation: {order, centre_px, score},
   model: {...}, disagreements: [...], per_part: {...}}

uv run python -m megaloman.symmetry [--budget 0.5]
"""
from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import models, store
from .ladder_sym import best_axis, mirror_iou

VERSION = "s1"
MODEL = "openai/gpt-5.6-sol"
VERIFY = 0.9          # a symmetry is verified if its measured IoU >= this
ALPHA = 128
ROOT = Path(__file__).resolve().parents[2]
REG = ROOT / "workspace/experiments/regions"
OUT = REG / "symmetry"
PROMPT = OUT / "prompts/symmetry.md"


def _png(im: Image.Image) -> bytes:
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def silhouette(concept_png: Path) -> np.ndarray:
    im = Image.open(concept_png).convert("RGBA")
    return np.asarray(im)[..., 3] >= ALPHA


# ---- model --------------------------------------------------------------------------------
def ask_model(concept_png: Path, legend: dict | None, ledger) -> dict:
    im = Image.open(concept_png).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    bg.alpha_composite(im)
    view = bg.convert("RGB")
    view.thumbnail((512, 512))
    png = _png(view)
    leg = ""
    if legend:
        leg = "The design's named parts (judge each one's own symmetry; say which are intentionally asymmetric):\n" + \
              "\n".join(f"- {p['name']}: {p['where']}" for p in legend["parts"])
    prompt = PROMPT.read_text().format(legend=leg)
    body = {"model": MODEL, "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt},
                                                      {"type": "image_url", "image_url": {"url": models.data_uri(png)}}]}]}
    ck = store.key({"symmetry_assess": VERSION, "prompt": prompt, "image": store.put_blob(png), "model": MODEL})
    resp, _, cost, cached = models.call(body, ledger, ck)
    txt = resp["choices"][0]["message"].get("content") or ""
    out = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
    return {**out, "_call": ck, "_cost": cost, "_cached": cached}


def ask_decisions(concept_png: Path, legend: dict | None, ledger, model: str | None = None) -> dict:
    """Decision-model classifier (default Jev): Choice for the global type, one Noul per legend part."""
    from . import decisions as D
    model = model or D.JEV
    im = Image.open(concept_png)
    qs = dict(D.SYMMETRY_Q)
    for i, p in enumerate((legend or {}).get("parts", [])):
        qs[f"part{i}"] = {"type": "noul",
                          "instructions": f"Is the part '{p['name']}' ({p['where']}) intended to be left-right mirror "
                                          "symmetric (or a mirrored pair)? Ignore drawing imperfections.",
                          "criteria": {"true": "symmetric / mirrored pair", "false": "intentionally asymmetric"}}
    r = D.decide(model, ["Concept image of a game character / object:", im], qs, ledger, tag="sym1")
    a = r["answers"]
    out = {"type": a["symmetry"]["choice"], "order": None, "probabilities": a["symmetry"].get("probabilities"),
           "confidence": a["symmetry"].get("confidence"), "classifier": model, "secs": r["secs"],
           "per_part": {p["name"]: {"symmetric": a[f"part{i}"]["noul"] >= 0.5, "p": a[f"part{i}"]["noul"]}
                        for i, p in enumerate((legend or {}).get("parts", []))}}
    return {**out, "_call": r["call"], "_cost": r["cost"], "_cached": r["cached"]}


# ---- measurement --------------------------------------------------------------------------
def rotation_iou(mask: np.ndarray, centre: tuple[float, float], deg: float) -> float:
    """IoU of mask with itself rotated by deg about centre (x, y) in source px."""
    im = Image.fromarray(mask.astype(np.uint8) * 255)
    r = np.asarray(im.rotate(deg, resample=Image.NEAREST, center=centre)) > 127
    u = (mask | r).sum()
    return float((mask & r).sum() / u) if u else 1.0


def best_rotation(mask: np.ndarray, order: int) -> tuple[tuple[float, float], float]:
    """Coarse-to-fine centre search near the centroid; score = min IoU over the n-1 non-trivial rotations."""
    ys, xs = np.nonzero(mask)
    c0 = (float(xs.mean()), float(ys.mean()))
    score = lambda c: min(rotation_iou(mask, c, 360 * k / order) for k in range(1, order))
    best = (c0, score(c0))
    for step in (16, 4, 1):
        cx, cy = best[0]
        for dx in (-2, -1, 0, 1, 2):
            for dy in (-2, -1, 0, 1, 2):
                c = (cx + dx * step, cy + dy * step)
                s = score(c)
                if s > best[1]:
                    best = (c, s)
    return (round(best[0][0], 1), round(best[0][1], 1)), round(best[1], 4)


def measure(mask: np.ndarray) -> dict:
    v2, vs = best_axis(mask)
    h2, hs = best_axis(mask.T)
    small = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(
        (mask.shape[1] // 4, mask.shape[0] // 4), Image.BILINEAR)) > 127      # rotation search at 1/4 res
    rot = {}
    for n in range(2, 9):
        c, s = best_rotation(small, n)
        rot[n] = {"centre_px": [c[0] * 4 + 1.5, c[1] * 4 + 1.5], "score": s}
    return {"vertical": {"axis_px": v2 / 2, "axis_frac": round(v2 / 2 / mask.shape[1], 4), "score": round(vs, 4)},
            "horizontal": {"axis_px": h2 / 2, "axis_frac": round(h2 / 2 / mask.shape[0], 4), "score": round(hs, 4)},
            "rotation": rot}


def classify(m: dict) -> tuple[str, int | None]:
    v, h = m["vertical"]["score"] >= VERIFY, m["horizontal"]["score"] >= VERIFY
    rot = [n for n, r in m["rotation"].items() if r["score"] >= VERIFY]
    n = max(rot) if rot else None
    if n and n > 2:
        return ("radial" if (v or h) else "rotational"), n
    if v and h:
        return "both", None
    if v:
        return "vertical", None
    if h:
        return "horizontal", None
    return ("rotational", 2) if n == 2 else ("none", None)


def assess_symmetry(concept_png, legend=None, regions=None, ledger=None, use_model=True, classifier="jev") -> dict:
    """Spec for ladder_sym. legend: dict or path; regions: optional label map path (same size) for per-part measures.
    classifier: "jev" (decision model, default since bake-off d1), "luna", or "chat" (gpt-5.6-sol prompt)."""
    from . import ladder as L
    concept_png = Path(concept_png)
    legend_d = json.loads(Path(legend).read_text()) if isinstance(legend, (str, Path)) else legend
    ledger = ledger or models.Ledger(0.5)
    mask = silhouette(concept_png)
    m = measure(mask)
    typ, order = classify(m)
    if not use_model:
        model = None
    elif classifier == "chat":
        model = ask_model(concept_png, legend_d, ledger)
    else:
        from . import decisions as D
        model = ask_decisions(concept_png, legend_d, ledger, D.LUNA if classifier == "luna" else D.JEV)
    dis = []
    if model:
        mt, mo = model.get("type"), model.get("order")
        if mt != typ or (typ in ("radial", "rotational") and mo and mo != order):
            dis.append({"what": "global type", "model": f"{mt}{'/' + str(mo) if mo else ''}",
                        "measured": f"{typ}{'/' + str(order) if order else ''}", "resolution": "measurement wins"})
    axes = {k: {**m[k], "verified": m[k]["score"] >= VERIFY} for k in ("vertical", "horizontal")}
    spec = {"version": VERSION, "concept": str(concept_png), "size": list(mask.shape[::-1]), "verify_threshold": VERIFY,
            "type": typ, "order": order, "axes": axes,
            "rotation": {"best_order": max(m["rotation"], key=lambda n: m["rotation"][n]["score"]),
                         "by_order": m["rotation"],
                         "enforced": False, "note": "radial/rotational is recorded only; ladder v2 enforces mirror axes"},
            "model": {k: v for k, v in (model or {}).items() if not k.startswith("_")} | (
                {"call": model["_call"], "cost": model["_cost"], "cached": model["_cached"]} if model else {}),
            "disagreements": dis, "per_part": {}}
    if legend_d:
        src = L.load_labels(Path(regions), legend_d) if regions else None
        for i, p in enumerate(legend_d["parts"]):
            rec = {}
            if model and p["name"] in (model.get("per_part") or {}):
                rec["model"] = model["per_part"][p["name"]]
            if src is not None:
                pm = src == i + 1
                for k, flip in (("vertical", False), ("horizontal", True)):
                    if axes[k]["verified"] and pm.any():
                        mm = pm.T if flip else pm
                        a2 = int(round(2 * axes[k]["axis_frac"] * mm.shape[1]))
                        rec[f"{k}_iou"] = round(mirror_iou(mm, a2), 4)
                rec["measured_symmetric"] = all(v >= VERIFY for k, v in rec.items() if k.endswith("_iou")) \
                    if any(k.endswith("_iou") for k in rec) else None
                ms = (rec.get("model") or {}).get("symmetric")
                if ms is not None and rec["measured_symmetric"] is not None and ms != rec["measured_symmetric"]:
                    dis.append({"what": f"part {p['name']}", "model": ms, "measured": rec["measured_symmetric"],
                                "resolution": "measurement wins"})
                    rec["symmetric"] = rec["measured_symmetric"]
                else:
                    rec["symmetric"] = rec["measured_symmetric"] if rec["measured_symmetric"] is not None else ms
            # exempt from mirror enforcement only if the model calls it intentionally asymmetric AND it measures so;
            # a model-symmetric part that measures low is a drawing/segmentation imperfection, still enforced
            ms = (rec.get("model") or {}).get("symmetric")
            rec["intentionally_asymmetric"] = bool(ms is False and rec.get("measured_symmetric") is not True)
            spec["per_part"][p["name"]] = rec
        if regions:
            spec["regions"] = str(regions)
    spec["cost"] = {"spent": round(ledger.spent, 5), "calls": ledger.calls, "cached": ledger.cached}
    return spec


def draw(spec: dict, out: Path):
    im = Image.open(spec["concept"]).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    bg.alpha_composite(im)
    d = ImageDraw.Draw(bg)
    w, h = im.size
    for k, a in spec["axes"].items():
        col = (0, 160, 0, 255) if a["verified"] else (200, 0, 0, 255)
        x = a["axis_px"]
        line = [(x, 0), (x, h)] if k == "vertical" else [(0, x), (w, x)]
        d.line(line, fill=col, width=3 if a["verified"] else 1)
        d.text((x + 6, 6) if k == "vertical" else (6, x + 6), f"{k} {a['score']:.3f}", fill=col)
    d.text((6, h - 18), f"type={spec['type']}  model={spec['model'].get('type')}", fill=(0, 0, 0, 255))
    bg.convert("RGB").resize((w // 2, h // 2), Image.LANCZOS).save(out)


if __name__ == "__main__":
    budget = float(sys.argv[sys.argv.index("--budget") + 1]) if "--budget" in sys.argv else 0.5
    OUT.mkdir(parents=True, exist_ok=True)
    spec = assess_symmetry(ROOT / "workspace/shared/sources/squid_concept_gpt_front.png", REG / "m1/legend.json",
                           regions=REG / "m1/gpt-s1/regions.png", ledger=models.Ledger(budget))
    (OUT / "spec.json").write_text(json.dumps(spec, indent=1, default=str))
    draw(spec, OUT / "axis.png")
    print(json.dumps({k: spec[k] for k in ("type", "order", "axes", "disagreements", "cost")}, indent=1))
    print("model", spec["model"].get("type"), spec["model"].get("reason"))
    print("best rotation", spec["rotation"]["best_order"], spec["rotation"]["by_order"][spec["rotation"]["best_order"]])
    for k, v in spec["per_part"].items():
        print(" ", k, v)
