"""Snipe-bot dry run: the whole region-ladder pipeline on a NEW enemy, to see how well it generalises.

Brief (Johnny): "a snipe bot: a robot for a marsh level. It's a bird (a snipe) robot that is also a sniper, so its
beak should be recognisably a gun. Because it's a sniper it should be facing horizontally (so not symmetric!)."
Style: a whimsical NES Mega Man-like robot enemy. Facing: LEFT (recorded in results.json).

Steps (each cached, each writing into workspace/experiments/regions/snipe/):
  concept   Nano Banana (BYOK) + gpt-5.4-image-2 candidates; gpt-5.6-sol picks; crop to a centred transparent square
  symmetry  symmetry.assess_symmetry (Jev + deterministic verification); the ladder honours the spec (no mirror)
  regions   generic legend (gpt-5.6-sol, depth / occluded_by), Nano Banana paint, snap / register / verify
  plan      generic sprite plan (gpt-5.6-sol): palette first, region priority, eye parts, significance guidance
  ladder    pipeline_gen.rung at 64 / 32 / 24 / 20 / 16 (generic versions of the v4 rules)
  render    pipeline_gen.render (render_v4.render with mirror off) + NES tile rule

Cost: OpenRouter reports usage.cost = 0 for BYOK calls (Nano Banana via Johnny's Google key) and the Ledger
undercounts, so every call's real cost is read from its record: usage.cost + cost_details.upstream_inference_cost
when is_byok (`call_cost`).

uv run python -m megaloman.snipe <step> [--budget 3]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from . import imaging, models, store

ROOT = Path(store.ROOT)
OUT = ROOT / "workspace/experiments/regions/snipe"
NB = "google/gemini-nano-banana-2.1"
GPT_IMG = "openai/gpt-5.4-image-2"
SOL = "openai/gpt-5.6-sol"
FACING = "left"
CALLS: dict[str, str] = {}       # cache key -> step tag, for every call this dry run used (cached or not)

CONCEPT_PROMPT = """Concept art for a video game enemy: a whimsical robot enemy in the style of the NES Mega Man games
(a classic Mega Man robot-master / small enemy design: chunky, friendly, toy-like, rounded armour plates, bold simple
shapes, big expressive eyes).

The character: SNIPE BOT, a robot for a marsh level. It is a robot snipe (the wading marsh bird with a very long,
straight beak, a round plump body, short tail and thin legs) that is also a SNIPER: its long straight beak is
clearly a rifle barrel (a sniper rifle muzzle at the tip, a barrel with a muzzle brake, maybe a small scope on top of
the head), so the bird's silhouette and the gun read at the same time.

Composition and rendering rules:
- ONE single character only, full body, strict SIDE VIEW (profile) facing LEFT: the beak/barrel points to the left
  edge of the image, horizontal.
- Shaded 3-D form: soft volumetric shading and highlights on every part (not flat cartoon colours), clean dark
  outlines.
- Marsh-themed colours: reed and moss greens, mud browns, a little rusty orange or cattail tan; the eye a bright
  contrasting colour.
- Plain pure white background, no ground, no shadow on the ground, no scenery, no text, no labels, no border.
- The whole character fits in the frame with margin on every side."""


# ------------------------------------------------------------------ cost (BYOK-aware)

def call_cost(ck: str) -> dict:
    rec = store.get_json("calls", ck) or {}
    u = (rec.get("response") or {}).get("usage") or {}
    up = (u.get("cost_details") or {}).get("upstream_inference_cost") or 0.0
    byok = bool(u.get("is_byok"))
    return {"openrouter": float(u.get("cost") or 0.0), "upstream": float(up) if byok else 0.0, "byok": byok,
            "total": float(u.get("cost") or 0.0) + (float(up) if byok else 0.0)}


def decision_cost(ck: str) -> float:
    rec = store.get_json("decisions", ck) or store.get_json("calls", ck) or {}
    return float(rec.get("cost") or 0.0)


def spend(extra: dict | None = None) -> dict:
    by = {}
    for ck, tag in CALLS.items():
        c = call_cost(ck)
        by.setdefault(tag, 0.0)
        by[tag] += c["total"]
    for tag, v in (extra or {}).items():
        by[tag] = by.get(tag, 0.0) + v
    return {"by_step": {k: round(v, 4) for k, v in by.items()}, "total": round(sum(by.values()), 4)}


def track(ck: str, tag: str):
    """Record a call this dry run used; persisted in OUT/calls.json so the total spans every step's process."""
    f = OUT / "calls.json"
    allc = json.loads(f.read_text()) if f.exists() else {}
    CALLS.update(allc)
    CALLS[ck] = tag
    OUT.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(CALLS, indent=1))
    return ck


# ------------------------------------------------------------------ 1. concept

def gen_concept(model: str, seed: int, ledger) -> dict:
    body = {"model": model, "modalities": ["image", "text"], "seed": seed,
            "messages": [{"role": "user", "content": [{"type": "text", "text": CONCEPT_PROMPT}]}]}
    ck = store.key({"step": "snipe.concept", "model": model, "prompt": CONCEPT_PROMPT, "seed": seed})
    try:
        resp, imgs, cost, cached = models.call(body, ledger, ck)
    except models.BudgetExceeded:
        raise
    except Exception as e:
        return {"model": model, "seed": seed, "error": str(e)[:300], "call": ck}
    track(ck, "concept")
    return {"model": model, "seed": seed, "call": ck, "cached": cached, "png": imgs[0] if imgs else None,
            "cost": call_cost(ck), "text": (resp["choices"][0]["message"].get("content") or "")[:200]}


def remove_white(im: Image.Image, tol=24) -> Image.Image:
    """White background -> transparent: near-white pixels connected to the border (flood fill), softened 1 px."""
    a = np.array(im.convert("RGBA")).astype(int)
    rgb = a[..., :3]
    white = (rgb.min(-1) >= 255 - tol) & ((rgb.max(-1) - rgb.min(-1)) <= 18)
    lab, _ = ndimage.label(white)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    bg = np.isin(lab, list(border))
    bg = ndimage.binary_opening(bg, iterations=1) | (bg & ~ndimage.binary_erosion(~bg, iterations=1))
    # drop specks of foreground left floating in the background
    fg = ~bg
    fl, k = ndimage.label(fg)
    if k > 1:
        areas = ndimage.sum_labels(np.ones_like(fl), fl, range(1, k + 1))
        keep = [i + 1 for i, s in enumerate(areas) if s >= 0.01 * areas.max()]
        fg = np.isin(fl, keep)
    a[..., 3] = np.where(fg, 255, 0)
    return Image.fromarray(a.astype(np.uint8), "RGBA")


def square_crop(im: Image.Image, side=1024, margin=0.06) -> Image.Image:
    """Crop to the alpha bbox, centre it on a square transparent canvas with `margin` on the long side."""
    a = np.array(im)[..., 3] > 0
    ys, xs = np.nonzero(a)
    box = im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    w, h = box.size
    s = int(round(max(w, h) / (1 - 2 * margin)))
    can = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    can.alpha_composite(box, ((s - w) // 2, (s - h) // 2))
    return can.resize((side, side), Image.LANCZOS)


JUDGE_PROMPT = """You are the art director for a NES Mega Man-style game. Brief: "SNIPE BOT: a robot for a marsh level.
A robot snipe (marsh wading bird with a very long straight beak) that is also a sniper, so its beak is recognisably a
rifle barrel. Side view, facing left. Whimsical Mega Man robot enemy, shaded 3-D concept art, marsh colours."

The candidates are labelled A, B, C, D in the image. The chosen concept will be shrunk to a 16-32 px NES sprite, so
also weigh: a bold, readable silhouette (bird + gun obvious from the outline alone), few clearly separated parts, a
clear single eye, distinct legs, no clutter, single character, strict side view.

Return JSON only: {"ranking": ["<best letter>", ...], "scores": {"A": 1-10, ...}, "reasons": {"A": "<one sentence>",
...}, "best_for_sprite": "<letter>"}"""


def judge_concepts(cands: list[dict], ledger) -> dict:
    from PIL import ImageDraw
    tiles = []
    for c in cands:
        im = Image.open(c["square"]).convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        bg.alpha_composite(im)
        tiles.append(bg.convert("RGB").resize((384, 384), Image.LANCZOS))
    sheet = Image.new("RGB", (len(tiles) * 400 + 16, 440), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    from .ladder import _font
    for k, t in enumerate(tiles):
        sheet.paste(t, (16 + k * 400, 48))
        d.text((16 + k * 400 + 180, 6), "ABCD"[k], fill=(0, 0, 0), font=_font(34))
    png = imaging.to_png(sheet)
    sheet.save(OUT / "concept/judge_sheet.png")
    body = {"model": SOL, "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": [{"type": "text", "text": JUDGE_PROMPT},
                                                      {"type": "image_url", "image_url": {"url": models.data_uri(png)}}]}]}
    ck = store.key({"step": "snipe.concept_judge", "model": SOL, "prompt": JUDGE_PROMPT, "image": store.sha(png)})
    resp, _, _, cached = models.call(body, ledger, ck)
    track(ck, "concept_judge")
    m = re.search(r"\{.*\}", resp["choices"][0]["message"].get("content") or "", re.S)
    return json.loads(m.group(0)) | {"call": ck, "cached": cached}


PICK = ("gpt-s1", "the judge's overall #1 (9/10: long rifle-beak, wading legs, single eye, clean profile); its "
        "sprite pick D (nb-s3) is the most compact, but its 2:1 aspect (barrel sets the width) fills only 0.50 of "
        "the square's height vs C's 0.76, i.e. ~4 rows fewer at 16 px; the NB candidates are also three-quarter-ish "
        "round birds ('generic round bird' per the judge), C is the recognisable snipe")


def step_concept(ledger, plan=((NB, 1), (NB, 2), (GPT_IMG, 1), (NB, 3)), pick=PICK) -> dict:
    d = OUT / "concept"
    d.mkdir(parents=True, exist_ok=True)
    cands = []
    for k, (model, seed) in enumerate(plan):
        r = gen_concept(model, seed, ledger)
        cid = f"{'nb' if model == NB else 'gpt'}-s{seed}"
        rec = {"id": cid, "letter": "ABCD"[k], "model": model, "seed": seed, "call": r["call"],
               "error": r.get("error"), "cost": r.get("cost"), "text": r.get("text")}
        if r.get("png"):
            raw = imaging.from_png(r["png"]).convert("RGBA")
            raw.save(d / f"{cid}_raw.png")
            sq = square_crop(remove_white(raw))
            sq.save(d / f"{cid}.png")
            rec.update(raw=str(d / f"{cid}_raw.png"), square=str(d / f"{cid}.png"), raw_size=list(raw.size))
        cands.append(rec)
    ok = [c for c in cands if c.get("square")]
    for k, c in enumerate(ok):
        c["letter"] = "ABCD"[k]
    j = judge_concepts(ok, ledger)
    jp = j.get("best_for_sprite") or j["ranking"][0]
    chosen = next(c for c in ok if c["id"] == pick[0]) if pick else next(c for c in ok if c["letter"] == jp)
    for c in ok:
        a = np.array(Image.open(c["square"]))[..., 3] > 127
        ys, xs = np.nonzero(a)
        c["bbox_frac"] = [round((np.ptp(xs) + 1) / a.shape[1], 3), round((np.ptp(ys) + 1) / a.shape[0], 3)]
    Image.open(chosen["square"]).save(d / "chosen.png")
    res = {"facing": FACING, "prompt": CONCEPT_PROMPT, "candidates": cands, "judge": j, "chosen": chosen["id"],
           "chosen_by": pick[1] if pick else "judge best_for_sprite", "cost": spend()}
    (d / "results.json").write_text(json.dumps(res, indent=1, default=str))
    return res


# ------------------------------------------------------------------ 2. symmetry

def step_symmetry(ledger, legend=None, regions=None) -> dict:
    from . import symmetry as S
    d = OUT / "symmetry"
    d.mkdir(parents=True, exist_ok=True)
    spec = S.assess_symmetry(OUT / "concept/chosen.png", legend, regions=regions, ledger=ledger)
    if spec["model"].get("call"):
        track(spec["model"]["call"], "symmetry")
    (d / "spec.json").write_text(json.dumps(spec, indent=1, default=str))
    S.draw(spec, d / "axis.png")
    return spec


# ------------------------------------------------------------------ 3. legend + region paint

def concept_inputs():
    from . import regions
    concept = Image.open(OUT / "concept/chosen.png").convert("RGBA")
    sil, shadow = regions.concept_masks(concept)
    return concept, sil, regions.concept_on_white(concept, sil), int(shadow.sum())


# Nano Banana refuses the chosen (GPT-made) concept as an input image: finish PROHIBITED_CONTENT on both v2-style
# paints, again with weapon words neutralised, and on a one-line "repaint this toy robot bird as a flat colour map"
# probe; the same probe on the NB-made candidate D passes. Blocked calls are free. So the paint falls back to
# gpt-5.4-image-2 (the M1 painter, which needs registration).
PAINTERS = ((NB, 1), (NB, 2), (GPT_IMG, 1))


def step_regions(ledger, painters=PAINTERS) -> dict:
    from . import pipeline_gen as G
    d = OUT / "regions"
    d.mkdir(parents=True, exist_ok=True)
    concept, sil, concept_rgb, shadow_px = concept_inputs()
    concept_rgb.save(d / "concept_on_white.png")
    leg = G.legend(concept_rgb, ledger)
    track(leg["call"], "legend")
    (d / "legend.json").write_text(json.dumps(leg, indent=1))
    (d / "paint_prompt.md").write_text(G.paint_prompt(leg))
    cands = []
    for model, sd in painters:
        c = G.region_candidate(d, f"{'nb' if model == NB else 'gpt'}-s{sd}", sd, concept, concept_rgb, sil, leg,
                               ledger, model)
        rec = (store.get_json("calls", c["call"]) or {}).get("response") or {}
        c["finish"] = ((rec.get("choices") or [{}])[0]).get("native_finish_reason")
        track(c["call"], "paint")
        c["cost"] = call_cost(c["call"])
        cands.append(c)
    ok = [c for c in cands if c["align"].get("pass")]
    best = max(ok or [c for c in cands if c["align"].get("iou")], key=lambda c: c["align"]["iou"], default=None)
    if best:
        Image.open(d / best["id"] / "regions.png").save(d / "regions.png")
        np.save(d / "labels.npy", np.load(d / best["id"] / "labels.npy"))
        Image.open(d / best["id"] / "overlay.png").save(d / "overlay.png")
    res = {"legend": leg, "layer_order": __import__("megaloman.regions_v2", fromlist=["x"]).layer_order(leg),
           "shadow_px": shadow_px, "candidates": cands, "best": best and best["id"], "cost": spend()}
    (d / "results.json").write_text(json.dumps(res, indent=1, default=str))
    return res


# ------------------------------------------------------------------ 4a. plan: palette first, priority, eyes, significance

BRIEF = ("SNIPE BOT, an enemy for a marsh level in a NES Mega Man-style game: a robot snipe (marsh wading bird with "
         "a very long straight beak) that is also a sniper, so its beak is recognisably a gun (a rifle barrel). Side "
         "view facing left, not symmetric. Whimsical Mega Man robot enemy.")
SIG_NOTES = ("The brief: a snipe (marsh wading bird: very long straight beak, round body, thin wading legs) that is also "
             "a sniper, so its beak must stay recognisably a gun (a rifle barrel with a muzzle). Decide which parts carry "
             "that identity at 16 px and how much they need to be exaggerated (into the empty background) to keep it.")


def prune_absent(leg: dict, labels: np.ndarray) -> tuple[dict, np.ndarray, list[str]]:
    """Drop legend parts the paint left out (0 px) and renumber the label map."""
    keep = [i for i in range(len(leg["parts"])) if (labels == i + 1).any()]
    gone = [p["name"] for i, p in enumerate(leg["parts"]) if i not in keep]
    lut = np.zeros(len(leg["parts"]) + 1, labels.dtype)
    for k, i in enumerate(keep):
        lut[i + 1] = k + 1
    out = json.loads(json.dumps(leg))
    out["parts"] = [leg["parts"][i] for i in keep]
    for p in out["parts"]:
        p["occluded_by"] = [o for o in p["occluded_by"] if o not in gone]
    return out, lut[labels], gone


def load_regions():
    from . import pipeline_gen as G
    from . import ladder_v2 as LV
    d = OUT / "regions"
    leg = json.loads((d / "legend.json").read_text())
    leg, labels, gone = prune_absent(leg, np.load(d / "labels.npy"))
    src = LV.square(G._centre(labels))
    return leg, labels, src, gone


def step_plan(ledger, version="v1") -> dict:
    """v1: the generic plan prompt. v2: + the eye-colour rule (pipeline_gen.EYE_RULE)."""
    from . import pipeline_gen as G
    d = OUT / ("plan" if version == "v1" else f"plan_{version}")
    d.mkdir(parents=True, exist_ok=True)
    leg, labels, src, gone = load_regions()
    concept, sil, concept_rgb, _ = concept_inputs()
    pl = G.plan(concept_rgb, OUT / "regions/regions.png", src, leg, BRIEF, ledger, eye_rule=version != "v1")
    track(pl["call"], "plan")
    legp, prec = G.apply_plan(leg, pl)
    specs, g = G.significance(src, legp, concept_rgb, OUT / "regions/regions.png", SIG_NOTES, prec["eyes"], ledger,
                              d / "prompts")
    track(g["call"], "significance")
    res = {"pruned_parts": gone, "plan": prec, "plan_raw": pl["raw"], "significance": specs,
           "significance_summary": g["raw"].get("summary"), "cost": spend()}
    (d / "plan.json").write_text(json.dumps(res, indent=1, default=str))
    (d / "legend_plan.json").write_text(json.dumps(legp, indent=1))
    return res


# ------------------------------------------------------------------ 4b. ladder + render

SIZES = (64, 32, 24, 20, 16)
RENDER_OPTS = {"contrast_below": 0, "speck_below": 32}    # render_v4.OPTS: lines between all regions, every size
CHECKS = {"tile_colors": {}, "orphans": {}, "holes": {}, "outline": {}, "colors": {"max_total": 4, "gate": True}}


VERSIONS = {"v1": {"plan": "plan", "eye_moat": False},          # generic rules, eye margin from lines only
            "v2": {"plan": "plan_v2", "eye_moat": True},        # + eye-colour rule in the plan, eye surround moat
            "v3": {"plan": "plan_v2", "eye_moat": True, "stroke_floor": True, "eye_nudge": True},  # + strokes, nudge
            "v3a": {"plan": "plan_v2", "eye_moat": True, "stroke_floor": True}}                    # strokes, no nudge


def step_ladder(version="v1") -> dict:
    from . import ladder as L, pipeline_gen as G, render as R1, render_v4 as R4
    V = VERSIONS[version]
    d = OUT / version
    (d / "ladder").mkdir(parents=True, exist_ok=True)
    (d / "render").mkdir(parents=True, exist_ok=True)
    pj = json.loads((OUT / V["plan"] / "plan.json").read_text())
    leg = json.loads((OUT / V["plan"] / "legend_plan.json").read_text())
    _, _, src_raw, _ = load_regions()
    spec = json.loads((OUT / "symmetry/spec.json").read_text())
    specs = pj["significance"]
    eyes = pj["plan"]["eyes"]
    surround = pj["plan"]["eye_surround"]
    ctx = G.prepare(src_raw, leg, spec, specs, SIZES)
    protected = set(eyes) | ({surround} if surround else set())
    mats = G.materials(leg, protected)
    L.colorize(ctx["src"], leg).save(d / "ladder/source.png")
    light = [tuple(imaging.rgb(pj["plan"]["palette"]["light"]["hex"]))]
    res = {"version": version, "spec": V, "palette": pj["plan"]["palette"], "sizes": SIZES, "symmetry": ctx["sym"], "protected": sorted(protected), "opts": RENDER_OPTS,
           "instances": {str(v): ctx["base_of"][v] for v in range(1, len(ctx["base_of"]))}, "rungs": {}}
    for N in SIZES:
        r = G.rung(ctx, N, eyes, {"eye_surround": surround if V["eye_moat"] else None,
                                  "stroke_floor": V.get("stroke_floor", False), "eye_nudge": V.get("eye_nudge", False)})
        L.colorize(r["base"], leg).save(d / f"ladder/rung_{N}.png")
        im, info = G.render(r["ext"], ctx["base_of"], mats, leg, N, "none", RENDER_OPTS)
        legal, tinfo = R1.tile_reduce(im, salient=light, salient_w=8.0, symmetric=False)
        im.save(d / f"render/render_{N}_free.png")
        legal.save(d / f"render/render_{N}.png")
        a = np.array(legal)
        info.update(tile=tinfo, checks=imaging.run_checks(legal, CHECKS), eyes=G.eye_check(info["grid"], leg, eyes),
                    colours=sorted(R1.hexs(c) for c in np.unique(a[a[..., 3] > 0][:, :3], axis=0)),
                    separation=R4.separation_check(info["grid"], leg))
        res["rungs"][str(N)] = {"k": r["k"], "log": r["log"], "health_score": r["health_score"],
                                "health": [{k: x[k] for k in ("part", "status", "area", "components", "reasons")}
                                           for x in r["health"]], "render": info}
    (d / "results.json").write_text(json.dumps(res, indent=1, default=str))
    return res


# ------------------------------------------------------------------ 5. an honest read per size (one model call)

READ_PROMPT = """Image 1 is the concept art of SNIPE BOT (a robot snipe, marsh wading bird with a very long beak, that is
also a sniper: its beak is a rifle barrel; facing left). Image 2 shows the same enemy as NES sprites at 64, 32, 24, 20
and 16 px (left to right), all enlarged to the same display size.

For EACH size, answer strictly from what is visible: does it read as a bird? Is the beak recognisably a gun barrel?
Is there a clear eye? Are the legs readable? Then one short verdict sentence. Be blunt; do not be polite.
Return JSON only: {"64": {"bird": true|false, "gun_beak": true|false, "eye": true|false, "legs": true|false,
"verdict": "..."}, "32": {...}, "24": {...}, "20": {...}, "16": {...}}"""


def step_read(ledger, version="v3") -> dict:
    concept, _, concept_rgb, _ = concept_inputs()
    c = concept_rgb.copy()
    c.thumbnail((512, 512))
    strip = Image.new("RGB", (5 * 270 + 10, 280), (255, 255, 255))
    for k, N in enumerate(SIZES):
        im = Image.open(OUT / version / "render" / f"render_{N}.png").convert("RGBA").resize((260, 260), Image.NEAREST)
        strip.paste(im, (10 + k * 270, 10), im)
    strip.save(OUT / version / "strip.png")
    imgs = [imaging.to_png(c), imaging.to_png(strip)]
    body = {"model": SOL, "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": [{"type": "text", "text": READ_PROMPT}] +
                          [{"type": "image_url", "image_url": {"url": models.data_uri(b)}} for b in imgs]}]}
    ck = store.key({"step": "snipe.read", "model": SOL, "prompt": READ_PROMPT, "images": [store.sha(b) for b in imgs]})
    resp, _, _, cached = models.call(body, ledger, ck)
    track(ck, "read")
    out = json.loads(re.search(r"\{.*\}", resp["choices"][0]["message"].get("content") or "", re.S).group(0))
    (OUT / version / "read.json").write_text(json.dumps(out | {"call": ck}, indent=1))
    return out


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step")
    ap.add_argument("--budget", type=float, default=3.0)
    ap.add_argument("--version", default="v1")
    a = ap.parse_args()
    ledger = models.Ledger(a.budget)
    if a.step == "concept":
        r = step_concept(ledger)
        for c in r["candidates"]:
            print(c["id"], c["letter"], c.get("error"), c.get("cost"))
        print(json.dumps(r["judge"], indent=1))
        print("chosen", r["chosen"], r["cost"])
    elif a.step == "regions":
        r = step_regions(ledger)
        for p in r["legend"]["parts"]:
            print(f"  {p['name']:22} {p['role']:8} pr{p['priority']:2} {p['color']} {p['material']:14} {p['depth']:5} "
                  f"occ_by={p['occluded_by']}")
        print("layers", r["layer_order"], "fixes", r["legend"]["color_fixes"])
        for c in r["candidates"]:
            print(c["id"], c.get("error"), c.get("iou_raw"), c.get("iou_registered"), c.get("transform"),
                  c["align"], c.get("islands_folded"), c["cost"])
        print("best", r["best"], r["cost"])
    elif a.step == "plan":
        r = step_plan(ledger, a.version)
        print("pruned", r["pruned_parts"])
        print(json.dumps({k: r["plan"][k] for k in ("palette", "palette_reason", "eyes", "eye_surround", "summary",
                                                    "repairs")}, indent=1))
        for n, a_ in r["plan"]["assign"].items():
            print(f"  {n:22} {a_['group']:6} {a_['why']}")
        print("priority", [q["part"] for q in r["plan"]["priority"]])
        for q in r["significance"]:
            print(f"  {q['name']:22} sig {q['significance']:.2f} {q['mode']:10} x{q['max_exaggeration']:.2f} "
                  f"{q['essential_shape']:13} {q['anchor']:20} bg {q['may_claim_background']!s:5} "
                  f"fp16 {q['min_footprint']['16']} fp32 {q['min_footprint']['32']}")
        print(r["significance_summary"], r["cost"])
    elif a.step == "ladder":
        r = step_ladder(a.version)
        print("sym", r["symmetry"], "instances", r["instances"])
        for N, g in r["rungs"].items():
            i = g["render"]
            print(N, "k", {k: v["k"] for k, v in g["k"].items()}, "eyes", g["log"]["eyes"], "floors", g["log"]["floors"])
            print("  tile", i["tile"]["changed_px"], i["colours"], "eyes ok", {e: v["ok"] for e, v in i["eyes"].items()},
                  "sep", i["separation"]["ok"], i["separation"]["touching"], "clumps", i["clumps"],
                  "checks", {k: v.get("ok", v) if isinstance(v, dict) else v for k, v in i["checks"].items()})
            print("  health", [(h["part"], h["status"], h["area"]) for h in g["health"] if h["status"] != "green"])
            if int(N) <= 32:
                print("\n".join("  " + row for row in i["grid"]))
    elif a.step == "read":
        print(json.dumps(step_read(ledger, a.version), indent=1))
    elif a.step == "spend":
        CALLS.update(json.loads((OUT / "calls.json").read_text()))
        print(json.dumps(spend(), indent=1))
    elif a.step == "symmetry":
        s = step_symmetry(ledger)
        print(json.dumps({k: s[k] for k in ("type", "order", "axes", "disagreements", "cost")}, indent=1))
        print("model", {k: s["model"].get(k) for k in ("type", "probabilities", "confidence")})
        print("best rotation", s["rotation"]["best_order"], s["rotation"]["by_order"][s["rotation"]["best_order"]])


if __name__ == "__main__":
    main()
