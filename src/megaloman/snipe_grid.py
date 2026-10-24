"""Snipe bot, grid route (Johnny, 2026-10-24): no image generation. A vision reasoning LLM places pixels symbolically.

  inputs   compact concept (with vestigial wing), region map + legend/priority, NES/Mega Man 16x16 style refs, guidance
  round 1  model returns a 16x16 grid of palette letters + palette (<= 5 colours incl. outline, '.' = transparent)
  round 2+ code validates and renders it 8x; the model sees its render next to the concept and revises (3 rounds max)
  manifest pipeline.json (megaloman.pipeline/1), rewritten after every round so the cockpit view live-refreshes

uv run python -m megaloman.snipe_grid [--models anthropic/claude-opus-5.5,openai/gpt-5.6-sol] [--budget 1.5]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw

from . import models, store, snipe, pipeline_manifest as PM
from .ladder import _font
from .snipe_boxy import PARTS

BOXY = snipe.OUT / "compact" / "boxy"
OUT = snipe.OUT / "compact" / "grid"
BENCH = Path(store.ROOT) / "workspace/games/megaman/classes/small_enemy/keyframe/benchmarks"
REFS = [("telly", "Telly (Mega Man 2)"), ("met_walk", "Met, walking"), ("pipi", "Pipi (bird)")]
ROUNDS, MAX_COLOURS = 3, 5

GUIDANCE = [
    "Fill the square: the silhouette is compact and occupies most of the 16x16 in both width and height.",
    "Preserve the silhouette and the big shapes and lines of the concept; it must read from its outline alone.",
    "Head and body are clearly separated by a different colour AND a dark line between them.",
    "The eye is big, bright and unmistakable: the top priority.",
    "The snub gun-nose is short and chunky (2+ px thick), not a 1-px dash.",
    "The scope on top is distinct: its own little block, separated from the head by outline.",
    "Legs folded/tucked under the body; feet just under it. Tail stubby.",
    "Breastplate is just a hint (a few pixels or a shade), never a big panel.",
    "Few large colour masses: one body colour, one accent, a bright eye; no fine patterns, no noise.",
    "Strong dark outline and value contrast between masses (NES style: <= 4-5 colours incl. outline).",
]

SYSTEM = "You are an expert NES pixel artist (Mega Man 2/3 enemy sprites). You place every pixel deliberately."

FORMAT = f"""Answer with ONE JSON object in a ```json fenced block, nothing after it:
{{"palette": {{"K": "#000000", "A": "#rrggbb", ...}},   // at most {MAX_COLOURS} colours INCLUDING the outline; single uppercase letters
 "grid": ["................", ... 16 strings],            // exactly 16 rows x 16 chars; '.' = transparent; other chars = palette letters
 "notes": "one or two sentences: what you changed and why"}}"""

FIRST = """Draw this enemy as a 16x16 NES sprite (Mega Man style), placing the pixels yourself.

Image 1: the concept (a round robot "snipe" bird: gun-nose beak, scope on top, big eye, folded side wing, small
vestigial wing nub up/right, tucked legs, stubby tail).
Image 2: the region map of the concept, with a legend. The legend order is the PRIORITY order (highest first):
{prio}. When pixels are scarce, higher-priority parts win; low-priority parts may shrink to a hint or vanish.
Image 3: real NES Mega Man enemy sprites at 16x16-ish (shown enlarged), for the target style: outline, colour count,
how features are abstracted.

Guidance:
{guidance}

Take colours from the concept (NES-ish versions of them). Think it through on a rough grid first, then give the
final grid.

"""

REVISE = """Revision round {r} of {n}. Your previous answer is below. Image 1 shows, left to right: the concept, the
region map, and YOUR sprite rendered at 8x (grid lines every pixel) and at 1x-ish thumbnail size. Image 2 is the
NES style reference again.

Compare your sprite with the concept honestly: does it read as this character? Check each priority part in order
({prio}) and the guidance:
{guidance}
{issues}
Fix the biggest problems; keep what works. Your previous answer:
```json
{prev}
```

"""


def on_white(im: Image.Image) -> Image.Image:
    bg = Image.new("RGB", im.size, "white")
    bg.paste(im, mask=im.split()[3] if im.mode == "RGBA" else None)
    return bg


def png(im) -> bytes:
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def regions_image() -> Image.Image:
    ov = Image.open(BOXY / "overlay.png").convert("RGB").resize((640, 640), Image.LANCZOS)
    W = 640 + 300
    im = Image.new("RGB", (W, 640), "white")
    im.paste(ov, (0, 0))
    d, f = ImageDraw.Draw(im), _font(26)
    d.text((660, 14), "legend = priority", fill="black", font=f)
    for i, (n, c, _) in enumerate(PARTS):
        y = 60 + i * 56
        d.rectangle([660, y, 696, y + 36], fill=c, outline="black")
        d.text((708, y + 2), f"{i + 1}. {n}", fill="black", font=f)
    return im


def refs_image() -> Image.Image:
    S, pad = 16, 30
    ims = [Image.open(BENCH / n / "subject.png").convert("RGBA") for n, _ in REFS]
    W = sum(i.width * S + pad for i in ims) + pad
    H = max(i.height for i in ims) * S + 70
    sheet = Image.new("RGB", (W, H), "white")
    d, f, x = ImageDraw.Draw(sheet), _font(22), pad
    for im, (_, label) in zip(ims, REFS):
        big = im.resize((im.width * S, im.height * S), Image.NEAREST)
        sheet.paste(big, (x, 20), big)
        d.text((x, H - 40), f"{label} {im.width}x{im.height}", fill="black", font=f)
        x += big.width + pad
    return sheet


def parse(text: str):
    m = re.findall(r"```json\s*(\{.*?\})\s*```", text, re.S) or re.findall(r"(\{\s*\"palette\".*\})", text, re.S)
    if not m:
        return None, ["no JSON block found"]
    try:
        doc = json.loads(m[-1])
    except Exception as e:
        return None, [f"JSON parse error: {e}"]
    return doc, validate(doc)


def validate(doc) -> list[str]:
    errs = []
    pal, grid = doc.get("palette") or {}, doc.get("grid") or []
    if not isinstance(pal, dict) or not pal:
        return ["palette missing"]
    if len(pal) > MAX_COLOURS:
        errs.append(f"palette has {len(pal)} colours (max {MAX_COLOURS})")
    for k, v in pal.items():
        if len(k) != 1 or k == "." or not re.fullmatch(r"#[0-9a-fA-F]{6}", str(v)):
            errs.append(f"bad palette entry {k!r}: {v!r}")
    if len(grid) != 16:
        errs.append(f"grid has {len(grid)} rows (need 16)")
    for i, row in enumerate(grid):
        if len(row) != 16:
            errs.append(f"row {i} has {len(row)} chars (need 16)")
        bad = set(row) - set(pal) - {"."}
        if bad:
            errs.append(f"row {i} uses unknown letters {sorted(bad)}")
    return errs


def render(doc, scale=1, lines=False) -> Image.Image:
    pal = {k: tuple(int(v[i:i + 2], 16) for i in (1, 3, 5)) + (255,) for k, v in doc["palette"].items()}
    im = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    for y, row in enumerate(doc["grid"][:16]):
        for x, ch in enumerate(row[:16]):
            if ch in pal:
                im.putpixel((x, y), pal[ch])
    if scale == 1:
        return im
    big = on_white(im.resize((16 * scale, 16 * scale), Image.NEAREST))
    if lines:
        d = ImageDraw.Draw(big)
        for i in range(17):
            d.line([(i * scale, 0), (i * scale, 16 * scale)], fill=(200, 200, 200))
            d.line([(0, i * scale), (16 * scale, i * scale)], fill=(200, 200, 200))
    return big


def compare_image(concept, regions, doc) -> Image.Image:
    T = 512
    c = on_white(concept).resize((T, T), Image.LANCZOS)
    r = regions.resize((int(regions.width * T / regions.height), T), Image.LANCZOS)
    s8 = render(doc, 32, lines=True)  # 512
    s3 = render(doc, 4)               # 64: near-real size
    W = c.width + r.width + s8.width + s3.width + 50
    im = Image.new("RGB", (W, T), (235, 235, 235))
    x = 0
    for p in (c, r, s8, s3):
        im.paste(p, (x, 0))
        x += p.width + 16
    return im


def call(model, content, ledger, key):
    body = {"model": model, "max_tokens": 16000, "reasoning": {"max_tokens": 8000},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]}
    ck = hashlib.sha256(("snipe_grid1|" + key + json.dumps(body)).encode()).hexdigest()
    resp, _, cost, cached = models.call(body, ledger, ck)
    return resp["choices"][0]["message"].get("content") or "", cost, cached


def img_part(im):
    return {"type": "image_url", "image_url": {"url": models.data_uri(png(im))}}


def run_model(model, tag, concept, regions, refs, ledger, stages, write):
    prio = " > ".join(n for n, _, _ in PARTS)
    guidance = "\n".join(f"- {g}" for g in GUIDANCE)
    d = OUT / tag
    d.mkdir(parents=True, exist_ok=True)
    doc, log = None, []
    for r in range(1, ROUNDS + 1):
        if r == 1:
            content = [{"type": "text", "text": FIRST.format(prio=prio, guidance=guidance) + FORMAT},
                       img_part(on_white(concept).resize((768, 768), Image.LANCZOS)), img_part(regions), img_part(refs)]
        else:
            issues = "" if not log[-1]["errors"] else "Validation problems last round: " + "; ".join(log[-1]["errors"])
            content = [{"type": "text", "text": REVISE.format(r=r, n=ROUNDS, prio=prio, guidance=guidance,
                                                              issues=issues, prev=json.dumps(doc, indent=0)) + FORMAT},
                       img_part(compare_image(concept, regions, doc)), img_part(refs)]
        new, errs, cost = None, ["not run"], 0.0
        for attempt in range(2):  # one repair retry when the answer doesn't validate
            text, c, cached = call(model, content, ledger, f"{tag}|r{r}|a{attempt}")
            cost += c
            new, errs = parse(text)
            (d / f"round{r}_a{attempt}.txt").write_text(text)
            if new is not None and not errs:
                break
            content = content + [{"type": "text", "text": "Your answer failed validation: " + "; ".join(errs)
                                  + ". Answer again, fully, in the required format."}]
        if new is None:
            log.append({"round": r, "errors": errs, "cost": cost})
            print(tag, r, "FAILED", errs)
            break
        doc = new
        (d / f"round{r}.json").write_text(json.dumps(doc, indent=1))
        render(doc).save(d / f"round{r}_1x.png")
        render(doc, 8).save(d / f"round{r}_8x.png")
        compare_image(concept, regions, doc).save(d / f"round{r}_compare.png")
        log.append({"round": r, "errors": errs, "cost": round(cost, 4), "notes": doc.get("notes", "")})
        print(tag, r, f"${cost:.3f}", "errors:", errs, "|", doc.get("notes", "")[:200])
        stages.append(PM.stage(f"{tag}_r{r}", f"{tag} round {r}", "images",
                               [{"path": str(d / f"round{r}_8x.png"), "caption": f"8x ({len(doc['palette'])} colours)",
                                 "pixel": True}],
                               notes=[doc.get("notes", "")[:220], f"${cost:.3f}" + (f" | invalid: {errs}" if errs else "")]))
        write()
    if doc:
        render(doc).save(d / "final_16.png")
        render(doc, 8).save(d / "final_8x.png")
        stages.append(PM.stage(f"{tag}_final", f"{tag} final", "images",
                               [{"path": str(d / "final_16.png"), "caption": "1x", "pixel": True},
                                {"path": str(d / "final_8x.png"), "caption": "8x", "pixel": True}]))
        write()
    return doc, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="anthropic/claude-opus-5.5,openai/gpt-5.6-sol")
    ap.add_argument("--budget", type=float, default=1.5)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ledger = models.Ledger(a.budget)
    concept = Image.open(BOXY / "concept.png").convert("RGBA")
    regions, refs = regions_image(), refs_image()
    on_white(concept).save(OUT / "concept.png")
    regions.save(OUT / "regions_legend.png")
    refs.save(OUT / "style_refs.png")
    stages = [
        PM.stage("concept", "1 concept", "image", [{"path": str(OUT / "concept.png"), "caption": "compact, vestigial wing"}]),
        PM.stage("regions", "2 regions + priority", "image", [{"path": str(OUT / "regions_legend.png")}]),
        PM.stage("refs", "3 style refs", "image", [{"path": str(OUT / "style_refs.png"), "caption": ", ".join(l for _, l in REFS),
                                                    "pixel": True}]),
        PM.stage("guidance", "4 guidance", "list", data={"items": GUIDANCE}),
    ]
    state = {"verdict": None}

    def write():
        PM.write(OUT, "Snipe grid: LLM places pixels (no image gen)", stages, verdict=state["verdict"],
                 cost_usd=round(ledger.spent, 4))

    write()
    results = {}
    for m in a.models.split(","):
        tag = m.split("/")[-1].replace(".", "_")
        try:
            doc, log = run_model(m, tag, concept, regions, refs, ledger, stages, write)
        except models.BudgetExceeded as e:
            print("BUDGET", e)
            results[m] = {"stopped": str(e)}
            break
        results[m] = {"log": log, "final": str(OUT / tag / "final_16.png") if doc else None}
    results["cost_usd"] = round(ledger.spent, 4)
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    write()
    print("spent", ledger.spent)


if __name__ == "__main__":
    main()
