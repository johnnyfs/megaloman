"""Snipe bot, grid LADDER route (Johnny, 2026-10-24): the grid route stepped down 64 -> 32 -> 24 -> 20 -> 16.

Each size is placed symbolically by a vision LLM (Opus 5.5 via OpenRouter) as a palette-letter grid; from 32 down the
previous size's grid + render is given as a reference along with concept, regions+priority, style refs, guidance.
Lessons from the 16x16 grid run: eye capped (~3x3 at 16, proportional), colours from the concept (sampled hexes given,
not the region legend), the model sees its render at 1x as well as upscaled. 1 revision round per size.
Johnny (live): revisions can lose the shape, so each size keeps its BEST round (silhouette IoU vs the downscaled concept
mask; ties go to the earlier round), and the shape reference for the next size is the best-IoU earlier size, not just
the previous one. --rebuild re-derives finals/manifest from saved rounds without API calls.

uv run python -m megaloman.snipe_grid_ladder [--sizes 64,32,24,20,16] [--budget 1.5]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re

from PIL import Image, ImageDraw

from . import models, snipe, pipeline_manifest as PM
from .ladder import _font
from .snipe_boxy import PARTS
from .snipe_grid import BOXY, REFS, SYSTEM, on_white, png, regions_image, refs_image, img_part

OUT = snipe.OUT / "compact" / "grid-ladder"
MODEL = "anthropic/claude-opus-5.5"
ROUNDS, MAX_COLOURS = 2, 5


def eye_px(n):
    e = max(3, round(3 * n / 16))
    return e


def guidance(n):
    e = eye_px(n)
    return [
        f"Fill the square: the silhouette is compact and occupies most of the {n}x{n} in both width and height.",
        "Preserve the silhouette and the big shapes and lines of the concept; it must read from its outline alone.",
        "Head and body are clearly separated by a different colour AND a dark line between them.",
        f"The eye is bright and unmistakable (top priority) but CAPPED: its bright part is about {e}x{e} px "
        f"(never more than {e + 1}x{e + 1}), inside its dark ring.",
        f"The snub gun-nose is short and chunky (at least {max(2, round(2 * n / 16))} px thick), not a thin dash.",
        "The scope on top is distinct: its own little block, separated from the head by outline.",
        "Legs folded/tucked under the body; feet just under it. Tail stubby.",
        "Breastplate is just a hint (a few pixels or a shade), never a big panel.",
        "Few large colour masses: one body colour, one accent, a bright eye; no fine patterns, no noise.",
        f"Strong dark outline and value contrast between masses (NES style: <= {MAX_COLOURS} colours incl. outline).",
        "Use the CONCEPT's colours (sampled hexes given below), never the region-map legend colours.",
    ]


def concept_colours(concept, k=8):
    im = on_white(concept).convert("RGB").resize((128, 128), Image.LANCZOS).quantize(k, method=Image.MEDIANCUT)
    pal = im.getpalette()[:k * 3]
    counts = sorted(im.getcolors(), reverse=True)
    out = []
    for c, i in counts:
        r, g, b = pal[i * 3:i * 3 + 3]
        if min(r, g, b) > 235:
            continue  # background white
        out.append(f"#{r:02x}{g:02x}{b:02x} ({100 * c / (128 * 128):.0f}%)")
    return out


def fmt(n):
    return f"""Answer with ONE JSON object in a ```json fenced block, nothing after it:
{{"palette": {{"K": "#000000", "A": "#rrggbb", ...}},   // at most {MAX_COLOURS} colours INCLUDING the outline; single uppercase letters
 "grid": ["{'.' * n}", ... {n} strings],   // exactly {n} rows x {n} chars; '.' = transparent; other chars = palette letters
 "notes": "one or two sentences: what you kept, what you dropped or merged at this size, and why"}}"""


FIRST = """Draw this enemy as a {n}x{n} NES-style sprite (Mega Man style), placing every pixel yourself.

Image 1: the concept (a round robot "snipe" bird: gun-nose beak, scope on top, eye, folded side wing, small
vestigial wing nub up/right, tucked legs, stubby tail).
Image 2: the region map of the concept, with a legend. The legend order is the PRIORITY order (highest first):
{prio}. When pixels are scarce, higher-priority parts win; low-priority parts may shrink to a hint or vanish.
The region map's colours are only labels: do NOT use them as sprite colours.
Image 3: real NES Mega Man enemy sprites at ~16x16 (shown enlarged), for the target style: outline, colour count,
how features are abstracted.
{prevtxt}
Concept colours (sampled from image 1, by area): {cols}. Build your palette from these (NES-ish versions).

Guidance:
{guidance}

This is one rung of a resolution ladder (64 -> 32 -> 24 -> 20 -> 16): at each size keep everything that still fits,
and decide deliberately what to merge or drop. Think it through on a rough grid first, then give the final grid.

"""

PREVTXT = """Image 4: YOUR sprite from an earlier, larger rung ({p}x{p}; the one that best kept the concept's silhouette), rendered upscaled and at 1x. Use it as the
reference: reduce it to {n}x{n} keeping the same design, colours and proportions, simplifying where pixels run out.
Its grid:
```
{grid}
```
"""

REVISE = """Revision round {r} of {rounds} for the {n}x{n} rung. Your answer is below. Image 1 shows, left to right: the
concept, the region map, YOUR sprite upscaled (grid lines every pixel), and YOUR sprite at true 1x and 2x size (how it
will really be seen). Image 2 is the NES style reference again.{prevnote}

Compare honestly: does it read as this character, at 1x too? Check each priority part in order ({prio}) and the
guidance:
{guidance}
{issues}
Fix the biggest problems; keep what works. Your previous answer:
```json
{prev}
```

"""


def parse(text, n):
    m = re.findall(r"```json\s*(\{.*?\})\s*```", text, re.S) or re.findall(r"(\{\s*\"palette\".*\})", text, re.S)
    if not m:
        return None, ["no JSON block found"]
    try:
        doc = json.loads(m[-1])
    except Exception as e:
        return None, [f"JSON parse error: {e}"]
    return doc, validate(doc, n)


def validate(doc, n):
    errs = []
    pal, grid = doc.get("palette") or {}, doc.get("grid") or []
    if not isinstance(pal, dict) or not pal:
        return ["palette missing"]
    if len(pal) > MAX_COLOURS:
        errs.append(f"palette has {len(pal)} colours (max {MAX_COLOURS})")
    for k, v in pal.items():
        if len(k) != 1 or k == "." or not re.fullmatch(r"#[0-9a-fA-F]{6}", str(v)):
            errs.append(f"bad palette entry {k!r}: {v!r}")
    if len(grid) != n:
        errs.append(f"grid has {len(grid)} rows (need {n})")
    for i, row in enumerate(grid):
        if len(row) != n:
            errs.append(f"row {i} has {len(row)} chars (need {n})")
        bad = set(row) - set(pal) - {"."}
        if bad:
            errs.append(f"row {i} uses unknown letters {sorted(bad)}")
    return errs


def render(doc, n, scale=1, lines=False):
    pal = {k: tuple(int(v[i:i + 2], 16) for i in (1, 3, 5)) + (255,) for k, v in doc["palette"].items()}
    im = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    for y, row in enumerate(doc["grid"][:n]):
        for x, ch in enumerate(row[:n]):
            if ch in pal:
                im.putpixel((x, y), pal[ch])
    if scale == 1:
        return im
    big = on_white(im.resize((n * scale, n * scale), Image.NEAREST))
    if lines:
        d = ImageDraw.Draw(big)
        for i in range(n + 1):
            d.line([(i * scale, 0), (i * scale, n * scale)], fill=(200, 200, 200))
            d.line([(0, i * scale), (n * scale, i * scale)], fill=(200, 200, 200))
    return big


def up(n):  # upscale factor to ~512
    return max(1, 512 // n)


def sprite_card(doc, n, lines=True):
    """Upscaled render + true 1x and 2x thumbnails on white."""
    s = render(doc, n, up(n), lines=lines)
    one, two = on_white(render(doc, n)), render(doc, n, 2)
    W = s.width + 16 + max(two.width, 40) + 16
    im = Image.new("RGB", (W, s.height), (235, 235, 235))
    im.paste(s, (0, 0))
    im.paste(two, (s.width + 16, 10))
    im.paste(one, (s.width + 16, 20 + two.height))
    return im


def compare_image(concept, regions, doc, n):
    T = 512
    c = on_white(concept).resize((T, T), Image.LANCZOS)
    r = regions.resize((int(regions.width * T / regions.height), T), Image.LANCZOS)
    card = sprite_card(doc, n)
    W = c.width + r.width + card.width + 32
    im = Image.new("RGB", (W, max(T, card.height)), (235, 235, 235))
    x = 0
    for p in (c, r, card):
        im.paste(p, (x, 0))
        x += p.width + 16
    return im


def iou(doc, n, concept):
    m = on_white(concept).convert("RGB").resize((n, n), Image.BOX)
    cm = {(x, y) for x in range(n) for y in range(n) if min(m.getpixel((x, y))) < 235}
    sp = render(doc, n)
    sm = {(x, y) for x in range(n) for y in range(n) if sp.getpixel((x, y))[3] > 0}
    return len(cm & sm) / max(1, len(cm | sm))


def finish_size(n, docs, log, total, concept, stages):
    """docs: {round: doc}. Keep the best-IoU round (earliest on ties); write finals and the stage."""
    d = OUT / f"s{n}"
    scores = {r: round(iou(doc, n, concept), 3) for r, doc in docs.items()}
    best = max(scores, key=lambda r: (scores[r], -r))
    doc = docs[best]
    render(doc, n).save(d / f"final_{n}.png")
    render(doc, n, up(n)).save(d / "final_up.png")
    ims = [{"path": str(d / f"round{r}_up.png"), "caption": f"round {r} ({up(n)}x) IoU {scores[r]}"
            + (" KEPT" if r == best else ""), "pixel": True} for r in sorted(docs)]
    ims.append({"path": str(d / f"final_{n}.png"), "caption": f"final 1x (round {best})", "pixel": True})
    why = (f"kept round {best}: best silhouette IoU vs concept ({', '.join(f'r{r} {v}' for r, v in sorted(scores.items()))})"
           + ("; tie -> earlier round" if list(scores.values()).count(scores[best]) > 1 else ""))
    stages[:] = [s for s in stages if s["id"] != f"s{n}"]
    stages.append(PM.stage(f"s{n}", f"{n}x{n}", "images", ims,
                           notes=[why] + [l.get("notes", "")[:240] for l in log if l.get("notes")]
                           + [f"${total:.3f}"] + [f"invalid r{l['round']}: {l['errors'][:3]}" for l in log if l.get("errors")]))
    return doc, scores[best], best


def call(content, ledger, key):
    body = {"model": MODEL, "max_tokens": 24000, "reasoning": {"max_tokens": 8000},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]}
    ck = hashlib.sha256(("snipe_gridladder1|" + key + json.dumps(body)).encode()).hexdigest()
    resp, _, cost, cached = models.call(body, ledger, ck)
    return resp["choices"][0]["message"].get("content") or "", cost


def run_size(n, prev, concept, regions, refs, cols, ledger, stages, write):
    prio = " > ".join(p for p, _, _ in PARTS)
    g = "\n".join(f"- {x}" for x in guidance(n))
    d = OUT / f"s{n}"
    d.mkdir(parents=True, exist_ok=True)
    doc, log, total, docs = None, [], 0.0, {}
    for r in range(1, ROUNDS + 1):
        if r == 1:
            pt = PREVTXT.format(p=prev[0], n=n, grid="\n".join(prev[1]["grid"])) if prev else ""
            content = [{"type": "text", "text": FIRST.format(n=n, prio=prio, guidance=g, cols=", ".join(cols),
                                                             prevtxt=pt) + fmt(n)},
                       img_part(on_white(concept).resize((768, 768), Image.LANCZOS)), img_part(regions), img_part(refs)]
            if prev:
                content.append(img_part(sprite_card(prev[1], prev[0], lines=False)))
        else:
            issues = "" if not log[-1]["errors"] else "Validation problems last round: " + "; ".join(log[-1]["errors"])
            pn = f" Image 3 is your previous {prev[0]}x{prev[0]} rung, for consistency." if prev else ""
            content = [{"type": "text", "text": REVISE.format(r=r, rounds=ROUNDS, n=n, prio=prio, guidance=g, issues=issues,
                                                              prevnote=pn, prev=json.dumps(doc, indent=0)) + fmt(n)},
                       img_part(compare_image(concept, regions, doc, n)), img_part(refs)]
            if prev:
                content.append(img_part(sprite_card(prev[1], prev[0], lines=False)))
        new, errs, cost = None, ["not run"], 0.0
        for attempt in range(2):
            text, c = call(content, ledger, f"{n}|r{r}|a{attempt}")
            cost += c
            new, errs = parse(text, n)
            (d / f"round{r}_a{attempt}.txt").write_text(text)
            if new is not None and not errs:
                break
            content = content + [{"type": "text", "text": "Your answer failed validation: " + "; ".join(errs)
                                  + ". Answer again, fully, in the required format."}]
        total += cost
        if new is None or errs:
            log.append({"round": r, "errors": errs, "cost": round(cost, 4)})
            print(n, r, "INVALID", errs)
            if new is None or len(new.get("grid") or []) != n:
                break
        doc = docs[r] = new
        (d / f"round{r}.json").write_text(json.dumps(doc, indent=1))
        render(doc, n).save(d / f"round{r}_1x.png")
        render(doc, n, up(n)).save(d / f"round{r}_up.png")
        compare_image(concept, regions, doc, n).save(d / f"round{r}_compare.png")
        if not log or log[-1].get("round") != r:
            log.append({"round": r, "errors": errs, "cost": round(cost, 4), "notes": doc.get("notes", "")})
        print(n, r, f"${cost:.3f}", errs, "|", doc.get("notes", "")[:200])
        write()
    best = None
    if docs:
        doc, score, best = finish_size(n, docs, log, total, concept, stages)
        write()
    return (doc, score, best) if docs else (None, 0, None), log, total


def ladder_sheet(finals):
    """All final sizes side by side, each scaled to the same display height (nearest), plus 1x row."""
    H = 256
    cells = [(n, render(doc, n, max(1, H // n))) for n, doc in finals]
    pad, f = 24, _font(22)
    W = sum(c.width for _, c in cells) + pad * (len(cells) + 1)
    sheet = Image.new("RGB", (W, H + 110), "white")
    d, x = ImageDraw.Draw(sheet), pad
    for (n, c), (_, doc) in zip(cells, finals):
        sheet.paste(c, (x, 10))
        one = render(doc, n)
        sheet.paste(one, (x, H + 20), one)
        d.text((x, H + 80), f"{n}x{n}", fill="black", font=f)
        x += c.width + pad
    return sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="64,32,24,20,16")
    ap.add_argument("--budget", type=float, default=1.5)
    ap.add_argument("--rebuild", action="store_true", help="no API calls: re-pick best rounds from saved JSON")
    a = ap.parse_args()
    sizes = [int(s) for s in a.sizes.split(",")]
    OUT.mkdir(parents=True, exist_ok=True)
    ledger = models.Ledger(a.budget)
    concept = Image.open(BOXY / "concept.png").convert("RGBA")
    regions, refs = regions_image(), refs_image()
    cols = concept_colours(concept)
    on_white(concept).save(OUT / "concept.png")
    regions.save(OUT / "regions_legend.png")
    refs.save(OUT / "style_refs.png")
    stages = [
        PM.stage("concept", "1 concept", "image", [{"path": str(OUT / "concept.png"), "caption": "compact, vestigial wing"}],
                 notes=["concept colours: " + ", ".join(cols)]),
        PM.stage("regions", "2 regions + priority", "image", [{"path": str(OUT / "regions_legend.png")}]),
        PM.stage("refs", "3 style refs", "image", [{"path": str(OUT / "style_refs.png"),
                                                    "caption": ", ".join(l for _, l in REFS), "pixel": True}]),
        PM.stage("guidance", "4 guidance (at 16; eye cap scales)", "list", data={"items": guidance(16)}),
    ]
    state = {"verdict": None}
    ladder_pos = len(stages)

    def write():
        PM.write(OUT, f"Snipe grid ladder: Opus 5.5 places pixels {'->'.join(map(str, sizes))}", stages,
                 verdict=state["verdict"], cost_usd=round(ledger.spent, 4))

    write()
    if a.rebuild:
        old = json.loads((OUT / "results.json").read_text()) if (OUT / "results.json").exists() else {}
    results, finals, prev, best_ref = {"model": MODEL, "sizes": sizes}, [], None, None
    for n in sizes:
        try:
            if a.rebuild:
                docs = {r: json.loads((OUT / f"s{n}" / f"round{r}.json").read_text()) for r in range(1, ROUNDS + 1)
                        if (OUT / f"s{n}" / f"round{r}.json").exists()}
                if not docs:
                    break
                log = (old.get(str(n)) or {}).get("log", [])
                cost = (old.get(str(n)) or {}).get("cost", 0.0)
                doc, score, kept = finish_size(n, docs, log, cost, concept, stages)
            else:
                (doc, score, kept), log, cost = run_size(n, prev, concept, regions, refs, cols, ledger, stages, write)
        except models.BudgetExceeded as e:
            print("BUDGET", e)
            results["stopped"] = str(e)
            break
        results[str(n)] = {"log": log, "cost": round(cost, 4), "kept_round": kept, "iou": score}
        if doc:
            finals.append((n, doc))
            if best_ref is None or score >= best_ref[2]:
                best_ref = (n, doc, score)
            prev = best_ref[:2]  # shape reference: best-preserving earlier size
            ladder_sheet(finals).save(OUT / "ladder.png")
            lad = PM.stage("ladder", "ladder: finals side by side", "images",
                           [{"path": str(OUT / f"s{m}" / f"final_{m}.png"), "caption": f"{m}x{m}", "pixel": True}
                            for m, _ in finals] + [{"path": str(OUT / "ladder.png"), "caption": "same display height + 1x"}])
            stages[:] = [s for s in stages if s["id"] != "ladder"]
            stages.insert(ladder_pos, lad)
            write()
    if a.rebuild:
        results["stopped"] = old.get("stopped")
        results["cost_usd"] = old.get("cost_usd", 0.0)
        ledger.spent = results["cost_usd"]
    else:
        results["cost_usd"] = round(ledger.spent, 4)
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    write()
    print("spent", ledger.spent)


if __name__ == "__main__":
    main()
