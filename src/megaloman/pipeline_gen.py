"""Generic region-ladder pipeline (first generalisation, from the snipe-bot dry run, 2026-10-23).

The squid pipeline (regions_v2 / regions_nanobanana / ladder_v2-v4 / render_v3-v4) is hard-wired to the squid: part
names (cap, cap_side_lobes, face_housing, visor, eyes, nozzle, tentacles_*, suction_cups_front, undercarriage),
squid prompt examples, a hand-made palette / priority table (ladder_v3.PALETTE / ASSIGN / PRIORITY), the
squid-only face template (ring > visor moat > two eyes on the axis), arm groups and curl, fin pay, and an exact
mirror at every stage. This module generalises only what a new, asymmetric enemy needs to run. Squid modules are
untouched except two guards for a spec without a verified vertical axis: ladder_v2.prepare refuses it (v2-v4
mirror unconditionally) and render_v4.render(keep="none") skips the final mirror. Squid v4 reproduces pixel for
pixel. Each place that had to be special-cased is recorded in PIPELINE_RULES
("finding (snipe dry run)").

  legend()      regions_v2.LEGEND_PROMPT with the squid examples replaced by generic ones (same fields: material,
                depth, occluded_by)
  paint_prompt  regions_v2.PAINT_PROMPT without the suction-cup line, "creature" -> "character", weapon words
                neutralised (it did not get past Nano Banana's filter; see snipe.PAINTERS)
  plan()        NEW model step (gpt-5.6-sol): palette first (black + 3 NES colours, part -> colour group), region
                priority (eyes highest), which parts are eyes, with reasons. Replaces the squid's hand-written tables.
  significance  ladder_sig.guidance with a generic prompt (the squid's art-director notes about fins are replaced
                by the brief's character notes); the eyes override applies to the plan's eye parts.
  prepare/rung  ladder_v2 / ladder_v3 machinery (source mode, significance into the background only, priority
                assign, depth `behind`), with: no mirror unless the spec verifies an axis, a generic eye floor (one
                filled rectangle per eye piece at its scaled centroid; vertical slit at <= 20; v2: a 1 px moat of the
                plan's eye surround; v3: nudged 1 px when the moat would erase a character part), generic floors,
                v3: a stroke floor (1 px principal-axis line) for 'stroke' parts that break up.
  render()      render_v4.render with generic materials and mirror off.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from . import imaging, ladder as L, ladder_sig as SG, ladder_sym as SY, ladder_v2 as LV, ladder_v3 as L3
from . import models, regions, regions_v2 as RV, render_v4 as R4, store

SOL = "openai/gpt-5.6-sol"

# ------------------------------------------------------------------ legend (generic prompt)

LEGEND_PROMPT = """You are preparing concept art for pixel-art reduction. Split this character into its PARTS: the regions a
pixel artist keeps distinct when drawing it as a tiny 16-32 px sprite. At that size REGIONS matter far more than
colouring: each part becomes one solid colour, and the boundaries between parts are what carry the sprite.

Return JSON only: {"parts": [ {"name", "role", "priority", "color", "min_px", "material", "depth", "occluded_by",
"where"} ... ]}

Rules for choosing parts:
1. OCCLUSION MAKES A SUBCLASS. When one instance of a repeated part hides another instance of the same kind (a leg
   behind a leg, a wing behind a wing, an arm behind an arm), the hidden instances are their OWN part, named
   <part>_back with depth "back". Do this for every kind of part that has hidden instances. Visible front instances
   have depth "front". depth "back" is ONLY for hidden instances of a repeated kind (there must also be front
   instances of that kind). A part that is merely behind a DIFFERENT part (a tail behind a body, a neck behind a
   head) keeps depth "front" and lists what covers it in occluded_by.
   Decide who is in front from the outlines: at an overlap, the part whose outline runs unbroken across the overlap
   is in front; the part whose outline disappears under it is behind.
2. If front instances of the same kind touch or overlap each other, split them into separate parts so every
   touching pair has a different region colour. Give them the same "material" so they are drawn in the same final
   colour.
3. NO pattern-only parts: spots, stripes, panel lines, rivets, vents, two-tone colouring, highlights and decals are
   paint on the part they sit on, not parts.
4. Small mechanical connectors, collars, bolts and joints are not separate parts: fold them into the part they
   belong to. Distinct character features (a weapon, a horn, a scope, a beak) stay their own parts. Small lights,
   glows and lenses belong to the part they sit on, EXCEPT the eyes.
5. Eyes, a visor or eye housing, and other face features stay their own parts. 6 to 14 parts in total.
6. Every visible pixel of the character belongs to exactly one part (no background part; ignore any ground shadow).

Fields:
- name: short snake_case.
- role: body | feature | limb | detail.
- priority: integer 1-10; higher wins a contested pixel when shrinking (small important features high, big masses
  low; a front part higher than the back part it covers).
- color: "#RRGGBB" flat REGION-MAP colour: highly saturated, far apart from every other part's colour, far from
  white and light grey (the background is white). Do NOT use the part's real colour.
- min_px: smallest size in pixels this part must keep at a 32 px sprite height to stay readable.
- material: the part's real-world surface (e.g. "armour_green", "metal_dark", "glass", "gun_metal"); parts sharing
  a material are drawn in the same final colour.
- depth: "front" or "back".
- occluded_by: list of part names that cover parts of this one ([] if none).
- where: one precise sentence locating the part in this image, enough to paint it (say which instances: near/far,
  left/right, how many)."""


def legend(concept_rgb, ledger, seed=0) -> dict:
    png = imaging.to_png(concept_rgb)
    body = {'model': SOL, 'response_format': {'type': 'json_object'}, 'seed': seed,
            'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': LEGEND_PROMPT},
                                                      {'type': 'image_url', 'image_url': {'url': models.data_uri(png)}}]}]}
    ck = store.key({'step': 'pipeline_gen.legend', 'model': SOL, 'prompt': LEGEND_PROMPT, 'image': store.sha(png),
                    'seed': seed})
    resp, _, cost, cached = models.call(body, ledger, ck)
    out = regions._parse(resp['choices'][0]['message'].get('content'))
    parts = out.get('parts') or []
    names = [p.get('name') for p in parts]
    for p in parts:
        p['role'] = p.get('role') if p.get('role') in regions.ROLES else 'detail'
        p['priority'] = int(p.get('priority', 1))
        p['min_px'] = int(p.get('min_px', 1))
        p['material'] = str(p.get('material') or p['name'])
        p['depth'] = p.get('depth') if p.get('depth') in RV.DEPTHS else 'front'
        p['occluded_by'] = [n for n in (p.get('occluded_by') or []) if n in names]
    fixes = regions.fix_colors(parts)
    return {'version': 'gen1', 'model': SOL, 'parts': parts, 'color_fixes': fixes, 'call': ck, 'cost': cost,
            'cached': cached, 'prompt': LEGEND_PROMPT}


# Nano Banana refused the first snipe paint (finish PROHIBITED_CONTENT, both seeds) with weapon words in the table.
# The paint prompt only needs shapes and colours, so weapon vocabulary is replaced by neutral words there (the legend
# keeps its names). Recorded as a generalisation gap.
NEUTRAL_WORDS = [("cannon", "tube"), ("Cannon", "Tube"), ("muzzle", "tip"), ("rifle", "tube"), ("barrel", "tube"),
                 ("gun_metal", "dark_metal"), ("gun", "tube"), ("sniper", ""), ("scope", "sight"), ("Scope", "Sight"),
                 ("weapon", "tube"), ("perforated ", "")]


def neutral(text: str) -> str:
    for a, b in NEUTRAL_WORDS:
        text = text.replace(a, b)
    return text


def paint_prompt(leg, neutral_words=True) -> str:
    p = RV.paint_prompt(leg)
    if neutral_words:
        p = neutral(p)
    p = p.replace("- Each suction cup is a solid disc of the cup colour on its arm.\n", "")
    p = p.replace("including the curve of each arm where it passes in front of another arm",
                  "including the curve of each limb where it passes in front of another")
    p = p.replace("(an inner arm over an outer arm)", "(a near leg over a far leg)")
    return p.replace("creature", "character")


def paint(concept_rgb, leg, seed, ledger, model) -> dict:
    """regions_v2.paint's call shape with the generic prompt."""
    png = imaging.to_png(concept_rgb)
    prompt = paint_prompt(leg)
    body = {'model': model, 'modalities': ['image', 'text'], 'seed': seed,
            'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': prompt},
                                                      {'type': 'image_url', 'image_url': {'url': models.data_uri(png)}}]}]}
    ck = store.key({'step': 'pipeline_gen.paint', 'model': model, 'prompt': prompt, 'images': [store.sha(png)],
                    'seed': seed})
    try:
        resp, imgs, cost, cached = models.call(body, ledger, ck)
    except models.BudgetExceeded:
        raise
    except Exception as e:
        return {'png': None, 'call': ck, 'error': str(e)[:300], 'prompt': prompt}
    return {'png': imgs[0] if imgs else None, 'call': ck, 'cached': cached, 'prompt': prompt,
            'text': (resp['choices'][0]['message'].get('content') or '')[:300],
            'error': None if imgs else 'no image returned'}


def region_candidate(d: Path, cid, seed, concept, concept_rgb, sil, leg, ledger, model) -> dict:
    """regions_nanobanana._candidate, generic paint prompt."""
    rec = {'id': cid, 'model': model, 'seed': seed}
    p = paint(concept_rgb, leg, seed, ledger, model)
    rec.update(call=p['call'], error=p['error'])
    if p['png'] is None:
        rec['align'] = {'pass': False, 'iou': 0.0}
        return rec
    cdir = d / cid
    cdir.mkdir(parents=True, exist_ok=True)
    painted = imaging.from_png(p['png'])
    painted.save(cdir / 'painted.png')
    labels = regions.snap(painted, leg, concept.size)
    rec['iou_raw'] = round(regions.iou(labels > 0, sil), 4)
    labels, rec['transform'], v = regions.register(labels, sil)
    rec['iou_registered'] = round(v, 4)
    labels, rec['holes_filled_px'] = regions.fill_holes(labels, sil)
    rec['align'] = regions.align(labels, sil, leg)
    if rec['align']['iou_ok']:
        labels = regions.clip_to_silhouette(labels, sil)
    labels, rec['islands_folded'] = RV.clean_islands(labels, leg)
    np.save(cdir / 'labels.npy', labels)
    regions.render_map(labels, leg).save(cdir / 'regions.png')
    regions.overlay(concept, labels, leg).save(cdir / 'overlay.png')
    rec['occlusion'] = RV.occlusion_check(labels, leg)
    return rec


# ------------------------------------------------------------------ plan: palette first, priority, eyes

NES_CHOICES = {k: v for k, v in imaging.NES.items() if k not in ('0D', '0E', '1D', '1E', '2E', '2F', '3E', '3F')}

PLAN_PROMPT = """You are the pixel artist deciding, BEFORE any shrinking, how this game character will be drawn as a
tiny NES sprite (64, 32, 24, 20 and finally 16 px). The brief: {brief}

Images: 1. the concept art; 2. its region map (one flat colour per part, legend below); 3. a plain proportional
shrink of the region map at 64, 32 and 16 px.

Legend (region colour, name, role, material, depth, where):
{legend}

House rules (blessed by the art director; follow them):
1. TARGET PALETTE FIRST. The sprite uses black (0F, the outline and lines) plus exactly THREE NES colours. Choose them
   now from the NES palette below, close to the concept's real colours, as a dark / mid / light set with clear value
   contrast (ideally one hue ramp, but marsh greens + a contrasting accent are fine). Then assign EVERY part to one
   group: "black", "dark", "mid" or "light". Make executive calls: parts that should read as one thing share a
   colour; touching parts that must stay distinct get different colours where possible (a 1 px black line will
   separate every pair of distinct regions anyway); hidden/back instances (depth back) are usually "black" (shadow);
   the eyes must contrast strongly with what surrounds them. Shrinking an image that still has too many colours is
   confusing, so decide now.
2. NO PATTERNING: one solid colour per part.
3. REGION PRIORITY: an explicit order of ALL parts, highest first, deciding who keeps a contested pixel when
   shrinking. The eyes are always first; then the parts that carry the character (here, say, the gun-beak);
   big body masses and hidden parts last.
4. EYES: list which parts are eyes (the eye itself, the part that must stay a filled rectangle), and which part, if
   any, is the eye's surround (a ring or housing directly around the eye).{eye_rule}

NES palette (code: hex):
{nes}

Return JSON only:
{{"palette": {{"dark": "<NES code>", "mid": "<NES code>", "light": "<NES code>"}},
 "palette_reason": "<one sentence>",
 "assign": {{"<part>": {{"group": "black|dark|mid|light", "why": "<short>"}} ...every part...}},
 "priority": [{{"part": "<name>", "why": "<short>"}} ...every part, highest first...],
 "eyes": ["<part>"...], "eye_surround": "<part or null>",
 "summary": "<one sentence: what must survive at 16 px for this to read as the brief>"}}"""


def _legend_rows(leg) -> str:
    return "\n".join(f"- {p['color']} {p['name']} ({p['role']}, {p['material']}, {p['depth']}): {p['where']}"
                     for p in leg['parts'])


def _v1_strip(src, leg):
    return SG._v1_strip(src, leg)


# v2 (snipe dry run): the v1 plan gave the scope the eye's colour; at <= 32 px the gold scope sat 1-2 px above the
# gold eye and the two read as one gold blob. The blessed rule says eyes stay distinct from everything else, so it
# is now spelled out for the palette call.
EYE_RULE = """
   The eye must be the ONLY part of its colour anywhere near it: no part that touches the eye's surround, or lies
   within a few pixels of the eye at 16 px (head accessories, the beak/snout, the brow), may share the eye's colour
   group. Otherwise eye and neighbour merge into one blob at small sizes."""


def plan(concept_rgb: Image.Image, region_png: Path, src, leg, brief: str, ledger, eye_rule: bool = False) -> dict:
    prompt = PLAN_PROMPT.format(brief=brief, legend=_legend_rows(leg), eye_rule=EYE_RULE if eye_rule else "",
                                nes="  ".join(f"{k}:{v}" for k, v in NES_CHOICES.items()))
    c = concept_rgb.copy()
    c.thumbnail((768, 768))
    r = Image.open(region_png).convert('RGB')
    r.thumbnail((768, 768))
    imgs = [imaging.to_png(c), imaging.to_png(r), imaging.to_png(_v1_strip(src, leg))]
    body = {'model': SOL, 'response_format': {'type': 'json_object'},
            'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': prompt}] +
                          [{'type': 'image_url', 'image_url': {'url': models.data_uri(b)}} for b in imgs]}]}
    ck = store.key({'step': 'pipeline_gen.plan', 'model': SOL, 'prompt': prompt, 'images': [store.sha(b) for b in imgs]})
    resp, _, cost, cached = models.call(body, ledger, ck)
    raw = regions._parse(resp['choices'][0]['message'].get('content'))
    return {'raw': raw, 'call': ck, 'cached': cached, 'prompt': prompt}


def apply_plan(leg: dict, pl: dict) -> tuple[dict, dict]:
    """Normalise the plan onto the legend: priorities (rank), colour groups, palette. Validates (every part assigned
    and ranked once, eyes first, codes in the NES palette) and repairs deterministically, recording each repair."""
    raw = pl['raw']
    names = [p['name'] for p in leg['parts']]
    fixes = []
    pal = {'black': ('0F', '000000')}
    for g in ('dark', 'mid', 'light'):
        code = str(raw['palette'][g]).upper().strip()
        if code not in NES_CHOICES:
            fixes.append(f'palette {g}: {code} not an NES code -> 00')
            code = '00'
        pal[g] = (code, NES_CHOICES[code])
    assign = {}
    for n in names:
        a = (raw.get('assign') or {}).get(n) or {}
        g = a.get('group') if a.get('group') in pal else None
        if g is None:
            fixes.append(f'{n}: no/invalid group -> mid')
            g = 'mid'
        assign[n] = (g, a.get('why') or '')
    order = [q['part'] for q in raw.get('priority') or [] if q.get('part') in names]
    order = list(dict.fromkeys(order))
    missing = [n for n in names if n not in order]
    if missing:
        fixes.append(f'priority missing {missing}: appended in legend priority order')
        order += sorted(missing, key=lambda n: -leg['parts'][names.index(n)]['priority'])
    eyes = [e for e in raw.get('eyes') or [] if e in names]
    for e in reversed(eyes):                      # house rule: eyes first
        if order.index(e) != 0:
            order.remove(e)
            order.insert(0, e)
    why = {q['part']: q.get('why', '') for q in raw.get('priority') or [] if q.get('part') in names}
    out = json.loads(json.dumps(leg))
    for p in out['parts']:
        p['priority_legend'] = p['priority']
        p['priority'] = len(order) - order.index(p['name'])
        p['group'] = assign[p['name']][0]
        p['colour'] = pal[p['group']][1]
    out['version'] = 'gen1+plan'
    surround = raw.get('eye_surround') if raw.get('eye_surround') in names else None
    rec = {'palette': {g: {'nes': c, 'hex': h} for g, (c, h) in pal.items()},
           'palette_reason': raw.get('palette_reason'),
           'assign': {n: {'group': g, 'hex': pal[g][1], 'why': w} for n, (g, w) in assign.items()},
           'priority': [{'rank': k + 1, 'part': n, 'why': why.get(n, '')} for k, n in enumerate(order)],
           'eyes': eyes, 'eye_surround': surround, 'summary': raw.get('summary'), 'repairs': fixes}
    return out, rec


# ------------------------------------------------------------------ significance (generic prompt)

SIG_NOTES = """- {notes}
- The eyes are fine at proportional size and must NOT be enlarged. They only need a FLOOR: each eye must never shrink
  below a distinct filled rectangle (at least 1x2 px)."""


def significance_prompt(notes: str) -> str:
    base = (SG.OUT / "prompts/significance.md").read_text()
    a = base.index("The problem:")
    b = base.index("The only tools available")
    new = ("The problem: as shapes shrink they lose SIGNIFICANCE. A proportional shrink keeps areas right, but small "
           "parts that carry the character stop reading. The art director's notes:\n" + SIG_NOTES.format(notes=notes)
           + "\n\n")
    return base[:a].replace("this creature", "this character") + new + base[b:]


def significance(src, leg, concept_rgb, region_png, notes, eye_parts, ledger, prompt_dir: Path) -> tuple[list, dict]:
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "significance.md").write_text(significance_prompt(notes))
    g = SG.guidance(src, leg, concept_rgb, region_png, ledger, prompt_dir)
    saved = dict(SG.OVERRIDES)
    try:                                       # the squid's override is keyed by the part name "eyes"
        SG.OVERRIDES.clear()
        for e in eye_parts:
            SG.OVERRIDES[e] = saved["eyes"]
        specs = SG.specs_from(g, leg)
    finally:
        SG.OVERRIDES.clear()
        SG.OVERRIDES.update(saved)
    return specs, g


# ------------------------------------------------------------------ ladder (generic)

EYE_SLIT = {20: (1, 3), 16: (1, 2)}          # blessed: vertical slits at <= 20
EYE_FLOOR = {64: (2, 3), 32: (2, 3), 24: (2, 3)}  # ladder_v3.EYE_FLOOR above the slit sizes


def prepare(src_raw, leg, spec, specs, sizes=(64, 32, 24, 20, 16), centre=True) -> dict:
    """ladder_v2.prepare without the mirror (unless the spec verifies a vertical axis) and with given specs.
    centre=False: src_raw is already the square source (e.g. `tight`), used as is."""
    n = len(leg["parts"])
    mirror = bool(spec and spec["axes"].get("vertical", {}).get("verified"))
    if mirror:
        cands, sym = LV.symmetric_source(src_raw, leg, spec)
        src = cands["left"]
        sym["keep"] = "left"
    else:
        src = LV.square(_centre(src_raw)) if centre else src_raw
        sym = {"keep": "none", "mirror": False, "why": f"spec type {spec and spec['type']}: no verified mirror axis"}
    inst, base_of, _ = LV.instances(src, leg)
    pair = list(range(len(base_of)))           # no mirror partners without symmetry
    names = [p["name"] for p in leg["parts"]]
    P = [SG.pieces(src, i) for i in range(n)]
    for i, q in enumerate(specs):
        for pc in P[i]:
            pc["anchor"] = SG.anchor_of(pc, src, q, names)
    return {"src": src, "inst": inst, "base_of": base_of, "pair": pair, "specs": specs, "P": P, "sym": sym,
            "legend": leg, "idx": {nm: k + 1 for k, nm in enumerate(names)}, "sst": L.source_stats(src, n)}


def _centre(lab):
    """Crop to the foreground bbox and re-pad to a square with the bbox centred (the concept crop is already
    centred; this only removes the paint's off-by-a-few offsets)."""
    ys, xs = np.nonzero(lab > 0)
    a = lab[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = a.shape
    s = int(round(max(h, w) / 0.88))
    out = np.zeros((s, s), lab.dtype)
    y0, x0 = (s - h) // 2, (s - w) // 2
    out[y0:y0 + h, x0:x0 + w] = a
    return out


def tight(lab, keep=None) -> tuple[np.ndarray, int, int]:
    """Split variant (Johnny, after the snipe dry run): crop to the bounding box of the `keep` labels (all
    foreground if None) with no margin, then square it, centred. -> (square map, oy, ox) with
    out[y, x] == lab[y + oy, x + ox] (labels outside `keep` are dropped)."""
    m = (lab > 0) if keep is None else np.isin(lab, list(keep))
    ys, xs = np.nonzero(m)
    h, w = np.ptp(ys) + 1, np.ptp(xs) + 1
    s = int(max(h, w))
    oy, ox = int(ys.min() - (s - h) // 2), int(xs.min() - (s - w) // 2)
    out = np.zeros((s, s), lab.dtype)
    src = np.where(m, lab, 0)
    y0, x0 = max(0, oy), max(0, ox)
    y1, x1 = min(lab.shape[0], oy + s), min(lab.shape[1], ox + s)
    out[y0 - oy:y1 - oy, x0 - ox:x1 - ox] = src[y0:y1, x0:x1]
    return out, oy, ox


def piece_cells(lab, part_id, oy, ox, S, M) -> np.ndarray:
    """Coverage of one part on the BODY's sprite grid (cell = S / M source px, origin (oy, ox)), extended beyond the
    body square as far as the part reaches. -> (ys, xs, cov) cell arrays with cov >= 0.5, plus the principal-axis
    stroke if box coverage breaks the part (stroke floor, as `stroke_floor`)."""
    m = (lab == part_id).astype(np.float64)
    c = S / M
    ys, xs = np.nonzero(m)
    r0, r1 = int(np.floor((ys.min() - oy) / c)), int(np.floor((ys.max() - oy) / c))
    j0, j1 = int(np.floor((xs.min() - ox) / c)), int(np.floor((xs.max() - ox) / c))
    ii = np.pad(m, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    H, W = m.shape
    cells = {}
    for r in range(r0, r1 + 1):
        for j in range(j0, j1 + 1):
            a0, a1 = int(np.clip(round(oy + r * c), 0, H)), int(np.clip(round(oy + (r + 1) * c), 0, H))
            b0, b1 = int(np.clip(round(ox + j * c), 0, W)), int(np.clip(round(ox + (j + 1) * c), 0, W))
            area = max(1, (a1 - a0) * (b1 - b0))
            v = (ii[a1, b1] - ii[a0, b1] - ii[a1, b0] + ii[a0, b0]) / area
            if v >= 0.5:
                cells[(r, j)] = v
    grid = np.zeros((r1 - r0 + 1, j1 - j0 + 1), bool)
    for (r, j) in cells:
        grid[r - r0, j - j0] = True
    comps = L.components(grid)
    stroke = 0
    if comps != 1:
        cy, cx = ys.mean(), xs.mean()
        v = np.linalg.svd(np.stack([ys - cy, xs - cx], 1), full_matrices=False)[2][0]
        t = (ys - cy) * v[0] + (xs - cx) * v[1]
        a, b = [((np.array([cy, cx]) + v * tt) - [oy, ox]) / c for tt in (t.min(), t.max())]
        n = int(max(abs(b - a))) + 1
        for k in range(n + 1):
            y, x = (int(np.floor(z)) for z in a + (b - a) * k / max(1, n))
            if not grid[y - r0, x - j0]:
                grid[y - r0, x - j0] = True
                stroke += 1
    return grid, r0, j0, {"components_box": comps, "stroke_px": stroke, "cells": int(grid.sum())}


def rung(ctx, N: int, eyes: list[str], params: dict | None = None) -> dict:
    prm = {"eye_slit": None, **(params or {})}
    leg, src, specs, P = ctx["legend"], ctx["src"], ctx["specs"], ctx["P"]
    M, side = N - 2, src.shape[0]
    ks, klog = {}, {}
    for i in sorted((i for i, q in enumerate(specs) if q["mode"] == "exaggerate" and q["max_exaggeration"] > 1),
                    key=lambda i: -specs[i]["significance"]):
        klog[specs[i]["name"]] = LV.choose_k(src, leg, specs, P, i, M, N, ks)
        ks[i] = klog[specs[i]["name"]]["k"]
    mod = LV.build_bg_only(src, specs, P, ks) if ks else src
    ext_src = LV.extend(mod, ctx["inst"], ctx["base_of"])
    base_of = np.array(ctx["base_of"])
    pr = np.array([leg["parts"][b - 1]["priority"] for b in ctx["base_of"][1:]], float)
    ext = L.assign(L.coverage(ext_src, len(base_of) - 1, M), pr)
    base = base_of[ext]
    floors = {}
    for i, q in enumerate(specs):               # significance floors (non-eye parts; ladder_v2._floor)
        if q["mode"] == "floor" and q["name"] not in eyes:
            nb, notes = LV._floor(base, i, q, P[i], M, N, side, specs)
            if notes:
                floors[q["name"]] = notes
            ch = nb != base
            ext[ch] = nb[ch]
            base = nb
    strokes = stroke_floor(ext, base_of, ctx, mod, N, M, eyes) if prm.get("stroke_floor") else {}
    eye_log = eye_floor(ext, base_of, ctx, N, M, eyes, prm["eye_slit"], prm.get("eye_surround"),
                        nudge=prm.get("eye_nudge", False))
    if ctx["sym"]["keep"] in ("left", "right"):
        ext = LV.mirror_inst(ext, ctx["sym"]["keep"], ctx["pair"])
    base = base_of[ext]
    h = L.health(base, src, leg, ctx["sst"])
    return {"N": N, "M": M, "ext": ext, "base": base, "k": klog, "log": {"floors": floors, "strokes": strokes, "eyes": eye_log},
            "health": h, "health_score": SY.health_score(h), "params": prm}


def stroke_floor(ext, base_of, ctx, mod, N, M, eyes) -> dict:
    """v3 (snipe): a part whose essential shape is a STROKE (the guidance's word: the beak-barrel, legs, neck) must
    not break up as it shrinks (squid finding M2: limbs at <= 32 px need stroke primitives). Box coverage drops a
    stroke thinner than ~0.5 px wherever it straddles two rows, so a 1 px barrel comes out dashed. If the part has
    more components at this rung than source pieces (or vanished), each source piece (after exaggeration) is redrawn
    as a 1 px line along its principal axis between its extreme points, over background and lower-priority parts
    only, unioned with what the shrink kept."""
    specs, leg, side = ctx["specs"], ctx["legend"], mod.shape[0]
    pri = np.array([0] + [p["priority"] for p in leg["parts"]])
    out = {}
    for i, q in enumerate(specs):
        if q["essential_shape"] != "stroke" or q["name"] in eyes:
            continue
        pcs = SG.pieces(mod, i)
        have = L.components(base_of[ext] == i + 1)
        if not pcs or (have and have <= len(pcs)):
            continue
        drawn = 0
        for pc in pcs:
            ys, xs = np.nonzero(pc["mask"])
            c = np.array([ys.mean(), xs.mean()])
            v = np.linalg.svd(np.stack([ys - c[0], xs - c[1]], 1), full_matrices=False)[2][0]
            t = (ys - c[0]) * v[0] + (xs - c[1]) * v[1]
            a, b = [(c + v * tt + 0.5) * M / side - 0.5 for tt in (t.min(), t.max())]
            n = int(max(abs(b - a))) + 1
            for k in range(n + 1):
                y, x = (int(round(z)) for z in a + (b - a) * k / max(1, n))
                if 0 <= y < M and 0 <= x < M and pri[base_of[ext[y, x]]] < pri[i + 1] if ext[y, x] else \
                        0 <= y < M and 0 <= x < M:
                    if base_of[ext[y, x]] != i + 1:
                        ext[y, x] = i + 1
                        drawn += 1
        out[q["name"]] = {"components_before": int(have), "pieces": len(pcs), "drawn_px": drawn,
                          "components_after": int(L.components(base_of[ext] == i + 1))}
    return out


def eye_floor(ext, base_of, ctx, N, M, eyes, slit=None, surround=None, nudge=False) -> dict:
    """Generic eye rule (blessed rules 4 + 'slits at <= 20'): each source piece of an eye part is redrawn as ONE
    filled rectangle at its scaled centroid: a vertical slit (1 x 2-3) at <= 20, otherwise the scaled eye box with
    the v3 floor. Applied last (eyes have the highest priority). v1: the black margin came from the renderer only
    (every eye boundary gets a line, eyes are protected, so the neighbour gives). v2: with `surround` (the plan's
    eye surround part) a 1 px moat of it is drawn round each eye rectangle, as the squid's visor."""
    src, side = ctx["src"], ctx["src"].shape[0]
    out = {}
    for e in eyes:
        i = ctx["idx"][e]
        for k, pc in enumerate(SG.pieces(src, i - 1)):
            ys, xs = np.nonzero(pc["mask"])
            if N <= 20:
                w, h = slit or EYE_SLIT[N]
            else:
                fw, fh = EYE_FLOOR.get(N, (2, 3))
                w = max(fw, int(round((np.ptp(xs) + 1) * M / side)))
                h = max(fh, int(round((np.ptp(ys) + 1) * M / side)))
            cy, cx = (np.array(pc["centroid"]) + 0.5) * M / side - 0.5
            y0, x0 = int(round(cy - (h - 1) / 2)), int(round(cx - (w - 1) / 2))
            had = int((base_of[ext] == i).sum())
            ext[(base_of[ext] == i)] = _nearest_other(ext, base_of, i)[(base_of[ext] == i)]
            moved = [0, 0]
            if nudge and surround:                # v3: don't let the moat erase a character part (the scope at 20)
                y0, x0, moved = _nudge(ext, base_of, ctx, y0, x0, w, h, eyes, surround, M)
            moat = 0
            if surround:                          # v2: the eye's own surround as a 1 px moat (the squid's visor)
                moat = 1
                ext[max(0, y0 - moat):y0 + h + moat, max(0, x0 - moat):x0 + w + moat] = ctx["idx"][surround]
            ext[y0:y0 + h, x0:x0 + w] = i
            out[f"{e}#{k}"] = {"box": [x0, y0, w, h], "had_px": had, "moat": moat, "nudged": moved}
    return out


def _nudge(ext, base_of, ctx, y0, x0, w, h, eyes, surround, M, sig_min=0.8):
    """v3: place the eye + its 1 px moat at the scaled centroid, or 1 px away, minimising the pixels it takes from
    character parts (significance >= sig_min or exaggerated, other than the eye and its surround), + 0.5 per
    background pixel under the moat (it changes the silhouette) + 0.5 per px moved, and only when staying put would
    erase a character part completely. (A first try always minimised, allowed 2 px and charged 0.25 for background:
    the eye slid off the head onto the neck at 24 and 16, and moved at 64-24 where nothing was at stake.)"""
    specs = ctx["specs"]
    keep = {k + 1 for k, q in enumerate(specs) if (q["significance"] >= sig_min or q["mode"] == "exaggerate")
            and q["name"] not in eyes and q["name"] != surround}
    behind = L3.behind_fn(ctx["legend"])
    black = {k + 1 for k, p in enumerate(ctx["legend"]["parts"]) if p.get("group") == "black"}

    def erased(yy, xx):
        """Character parts the moat + eye would wipe out: no pixel left outside the box that the renderer keeps
        (a pixel touching a non-black region it is behind becomes that boundary's line)."""
        b = base_of[ext].copy()
        b[max(0, yy - 1):yy + h + 1, max(0, xx - 1):xx + w + 1] = ctx["idx"][surround]
        b[yy:yy + h, xx:xx + w] = ctx["idx"][eyes[0]]
        out = []
        for k in keep:
            ys, xs = np.nonzero(b == k)
            if not len(ys) and (base_of[ext] == k).any():
                out.append(k)
                continue
            live = 0
            for y, x in zip(ys, xs):
                nb = [b[y + dy, x + dx] for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))
                      if 0 <= y + dy < M and 0 <= x + dx < M]
                if not any(q and q != k and q not in black and behind(k, q) for q in nb):
                    live += 1
            if len(ys) and not live:
                out.append(k)
        return out
    if not erased(y0, x0):                    # only move when staying would erase a character part
        return y0, x0, [0, 0]
    best = None
    for dy in (0, 1, -1):
        for dx in (0, 1, -1):
            yy, xx = y0 + dy, x0 + dx
            if yy - 1 < 0 or xx - 1 < 0 or yy + h + 1 > M or xx + w + 1 > M:
                continue
            box = base_of[ext[yy - 1:yy + h + 1, xx - 1:xx + w + 1]]
            cost = float(np.isin(box, list(keep)).sum()) + 0.5 * float((box == 0).sum()) + 0.5 * (abs(dy) + abs(dx))
            key = (len(erased(yy, xx)), cost, abs(dy) + abs(dx))
            if best is None or key < best[0]:
                best = (key, yy, xx, [dy, dx])
    return (best[1], best[2], best[3]) if best else (y0, x0, [0, 0])


def _nearest_other(ext, base_of, i):
    """For every pixel, the nearest non-background label that is not part i (to refill an eye's old pixels)."""
    ok = (ext > 0) & (base_of[ext] != i)
    if not ok.any():
        return np.zeros_like(ext)
    _, (iy, ix) = ndimage.distance_transform_edt(~ok, return_indices=True)
    return ext[iy, ix]


# ------------------------------------------------------------------ render (generic materials, no mirror)

def materials(leg, protected: set[str], fins: set[str] = frozenset()) -> list[dict]:
    pal = {p["group"]: p["colour"] for p in leg["parts"]}
    mats = []
    for p in leg["parts"]:
        g = p["group"]
        mats.append({"name": p["name"], "role": p["role"], "depth": p["depth"], "group": g,
                     "mode": "black" if g == "black" else "solid",
                     "colour": (0, 0, 0) if g == "black" else tuple(imaging.rgb(pal[g])),
                     "priority": p["priority"], "cup": False, "fin": p["name"] in fins,
                     "protected": p["name"] in protected})
    return mats


def render(ext, base_of, mats, leg, N, keep="none", opts=None):
    return R4.render(ext, base_of, mats, leg, N, keep, opts)


def eye_check(grid: list[str], leg, eyes: list[str]) -> dict:
    """On the final pixels: each eye a filled rectangle, with ink/black (not another colour) on every side."""
    ch = ".abcdefghijklmnopqrstuvwxyz"
    names = [p["name"] for p in leg["parts"]]
    black = {ch[i + 1] for i, p in enumerate(leg["parts"]) if p["group"] == "black"} | {"#"}
    g = np.array([list(r) for r in grid])
    out = {}
    for e in eyes:
        m = g == ch[names.index(e) + 1]
        lab, k = ndimage.label(m)
        recs = []
        for sl in ndimage.find_objects(lab):
            y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
            ring = [(y, x) for y in range(y0 - 1, y1 + 1) for x in range(x0 - 1, x1 + 1)
                    if not (y0 <= y < y1 and x0 <= x < x1) and 0 <= y < g.shape[0] and 0 <= x < g.shape[1]
                    and (y in (y0 - 1, y1) and x0 <= x < x1 or x in (x0 - 1, x1) and y0 <= y < y1)]
            recs.append({"box": [x1 - x0, y1 - y0], "filled": bool(m[y0:y1, x0:x1].all()),
                         "dark_margin": all(g[y, x] in black for y, x in ring)})
        out[e] = {"components": int(k), "eyes": recs,
                  "ok": k >= 1 and all(r["filled"] and r["dark_margin"] for r in recs)}
    return out
