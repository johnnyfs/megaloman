"""Snipe bot v3: a transparent pipeline sheet, and two image-model routes to 16x16 (Johnny, 2026-10-24).

Johnny's design for the compact snipe (priority order, highest first):
  eye > gun nose > scope > head|body line > head > body > side wing > vestigial wing > legs > tail > breastplate

  concept   Nano Banana, COMPACT_PROMPT (+ vestigial wing nub up/right), white removed, TIGHT trim (margin 0)
  regions   Nano Banana paints a flat region map with Johnny's fixed legend; pixels -> nearest legend colour inside
            the concept silhouette; eye-on-head sanity check (eye is one small blob, inside/next to the head)
  A         Nano Banana draws the sprite directly at 16x16 on a visible grid (given the regions + priorities); snap
  B         Nano Banana draws it at 32x32, boxy, on a grid; snap to the grid + NES palette; priority-aware reduce to
            24/20/16; cells where the reduce is ambiguous get two resolutions and gpt-5.6-sol picks (expert pass)
  sheet     one PNG per route: concept | region overlay + legend | priority list | model raw | each rung | final

uv run python -m megaloman.snipe_boxy [--budget 2.5]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from . import imaging, models, store, snipe, pipeline_manifest as PM
from .ladder import _font

OUT = snipe.OUT / "compact" / "boxy"
NB, SOL = snipe.NB, snipe.SOL

# Johnny's parts, in priority order (index 0 = highest). Colours are the flat paint-map colours (far apart).
PARTS = [
    ("eye", "#FFE600", "the big round glowing eye on the head (only the eye glass, not its socket)"),
    ("gun_nose", "#0050FF", "the snub-nosed gun barrel that is the beak, sticking out of the head to the left"),
    ("scope", "#A000FF", "the little telescope scope and its mount on top of the head"),
    ("head", "#00C8C8", "the head: the top/front part of the round body (head and body are carved from ONE circle; "
                        "the head is the upper-left cap, around the eye)"),
    ("body", "#FF0000", "the body: the rest of the round body below/behind the head"),
    ("side_wing", "#FF80C0", "the folded side wing on the flank, with its shape"),
    ("vestigial_wing", "#FF8000", "the small vestigial wing nub sticking up toward the upper right"),
    ("legs", "#704000", "the little folded legs and feet"),
    ("tail", "#003080", "the short stubby tail at the back"),
    ("breastplate", "#00A000", "the breastplate / belly armour panel, if any"),
]
PRIO = {n: i for i, (n, _, _) in enumerate(PARTS)}
PALETTE_ORDER = ["eye", "head", "body", "gun_nose", "scope", "side_wing", "vestigial_wing", "legs", "tail",
                 "breastplate"]   # colour budget: eye, then the two big masses (head vs body), then the gun
KEEP_SMALL = {"eye", "gun_nose", "scope"}       # if present in a cell at all, these win it

PRIORITY_TEXT = """Region priority, highest first (who keeps pixels when the sprite gets small):
1 eye: one clear bright eye, kept as a distinct block with a dark edge.
2 gun nose: the snub gun barrel, sticking out left; must read as a gun and a beak.
3 scope on top of the head: must stay its own distinct little shape, separated from the head by an outline.
4 the head|body separation: head and body are carved from ONE circle; they differ by colour and the line between.
5 head  6 body  7 side wing (keep its shape)  8 vestigial wing nub (up and right, keeps it a bird)
9 little legs  10 tail  11 breastplate (lowest: at small sizes at most a hint, an extra colour line)."""

CONCEPT_PROMPT = snipe.COMPACT_PROMPT  # already has the vestigial-wing line

PAINT_PROMPT = """Repaint this exact character as a FLAT COLOUR REGION MAP. Keep the outline, pose and every shape exactly
where it is (trace it, do not redraw). Fill each part with one solid flat colour, no shading, no outlines, no
highlights, plain white background:
{rows}
Every pixel of the character belongs to exactly one part. Nothing else in the image."""

A_PROMPT = """Redraw this character as a finished NES-style pixel-art game sprite of exactly 16 x 16 pixels, shown
enlarged on a 16 x 16 grid (each sprite pixel one large flat square, all squares the same size, filling the image).
Facing left, side view. Black 1-pixel outline around the sprite and between parts. At most 4 colours plus black.
The sprite fills the 16 x 16 box (no empty margin). Plain white background.
Composition matters more than detail at this size. Parts, and what must survive:
{prio}
The second image is ONLY a map of where the parts are ({legend}). Do NOT use its colours: colour the sprite with
the concept's own colours (image 1), with the head and the body clearly different colours."""

B_PROMPT = """Redraw this character as BOXY NES-style pixel art of exactly 32 x 32 pixels, shown enlarged on a 32 x 32
grid (each sprite pixel one large flat square, all squares the same size, filling the image). Big simple blocks, flat
colours, no dithering, no anti-aliasing, no gradients. Facing left, side view. Black 1-pixel outline around the
sprite and between parts. At most 4 colours plus black. The sprite fills the 32 x 32 box (no empty margin).
Plain white background. It will be reduced to 16 x 16 afterwards, so design every part at least 2 pixels thick and
keep the parts' shapes simple and distinct. Parts, and what must survive:
{prio}
The second image is ONLY a map of where the parts are ({legend}). Do NOT use its colours: colour the sprite with
the concept's own colours (image 1), with the head and the body clearly different colours."""

JUDGE_PROMPT = """Two 16x16 pixel sprites of SNIPE BOT (a round robot marsh bird facing left, snub gun for a beak, big
eye, scope on its head, small wing nub up-right, little legs, tail), shown enlarged: A on the left, B on the right.
They differ only in a few ambiguous pixels. Which reads better as that character at 16x16 (eye distinct, gun nose,
scope separate from head, bird silhouette)? JSON only: {"pick": "A" or "B", "why": "<one sentence>"}"""


def hexrgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def tight_square(im: Image.Image, side=1024) -> Image.Image:
    return snipe.square_crop(im, side=side, margin=0.0)


def call_img(prompt, images, seed, tag, ledger):
    content = [{"type": "text", "text": prompt}] + [
        {"type": "image_url", "image_url": {"url": models.data_uri(imaging.to_png(i))}} for i in images]
    body = {"model": NB, "modalities": ["image", "text"], "seed": seed, "messages": [{"role": "user", "content": content}]}
    ck = store.key({"step": f"snipe_boxy.{tag}", "model": NB, "prompt": prompt, "seed": seed,
                    "images": [store.sha(imaging.to_png(i)) for i in images]})
    resp, imgs, _, cached = models.call(body, ledger, ck)
    snipe.track(ck, tag)
    if not imgs:
        raise RuntimeError(f"{tag}: no image ({(resp['choices'][0]['message'].get('content') or '')[:200]})")
    return imaging.from_png(imgs[0]).convert("RGBA"), ck


def on_white(im):
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    bg.alpha_composite(im)
    return bg.convert("RGB")


# ------------------------------------------------------------------ regions + sanity

def label_map(painted: Image.Image, concept: Image.Image) -> np.ndarray:
    """Painted map -> labels (-1 background) by nearest legend colour, inside the concept silhouette."""
    a = np.array(painted.convert("RGB").resize(concept.size, Image.NEAREST)).astype(int)
    cols = np.array([hexrgb(c) for _, c, _ in PARTS])
    d = ((a[:, :, None, :] - cols[None, None]) ** 2).sum(-1)
    lab = d.argmin(-1)
    white = a.min(-1) > 225
    sil = np.array(concept)[..., 3] > 127
    lab = np.where(sil, lab, -1)
    # silhouette pixels the painter left white: nearest painted label
    hole = sil & white
    if hole.any():
        idx = ndimage.distance_transform_edt(hole | ~sil, return_distances=False, return_indices=True)
        lab = np.where(hole, lab[idx[0], idx[1]], lab)
    return lab


def eye_sanity(lab: np.ndarray, eye_id=None, head_ids=None) -> dict:
    """Eye-on-head check (suggested after the compact run, where `eye` took the orange wing patch): the eye is ONE
    main blob, small (< 6 % of the figure), and its ring (6 px outside it) is mostly head (or eye socket)."""
    eye_id = PRIO["eye"] if eye_id is None else eye_id
    head_ids = [PRIO["head"]] if head_ids is None else head_ids
    eye = lab == eye_id
    fig = (lab >= 0).sum()
    comps, k = ndimage.label(eye)
    if not k:
        return {"ok": False, "pieces": 0}
    sizes = ndimage.sum_labels(eye, comps, range(1, k + 1))
    m = comps == int(np.argmax(sizes)) + 1
    frac = float(m.sum() / max(fig, 1))
    ring = ndimage.binary_dilation(m, iterations=6) & ~m & (lab >= 0)
    head_share = float(np.isin(lab[ring], head_ids).mean()) if ring.any() else 0.0
    pieces = int((sizes > 0.1 * sizes.max()).sum())
    ok = pieces == 1 and frac < 0.06 and head_share >= 0.5
    return {"ok": bool(ok), "pieces": pieces, "area_frac": round(frac, 4), "head_share_of_ring": round(head_share, 3)}


def overlay(concept: Image.Image, lab: np.ndarray) -> Image.Image:
    cols = np.array([hexrgb(c) for _, c, _ in PARTS], np.uint8)
    base = np.array(on_white(concept)).astype(float)
    col = cols[np.clip(lab, 0, None)].astype(float)
    m = (lab >= 0)[..., None]
    out = np.where(m, 0.35 * base + 0.65 * col, base)
    edge = (lab != np.roll(lab, 1, 0)) | (lab != np.roll(lab, 1, 1))
    out[edge] = 0
    return Image.fromarray(out.astype(np.uint8))


# ------------------------------------------------------------------ pixels

def snap_sprite(raw: Image.Image, max_colors=5):
    """Model 'pixel art' -> 1x sprite. Grid pitch/phase from the edge profile (vendor gridsnap); each cell = median
    colour of its central half; white connected to the border -> transparent; near-black -> black; the rest snapped
    to the nearest NES colour. (imaging.snap dropped every black outline pixel and kept the white background here.)"""
    from .vendor import gridsnap
    a = np.array(on_white(raw)).astype(float)
    gx = np.abs(np.diff(a, axis=1)).sum(-1).sum(0)
    gy = np.abs(np.diff(a, axis=0)).sum(-1).sum(1)
    (px, fx), (py, fy) = gridsnap.pitch_phase(gx, 6.0, 40.0)[:2], gridsnap.pitch_phase(gy, 6.0, 40.0)[:2]
    p = (px + py) / 2
    fx, fy = fx % p, fy % p
    nx, ny = int((a.shape[1] - fx) // p), int((a.shape[0] - fy) // p)
    cells = np.zeros((ny, nx, 3))
    for j in range(ny):
        for i in range(nx):
            y0, x0 = fy + j * p, fx + i * p
            c = a[int(y0 + p / 4):int(y0 + 3 * p / 4) + 1, int(x0 + p / 4):int(x0 + 3 * p / 4) + 1]
            cells[j, i] = np.median(c.reshape(-1, 3), 0)
    white = cells.min(-1) > 200
    lab, _ = ndimage.label(white)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    bg = np.isin(lab, list(border))
    nes = np.array(sorted({imaging.rgb(h) for h in imaging.NES.values()}), float)
    near = ((imaging._lab(cells.reshape(-1, 3))[:, None] - imaging._lab(nes)[None]) ** 2).sum(-1).argmin(1)
    rgbc = nes[near].reshape(ny, nx, 3)
    rgbc[cells.max(-1) < 60] = 0
    out = np.dstack([rgbc, np.where(bg, 0, 255)]).astype(np.uint8)
    out[bg] = 0
    im = Image.fromarray(out, "RGBA")
    im = im.crop(im.getbbox())
    return im, {"pitch": round(float(p), 2), "grid": [nx, ny], "size": list(im.size)}


def priority_palette(im: Image.Image, regions: np.ndarray, max_colors=5):
    """Palette lock that keeps the colours the priority list needs: black, then the dominant colour of each part in
    priority order (eye first), then remaps the rest by Lab distance. (imaging.palette_lock keeps the most FREQUENT
    colours, which dropped the eye, gun and scope colours in the first run.)"""
    a = np.array(im)
    op = a[..., 3] > 0
    keep = []
    if (op & (a[..., :3].sum(-1) < 30)).any():
        keep.append((0, 0, 0))
    for n in PALETTE_ORDER:
        i = PRIO[n]
        sel = op & (regions == i) & (a[..., :3].sum(-1) >= 30)
        if not sel.any() or len(keep) >= max_colors:
            continue
        cs, cnt = np.unique(a[sel][:, :3], axis=0, return_counts=True)
        for j in np.argsort(-cnt):
            c = tuple(int(v) for v in cs[j])
            if c not in keep:
                keep.append(c)
                break
    kc = np.array(keep, float)
    px = a[op][:, :3].astype(float)
    near = ((imaging._lab(px)[:, None] - imaging._lab(kc)[None]) ** 2).sum(-1).argmin(1)
    out = a.copy()
    out[op, :3] = kc[near].astype(np.uint8)
    return Image.fromarray(out, "RGBA"), {"kept": [n for n in keep]}


def region_of_sprite(sprite: Image.Image, lab: np.ndarray) -> np.ndarray:
    """Register the concept's label map to the sprite box (both tight-trimmed): per sprite pixel the modal region."""
    ys, xs = np.nonzero(lab >= 0)
    L = lab[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    w, h = sprite.size
    out = np.full((h, w), -1)
    for y in range(h):
        for x in range(w):
            c = L[int(y * L.shape[0] / h):max(int((y + 1) * L.shape[0] / h), int(y * L.shape[0] / h) + 1),
                  int(x * L.shape[1] / w):max(int((x + 1) * L.shape[1] / w), int(x * L.shape[1] / w) + 1)]
            c = c[c >= 0]
            if c.size:
                out[y, x] = np.bincount(c).argmax()
    return out


def reduce(sprite: Image.Image, regions: np.ndarray, N: int):
    """Priority-aware box reduce of a trimmed sprite to fit N (long side = N). Per target cell: transparent if most
    of it is; a KEEP_SMALL part's colour if any of its pixels is there (highest priority first); else the majority
    colour. Ties (no strict majority) are recorded as ambiguous with both options."""
    a = np.array(sprite)
    h, w = a.shape[:2]
    s = N / max(w, h)
    W, H = max(1, round(w * s)), max(1, round(h * s))
    out = np.zeros((H, W, 4), np.uint8)
    amb = []
    for Y in range(H):
        for X in range(W):
            y0, y1 = int(Y * h / H), max(int((Y + 1) * h / H), int(Y * h / H) + 1)
            x0, x1 = int(X * w / W), max(int((X + 1) * w / W), int(X * w / W) + 1)
            blk, reg = a[y0:y1, x0:x1].reshape(-1, 4), regions[y0:y1, x0:x1].reshape(-1)
            op = blk[:, 3] > 0
            if op.sum() * 2 < len(blk):
                continue
            px, rg = blk[op], reg[op]
            pick = None
            for name in sorted(KEEP_SMALL, key=PRIO.get):
                sel = (rg == PRIO[name]) & (px[:, :3].sum(1) > 40)
                if sel.any():
                    cs, n = np.unique(px[sel], axis=0, return_counts=True)
                    pick = cs[n.argmax()]
                    break
            if pick is None:
                cs, n = np.unique(px, axis=0, return_counts=True)
                order = np.argsort(-n)
                pick = cs[order[0]]
                if len(n) > 1 and n[order[0]] == n[order[1]]:
                    # tie: default = higher-priority region's colour; alternate recorded for the expert pass
                    pri = lambda c: min([PRIO_SAFE(r) for r, p in zip(rg, px) if (p == c).all()] or [99])
                    c0, c1 = cs[order[0]], cs[order[1]]
                    pick, alt = (c0, c1) if pri(c0) <= pri(c1) else (c1, c0)
                    amb.append({"y": Y, "x": X, "alt": alt.tolist()})
            out[Y, X] = pick
    return Image.fromarray(out, "RGBA"), amb


def PRIO_SAFE(r):
    return int(r) if r >= 0 else 99


def with_alt(im: Image.Image, amb) -> Image.Image:
    a = np.array(im)
    for c in amb:
        a[c["y"], c["x"]] = c["alt"]
    return Image.fromarray(a, "RGBA")


def big(im: Image.Image, box=256) -> Image.Image:
    k = max(1, box // max(im.size))
    return on_white(im.resize((im.width * k, im.height * k), Image.NEAREST))


def judge_pair(a: Image.Image, b: Image.Image, ledger) -> dict:
    sheet = Image.new("RGB", (560, 300), (255, 255, 255))
    sheet.paste(big(a), (16, 30)); sheet.paste(big(b), (290, 30))
    d = ImageDraw.Draw(sheet)
    d.text((130, 2), "A", fill=0, font=_font(24)); d.text((400, 2), "B", fill=0, font=_font(24))
    png = imaging.to_png(sheet)
    body = {"model": SOL, "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": [{"type": "text", "text": JUDGE_PROMPT},
                                                      {"type": "image_url", "image_url": {"url": models.data_uri(png)}}]}]}
    ck = store.key({"step": "snipe_boxy.judge", "prompt": JUDGE_PROMPT, "image": store.sha(png)})
    resp, _, _, _ = models.call(body, ledger, ck)
    snipe.track(ck, "expert_judge")
    m = re.search(r"\{.*\}", resp["choices"][0]["message"].get("content") or "", re.S)
    return json.loads(m.group(0)) | {"call": ck}


# ------------------------------------------------------------------ sheet

def sheet(route: str, concept, ov, raw, rungs: list[tuple[str, Image.Image]], final, notes: list[str], path: Path):
    T = 300
    font, small = _font(22), _font(16)
    cols = [("1 concept (tight trim)", on_white(concept).resize((T, T), Image.LANCZOS)),
            ("2 regions (classified)", ov.resize((T, T), Image.LANCZOS)),
            (f"4 model raw ({route})", on_white(raw).resize((T, T), Image.LANCZOS))]
    W = 16 + (T + 16) * 3 + 360
    Hh = 40 + T + 60 + 300 + 16
    sh = Image.new("RGB", (W, Hh), (24, 24, 30))
    d = ImageDraw.Draw(sh)
    x = 16
    for i, (t, im) in enumerate(cols):
        if i == 2:
            x += 360  # priority column goes between regions and raw
        d.text((x, 8), t, fill=(230, 230, 230), font=font)
        sh.paste(im, (x, 40))
        x += T + 16
    # legend + priority list
    px = 16 + 2 * (T + 16)
    d.text((px, 8), "3 legend = priority", fill=(230, 230, 230), font=font)
    for i, (n, c, _) in enumerate(PARTS):
        d.rectangle((px, 44 + i * 28, px + 20, 64 + i * 28), fill=hexrgb(c), outline=(0, 0, 0))
        d.text((px + 28, 44 + i * 28), f"{i + 1}. {n}", fill=(230, 230, 230), font=small)
    # rungs
    y = 40 + T + 20
    d.text((16, y), "5 ladder (snapped / cleaned / reduced)  ->  6 final 16x16", fill=(230, 230, 230), font=font)
    y += 34
    x = 16
    for t, im in rungs + [("FINAL", final)]:
        b = big(im, 220 if t != "FINAL" else 256)
        sh.paste(b, (x, y))
        d.text((x, y + b.height + 2), f"{t} ({im.width}x{im.height})", fill=(230, 230, 230), font=small)
        x += b.width + 16
    for i, n in enumerate(notes[:6]):
        d.text((x + 8, y + i * 24), n[:44], fill=(200, 200, 160), font=small)
    sh.save(path)
    return path


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=2.5)
    ap.add_argument("--routes", default="B,A")
    a = ap.parse_args()
    ledger = models.Ledger(a.budget)
    snipe.OUT = OUT
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"parts_priority": [n for n, _, _ in PARTS]}

    # concept
    raw, ck = call_img(CONCEPT_PROMPT, [], 11, "concept", ledger)
    raw.save(OUT / "concept_raw.png")
    concept = tight_square(snipe.remove_white(raw))
    concept.save(OUT / "concept.png")
    al = np.array(concept)[..., 3] > 127
    ys, xs = np.nonzero(al)
    res["concept"] = {"call": ck, "bbox_frac": [round((np.ptp(xs) + 1) / 1024, 3), round((np.ptp(ys) + 1) / 1024, 3)]}

    # regions (2 seeds max: first that passes the eye check)
    rows = "\n".join(f"- {n}: {c} = {w}" for n, c, w in PARTS)
    for seed in (1, 2):
        painted, ck = call_img(PAINT_PROMPT.format(rows=rows), [on_white(concept)], seed, "paint", ledger)
        lab = label_map(painted, concept)
        chk = eye_sanity(lab)
        res.setdefault("paint", []).append({"seed": seed, "call": ck, "eye_check": chk})
        if chk["ok"]:
            break
    painted.save(OUT / "painted.png")
    np.save(OUT / "labels.npy", lab)
    ov = overlay(concept, lab)
    ov.save(OUT / "overlay.png")
    area = {n: round(float((lab == i).sum() / (lab >= 0).sum()), 4) for i, (n, _, _) in enumerate(PARTS)}
    res["region_area"] = area
    legend_txt = ", ".join(f"{n} {c}" for n, c, _ in PARTS)
    snipe_spend = lambda: snipe.spend()["total"]

    for route in a.routes.split(","):
        d = OUT / route
        d.mkdir(exist_ok=True)
        prompt = (B_PROMPT if route == "B" else A_PROMPT).format(prio=PRIORITY_TEXT, legend=legend_txt)
        mraw = None
        for seed in (1, 2, 3):
            try:
                mraw, ck = call_img(prompt, [on_white(concept), on_white(ov.convert("RGBA"))], seed, f"route_{route}", ledger)
                break
            except models.BudgetExceeded:
                raise
            except Exception as e:
                res.setdefault(f"{route}_errors", []).append(str(e)[:200])
        if mraw is None:
            continue
        mraw.save(d / "model_raw.png")
        spr, info = snap_sprite(mraw)
        spr.save(d / "snapped.png")
        regs = region_of_sprite(spr, lab)
        spr, pl = priority_palette(spr, regs)
        spr.save(d / "snapped.png")
        info |= pl
        rungs, notes, r = [("snapped", spr)], [f"grid {info.get('size')} pitch {info.get('pitch')}"], {}
        final = spr
        if max(spr.size) > 16:
            for N in ((24, 20, 16) if route == "B" else (16,)):
                im, amb = reduce(spr, regs, N)
                im.save(d / f"rung_{N}.png")
                rungs.append((f"{N}", im))
                r[N] = {"ambiguous": len(amb)}
                if N == 16:
                    final = im
                    if amb:
                        alt = with_alt(im, amb)
                        alt.save(d / "rung_16_alt.png")
                        j1, j2 = judge_pair(im, alt, ledger), judge_pair(alt, im, ledger)
                        j = {"pick": "B" if (j1.get("pick") == "B" and j2.get("pick") == "A") else "A",
                             "orders": [j1, j2], "rule": "alt only if both orders agree (position bias)"}
                        r["expert"] = j
                        if j.get("pick") == "B":
                            final = alt
                        notes.append(f"16: {len(amb)} ambiguous -> {'alt' if j['pick']=='B' else 'priority default'} ({j1.get('pick')}/{j2.get('pick')})")
        final.save(d / "final_16.png")
        notes.append(f"final {final.size[0]}x{final.size[1]}, colours {len(imaging.colors_of(final))}")
        notes.append(f"spend so far ${snipe_spend():.2f}")
        sheet(route, concept, ov, mraw, rungs, final, notes, d / "pipeline_sheet.png")
        PM.write(d, f"Snipe bot, route {route} ({'boxy 32 then clean' if route == 'B' else 'direct 16'})", [
            PM.stage("concept", "1 concept (tight trim)", images=[OUT / "concept.png"],
                     notes=[f"bbox {res['concept']['bbox_frac']} of square"]),
            PM.stage("regions", "2 regions", "legend", images=[OUT / "overlay.png"],
                     notes=[f"eye check {'ok' if res['paint'][-1]['eye_check']['ok'] else 'FAIL'}"],
                     data={"items": [{"name": n, "color": c, "area": area[n]} for n, c, _ in PARTS]}),
            PM.stage("priority", "3 priority", "list", data={"items": [n for n, _, _ in PARTS]},
                     notes=["keep-if-present: " + ", ".join(sorted(KEEP_SMALL, key=PRIO.get))]),
            PM.stage("model", f"4 model raw ({route})", images=[d / "model_raw.png"], notes=[prompt.splitlines()[0][:60]]),
            *[PM.stage(f"rung_{t}", f"5 {t}", images=[{"path": str(d / ('snapped.png' if t == 'snapped' else f'rung_{t}.png')),
                                                       "caption": f"{im.width}x{im.height}", "pixel": True}],
                       notes=[f"{r.get(int(t), {}).get('ambiguous')} ambiguous cells"] if t.isdigit() else [notes[0]])
              for t, im in rungs],
            PM.stage("final", "6 final", images=[{"path": str(d / "final_16.png"), "caption": f"{final.width}x{final.height}",
                                                  "pixel": True}], notes=notes[1:]),
        ], cost_usd=snipe.spend()["total"])
        res[route] = {"call": ck, "snap": info, "rungs": r, "final_size": list(final.size), "notes": notes}
    res["cost"] = snipe.spend()
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=str))
    md = [f"# Snipe boxy run", f"cost: ${res['cost']['total']}", f"priority: {' > '.join(res['parts_priority'])}",
          f"eye checks: {[p['eye_check'] for p in res['paint']]}", f"region area: {area}"]
    for route in a.routes.split(","):
        md.append(f"- {route}: {json.dumps(res.get(route), default=str)[:400]}")
    (OUT / "README.md").write_text("\n".join(md) + "\n")
    print(json.dumps(res, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
