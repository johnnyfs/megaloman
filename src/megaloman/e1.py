"""E1 "flag, then critic or refine" (MODEL_IN_LOOP_PLAN section 4, plus the 2026-10-23 addendum knobs).

Baseline A = pipeline v4 at 24 / 20 / 16 (32 and 64 are frozen). Arms:
  B1 `v4-critic`   : knob sweep over v4's free choices -> blessed hard gate -> decision-model critic (Jev + Luna,
                     absolute yes/no questions, one sprite per call, so no position bias). No clear winner -> keep v4.
  B2 `v4-nbrefine` : Nano Banana refines the v4 sprite freely at ~512 px (no grid overlay), then snap to the grid,
                     recover labels, deterministic repair (separation lines, specks, mirror), hard gate.
Plus: a flagger (yes/no + Box) validated on synthetic defects with known locations, the gpt-5.6-sol pairwise judge
(both orders x 2 samples), and the blind-pick set for Johnny (`ab-blind` on the cockpit).

Knobs (models choose only where the rules leave freedom, or where a rule is a trial):
  fin_dk      fins' exaggeration k offset (-0.2, -0.1, 0, +0.1; +0.1 passes the frame margin)
  fin_depth   rows of black under the fins (1, 2)
  fin_pay     fins re-grow their line pixels from the background (on / off)
  eye_slit    20: 1x3 / 1x2 / 2x2; 16: 1x2 / 2x2
  curl_run    inner-arm cup run above the tip row (0, 1, 2), at 20 and 16
  ring_edge   addendum + trial rule: the face ring becomes an edge at this size (corners clipped, fins then head
              fill the freed space). Chosen per size, so the per-size picks give the switch rung.
  size_yield  addendum: which part keeps contested cells between head and fins (head = v4, fins)
  cup_line    addendum (which boundary is sacrificed), 24 only: edge the cups on their arms at 24 (v4: unlined
              below 32; blessed unlined at the small sizes 20 / 16, so not a knob there)

uv run python -m megaloman.e1 [--budget 3]
Writes workspace/experiments/regions/e1/ (results.json, candidates/, flags/, nbrefine/, blind/ with key.json).
"""
from __future__ import annotations

import argparse
import io
import itertools
import json
import random
import re
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from . import decisions as D, imaging, models, store
from . import ladder_v2 as LV, ladder_v3 as L3, ladder_v4 as L4, render as R1, render_v3 as R3, render_v4 as R4

ROOT = Path(store.ROOT)
REG = R4.REG
OUT = REG / "e1"
CONCEPT = ROOT / "workspace/shared/sources/squid_concept_gpt_front.png"
SIZES = (24, 20, 16)
GPT = "openai/gpt-5.6-sol"
NB = "google/gemini-nano-banana-2.1"
CH = ".abcdefghijklmnopqrstuvwxyz"
CHECKS = {"tile_colors": {}, "orphans": {}, "holes": {}, "outline": {}, "colors": {"max_total": 4, "gate": True}}
CLEAR_MARGIN = 0.03      # critic: a candidate must beat A's mean p by this much to count as a clear winner
SEED = 1023
CALL_KEYS: set = set()   # every model call this experiment used (cached or not): the cost is summed from these


def spend() -> dict:
    """Total spend of the experiment from the call records (so calls cached from an earlier, interrupted run still
    count), with BYOK Nano Banana billed at its upstream (Google) cost, which the OpenRouter usage.cost leaves at 0."""
    by = {}
    for k in CALL_KEYS:
        r = store.get_json("calls", k) or {}
        m = (r.get("request") or {}).get("model", "?")
        u = (r.get("response") or {}).get("usage") or {}
        c = (r.get("cost") or 0.0) + ((u.get("cost_details") or {}).get("upstream_inference_cost") or 0.0
                                      if u.get("is_byok") else 0.0)
        b = by.setdefault(m, {"calls": 0, "usd": 0.0})
        b["calls"] += 1
        b["usd"] += c
    return {"total_usd": round(sum(b["usd"] for b in by.values()), 4),
            "by_model": {m: {"calls": b["calls"], "usd": round(b["usd"], 4)} for m, b in by.items()}}


# ------------------------------------------------------------------ setup / knobs

def setup():
    rdir = REG / "v2" / "regions"
    legend_v2 = json.loads((rdir / "legend.json").read_text())
    src_raw = np.load(rdir / "labels.npy")
    spec = json.loads(LV.SPEC.read_text())
    ctx = L4.prepare(src_raw, legend_v2, spec, (64, 32) + SIZES)
    mats = R3.materials(ctx["legend"], ctx["src"])
    return ctx, mats


def defaults(N) -> dict:
    d = {"fin_dk": 0.0, "fin_depth": 2, "fin_pay": True, "ring_edge": False, "size_yield": "head"}
    if N == 24:
        d["cup_line"] = False
    if N <= L4.SLIT_MAX:
        d["eye_slit"] = tuple(L4.EYE_SLIT[N])
    if N <= L4.CURL_MAX:
        d["curl_run"] = L4.CURL_RUN[N]
    return d


def space(N) -> dict:
    s = {"fin_dk": [-0.2, -0.1, 0.0, 0.1], "fin_depth": [1, 2], "fin_pay": [True, False], "ring_edge": [False, True],
         "size_yield": ["head", "fins"]}
    if N == 24:
        s["cup_line"] = [False, True]
    if N == 20:
        s["eye_slit"] = [(1, 3), (1, 2), (2, 2)]
    if N == 16:
        s["eye_slit"] = [(1, 2), (2, 2)]
    if N <= L4.CURL_MAX:
        s["curl_run"] = [0, 1, 2]
    return s


def combos(N):
    s = space(N)
    ks = list(s)
    for vals in itertools.product(*(s[k] for k in ks)):
        yield dict(zip(ks, vals))


def diff(N, kn) -> dict:
    d = defaults(N)
    return {k: v for k, v in kn.items() if d.get(k) != v}


def to_png(im) -> bytes:
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def gate(legal, info, leg) -> dict:
    a = np.array(legal)
    checks = imaging.run_checks(legal, CHECKS)
    eyes = R3.eye_check(legal, info["grid"])
    sep = R4.separation_check(info["grid"], leg)
    mirror = bool((a == a[:, ::-1]).all())
    fails = [k for k, v in checks.items() if v["gate"] and not v["pass"]]
    if not sep["ok"]:
        fails.append("separation")
    if not eyes["ok"]:
        fails.append("eyes")
    if not mirror:
        fails.append("mirror")
    if info.get("tile", {}).get("changed_px"):
        fails.append("tile_changed")
    return {"ok": not fails, "fails": fails, "separation": sep["touching"], "eyes": [e["box"] for e in eyes["eyes"]]}


def render_one(ctx, mats, N, kn, cache):
    lp = {k: kn[k] for k in ("fin_dk", "eye_slit", "curl_run", "fin_pay", "size_yield") if k in kn}
    key = json.dumps(lp, sort_keys=True)
    if key not in cache:
        cache[key] = L4.rung(ctx, N, lp)
    r = cache[key]
    opts = {**R4.OPTS, "fin_depth": kn["fin_depth"], "ring_edge": kn["ring_edge"]}
    if kn.get("cup_line"):
        opts["cup_edge_min"] = 24
    im, info = R4.render(r["ext"], ctx["base_of"], mats, ctx["legend"], N, ctx["sym"]["keep"], opts)
    light = imaging.rgb(L4.PALETTE["light"][1])
    legal, tinfo = R1.tile_reduce(im, salient=[light], salient_w=R3.SALIENT_W, symmetric=True)
    info["tile"] = tinfo
    hs = r["health"]
    return legal, info, {"health_score": r["health_score"], "k": r["k"].get("cap_side_lobes", {}).get("k"),
                         "reds": sum(h["status"] == "red" for h in hs), "ambers": sum(h["status"] == "amber" for h in hs)}


def sweep(ctx, mats) -> dict:
    """Steps 1-2: plumbing identity check, enumerate, render, dedupe, hard gate."""
    leg = ctx["legend"]
    out = {}
    for N in SIZES:
        cache, uniq = {}, {}
        n_all = n_fail = 0
        fail_why = {}
        for kn in combos(N):
            n_all += 1
            legal, info, h = render_one(ctx, mats, N, kn, cache)
            g = gate(legal, info, leg)
            if not g["ok"]:
                n_fail += 1
                for f in g["fails"]:
                    fail_why[f] = fail_why.get(f, 0) + 1
                continue
            a = np.array(legal)
            k = a.tobytes()
            if k not in uniq:
                uniq[k] = {"png": to_png(legal), "grid": info["grid"], "knobs": [], "health": h,
                           "ring_edge_log": info.get("ring_edge"), "ink_px": info["ink_px"],
                           "fin_depth_px": info["fin_depth_px"]}
            uniq[k]["knobs"].append(diff(N, kn))
        a_png = (R4.V4 / "render" / f"render_{N}.png").read_bytes()
        a_arr = np.array(Image.open(io.BytesIO(a_png)))
        assert a_arr.tobytes() in uniq, f"v4 defaults must reproduce v4 at {N}"
        cands = []
        for i, (k, u) in enumerate(sorted(uniq.items(), key=lambda kv: min(len(x) for x in kv[1]["knobs"]))):
            u["knobs"].sort(key=len)
            u["id"] = f"{N}-c{i:03d}"
            u["is_A"] = k == a_arr.tobytes()
            u["changed_px_vs_A"] = int((np.array(Image.open(io.BytesIO(u["png"]))) != a_arr).any(-1).sum())
            cands.append(u)
        d = OUT / "candidates" / str(N)
        d.mkdir(parents=True, exist_ok=True)
        for c in cands:
            (d / f"{c['id']}.png").write_bytes(c["png"])
        out[N] = {"n_combos": n_all, "n_gate_fail": n_fail, "gate_fail_reasons": fail_why, "n_survivors": len(cands),
                  "candidates": cands}
        print(N, "combos", n_all, "gate fail", n_fail, fail_why, "unique survivors", len(cands))
    return out


# ------------------------------------------------------------------ flagger (steps 3-4)

FLAG_Q = {
    "speck": "a stray single pixel or tiny speck that does not belong to any shape",
    "noise": "scattered noise: many random pixels sprinkled over the sprite",
    "lost_line": "two differently coloured parts touching with no dark separating line between them",
    "merged_arms": "two neighbouring tentacles merged into one shape, with no gap or line between them",
    "heavy_black": "a heavy solid black blob or block (thicker than a line) inside the sprite",
    "eyes_bad": "an eye that is missing, merged or not clearly one of two separate eyes",
    "asymmetric": "a left half that is not the mirror image of the right half",
    "fin_small": "side fins that are too small or missing to read as fins",
}
DEFECT_TO_Q = {"speck": "speck", "noise": "noise", "lost_line": "lost_line", "merged_arms": "merged_arms",
               "heavy_black": "heavy_black", "eye_removed": "eyes_bad", "copied_half": "asymmetric"}
RGB = {"black": (0, 0, 0), "head": (0xF8, 0x38, 0x00), "mid": (0xF8, 0x78, 0x58), "light": (0xF0, 0xD0, 0xB0)}


def flag_questions(boxes=True) -> dict:
    q = {}
    for k, desc in FLAG_Q.items():
        q[k] = {"type": "noul", "instructions": f"Does this small pixel-art game sprite have {desc}?",
                "criteria": {"true": "yes", "false": "no"}}
        if boxes:
            q[k + "_box"] = {"type": "box", "instructions": f"Draw a box around the worst place where the sprite has "
                                                            f"{desc}. If it has none, box the most suspicious area."}
    return q


def _rgba(path):
    return np.array(Image.open(path).convert("RGBA"))


def defects(a: np.ndarray, grid: list[str], rng) -> dict:
    """Synthetic defects with known locations on a v4 render -> {name: (image array, [y0, x0, y1, x1] inclusive)}."""
    H, W = a.shape[:2]
    g = np.array([list(r) for r in grid])
    col = lambda c: (a[..., 3] > 0) & (a[..., :3] == c).all(-1)
    out = {}
    # speck: one coral pixel in the background, 2 px clear of the sprite, left side, mid height
    fg = a[..., 3] > 0
    for it in (2, 1):
        far = ~ndimage.binary_dilation(fg, structure=np.ones((3, 3), bool), iterations=it)
        cand = [(y, x) for y, x in zip(*np.nonzero(far)) if x < W // 2 and H * 0.2 < y < H * 0.9]
        if cand:
            break
    if not cand:
        cand = [(y, x) for y, x in zip(*np.nonzero(~fg)) if x < W // 2]
    y, x = cand[len(cand) // 2]
    b = a.copy(); b[y, x] = (*RGB["mid"], 255)
    out["speck"] = (b, [y, x, y, x])
    # noise: 8 % of the sprite's pixels set to random palette colours
    b = a.copy()
    ys, xs = np.nonzero(fg)
    pick = rng.sample(range(len(ys)), max(3, int(0.08 * len(ys))))
    pal = [RGB[k] for k in RGB]
    for i in pick:
        b[ys[i], xs[i]] = (*pal[rng.randrange(4)], 255)
    out["noise"] = (b, [int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())])
    # lost line: the left head | fin line pixels become head colour
    ink = g == "#"
    head, fin = g == "a", g == "b"
    nb = lambda m: ndimage.binary_dilation(m, structure=ndimage.generate_binary_structure(2, 1))
    line = ink & nb(head) & nb(fin)
    line[:, W // 2:] = False
    if line.any():
        b = a.copy(); b[line] = (*RGB["head"], 255)
        ly, lx = np.nonzero(line)
        out["lost_line"] = (b, [int(ly.min()), int(lx.min()), int(ly.max()), int(lx.max())])
    # merged arms: on the left, ink pixels between two arm pixels (left and right neighbours) become arm colour
    arm = np.isin(g, list("hij"))
    m = np.zeros_like(ink)
    for y in range(H):
        for x in range(1, W // 2):
            if ink[y, x] and arm[y, x - 1] and x + 1 < W and arm[y, x + 1] and y > H * 0.55:
                m[y, x] = True
    if m.any():
        b = a.copy(); b[m] = (*RGB["light"], 255)
        my_, mx_ = np.nonzero(m)
        out["merged_arms"] = (b, [int(my_.min()), int(mx_.min()), int(my_.max()), int(mx_.max())])
    # heavy black: a 2x2 (3x3 at 24) black block inside the head, left of the axis
    s = 3 if H >= 24 else 2
    hy, hx = np.nonzero(head & ~nb(~head))
    best = None
    for y, x in zip(hy, hx):
        if x + s <= W // 2 and head[y:y + s, x:x + s].all() and (best is None or x < best[1]):
            best = (y, x)
    if best is None:
        hy, hx = np.nonzero(head)
        best = (int(hy.mean()), W // 2 - s)
    y, x = best
    b = a.copy(); b[y:y + s, x:x + s] = (0, 0, 0, 255)
    out["heavy_black"] = (b, [y, x, y + s - 1, x + s - 1])
    # eye removed: the right eye becomes visor black
    e = g == "e"
    e[:, : W // 2] = False
    b = a.copy(); b[e] = (0, 0, 0, 255)
    ey, ex = np.nonzero(e)
    out["eye_removed"] = (b, [int(ey.min()), int(ex.min()), int(ey.max()), int(ex.max())])
    # copied half: the right half is a copy (not a mirror) of the left half
    b = a.copy(); b[:, W - W // 2:] = a[:, : W // 2]
    out["copied_half"] = (b, [0, W - W // 2, H - 1, W - 1])
    return out


def box_px(bx, N):
    """Decision box (fractions of the fitted image) -> [y0, x0, y1, x1] in sprite pixels (inclusive, clamped)."""
    t, b = sorted((bx["top"], bx["bottom"]))
    l, r = sorted((bx["left"], bx["right"]))
    c = lambda v: int(min(N - 1, max(0, v)))
    y0, x0 = c(np.floor(t * N)), c(np.floor(l * N))
    return [y0, x0, max(y0, c(np.ceil(b * N) - 1)), max(x0, c(np.ceil(r * N) - 1))]


def iou(a, b):
    y0, x0, y1, x1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, y1 - y0 + 1) * max(0, x1 - x0 + 1)
    ar = lambda r: (r[2] - r[0] + 1) * (r[3] - r[1] + 1)
    return inter / (ar(a) + ar(b) - inter)


def hits(pred, true, pad=1):
    """The predicted box (padded by 1 px) overlaps the defect's box."""
    return not (pred[2] + pad < true[0] or pred[0] - pad > true[2] or pred[3] + pad < true[1] or pred[1] - pad > true[3])


def flag(img, N, led, tag):
    st = [f"A {N}x{N} pixel-art game sprite (a robotic squid enemy), shown enlarged:", img]
    jev = D.decide(D.JEV, st, flag_questions(True), led, tag=tag)
    luna = D.decide(D.LUNA, st, flag_questions(False), led, tag=tag)
    CALL_KEYS.update((jev["call"], luna["call"]))
    out = {}
    for k in FLAG_Q:
        out[k] = {"jev": jev["answers"][k]["noul"], "luna": luna["answers"][k]["noul"],
                  "box": box_px(jev["answers"][k + "_box"]["box"], N),
                  "box_conf": jev["answers"][k + "_box"].get("confidence")}
    return out, jev["cost"] + luna["cost"]


def run_flagger(led) -> dict:
    rng = random.Random(SEED)
    res = json.loads((R4.V4 / "results.json").read_text())
    (OUT / "flags").mkdir(parents=True, exist_ok=True)
    jobs = []
    for N in SIZES:
        a = _rgba(R4.V4 / "render" / f"render_{N}.png")
        jobs.append((N, "clean", a, None))
        for name, (b, bx) in defects(a, res["renders"][str(N)]["grid"], rng).items():
            jobs.append((N, name, b, [int(v) for v in bx]))
            Image.fromarray(b.astype(np.uint8), "RGBA").save(OUT / "flags" / f"defect_{N}_{name}.png")

    def one(j):
        N, name, arr, bx = j
        f, cost = flag(Image.fromarray(arr.astype(np.uint8), "RGBA"), N, led, "e1flag1")
        return N, name, bx, f

    with ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(one, jobs))
    items, clean = [], {}
    for N, name, bx, f in rows:
        if name == "clean":
            clean[N] = f
            continue
        q = DEFECT_TO_Q[name]
        items.append({"size": N, "defect": name, "question": q, "true_box": bx, "jev_p": f[q]["jev"],
                      "luna_p": f[q]["luna"], "box": f[q]["box"], "box_iou": round(iou(f[q]["box"], bx), 3),
                      "box_hit": hits(f[q]["box"], bx),
                      "other_yes_jev": [k for k in FLAG_Q if k != q and f[k]["jev"] >= 0.5]})
    fp = {m: [k for N in SIZES for k in FLAG_Q if clean[N][k][m] >= 0.5] for m in ("jev", "luna")}
    n = len(items)
    summ = {
        "n_defects": n,
        "jev_hit_rate": round(sum(i["jev_p"] >= 0.5 for i in items) / n, 3),
        "luna_hit_rate": round(sum(i["luna_p"] >= 0.5 for i in items) / n, 3),
        "both_hit_rate": round(sum(i["jev_p"] >= 0.5 and i["luna_p"] >= 0.5 for i in items) / n, 3),
        "jev_hit_rate_p08": round(sum(i["jev_p"] >= 0.8 for i in items) / n, 3),
        "box_hit_rate": round(sum(i["box_hit"] for i in items) / n, 3),
        "box_iou_mean": round(statistics.mean(i["box_iou"] for i in items), 3),
        "clean_false_flags": {m: f"{len(v)}/{len(SIZES) * len(FLAG_Q)}" for m, v in fp.items()},
        "by_defect": {d: {"jev": [round(i["jev_p"], 2) for i in items if i["defect"] == d],
                          "luna": [round(i["luna_p"], 2) for i in items if i["defect"] == d],
                          "iou": [i["box_iou"] for i in items if i["defect"] == d]} for d in DEFECT_TO_Q},
    }
    # step 4: flags on A (both models >= 0.5), drawn on the sprite
    flags_A = {}
    for N in SIZES:
        f = clean[N]
        flags_A[N] = {k: {"jev": round(v["jev"], 2), "luna": round(v["luna"], 2), "box": v["box"],
                          "flag": v["jev"] >= 0.5 and v["luna"] >= 0.5} for k, v in f.items()}
        draw_flags(N, flags_A[N])
    return {"items": items, "summary": summ, "flags_A": flags_A}


def draw_flags(N, f):
    im = Image.open(R4.V4 / "render" / f"render_{N}.png").convert("RGBA")
    z = 240 // N
    bg = Image.new("RGBA", im.size, (0x5a, 0x5a, 0x6e, 255))
    bg.alpha_composite(im)
    big = bg.resize((N * z, N * z), Image.NEAREST)
    d = ImageDraw.Draw(big)
    cols = ["#ff00ff", "#00ffff", "#ffff00", "#00ff00", "#ff8800", "#ffffff", "#8888ff", "#ff4444"]
    for (k, v), c in zip(f.items(), cols):
        if v["jev"] >= 0.5 or v["luna"] >= 0.5:
            y0, x0, y1, x1 = v["box"]
            d.rectangle([x0 * z, y0 * z, (x1 + 1) * z - 1, (y1 + 1) * z - 1], outline=c, width=3 if v["flag"] else 1)
    big.save(OUT / "flags" / f"flags_A_{N}.png")


# ------------------------------------------------------------------ critic (step 5)

CRITIC_Q = {
    "eyes": "Do the eyes read as two clear, separate eyes?",
    "fins": "Do the side fins read clearly as jaunty fins, distinct from the head?",
    "arms": "Are the tentacles distinct from each other (separate arms, not one merged skirt)?",
    "clean": "Is the sprite clean: no heavy black blobs, stray specks or muddy clumps?",
    "reads": "Does the sprite read clearly as the robotic squid in the concept art, at game size?",
}


def critic_score(png: bytes, N, led) -> dict:
    q = {k: {"type": "noul", "instructions": v, "criteria": {"true": "yes", "false": "no"}} for k, v in CRITIC_Q.items()}
    st = ["Concept art of a game enemy:", str(CONCEPT),
          f"A {N}x{N} pixel-art sprite of it for an NES-style game, shown enlarged:", png]
    out = {}
    for m, tag in ((D.JEV, "jev"), (D.LUNA, "luna")):
        r = D.decide(m, st, q, led, tag="e1crit1")
        CALL_KEYS.add(r["call"])
        out[tag] = {k: r["answers"][k]["noul"] for k in CRITIC_Q}
    out["score"] = round(statistics.mean([v for t in ("jev", "luna") for v in out[t].values()]), 4)
    out["score_jev"] = round(statistics.mean(out["jev"].values()), 4)
    out["score_luna"] = round(statistics.mean(out["luna"].values()), 4)
    return out


def run_critic(sw, led) -> dict:
    sentinels = {"v3_alt_finline_16": REG / "v3" / "alt_finline" / "render" / "render_16.png",
                 "v4_alt_cupline_16": R4.V4 / "alt_cupline" / "render" / "render_16.png"}
    jobs = [(N, c) for N in SIZES for c in sw[N]["candidates"]]

    def one(j):
        N, c = j
        return critic_score(c["png"], N, led)

    with ThreadPoolExecutor(8) as ex:
        scores = list(ex.map(one, jobs))
    for (N, c), s in zip(jobs, scores):
        c["critic"] = s
    sent = {k: critic_score(p.read_bytes(), 16, led) for k, p in sentinels.items()}
    rng = random.Random(SEED)
    out = {}
    for N in SIZES:
        cs = sw[N]["candidates"]
        A = next(c for c in cs if c["is_A"])
        ranked = sorted(cs, key=lambda c: -c["critic"]["score"])
        for r, c in enumerate(ranked):
            c["rank"] = r + 1
        top = [c for c in ranked if not c["is_A"]][:2]
        rest = [c for c in cs if not c["is_A"] and c not in top]
        rand = rng.sample(rest, min(2, len(rest)))
        sc = [c["critic"]["score"] for c in cs]
        clear = top[0]["critic"]["score"] - A["critic"]["score"] >= CLEAR_MARGIN
        out[N] = {"A": {"id": A["id"], "score": A["critic"]["score"], "rank": A["rank"]},
                  "top2": [{"id": c["id"], "score": c["critic"]["score"], "knobs": c["knobs"][0],
                            "changed_px": c["changed_px_vs_A"]} for c in top],
                  "random2": [{"id": c["id"], "score": c["critic"]["score"], "knobs": c["knobs"][0], "rank": c["rank"],
                               "changed_px": c["changed_px_vs_A"]} for c in rand],
                  "spread": round(max(sc) - min(sc), 4), "sd": round(statistics.pstdev(sc), 4),
                  "clear_winner": clear, "decision": top[0]["id"] if clear else "keep v4 (A)",
                  "knob_effects": knob_effects(N, cs)}
    a16 = out[16]["A"]["score"]
    out["sentinels"] = {k: {"score": v["score"], "below_A": v["score"] < a16} for k, v in sent.items()}
    if not all(v["below_A"] for v in out["sentinels"].values()):     # plan: a run that misses its sentinels is
        for N in SIZES:                                                # dropped, so the critic cannot overrule v4
            out[N]["decision"] = "keep v4 (A): sentinels failed"
    return out


def knob_effects(N, cs) -> dict:
    """Mean critic score per knob value, over every knob setting that produced a surviving grid."""
    acc = {}
    d = defaults(N)
    for c in cs:
        for kd in c["knobs"]:
            full = {**d, **kd}
            for k, v in full.items():
                acc.setdefault(k, {}).setdefault(str(v), []).append(c["critic"]["score"])
    return {k: {v: round(statistics.mean(x), 4) for v, x in vs.items()} for k, vs in acc.items()}


# ------------------------------------------------------------------ Nano Banana refine probe (step 6)

GREEN = (0x00, 0xB0, 0x00)
PAL5 = {"bg": GREEN, "#": (0, 0, 0), "head": RGB["head"], "mid": RGB["mid"], "light": RGB["light"]}
NB_PROMPT = """The first image is a {N}x{N} pixel-art sprite of a robotic squid enemy for an NES-style game, enlarged
{f}x with nearest-neighbour, so every sprite pixel is a flat {f}x{f} square on a flat green background. The second
image is the concept art it was made from.

Refine the sprite into the best {N}x{N} pixel-art sprite of this character that an expert pixel artist would make.
You may move any pixels you like, but keep these exactly:
- the same {N}x{N} pixel grid: every pixel a flat {f}x{f} square on the same grid, same framing and size;
- only these colours: black #000000, red #F83800, coral #F87858, cream #F0D0B0, on the flat green #00B000 background;
- exact left-right mirror symmetry;
- a 1 px black outline around the sprite, and a black line between any two differently coloured parts;
- two eyes as cream shapes inside a black visor, framed by a cream ring; jaunty coral side fins; distinct tentacles.
No anti-aliasing, gradients, shading, texture or new colours. Return only the edited image."""


def nb_input(N):
    f = 512 // N
    im = Image.open(R4.V4 / "render" / f"render_{N}.png").convert("RGBA")
    bg = Image.new("RGBA", im.size, (*GREEN, 255))
    bg.alpha_composite(im)
    return bg.convert("RGB").resize((N * f, N * f), Image.NEAREST), f


def nb_call(N, seed, led):
    img, f = nb_input(N)
    concept = Image.open(CONCEPT).convert("RGBA")
    cb = Image.new("RGBA", concept.size, (255, 255, 255, 255))
    cb.alpha_composite(concept)
    cb = cb.convert("RGB")
    cb.thumbnail((768, 768))
    prompt = NB_PROMPT.format(N=N, f=f)
    p1, p2 = imaging.to_png(img), imaging.to_png(cb)
    body = {"model": NB, "modalities": ["image", "text"], "seed": seed,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt},
                                                      {"type": "image_url", "image_url": {"url": models.data_uri(p1)}},
                                                      {"type": "image_url", "image_url": {"url": models.data_uri(p2)}}]}]}
    ck = store.key({"step": "e1.nbrefine", "model": NB, "prompt": prompt, "images": [store.sha(p1), store.sha(p2)],
                    "seed": seed})
    CALL_KEYS.add(ck)
    try:
        resp, imgs, cost, cached = models.call(body, led, ck)
    except models.BudgetExceeded:
        raise
    except Exception as e:  # noqa: BLE001 - record and continue
        return {"error": str(e)[:300], "upstream_cost": 0.0, "cached": False, "call": ck}
    u = resp.get("usage") or {}
    up = (u.get("cost_details") or {}).get("upstream_inference_cost") or 0.0
    if not cached:
        led.add(up)            # BYOK: OpenRouter reports usage.cost 0; the real (Google) cost is the upstream cost
    return {"png": imgs[0] if imgs else None, "upstream_cost": up if not cached else 0.0, "upstream_cost_billed": up,
            "openrouter_cost": cost, "is_byok": u.get("is_byok"), "cached": cached, "call": ck,
            "text": (resp["choices"][0]["message"].get("content") or "")[:200],
            "error": None if imgs else "no image returned"}


def snap(png: bytes, N):
    """Fit the grid (scale + offset search maximising per-cell colour purity), then the majority palette colour per
    cell. -> (cells [N][N] of palette keys, purity per cell, fit)."""
    a = np.asarray(Image.open(io.BytesIO(png)).convert("RGB")).astype(float)
    H, W = a.shape[:2]
    keys = list(PAL5)
    pal = np.array([PAL5[k] for k in keys], float)
    q = np.argmin(((a[:, :, None, :] - pal[None, None]) ** 2).sum(-1), -1)       # nearest palette colour per px
    best = None
    for s in np.arange(0.95, 1.051, 0.01):
        cell = min(H, W) / N * s
        for oy in np.arange(-0.5, 0.51, 0.125):
            for ox in np.arange(-0.5, 0.51, 0.125):
                y0 = (H - cell * N) / 2 + oy * cell
                x0 = (W - cell * N) / 2 + ox * cell
                ys = (y0 + (np.arange(N)[:, None] + np.linspace(0.25, 0.75, 4)[None]) * cell).astype(int).ravel()
                xs = (x0 + (np.arange(N)[:, None] + np.linspace(0.25, 0.75, 4)[None]) * cell).astype(int).ravel()
                if ys.min() < 0 or xs.min() < 0 or ys.max() >= H or xs.max() >= W:
                    continue
                smp = q[np.ix_(ys, xs)].reshape(N, 4, N, 4).transpose(0, 2, 1, 3).reshape(N, N, 16)
                cnt = np.stack([(smp == k).sum(-1) for k in range(len(keys))], -1)
                pur = cnt.max(-1) / 16
                if best is None or pur.mean() > best[0]:
                    best = (pur.mean(), cnt.argmax(-1), pur, {"scale": round(float(s), 3), "oy": float(oy),
                                                             "ox": float(ox), "cell_px": round(float(cell), 2)})
    _, lab, pur, fit = best
    cells = [[keys[lab[y, x]] for x in range(N)] for y in range(N)]
    fit["purity_mean"] = round(float(pur.mean()), 3)
    return cells, pur, fit


def recover(cells, grid_A, leg):
    """Label recovery: a cell whose colour matches its old letter's colour keeps the letter; black -> ink '#' (or the
    old black-group letter); background -> '.'; otherwise the nearest old letter of that colour."""
    parts = leg["parts"]
    grp = {CH[i + 1]: p["group"] for i, p in enumerate(parts)}
    colour_of = lambda l: "bg" if l == "." else "#" if l == "#" or grp[l] == "black" else \
        {"dark": "head", "mid": "mid", "light": "light"}[grp[l]]
    N = len(grid_A)
    gA = np.array([list(r) for r in grid_A])
    colA = np.vectorize(colour_of)(gA)
    out = np.full((N, N), ".", dtype="<U1")
    for y in range(N):
        for x in range(N):
            c, old = cells[y][x], gA[y, x]
            if colour_of(old) == c:
                out[y, x] = old
            elif c == "bg":
                out[y, x] = "."
            elif c == "#":
                out[y, x] = "#"
            else:
                m = (colA == c) & (gA != "e")     # a new cream pixel is never an eye unless it was one
                if not m.any():
                    out[y, x] = "#"
                    continue
                ys, xs = np.nonzero(m)
                k = np.argmin((ys - y) ** 2 + (xs - x) ** 2)
                out[y, x] = gA[ys[k], xs[k]]
    return out


def repair(g, leg):
    """Deterministic repair on the final pixels: separation lines wherever two distinct non-black regions touch
    (cups on arms excepted, as in v4), the pixel coming from the side that gives (protected template parts never;
    otherwise the region behind); body specks -> line; mirror the left half. -> (grid, log)."""
    parts = leg["parts"]
    names = [p["name"] for p in parts]
    behind = L4.behind_fn(leg)
    black = {CH[i + 1] for i, p in enumerate(parts) if p["group"] == "black"}
    prot = {CH[names.index(n) + 1] for n in R3.PROTECTED}
    idx = lambda l: CH.index(l)
    cupish = lambda a, b: {names[idx(a) - 1], names[idx(b) - 1]} & {"suction_cups_front"} and \
        any(n.startswith("tentacles_front") for n in (names[idx(a) - 1], names[idx(b) - 1]))
    g = g.copy()
    H, W = g.shape
    lines = 0
    changed = True
    while changed:
        changed = False
        for y in range(H):
            for x in range(W):
                a = g[y, x]
                if a in ".#" or a in black:
                    continue
                for dy, dx in ((1, 0), (0, 1), (-1, 0), (0, -1)):
                    yy, xx = y + dy, x + dx
                    if not (0 <= yy < H and 0 <= xx < W):
                        continue
                    b = g[yy, xx]
                    if b in ".#" or b in black or b == a or cupish(a, b):
                        continue
                    if a in prot and b in prot:
                        continue          # template internals (ring | eye are never adjacent; the moat is black)
                    give = (y, x) if (b in prot and a not in prot) or (a not in prot and b not in prot and
                                                                      behind(idx(a), idx(b))) else (yy, xx)
                    g[give] = "#"
                    lines += 1
                    changed = True
                    break
    specks = 0
    for y in range(H):
        for x in range(W):
            a = g[y, x]
            if a in ".#" or a in black or a in prot:
                continue
            if parts[idx(a) - 1]["role"] != "body":
                continue
            if not any(0 <= y + dy < H and 0 <= x + dx < W and g[y + dy, x + dx] == a
                       for dy, dx in ((1, 0), (0, 1), (-1, 0), (0, -1))):
                g[y, x] = "#"
                specks += 1
    # outline: any sprite pixel 4-touching the background must be dark
    outl = 0
    for y in range(H):
        for x in range(W):
            if g[y, x] in ".#" or g[y, x] in black:
                continue
            if any(not (0 <= y + dy < H and 0 <= x + dx < W) or g[y + dy, x + dx] == "."
                   for dy, dx in ((1, 0), (0, 1), (-1, 0), (0, -1))):
                g[y, x] = "#"
                outl += 1
    half = g[:, : W // 2].copy()
    g[:, W - W // 2:] = half[:, ::-1]
    return g, {"lines_added": lines, "specks_to_line": specks, "outline_px": outl}


def grid_image(g, leg) -> Image.Image:
    parts = leg["parts"]
    N = g.shape[0]
    out = np.zeros((N, N, 4), np.uint8)
    for y in range(N):
        for x in range(N):
            l = g[y, x]
            if l == ".":
                continue
            if l == "#":
                out[y, x] = (0, 0, 0, 255)
            else:
                p = parts[CH.index(l) - 1]
                out[y, x] = (*imaging.rgb(L3.PALETTE[p["group"]][1]), 255)
    return Image.fromarray(out, "RGBA")


def run_nb(led, seeds=(1, 2)) -> dict:
    leg = json.loads((R4.V4 / "legend_v4.json").read_text())
    res = json.loads((R4.V4 / "results.json").read_text())
    d = OUT / "nbrefine"
    d.mkdir(parents=True, exist_ok=True)
    jobs = [(N, s) for N in SIZES for s in seeds]
    with ThreadPoolExecutor(3) as ex:
        calls = list(ex.map(lambda j: nb_call(j[0], j[1], led), jobs))
    out = {}
    for (N, s), c in zip(jobs, calls):
        rec = {k: v for k, v in c.items() if k != "png"}
        rec["seed"] = s
        nb_input(N)[0].save(d / f"input_{N}.png")
        if not c.get("png"):
            rec["gate"] = {"ok": False, "fails": ["no image: " + str(c.get("error"))]}
            out.setdefault(N, []).append(rec)
            continue
        raw = imaging.from_png(c["png"])
        raw.save(d / f"raw_{N}_s{s}.png")
        rec["raw_size"] = list(raw.size)
        cells, pur, fit = snap(c["png"], N)
        rec["fit"] = fit
        gA = res["renders"][str(N)]["grid"]
        colA = np.array([[("bg" if l == "." else "#" if l in "#dgk" else
                           {"a": "head", "b": "mid", "f": "mid", "j": "mid", "l": "mid"}.get(l, "light")) for l in r]
                         for r in gA])
        cn = np.array(cells)
        rec["cells_changed_colour"] = int((cn != colA).sum())
        g0 = recover(cells, gA, leg)
        grid_image(g0, leg).save(d / f"snapped_{N}_s{s}.png")
        g1, rlog = repair(g0, leg)
        rec["repair"] = rlog
        im = grid_image(g1, leg)
        im.save(d / f"repaired_{N}_s{s}.png")
        grid = ["".join(r) for r in g1]
        info = {"grid": grid, "tile": {}}
        light = imaging.rgb(L4.PALETTE["light"][1])
        legal, tinfo = R1.tile_reduce(im, salient=[light], salient_w=R3.SALIENT_W, symmetric=True)
        info["tile"] = tinfo
        rec["gate"] = gate(legal, info, leg)
        a_png = np.array(Image.open(R4.V4 / "render" / f"render_{N}.png"))
        rec["changed_px_vs_A"] = int((np.array(legal) != a_png).any(-1).sum())
        rec["changed_px_snapped_vs_A"] = int((np.array(grid_image(g0, leg)) != a_png).any(-1).sum())
        rec["grid"] = grid
        rec["png_path"] = str(d / f"repaired_{N}_s{s}.png")
        legal.save(d / f"final_{N}_s{s}.png")
        rec["final_path"] = str(d / f"final_{N}_s{s}.png")
        out.setdefault(N, []).append(rec)
        print("NB", N, s, rec["fit"], "changed colour cells", rec["cells_changed_colour"], rlog, rec["gate"]["fails"],
              "px vs A", rec["changed_px_vs_A"])
    return out


# ------------------------------------------------------------------ judge (step 7)

def judge_pair(cand_png: bytes, a_png: bytes, N, led) -> dict:
    """gpt-5.6-sol, both orders x 2 samples. Win only if the candidate wins (net) in both orders; loss likewise."""
    q = (f"The first image is concept art of a robotic squid enemy for an NES-style game. Then come two {N}x{N} "
         "pixel-art sprites of it, A and B (shown enlarged). Which is the better sprite of this character at game size: "
         "it reads instantly as this squid, with clear eyes, jaunty fins and distinct tentacles, clean lines, and no "
         "muddy black clumps or stray pixels? Reply with JSON only: "
         '{"answer": "A" | "B" | "tie", "reason": "<one short sentence>"}')
    votes = {"cand_first": [], "A_first": []}
    reasons = []
    cost = 0.0
    for order in ("cand_first", "A_first"):
        first, second = (cand_png, a_png) if order == "cand_first" else (a_png, cand_png)
        for smp in range(2):
            content, red = [{"type": "text", "text": q}], []
            for s in ("Concept:", str(CONCEPT), "Sprite A:", first, "Sprite B:", second):
                if isinstance(s, str) and not s.endswith(".png"):
                    content.append({"type": "text", "text": s}); red.append(s)
                else:
                    p, h = D.image_part(s)
                    content.append(p); red.append(h)
            body = {"model": GPT, "response_format": {"type": "json_object"},
                    "messages": [{"role": "user", "content": content}]}
            ck = store.key({"e1_judge": q, "state": red, "sample": smp})
            CALL_KEYS.add(ck)
            resp, _, c, cached = models.call(body, led, ck)
            cost += 0 if cached else c
            txt = resp["choices"][0]["message"].get("content") or ""
            m = re.search(r"\{.*\}", txt, re.S)
            ans = (json.loads(m.group(0)).get("answer") if m else None) or "tie"
            ans = str(ans).strip().upper()
            pick = "tie" if ans not in ("A", "B") else ("cand" if (ans == "A") == (order == "cand_first") else "A")
            votes[order].append(pick)
            reasons.append(f"{order}#{smp}: {pick} - {(json.loads(m.group(0)).get('reason') if m else txt)[:140]}")
    net = {o: v.count("cand") - v.count("A") for o, v in votes.items()}
    res = "win" if all(n > 0 for n in net.values()) else "loss" if all(n < 0 for n in net.values()) else "tie"
    return {"result": res, "votes": votes, "net": net, "reasons": reasons, "cost": round(cost, 4)}


def run_judge(entries, led) -> dict:
    """entries: [(N, name, png bytes)] judged against A at N."""
    def one(e):
        N, name, png = e
        a = (R4.V4 / "render" / f"render_{N}.png").read_bytes()
        return N, name, judge_pair(png, a, N, led)

    with ThreadPoolExecutor(6) as ex:
        rows = list(ex.map(one, entries))
    out = {}
    for N, name, r in rows:
        out.setdefault(N, {})[name] = r
    return out


# ------------------------------------------------------------------ blind set (step 8)

def blind(sw, cr, nb, extra) -> dict:
    """Per size: A + critic top 2 + 2 random survivors + the best gated NB refine (+ extras), shuffled, lettered.
    Images are copied to blind/<N>_<letter>.png (the path names no candidate); key.json maps letters back."""
    rng = random.Random(SEED + 1)
    d = OUT / "blind"
    d.mkdir(parents=True, exist_ok=True)
    for p in d.glob("*.png"):
        p.unlink()
    key, panel = {}, {}
    for N in SIZES:
        byid = {c["id"]: c for c in sw[N]["candidates"]}
        items = {}   # png bytes -> {roles, source}

        def add(png, role, src):
            k = np.array(Image.open(io.BytesIO(png)).convert("RGBA")).tobytes()
            items.setdefault(k, {"png": png, "roles": [], "sources": []})
            items[k]["roles"].append(role)
            items[k]["sources"].append(src)

        add((R4.V4 / "render" / f"render_{N}.png").read_bytes(), "A (v4 control)", str(R4.V4 / "render" / f"render_{N}.png"))
        for i, t in enumerate(cr[N]["top2"]):
            add(byid[t["id"]]["png"], f"B1 critic top {i + 1}", t["id"])
        for i, t in enumerate(cr[N]["random2"]):
            add(byid[t["id"]]["png"], f"random survivor {i + 1} (control)", t["id"])
        good = [r for r in nb.get(N, []) if r["gate"]["ok"]]
        if good:
            b = min(good, key=lambda r: r["changed_px_vs_A"] if r["changed_px_vs_A"] else 10 ** 6)
            add(Path(b["final_path"]).read_bytes(), "B2 Nano Banana refine (best gated)", b["final_path"])
        for (n2, role, path) in extra:
            if n2 == N:
                add(Path(path).read_bytes(), role, str(path))
        lst = list(items.values())
        rng.shuffle(lst)
        key[N], panel[N] = {}, []
        for i, it in enumerate(lst):
            L = "abcdefgh"[i]
            p = d / f"{N}_{L}.png"
            p.write_bytes(it["png"])
            key[N][L] = {"roles": it["roles"], "sources": it["sources"]}
            panel[N].append((L, str(p)))
    (d / "key.json").write_text(json.dumps({"note": "letter -> candidate for the ab-blind panel; do not show on screen",
                                            "key": key}, indent=1))
    return panel


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=3.0)
    ap.add_argument("--skip-nb", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    led = models.Ledger(a.budget)
    ctx, mats = setup()
    sw = sweep(ctx, mats)
    fl = run_flagger(led)
    print("flagger", json.dumps(fl["summary"]), "spent", round(led.spent, 4))
    cr = run_critic(sw, led)
    print("critic", json.dumps({str(k): {kk: v[kk] for kk in ("A", "top2", "spread", "clear_winner")} if k != "sentinels"
                                else v for k, v in cr.items()}, default=str)[:3000], "spent", round(led.spent, 4))
    nb = {} if a.skip_nb else run_nb(led)
    print("nb spent", round(led.spent, 4))
    j16b = R4.V4 / "render" / "render_16b.png"
    entries = []
    for N in SIZES:
        byid = {c["id"]: c for c in sw[N]["candidates"]}
        for i, t in enumerate(cr[N]["top2"]):
            entries.append((N, f"critic_top{i + 1}:{t['id']}", byid[t["id"]]["png"]))
        for i, t in enumerate(cr[N]["random2"]):
            entries.append((N, f"random{i + 1}:{t['id']}", byid[t["id"]]["png"]))
        good = [r for r in nb.get(N, []) if r["gate"]["ok"]]
        if good:
            b = min(good, key=lambda r: r["changed_px_vs_A"] if r["changed_px_vs_A"] else 10 ** 6)
            entries.append((N, f"nb_refine:s{b['seed']}", Path(b["final_path"]).read_bytes()))
    entries.append((16, "johnny_16b", j16b.read_bytes()))
    jd = run_judge(entries, led)
    print("judge", {N: {k: v["result"] for k, v in r.items()} for N, r in jd.items()}, "spent", round(led.spent, 4))
    extra = [(16, "Johnny's hand edit 16b (reference)", j16b)]
    # the trial rule on its own at 20 (ring as edge, everything else v4), if it survived
    re20 = next((c for c in sw[20]["candidates"] if {"ring_edge": True} in c["knobs"]), None)
    if re20:
        p = OUT / "candidates" / "20" / f"{re20['id']}.png"
        extra.append((20, "ring-as-edge trial alone (v4 + ring_edge)", p))
    panel = blind(sw, cr, nb, extra)
    # which candidate does Johnny's 16b equal?
    j_arr = np.array(Image.open(j16b).convert("RGBA")).tobytes()
    j_match = [c["id"] for c in sw[16]["candidates"] if np.array(Image.open(io.BytesIO(c["png"]))).tobytes() == j_arr]
    sp = spend()
    res = {"budget": a.budget, "spent_total_incl_upstream": sp["total_usd"], "spend": sp,
           "ledger_this_run": {"spent": round(led.spent, 4), "calls": led.calls, "cached": led.cached},
           "sweep": {N: {k: v for k, v in s.items() if k != "candidates"} |
                     {"candidates": [{k: v for k, v in c.items() if k != "png"} for c in s["candidates"]]}
                     for N, s in sw.items()},
           "flagger": fl, "critic": cr, "nbrefine": nb, "judge": jd, "blind_panel": panel,
           "johnny_16b_equals": j_match}
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=str))
    print("spend", json.dumps(sp), "| this run", round(led.spent, 4), "calls", led.calls, "cached", led.cached)


if __name__ == "__main__":
    main()
