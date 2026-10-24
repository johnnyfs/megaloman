"""Ladder v2: symmetry preservation on top of the milestone-2 ladder (ladder.py, v1).

1. Detect the source's mirror symmetry: for each candidate vertical axis (half-pixel steps)
   score IoU(mask, mirror(mask)) for the silhouette and each part; keep the best global axis.
2. At each rung (source mode), measure mirror agreement of parts judged symmetric. If any
   breaks, build two candidates (left->right, right->left mirrored across the rung's axis,
   quantised to a pixel edge or centre) and choose among {original, L, R}: a vision judge
   (N=3, majority, cached, ledger-budgeted) with a deterministic fallback (health score).
"""
from __future__ import annotations

import io
import re
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import ladder as L

VERSION = "v2"
SYM_PART = 0.9         # part counts as symmetric if its own-axis mirror IoU >= this (axis near global)
TOL = 0.06             # a rung breaks if part IoU drops more than this below its source score
JUDGE_MODEL = "openai/gpt-5.6-sol"
SAMPLES = 3


# ---- symmetry measurement -------------------------------------------------------------
def mirror_cols(w: int, axis2: int) -> np.ndarray:
    """axis2 = 2*axis in pixel-edge coords (pixel c spans [c, c+1]). Mirror of c is axis2-1-c; -1 if off-grid."""
    m = axis2 - 1 - np.arange(w)
    m[(m < 0) | (m >= w)] = -1
    return m


def mirror_iou(mask: np.ndarray, axis2: int) -> float:
    w = mask.shape[1]
    mc = mirror_cols(w, axis2)
    valid = mc >= 0
    a = mask[:, valid]
    b = mask[:, mc[valid]]
    out = mask[:, ~valid].sum()          # pixels with no partner count as mismatches
    u = (a | b).sum() + out
    return float((a & b).sum() / u) if u else 1.0


def best_axis(mask: np.ndarray, lo=0.35, hi=0.65) -> tuple[int, float]:
    w = mask.shape[1]
    best = (w, -1.0)
    for a2 in range(int(2 * w * lo), int(2 * w * hi) + 1):
        s = mirror_iou(mask, a2)
        if s > best[1]:
            best = (a2, s)
    return best


def source_symmetry(src: np.ndarray, legend: dict) -> dict:
    a2, g = best_axis(src > 0)
    parts = {}
    for i, p in enumerate(legend["parts"]):
        m = src == i + 1
        own_a2, own = best_axis(m) if m.any() else (a2, 0.0)
        parts[p["name"]] = {"iou_global_axis": round(mirror_iou(m, a2), 4), "best_axis_px": own_a2 / 2,
                            "iou_own_axis": round(own, 4)}
        # symmetric = the part's own shape mirrors well about an axis within 1% width of the global one
        parts[p["name"]]["symmetric"] = bool(own >= SYM_PART and abs(own_a2 - a2) / 2 <= 0.01 * src.shape[1])
    # label-level symmetry: fraction of fg pixels whose mirror has the same label
    mc = mirror_cols(src.shape[1], a2)
    v = mc >= 0
    A, B = src[:, v], src[:, mc[v]]
    fg = (A > 0) | (B > 0)
    lab_agree = float(((A == B) & fg).sum() / fg.sum())
    return {"width": src.shape[1], "axis_px": a2 / 2, "axis_frac": round(a2 / 2 / src.shape[1], 4),
            "silhouette_iou": round(g, 4), "label_agreement": round(lab_agree, 4), "parts": parts}


def rung_axis2(sym: dict, s: int) -> int:
    """Quantise the source axis to this rung: nearest half-pixel (edge if even-ish, centre if odd-ish)."""
    return int(round(2 * sym["axis_frac"] * s))


def rung_symmetry(lab: np.ndarray, legend: dict, sym: dict, a2: int) -> dict:
    out = {}
    for i, p in enumerate(legend["parts"]):
        if sym["parts"][p["name"]]["symmetric"]:
            out[p["name"]] = round(mirror_iou(lab == i + 1, a2), 4)
    return out


def broken(rs: dict, sym: dict) -> list[str]:
    return [k for k, v in rs.items() if v < sym["parts"][k]["iou_global_axis"] - TOL]


def mirror_half(lab: np.ndarray, a2: int, keep: str) -> np.ndarray:
    """keep='left': copy the left half onto the right (and vice versa). Axis column (odd a2) is kept."""
    w = lab.shape[1]
    mc = mirror_cols(w, a2)
    out = lab.copy()
    cols = np.arange(w)
    tgt = (2 * cols + 1 > a2) if keep == "left" else (2 * cols + 1 < a2)
    tgt &= mc >= 0
    out[:, cols[tgt]] = lab[:, mc[tgt]]
    return out


# ---- choosing --------------------------------------------------------------------------
def health_score(h: list[dict]) -> int:
    return sum({"green": 2, "amber": 1, "red": 0}[r["status"]] for r in h)


def fallback(cands: dict, src, legend, sst) -> tuple[str, dict]:
    """Deterministic: health score, then mean plain-IoU, then smaller area error. Original wins ties."""
    m = {}
    for k, lab in cands.items():
        h = L.health(lab, src, legend, sst)
        aerr = float(np.mean([abs(r["area"] - r["expected"]) / max(r["expected"], 1) for r in h]))
        m[k] = {"health_score": health_score(h), "mean_iou": round(float(np.mean([r["iou_plain"] for r in h])), 4),
                "mean_area_err": round(aerr, 4)}
    order = {"original": 0, "left": 1, "right": 2}
    best = max(m, key=lambda k: (m[k]["health_score"], m[k]["mean_iou"], -m[k]["mean_area_err"], -order[k]))
    return best, m


PROMPT = """You are judging tiny pixel-art region maps of one game enemy: {concept}
Each flat colour is one named part:
{legend}
The first image is the concept art. Next are three candidate region maps of the same {s}x{s} grid,
shown enlarged with nearest-neighbour, labelled A, B, C. Which candidate best reads as the squid,
with clean, recognisable parts (and a sensible left/right balance)? Reply with JSON only:
{{"choice": "A"|"B"|"C", "reason": "<one short sentence>"}}"""
CONCEPT = "a front view of a cartoon robotic squid: a red domed mantle with side fins, a pale face frame around a dark visor with two glowing eyes, side pods, and four tentacles with joints and suction cups."


def panel(cands: list[np.ndarray], legend, h=240) -> bytes:
    img = Image.new("RGB", (len(cands) * (h + 20) + 20, h + 50), (90, 90, 96))
    d = ImageDraw.Draw(img)
    f = L._font(28)
    for k, lab in enumerate(cands):
        x = 20 + k * (h + 20)
        d.text((x, 4), "ABC"[k], fill=(255, 255, 255), font=f)
        img.paste(L.colorize(lab, legend).resize((h, h), Image.NEAREST), (x, 40))
    b = io.BytesIO()
    img.save(b, "PNG")
    return b.getvalue()


def vision_judge(cands: dict, s: int, legend, concept_png: bytes, ledger, rng_seed: int) -> dict:
    from . import models, store
    keys = ["original", "left", "right"]
    # deterministic per-rung shuffle so position bias doesn't always favour one candidate
    perm = list(np.random.default_rng(rng_seed).permutation(3))
    order = [keys[j] for j in perm]
    png = panel([cands[k] for k in order], legend)
    leg = "\n".join(f"- {p['color']} {p['name']}: {p['where']}" for p in legend["parts"])
    prompt = PROMPT.format(concept=CONCEPT, legend=leg, s=s)
    body = {"model": JUDGE_MODEL, "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": models.data_uri(concept_png)}},
                {"type": "image_url", "image_url": {"url": models.data_uri(png)}}]}]}
    pblob, cblob = store.put_blob(png), store.put_blob(concept_png)
    recs = []
    for i in list(range(SAMPLES)) + [SAMPLES + k for k in range(2)]:   # up to 2 retries replace failed samples
        if len([r for r in recs if r.get("choice")]) >= SAMPLES or (i >= SAMPLES and len(recs) >= SAMPLES + 2):
            break
        if i >= SAMPLES and all(r.get("choice") for r in recs):
            break
        ck = store.key({"ladder_sym_judge": VERSION, "prompt": prompt, "images": [cblob, pblob], "model": JUDGE_MODEL, "sample": i})
        try:
            resp, _, cost, cached = models.call(body, ledger, ck)
            m = json.loads(re.search(r"\{.*\}", resp["choices"][0]["message"].get("content") or "", re.S).group(0))
            letter = str(m.get("choice", "")).strip().upper()[:1]
            pick = order["ABC".index(letter)] if letter in "ABC" and letter else None
            recs.append({"sample": i, "choice": pick, "letter": letter, "reason": m.get("reason"), "cost": cost, "cached": cached})
        except Exception as e:  # budget or parse failure: sample abstains
            recs.append({"sample": i, "choice": None, "error": str(e)[:200]})
    votes = Counter(r["choice"] for r in recs if r.get("choice"))
    top = votes.most_common(1)
    winner = top[0][0] if top and top[0][1] >= 2 else None
    return {"order_shown": order, "panel_blob": pblob, "samples": recs, "votes": dict(votes), "majority": winner,
            "cost": round(sum(r.get("cost") or 0 for r in recs), 5)}


# ---- run -------------------------------------------------------------------------------
def run(region_path: Path, legend_path: Path, concept_path: Path, out: Path, targets=(32, 24, 20, 16),
        budget=1.0, use_judge=True) -> dict:
    from . import models
    out.mkdir(parents=True, exist_ok=True)
    legend = json.loads(legend_path.read_text())
    src = L.load_labels(region_path, legend)
    n = len(legend["parts"])
    sst = L.source_stats(src, n)
    sym = source_symmetry(src, legend)
    ledger = models.Ledger(budget)
    cim = Image.open(concept_path).convert("RGB")
    cim.thumbnail((256, 256))
    cb = io.BytesIO()
    cim.save(cb, "PNG")
    concept_png = cb.getvalue()

    decisions: dict[int, dict] = {}   # rung size -> decision (source mode: identical across targets)

    def decide(s: int, lab: np.ndarray) -> np.ndarray:
        if s in decisions:
            return decisions[s]["_lab"]
        a2 = rung_axis2(sym, s)
        rs = rung_symmetry(lab, legend, sym, a2)
        br = broken(rs, sym)
        rec = {"axis_px": a2 / 2, "axis_on": "pixel centre" if a2 % 2 else "pixel edge", "part_mirror_iou": rs,
               "broken": br, "mirrored": False, "choice": "original"}
        chosen = lab
        if br:
            cands = {"original": lab, "left": mirror_half(lab, a2, "left"), "right": mirror_half(lab, a2, "right")}
            fb, metrics = fallback(cands, src, legend, sst)
            rec["fallback_choice"], rec["candidate_metrics"] = fb, metrics
            rec["candidate_part_mirror_iou"] = {k: rung_symmetry(v, legend, sym, a2) for k, v in cands.items()}
            j = vision_judge(cands, s, legend, concept_png, ledger, s) if use_judge else None
            rec["judge"] = j
            pick = j["majority"] if j and j["majority"] else fb
            rec["choice"], rec["chosen_by"] = pick, "judge" if j and j["majority"] else "fallback"
            rec["mirrored"] = pick != "original"
            chosen = cands[pick]
            cd = out / "candidates"
            cd.mkdir(exist_ok=True)
            for k, v in cands.items():
                L.colorize(v, legend).save(cd / f"rung_{s}_{k}.png")
        rec["_lab"] = chosen
        decisions[s] = rec
        return chosen

    res = {"version": VERSION, "base": "ladder.py v1 (source mode)", "source": str(region_path), "legend": str(legend_path),
           "fg_threshold": L.FG_THRESHOLD, "min_share": L.MIN_SHARE,
           "source_components": {p["name"]: sst["parts"][i]["comps"] for i, p in enumerate(legend["parts"])}, "targets": {}}
    for t in targets:
        rs = [(s, decide(s, lab)) for s, lab in L.ladder(src, legend, t, mode="source")]
        hs = [L.health(lab, src, legend, sst) for _, lab in rs]
        d = out / f"t{t}"
        d.mkdir(parents=True, exist_ok=True)
        for s, lab in rs:
            L.colorize(lab, legend).save(d / f"rung_{s}.png")
        L.filmstrip(rs, hs, legend, out / f"filmstrip_t{t}.png")
        first = {}
        for i, p in enumerate(legend["parts"]):
            first[p["name"]] = {"first_degrade": next((s for (s, _), h in zip(rs, hs) if h[i]["status"] != "green"), None),
                                "first_break": next((s for (s, _), h in zip(rs, hs) if h[i]["status"] == "red"), None)}
        res["targets"][t] = {"rungs": [s for s, _ in rs], "score": sum(health_score(h) for h in hs),
                             "mean_iou": round(float(np.mean([r["iou_plain"] for h in hs for r in h])), 4),
                             "first": first, "health": {str(s): h for (s, _), h in zip(rs, hs)}}
    L.legend_strip(legend, out / "legend_strip.png")
    (out / "results.json").write_text(json.dumps(res, indent=1, default=str))
    symout = {"version": VERSION, "params": {"sym_part": SYM_PART, "tol": TOL, "judge_model": JUDGE_MODEL, "samples": SAMPLES,
                                             "budget": budget},
              "source": sym,
              "rungs": {str(s): {k: v for k, v in r.items() if k != "_lab"} for s, r in sorted(decisions.items(), reverse=True)},
              "cost": {"spent": round(ledger.spent, 5), "calls": ledger.calls, "cached": ledger.cached}}
    (out / "symmetry.json").write_text(json.dumps(symout, indent=1, default=str))
    return {"results": res, "symmetry": symout}


if __name__ == "__main__":
    import sys
    root = Path(__file__).resolve().parents[2] / "workspace/experiments/regions"
    r = run(root / "m1/gpt-s1/regions.png", root / "m1/legend.json", root / "m1/concept_on_white.png",
            root / "ladder/v2_symmetry", use_judge="--no-judge" not in sys.argv)
    s = r["symmetry"]
    print("source", {k: s["source"][k] for k in ("axis_px", "silhouette_iou", "label_agreement")})
    for k, v in s["source"]["parts"].items():
        print("  ", k, v)
    for sz, d in s["rungs"].items():
        print(sz, d["axis_px"], d["broken"], d["choice"], d.get("chosen_by"), d.get("fallback_choice"),
              (d.get("judge") or {}).get("votes"))
    print("cost", s["cost"])
    for t, v in r["results"]["targets"].items():
        print(t, v["score"], v["mean_iou"])
