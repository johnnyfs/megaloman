"""Region-aware downscale ladder with per-part health (milestone 2, observe only).

No model calls. Input: a flat region map (legend colours, transparent/white
background) plus legend.json. Output: per-rung label maps and health records.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

FG_THRESHOLD = 0.5      # foreground if summed part coverage >= this
MIN_SHARE = 0.3         # higher-priority part wins a contested pixel above this share of fg
EIGHT = np.ones((3, 3), int)
STATUS_RGB = {"green": (40, 180, 70), "amber": (240, 170, 20), "red": (220, 40, 40)}


def hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def load_labels(path: Path, legend: dict) -> np.ndarray:
    """Label map: 0 = background, i+1 = legend part i (nearest legend colour)."""
    im = np.array(Image.open(path).convert("RGBA")).astype(int)
    rgb, a = im[..., :3], im[..., 3]
    cols = np.array([hex_rgb(p["color"]) for p in legend["parts"]])
    d = ((rgb[:, :, None, :] - cols[None, None]) ** 2).sum(-1)
    lab = d.argmin(-1) + 1
    bg = (a < 128) | (rgb.min(-1) > 245)
    lab[bg] = 0
    return lab.astype(np.int32)


def coverage(labels: np.ndarray, n: int, size: int) -> np.ndarray:
    """Per-part area coverage (n, size, size) via box filter of one-hot masks."""
    out = np.zeros((n, size, size), np.float32)
    for i in range(n):
        m = Image.fromarray((labels == i + 1).astype(np.float32), "F")
        out[i] = np.array(m.resize((size, size), Image.BOX))
    return out


def assign(cov: np.ndarray, priority: np.ndarray, fg_thr=FG_THRESHOLD, min_share=MIN_SHARE) -> np.ndarray:
    fg = cov.sum(0)
    share = cov / np.maximum(fg, 1e-9)
    # rank: priority dominates among parts passing min_share; else majority
    eligible = share >= min_share
    score = np.where(eligible, priority[:, None, None] * 10.0 + share, -1.0)
    best = score.argmax(0)
    has = eligible.any(0)
    best = np.where(has, best, cov.argmax(0))
    lab = best + 1
    lab[fg < fg_thr] = 0
    return lab.astype(np.int32)


def derive_rungs(src: int, target: int) -> list[int]:
    rungs, s = [], 512
    while s > target:
        if s < src:
            rungs.append(s)
        s //= 2
    return rungs + [target]


def ladder(region_map: np.ndarray, legend: dict, target: int, rungs=None, mode="source") -> list[tuple[int, np.ndarray]]:
    """mode 'source': each rung from the source; 'chain': each rung from the previous."""
    n = len(legend["parts"])
    pr = np.array([p["priority"] for p in legend["parts"]], float)
    rungs = rungs or derive_rungs(region_map.shape[0], target)
    out, prev = [], region_map
    for s in rungs:
        base = region_map if mode == "source" else prev
        lab = assign(coverage(base, n, s), pr)
        out.append((s, lab))
        prev = lab
    return out


def adjacency(lab: np.ndarray) -> set[tuple[int, int]]:
    pairs = set()
    for a, b in ((lab[:, :-1], lab[:, 1:]), (lab[:-1], lab[1:])):
        m = (a != b) & (a > 0) & (b > 0)
        for x, y in set(zip(a[m].tolist(), b[m].tolist())):
            pairs.add((min(x, y), max(x, y)))
    return pairs


def components(mask: np.ndarray, min_area: float = 0) -> int:
    lab, k = ndimage.label(mask, EIGHT)
    if k == 0:
        return 0
    areas = ndimage.sum(mask, lab, range(1, k + 1))
    return int((areas >= max(min_area, 1)).sum())


def thickness(mask: np.ndarray) -> int:
    """Thickness of the thinnest component = 2*max chessboard inset - 1."""
    lab, k = ndimage.label(mask, EIGHT)
    if k == 0:
        return 0
    dt = ndimage.distance_transform_cdt(np.pad(mask, 1), "chessboard")[1:-1, 1:-1]
    return int(min(2 * dt[lab == i].max() - 1 for i in range(1, k + 1)))


def source_stats(src: np.ndarray, n: int) -> dict:
    tot = src.size
    st = {"adj": adjacency(src), "parts": []}
    for i in range(n):
        m = src == i + 1
        lab, k = ndimage.label(m, EIGHT)
        areas = sorted(ndimage.sum(m, lab, range(1, k + 1)).tolist(), reverse=True) if k else []
        st["parts"].append({"frac": m.sum() / tot, "comp_areas": areas, "side": src.shape[0],
                            "comps": sum(a >= 0.002 * m.sum() for a in areas)})
    return st


def health(lab: np.ndarray, src: np.ndarray, legend: dict, sst: dict) -> list[dict]:
    s = lab.shape[0]
    n = len(legend["parts"])
    adj = adjacency(lab)
    cov = coverage(src, n, s)
    recs = []
    for i, p in enumerate(legend["parts"]):
        m = lab == i + 1
        area = int(m.sum())
        exp = sst["parts"][i]["frac"] * s * s
        plain = cov[i] >= 0.5
        union = (m | plain).sum()
        iou = float((m & plain).sum() / union) if union else 1.0
        comps = components(m)
        # components expected to survive at this rung: those >= 2 px after scaling
        scale = (s / sst["parts"][i]["side"]) ** 2
        sc = sum(a * scale >= 2 for a in sst["parts"][i]["comp_areas"])
        src_nb = {b if a == i + 1 else a for a, b in sst["adj"] if i + 1 in (a, b)}
        nb = {b if a == i + 1 else a for a, b in adj if i + 1 in (a, b)}
        lost = sorted(legend["parts"][j - 1]["name"] for j in src_nb - nb)
        th = thickness(m)
        reasons_r, reasons_a = [], []
        if area == 0:
            reasons_r.append("vanished")
        else:
            if area < p["min_px"] ** 2:
                reasons_r.append(f"area {area} < min_px^2 {p['min_px']**2}")
            if comps < sc:
                reasons_r.append(f"merged {sc}->{comps}")
            if comps > sc > 0:
                reasons_a.append(f"split {sc}->{comps}")
            ratio = area / exp if exp else 0
            if not 0.6 <= ratio <= 1.6:
                reasons_a.append(f"area x{ratio:.2f}")
            if iou < 0.6:
                reasons_a.append(f"iou {iou:.2f}")
            if th <= 1 and p["role"] in ("limb", "body"):
                reasons_a.append("1px thin")
            if lost:
                reasons_a.append("lost adj " + ",".join(lost))
        status = "red" if reasons_r else "amber" if reasons_a else "green"
        recs.append({"part": p["name"], "status": status, "present": area > 0, "area": area,
                     "expected": round(exp, 1), "components": comps, "src_components": sc,
                     "thickness": th, "iou_plain": round(iou, 3), "lost_adjacency": lost,
                     "reasons": reasons_r + reasons_a})
    return recs


def colorize(lab: np.ndarray, legend: dict) -> Image.Image:
    pal = np.array([(255, 255, 255)] + [hex_rgb(p["color"]) for p in legend["parts"]], np.uint8)
    return Image.fromarray(pal[lab])


def _font(sz):
    for f in ("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"):
        if Path(f).exists():
            return ImageFont.truetype(f, sz)
    return ImageFont.load_default()


def filmstrip(rungs, healths, legend, path: Path, h=200):
    n = len(legend["parts"])
    dot, gap, lab_h = 16, 4, 30
    W = len(rungs) * (h + 14) + 14
    H = lab_h + h + 10 + dot + 14
    img = Image.new("RGB", (W, H), (24, 24, 28))
    d = ImageDraw.Draw(img)
    f = _font(22)
    for k, ((s, lab), hl) in enumerate(zip(rungs, healths)):
        x = 14 + k * (h + 14)
        d.text((x, 3), f"{s}px", fill=(230, 230, 230), font=f)
        img.paste(colorize(lab, legend).resize((h, h), Image.NEAREST), (x, lab_h))
        dw = min(dot, (h - gap * (n - 1)) // n)
        for j, r in enumerate(hl):
            dx = x + j * (dw + gap)
            d.ellipse([dx, lab_h + h + 8, dx + dw, lab_h + h + 8 + dw], fill=STATUS_RGB[r["status"]])
    img.save(path)


def legend_strip(legend, path: Path):
    f = _font(18)
    img = Image.new("RGB", (len(legend["parts"]) * 150, 30), (24, 24, 28))
    d = ImageDraw.Draw(img)
    for j, p in enumerate(legend["parts"]):
        d.rectangle([j * 150, 6, j * 150 + 16, 22], fill=hex_rgb(p["color"]))
        d.text((j * 150 + 22, 4), p["name"][:13], fill=(220, 220, 220), font=f)
    img.save(path)


def run(region_path: Path, legend_path: Path, out: Path, targets=(32, 24, 20, 16)) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    legend = json.loads(legend_path.read_text())
    src = load_labels(region_path, legend)
    n = len(legend["parts"])
    sst = source_stats(src, n)
    res = {"source": str(region_path), "legend": str(legend_path), "fg_threshold": FG_THRESHOLD,
           "min_share": MIN_SHARE, "source_components": {p["name"]: sst["parts"][i]["comps"] for i, p in enumerate(legend["parts"])},
           "targets": {}}
    for t in targets:
        res["targets"][t] = {}
        for mode in ("source", "chain"):
            rs = ladder(src, legend, t, mode=mode)
            hs = [health(lab, src, legend, sst) for _, lab in rs]
            d = out / f"t{t}" / mode
            d.mkdir(parents=True, exist_ok=True)
            for s, lab in rs:
                colorize(lab, legend).save(d / f"rung_{s}.png")
            filmstrip(rs, hs, legend, out / f"filmstrip_t{t}_{mode}.png")
            first = {}
            for i, p in enumerate(legend["parts"]):
                fa = next((s for (s, _), h in zip(rs, hs) if h[i]["status"] != "green"), None)
                fr = next((s for (s, _), h in zip(rs, hs) if h[i]["status"] == "red"), None)
                first[p["name"]] = {"first_degrade": fa, "first_break": fr}
            iou = float(np.mean([r["iou_plain"] for h in hs for r in h]))
            score = sum({"green": 2, "amber": 1, "red": 0}[r["status"]] for h in hs for r in h)
            res["targets"][t][mode] = {"rungs": [s for s, _ in rs], "score": score, "mean_iou": round(iou, 4), "first": first,
                                       "health": {str(s): h for (s, _), h in zip(rs, hs)}}
    legend_strip(legend, out / "legend_strip.png")
    (out / "results.json").write_text(json.dumps(res, indent=1, default=str))
    return res


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2] / "workspace/experiments/regions"
    r = run(root / "m1/gpt-s1/regions.png", root / "m1/legend.json", root / "m2")
    for t, v in r["targets"].items():
        print(t, {m: (v[m]["score"], v[m]["mean_iou"]) for m in v})
        for name, f in v["source"]["first"].items():
            print("  ", name, f, "| chain", v["chain"]["first"][name])
