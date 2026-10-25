"""Squid + snipe through the two untried open routes (survey #1 SDXL + Pixel Icon LoRA, #3 Qwen-Image-Edit-2511 +
Render It) at 64/32/16, one run per (model, subject, size), best seed by grid cleanliness. Snap + palette lock.

    uv run --with modal python -m megaloman.open_models_experiment

Writes <OUT>/<model>/pipeline.json per model and <OUT>/compare/pipeline.json (best picks side by side).
"""
from __future__ import annotations

import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from . import pipeline_manifest as pm
from .flux_kontext_experiment import CONCEPTS, block_mode, concept_palette, lock, on_white, up

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "workspace/experiments/regions/snipe/compact/open-models"
STYLE = ROOT / "workspace/experiments/regions/snipe/compact/grid-ladder/style_refs.png"
SIZES = (64, 32, 16)
SUBJ = {"snipe": "round bird-like robot creature with one eye, a scope on top, short gun-nose, tucked legs",
        "squid": "squid robot creature"}
MODELS = {"sdxl-icon": ("SDXL + Pixel Icon LoRA (img2img)", "L4"),
          "qwen-edit": ("Qwen-Image-Edit-2511 + Render It (multi-ref edit)", "H100")}
VIEW = "http://127.0.0.1:8765/api/view/show"


def png(im):
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def cleanliness(im, n, pal):
    """Mean share of each grid cell held by its dominant palette colour (1.0 = perfect lattice)."""
    c = 16
    a = np.asarray(lock(im.resize((n * c, n * c), Image.NEAREST), pal)).reshape(n, c, n, c, 3)
    a = a.transpose(0, 2, 1, 3, 4).reshape(n * n, c * c, 3)
    k = (a[..., 0].astype(int) << 16) | (a[..., 1].astype(int) << 8) | a[..., 2]
    return float(np.mean([np.bincount(np.unique(r, return_inverse=True)[1]).max() / (c * c) for r in k]))


def sdxl_job(k, src, pp, n, name):
    size = 512 if n == 16 else 1024
    strength = {16: 0.6, 32: 0.55, 64: 0.4}[n]
    prompt = (f"yotoicon, {n}x{n} pixel art NES game sprite of a {SUBJ[name]}, low resolution game sprite, "
              f"flat colours, hard edges, thick black outline, centred, plain white background")
    neg = "blurry, gradient, dithering, noise, photo, 3d, soft shading, text"
    r = k.img2img.remote(png(pp), prompt, neg, [1, 2, 3, 4], size, strength)
    return r, {"prompt": prompt, "size": size, "strength": strength, "lora": 0.7, "seeds": 4,
               "refs": f"pre-pixelated concept ({n}px, palette-locked) as img2img init"}


def qwen_job(k, src, pp, n, name):
    prompt = (f"Render this pixelart, redraw the creature in image 1 as a clean {n}x{n} pixel NES sprite in the "
              f"exact style of the sprites in image 2: chunky square pixels on a {n}x{n} grid, thick black "
              f"outline, 4 flat colours from image 1, no gradients, no dithering, plain white background. "
              f"Keep the silhouette, pose and colour regions of image 1. The sprite fills the frame.")
    style = on_white(Image.open(STYLE))
    r = k.edit.remote([png(pp), png(style)], prompt, [1, 2], 768)
    return r, {"prompt": prompt, "size": 768, "lora": 1.0, "seeds": 2, "cfg": 4.0, "steps": 25,
               "refs": "image1 pre-pixelated concept, image2 NES style sheet (Telly, Met, Pipi)"}


def main():
    import modal
    from .modal_pixel_open import USD_PER_HOUR, QwenEdit, SdxlIcon, app
    OUT.mkdir(parents=True, exist_ok=True)
    prep = {}
    for name, path in CONCEPTS.items():
        src = on_white(Image.open(path)).resize((1024, 1024), Image.LANCZOS)
        pal = concept_palette(src)
        for n in SIZES:
            pp = lock(src.resize((n, n), Image.LANCZOS), pal).resize((1024, 1024), Image.NEAREST)
            prep[name, n] = (src, pp, pal)
    log = {"runs": []}
    t0 = time.time()
    with modal.enable_output(), app.run():
        ks = {"sdxl-icon": (SdxlIcon(), sdxl_job), "qwen-edit": (QwenEdit(), qwen_job)}

        def run_model(m):
            k, job = ks[m]
            gpu_s, rows = 0.0, []
            for (name, n), (src, pp, pal) in prep.items():
                d = OUT / m / name
                d.mkdir(parents=True, exist_ok=True)
                src.save(d / "input_orig.png")
                pp.save(d / f"input_pp{n}.png")
                r, meta = job(k, src, pp, n, name)
                gpu_s += r["gen_s"] + (r["load_s"] if not rows else 0)
                scores = []
                for i, b in enumerate(r["images"]):
                    raw = Image.open(io.BytesIO(b)).convert("RGB")
                    raw.save(d / f"raw_{n}_s{i + 1}.png")
                    scores.append(cleanliness(raw, n, pal))
                best = int(np.argmax(scores))
                raw = Image.open(d / f"raw_{n}_s{best + 1}.png")
                snap = block_mode(raw, n, pal)
                snap.save(d / f"snap_{n}.png")
                up(snap, 256 // n).save(d / f"snap_{n}_x.png")
                rows.append({"model": m, "concept": name, "n": n, "gen_s": r["gen_s"], "load_s": r["load_s"],
                             "scores": scores, "best_seed": best + 1, **meta})
                print(f"{m} {name} {n}: {r['gen_s']:.0f}s scores {[round(s, 3) for s in scores]}", flush=True)
            return rows, gpu_s, gpu_s / 3600 * USD_PER_HOUR[MODELS[m][1]]

        with ThreadPoolExecutor(2) as ex:
            res = dict(zip(MODELS, ex.map(run_model, MODELS)))
    log["wall_s"] = time.time() - t0
    for m, (rows, gpu_s, usd) in res.items():
        log["runs"] += rows
        log[m] = {"gpu_s": gpu_s, "usd": round(usd, 3)}
        write_manifest(m, rows, usd)
    (OUT / "results.json").write_text(json.dumps(log, indent=1))
    write_compare(log)
    print(json.dumps({m: log[m] for m in MODELS}))


def write_manifest(m, rows, usd):
    d = OUT / m
    title, gpu = MODELS[m]
    st = [pm.stage("inputs", "1 inputs (pre-pixelated)", "images",
                   [{"path": str(d / c / f"input_pp{n}.png"), "caption": f"{c} pp{n}", "pixel": True}
                    for c in CONCEPTS for n in SIZES], notes=[rows[0]["refs"]])]
    for n in SIZES:
        st.append(pm.stage(f"raw{n}", f"2 raw best seed, {n}", "images",
                           [{"path": str(d / r["concept"] / f"raw_{n}_s{r['best_seed']}.png"),
                             "caption": f"{r['concept']} s{r['best_seed']} clean {max(r['scores']):.2f}"}
                            for r in rows if r["n"] == n]))
    st.append(pm.stage("snap", "3 snapped + palette-locked", "images",
                       [{"path": str(d / r["concept"] / f"snap_{r['n']}_x.png"), "caption": f"{r['concept']} {r['n']}",
                         "pixel": True} for r in rows],
                       notes=[f"{gpu}; known-grid dominant-colour snap; concept palette"]))
    pm.write(d, f"Snipe/squid via {title}", st, cost_usd=round(usd, 3))


def write_compare(log):
    d = OUT / "compare"
    d.mkdir(exist_ok=True)
    st = []
    for n in SIZES:
        ims = []
        for c in CONCEPTS:
            for m in MODELS:
                ims.append({"path": str(OUT / m / c / f"snap_{n}_x.png"), "caption": f"{c} {m}", "pixel": True})
        st.append(pm.stage(f"s{n}", f"{n}x{n}", "images", ims))
    pm.write(d, "Open models compared: SDXL Pixel Icon vs Qwen-Image-Edit Render It",
             st, cost_usd=round(sum(log[m]["usd"] for m in MODELS), 3))


if __name__ == "__main__":
    main()
