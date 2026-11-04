# nespix: text / image → NES pixel art

`src/megaloman/nespix/` is a model that turns a text description and/or a hi-res example image into
NES-legal pixel art at any size from 8×8 to 256×256: at most 4 opaque colours from the 2C02 palette, at most 3
per 8×8 tile, a transparent background and a black outline.

It is built "as directly as possible". The part that makes every pixel-art decision is a **fixed computational
graph with hand-set weights** (no training). Pretrained networks are used only for the things hand-set
weights can't do: imagining a concept from text (SDXL-Turbo) and judging whether a sprite still reads as
its subject (CLIP). Everything runs on CPU.

```
text ──► imagine (SDXL-Turbo, 6 seeds) ──► pick concept (CLIP text match + sanity)
                                                   │
image ─────────────────────────────────────────────┤
                                                   ▼
                     render: 32 hand-set weight variants per size  ──►  judge  ──►  sprite
```

## Usage

```bash
uv venv && uv pip install --index-url https://download.pytorch.org/whl/cpu torch \
  && uv pip install -e '.[nespix]'

nespix "a blue slime monster" --sizes 16 24 32 -o out/slime     # text -> sprites (≈1 min on 4 CPUs)
nespix --image concept.png --sizes 8 16 32 64 -o out/x          # image -> sprites (≈2 s per size)
nespix "fire knight" --image photo.jpg --restyle 0.5 -o out/k   # flatten a photo/painting first (img2img)
nespix --image concept.png --sizes 16 --fast                    # renderer only: no torch, ~30 ms
nespix --image concept.png --palette 0F 16 27 30                # lock the NES palette
```

```python
from megaloman.nespix import generate, pixelate, render
res = generate('a red mushroom with white spots', sizes=(16, 32))
res['sprites'][16].rgba().save('mushroom16.png')
sp = pixelate('concept.png', 24)          # judged variant search
sp = render('concept.png', 24)            # one pass with the default weights (numpy + scipy only)
```

Each output folder holds `concept.png`, `sprite_<N>.png` (1×, RGBA), `sprite_<N>_preview.png`,
`sheet.png` and `meta.json` (palette codes, the chosen weight variant, judge terms).

## The renderer (`render.py`): the hand-weighted model

Each stage is a deterministic function. Its weights live in `Params` and were set by hand against the
repo's concepts and the rules in `PIPELINE_RULES.md`.

| # | stage | what it does | why (rule) |
|---|---|---|---|
| 1 | matte | alpha, or a border flood fill on a median-smoothed copy with a tolerance adapted to the border noise; keep the main components | backgrounds from image models are textured |
| 2 | parts | regions of the foreground between the concept's ink lines (dark or *locally* dark: grey closing − L > 0.12) | "regions over coloration" |
| 3 | flatten | each part becomes one solid colour (its median); a two-tone part is split by 2-means; outlier pixels (eye whites, highlights) keep their colour | regions are one solid colour; no patterning |
| 4 | palette | **palette first**: an *exhaustive* search over all NES colours for black + 2 or 3 fills that minimise a truncated, saliency- and part-weighted OKLab error. Fills must stand clear of black (black does the shadows) and of each other. Chroma ×1.5 first, because NES colours are saturated and concepts are muddy | "target palette first" |
| 5 | classify | nearest palette entry per hi-res pixel. Black matches as if its lightness were 0.22, so deep shadows become ink. The concept's own outer outline is ignored (we draw our own) | |
| 6 | ink policy | below 48 px, interior ink thinner than 0.7 cells is opened away (lines vanish; eyes, visors and mouths survive) | |
| 7 | pool | area-pool class probabilities into the target grid. If the source is mirror-symmetric (IoU ≥ 0.88), pooling is symmetric, so the result mirrors **exactly** | "symmetry: pixel-exact at small sizes" |
| 8 | sharpen | a fixed 3×3 unsharp kernel on every class map and on the coverage map, then argmax. This is the "detail-preserving downsampler": thin limbs and small features win cells they would lose to a plain vote | eyes are a floor |
| 9 | clean | drop specks, fill pinholes, absorb orphan pixels, 3×3 majority smoothing of fill boundaries | |
| 10 | outline | a 1 px black outline drawn *outside* the fill (4-connected dilation: thin, no clumps, rounded corners), then corner tidy | "draw the outline around the label map, not inset" |
| 11 | tile rule | ≤ 3 opaque colours per 8×8 tile (metasprite tiles aligned to the canvas); the least-used fill in an over-full tile is remapped | NES hardware |

`mode='tile'` renders opaque 4-colour background tiles instead (no matte, no outline).

## The variant search and judge (`model.py`, `score.py`)

At small sizes the right trade-off depends on the subject, so the model renders a fixed 2⁵ grid of weight
variants per size: 3 vs. 4 colours, sharpen 0.4 or 1.1, ink lines kept or dropped, smoothing 0.3 or 0.8, and a
fat or tight silhouette. Duplicates are removed. The judge then scores each variant (hand-set weights):

- `25 ×` CLIP image similarity between the sprite and the concept (does it still read as the same thing?)
- `15 ×` CLIP text similarity to "pixel art of <description>"
- `−8 ×` fidelity: mean OKLab error against the concept box-downsampled to the same grid
- structural checks: fill ratio, fragments, orphan noise, an unused palette colour, a mostly-black blob

For text, SDXL-Turbo (bf16 on CPU, ~6 s per image with AMX) imagines 6 concepts. The prompt asks for what
the renderer converts best: one subject, centred, flat colours, thick black outlines, a white background. The
concepts are ranked by CLIP text match plus sanity terms (cropped subject, coverage, clutter). The best 2 are
rendered at every size, and the concept with the best mean judge score wins, so the whole size ladder comes from
one design.

## Guarantees (tested, `tests/test_nespix.py`)

At every size from 8 to 256: an N×N output; only NES colours; ≤ 4 opaque colours; ≤ 3 per 8×8 tile; every fill
pixel 4-separated from transparency by black; an exact mirror when the source is symmetric; deterministic; eyes
survive at 16 px.

## Results (2026-11-04)

- `workspace/experiments/nespix/gallery/gallery.png`: 12 text prompts plus the repo's squid and snipe concepts.
  Each row shows the concept, then 8, 16, 24, 32, 64 and 128 px. Per-example folders hold the 1× PNGs and
  `meta.json`. Regenerate with `python -m megaloman.nespix.gallery` (about 1 min per prompt on 4 CPUs).
- `workspace/experiments/nespix/range_mushroom/`: one design at 8, 12, 16, 20, 24, 32, 48, 64, 96, 128, 192 and
  256 px.

Honest assessment, by size:

| size | quality |
|---|---|
| 64–256 | Clean and NES-legal. It reads like a good hand-cleaned conversion: crisp outline, flat regions, correct palette. |
| 32 | Good for simple and medium subjects (mushroom, skull, ghost, heart, knight, slime, squid). Busy subjects get some interior noise. |
| 16–24 | Readable for simple, iconic subjects (heart, ghost, skull, mushroom, slime). Complex subjects (frog, dragon, robot, snipe) become recognisable blobs, not designed icons. |
| 8 | Silhouette plus a colour split. Fine for items and bullets; characters are not really legible. |

The gap to professional NES work at ≤ 24 px is **redesign**. A pixel artist doesn't shrink the concept; they
re-draw the subject as an icon (oversized eyes, fewer limbs, exaggerated key parts: see "preserve
significance" and the face templates in `PIPELINE_RULES.md`). The renderer only preserves and simplifies what
is there. Other known weaknesses:

- Ground shadows and puddles that the image model adds are kept as part of the sprite (the ghost, the slime).
- A palette choice can drift in hue: the yellow star came out lime (29), because no bright NES yellow exists
  near its lightness.
- The CLIP ViT-B/32 judge is weak at 8–16 px. There, fidelity and the structural checks do most of the work.

Next levers, in order:

1. A small-size concept prompt ("chunky icon, huge eyes, 2 colours") for N ≤ 24, so the redesign happens in the
   concept and not in the downsampler.
2. Ground-shadow removal in the matte (a flat, low-contrast blob under the subject).
3. Port the squid-specific significance and eye-floor rules from `ladder_v4`/`render_v4` into general
   part-level rules, now that parts exist (stage 2).
