# nesflat: flat / grayscale art on a solid background → NES pixel art (experiment)

This is a fresh model that shares no code with the rest of megaloman. Input: high-contrast flat-colour or
grayscale art with black ink lines on one solid background colour. Output: N×N sprites (8..256), black ink plus
up to 3 tones mapped to distinct NES colours, ≤ 3 opaque colours per 8×8 tile. Requirements: numpy, scipy, pillow.

```python
from nesflat.model import render            # energy model
from nesflat.regions import render_regions  # region model (vectorise -> rasterise)
render_regions('examples/snipe_flat.png', 32).rgba().save('snipe32.png')
```

Two hand-weighted models:

- `model.py`, **energy**: discrete labels minimise multi-scale block fidelity (1/2/4 px), local-contrast
  fidelity, Potts, orphan, closed-outline and ink-clump terms. Solved by vectorised ICM on a stride-4
  independent-set lattice; exact mirror when the source is symmetric.
- `regions.py`, **regions**: removes the source's ink lines, keeps the flat regions (solid black areas stay),
  rasterises the regions at the target size (small-region priority and a feature floor), redraws 1 px lines
  between regions (the pixel comes from the larger region), then draws the outline outside.
- `imagine.py`: SDXL-Turbo prompt for this input type, plus a concept-quality score.

## Findings (snipe concepts, `examples/`)

`comparison_snipe.png`: rows are flat/energy, flat/regions, depth/energy and depth/regions; columns are 16,
24, 32, 48 and 64 px.

- Both models behave as designed. The tone extraction and region map of the flat snipe are clean: 4 tones
  and about 55 regions.
- 48 and 64 px are acceptable. At 32 px and below, neither model gives clean sprites. The design has about
  55 regions, and a 16×16 grid has 256 pixels, so most regions get 1–4 pixels plus a line. The result is
  confetti, not an icon. Tuning the weights (`regions_weight_sweep.png`) moves the noise around but doesn't
  remove it.
- The constrained input fixed the problems it can fix: matting, palette and tone extraction are now exact.
  It didn't fix the small-size problem, because that problem is **editorial**. Deciding which parts to drop,
  merge or exaggerate (one eye, one cannon, one body split, the wing) is a design decision that local
  downsampling, however well weighted, doesn't make.
- The depth-shaded version is worse at small sizes. With 4 colours, a tone goes to the shade (dark teal),
  white is lost, and the shading adds more boundaries.
