# Pipeline rules: the running record

Every rule and finding for the concept → pixel-art pipeline, kept here so nothing is lost
when a session ends. When the pipeline is perfected, this becomes the source for a
skill. Mark each rule with its status: **rule** (blessed by Johnny), **trial** (being
tested in a version), or **finding** (measured, not yet a rule). Add a date and a source.

## Concept

- **rule** (2026-10-23, Johnny): concept art keeps its 3-D form and shading; no flat
  cartoon. The shading is signal for depth and shading at small sizes.
- **rule** (2026-10-23): work from a single view (front), cropped to a centred square with
  a transparent background. Drop the ground shadow before making regions.
- **rule** (2026-10-23): target size is a parameter. Sweep 32 / 24 / 20 / 16 px; 16×16 is
  the goal.

- **rule** (Johnny): **assess symmetry at the concept stage.** A vision model classifies
  the concept's symmetry (none / vertical axis / horizontal axis / radial / per part),
  and a deterministic mirror check verifies it. The result feeds the ladder's symmetry
  preservation, so that step stays deterministic.

- **finding** (2026-10-23): **decision models** (OpenRouter Decisions API,
  `POST /api/alpha/decisions`) return typed answers with probabilities: Choice, yes/no,
  Score, Box. `typesafe/jev-1.14` takes images (each side 256–1568 px, so upscale
  sprites with nearest-neighbour to ≥ 280 px), takes ~0.2 s and costs ~$0.04 per 1M
  input tokens. A test sprite came back "vertical symmetry" at p=0.98. Candidate for the
  symmetry classifier and a cheap pairwise judge; validate against the current judge
  first. Alternative: `openai/gpt-6-luna-decisions`.

## Regions

- **finding** (M1): `gpt-5.4-image-2` paints clean, distinct flat-colour part maps, but
  redraws shapes (silhouette IoU 0.86 vs the concept), and small details can pick up the
  wrong part's colour (face-frame rivets painted as the mantle).
- **finding** (M1): `gemini-nano-banana-2.1` hit HTTP 429 rate limits on OpenRouter at
  very low call volume (~2 parallel calls).
- **finding** (M2): fine-detail parts (suction cups, tentacle joints) merge from 512 →
  256 px down. They should be folded into their parent part or treated as
  shading/texture, not regions.
- **finding** (M2): a part painted as several pieces (mantle: 3) merges at 64 px; it
  should be one region with markings.

- **rule** (M1): image models redraw rather than trace (gpt-5.4-image-2 drew about 4–5%
  smaller and shifted 12 px). Always **register** the painted map to the concept outline
  (fit scale plus shift, refuse corrections over 15%) before snapping. IoU went
  0.83 → 0.96.
- **rule** (M1): markings take their part's colour in the paint prompt, or they read as
  holes.
- **rule** (M1): vision-model legend colours aren't reliably distinct; enforce minimum
  colour distance deterministically.
- **finding** (M1): the Nano Banana 429s were OpenRouter's shared Google quota
  (`upstream_provider_shared_pool`). Our own Google key (BYOK) would avoid them.

## Ladder (shrinking)

- **finding** (M2): compute each rung from the source, not from the previous rung
  (shape IoU 0.88–0.94 vs 0.83–0.89; chaining hides breakage).
- **finding** (M2): down to 64 px the ladder is effectively lossless; shape holds to
  32 px; the 32 → 16 step is where character is lost (tentacles merge, the face
  squeezes).
- **rule** (v2 blessed 2026-10-23, Johnny): **preserve symmetry.** If the source is symmetric (e.g. across
  a vertical axis), check left vs right at each rung. When it breaks, build two
  candidates (left half mirrored, right half mirrored) and let a judge pick.
- **rule** (Johnny): symmetry matters more the smaller the sprite. It's noticeable at
  64 px, bad at 32, and looks amateur at 16, where every pixel counts and the sprite must
  read as an icon, not just a likeness. So when the concept is symmetric, every rung must
  mirror **exactly**, pixel for pixel (v2b), not merely stay within a tolerance.
- **rule** (v3 blessed 2026-10-23 by Johnny, overriding the judge's "too wing-like"): **preserve significance.** As shapes shrink they lose
  significance; character-carrying parts (e.g. the jaunty side fins) are kept larger
  than their proportional size, with a minimum footprint and an essential shape.
- **rule** (Johnny): eyes are a *floor*, not an exaggeration. They never shrink below
  two distinct rectangles, but otherwise scale normally.
- **finding** (M2): at 20/16 px the face (visor and eyes) needs a fixed template, e.g. a
  visor ≥ 5 px wide with two 1×2 eyes.
- **finding** (M2): tentacles at ≤ 32 px need stroke primitives: a few distinct arms,
  ≥ 2 px, with gaps kept open.

## Render (regions → pixel art)

- **rule** (2026-10-23, Johnny):
  - **Outline:** black around the sprite. The default is thin: 1 px lines where each
    outline pixel touches at most two others (8-connectivity), with no clumps.
  - **Outlines separate regions of different colour by default, but not always.**
    Markings within a part (the spots on the squid's head) get no outline.
  - **Outlines thicken to show depth,** e.g. tentacle outlines get thicker nearer the
    body, where they recede or overlap.
  - **Some areas are simply black,** e.g. the visor around the squid's eyes.
  - **Regions are one solid colour,** sampled from the concept and snapped to the NES
    palette, unless the part is spotted or striped and the size allows the pattern.
- **trial**: at most one darker shade per part, where the concept's 3-D form justifies
  it and it doesn't muddy small sizes.

- **finding** (render v1): draw the outline *around* a label map computed at N-2, not
  inset into the rung, or it eats 1–2 px limbs.
- **finding** (render v1): below 32 px, drop the line between part pairs whose colours
  already contrast strongly (Lab dE ≥ 30); that line costs a whole row.
- **finding** (render v1): the NES tile rule (≤ 3 colours per 8×8) forces the palette
  down to black, salmon and cream at 20 and 16 px; minor colours (nozzle cyan, pod grey)
  get dropped.
- **finding** (render v1): 32 px reads well. 16 px doesn't read as a squid: the tentacles
  blur into one skirt and the eyes are 1×1. That's the region map's fault, not the
  renderer's. It needs the reshape step: a face template, 3–4 distinct tentacle strokes,
  and a head kept big.

### Johnny's critique of render v1 (2026-10-23): regions over coloration

- **rule:** at small sizes **regions matter far more than coloration**. Drop the
  concept's patterning (polka dots, stripes, two-tone limbs). Each region is one solid
  colour; distinctness between regions is what carries the sprite (Met and Telly keep
  every feature with very few pixels).
- **rule:** **occlusion makes a subclass.** When one instance of a part hides another
  (a tentacle behind a tentacle), the hidden one is a background subclass, rendered
  darker or as black/shading. Merging them into one shape kills the curve of the front
  tentacle. This must be captured at the regions stage (legend plus paint), not
  patched at render.
- **rule:** suction cups are their own solid colour, edged, and distinct from the
  tentacles. Tentacles are another solid colour.
- **rule:** tentacles need clear, solid separating lines as they shrink, and more of the
  line budget goes to shading and separation.
- **rule:** depth lines go where the physical gap is: e.g. a thicker line under the side
  fins (bottom), where fin and head separate most.
- **rule:** exaggeration (e.g. the fins' jauntiness) takes pixels from the empty
  background, never from the head or body.
- **finding:** 64 px almost works. 32 lacks region distinction. 16 is "a little triangle
  man".

## Process

- **rule** (Johnny): each alternative to a step is a **version** (ladder v1 / v2 / v3…).
  Versions are compared side by side in the cockpit with a version switcher. Johnny
  **blesses** a winner, and blessed versions are then combined and re-checked together.
