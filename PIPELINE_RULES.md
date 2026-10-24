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
- **finding** (2026-10-23, bake-off d1, `workspace/experiments/decisions/`): 29 items × 3
  repeats. Accuracy on known answers (12 synthetic pairs, 8 yes/no rubric checks):
  gpt-5.6-sol 95%, Jev 85%, Luna 85%. Jev and Luna cost ~$0.00005 a call (~140× cheaper)
  and take 0.3 s (vs 3.6 s), and they are nearly deterministic (0–1 of 29 items unstable,
  vs 10 for gpt). Both decision models catch noise and a broken (copied) half reliably
  (p ≥ 0.84). Jev is weak on merged parts (p ≈ 0.55). Luna answered "not symmetric" at
  p = 0 on exactly mirrored maps. On ladder preference items both disagree with gpt
  (1–2 of 4 agree), and both show **position bias**: Luna always picks A, Jev mostly
  picks B. **Use:** Jev is now the classifier behind `symmetry.assess_symmetry`
  (`classifier="jev"`; the deterministic verification still wins). On the squid all
  three models say vertical. Decision models are good for cheap defect screens. Keep
  gpt-5.6-sol for preference and character judging.

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

### Proposed refinements from the v2 pipeline (status: trial, awaiting Johnny)

- Front instances of the same kind that touch (each arm) are separate parts sharing a
  `material`, and keep their own identity through the ladder so the separating line can
  be drawn.
- Colour distinctness is judged at the target size: parts within one pixel of each other
  at 16 px must differ in colour or get a line.
- Floors must survive render: floored features (eyes, nozzle) are always ringed in ink
  and protected in the NES tile reduction.
- Exaggeration grows into the background only, never over lower-significance parts.
- Diagonal separating lines may clump; prefer 4-connected 1 px lines, or accept the
  clumps as depth.
- **finding** (v2): 64 px good; 32 px clearly better than v1; 16 px still not icon
  quality. Telly and Met spend most of their area on two huge eyes, and we don't. The
  missing piece is the face/arm template step.

- **finding** (Johnny on v2, 2026-10-23): much better overall, especially 24/20, and
  16 has personality. 32 may be marginally better in v1. At 24 and 20 the eyes aren't
  distinct enough from the surrounding shape. Eye contrast and separation is the open
  problem (see the face template).

### Johnny's distilled rules after v2 (2026-10-23)

1. **Target palette first.** Choose the final small palette at the start, not as a
   late reduction. Make executive calls on which parts share a colour, e.g. fins, nozzle
   and cups one colour; face ring and tentacles one colour; head the darker colour.
   Shrinking an image that still has too many colours is confusing.
2. **No patterning for now** (spots or stripes). Maybe revisit on large areas later.
3. **Region priority.** Parts have an explicit priority that decides who keeps pixels.
4. **Eyes:** keep them boxy, filled rectangles (unless the expression demands
   otherwise), and keep a clear margin around them so they stay distinct from everything
   else. v1's 32 px won on exactly this.
5. **Keep like instances distinct as they shrink.** Cut into shapes (gaps, notches,
   little curls) so tentacles don't grow together, and keep a hint of each one's shape.
6. **Line sacrifice by depth.** Where two regions meet, the separating line comes out of
   the region that's underneath or behind.

## Process

- **rule** (Johnny): each alternative to a step is a **version** (ladder v1 / v2 / v3…).
  Versions are compared side by side in the cockpit with a version switcher. Johnny
  **blesses** a winner, and blessed versions are then combined and re-checked together.

- **finding (v3)** (2026-10-23, `ladder_v3.py`, `render_v3.py`, `workspace/experiments/regions/v3/`): the six
  distilled rules applied to v2's region map. No new paint and $0 spent (no model calls).
  - **Palette, decided first** (`v3/palette.json`), one NES hue ramp plus black: **0F** black for the outline, visor
    moat, undercarriage and back arms; **16** `F83800` for the head (the darkest); **26** `F87858` for the fins,
    nozzle, cups and side pods; **36** `F0D0B0` for the face ring, the arms and the eyes. Pods joined the accents
    so the cream face ring stays a clean ring. With 3 colours placed by region, the tile rule is nearly free: it
    changes 8 px at 24 and none at the other sizes.
  - **Priority** (highest first): eyes > visor > face ring > nozzle > head > fins > inner arms > outer arms >
    cups > pods > undercarriage > back arms. It replaces the legend's model-chosen priorities in the shrink.
  - **Eyes**: a face template, applied last. Two filled boxes with a ≥ 1 px black moat on every side and a
    face-colour ring: 3×6 at 64, 2×3 at 32 and 24, 2×2 at 20 and 16, gap ≥ 2 (even, for exact symmetry). The
    ring's top and bottom rows are kept only at ≥ 24. Below that a ring row costs two rows (the ring plus a line
    to the cream arms), which starved the arms at 16. There the moat meets the dark head and the black
    undercarriage directly. A check on the final pixels (two filled boxes, black all round) passes at every size.
  - **Arms**: each front arm plus its cups is one arm group. Where two groups touch, a 1 px gap is cut from the
    arm behind. At 64 and 32 the gap widens to a notch at the tips. Each arm keeps a cup pixel at its tip (the
    curl hint). The renderer's no-clump clean-up must never refill those gaps. It did in the first run, and with
    the exact mirror the two inner arms joined back into a skirt at 16. Enclosed background between arms is
    filled black (shadow), so the holes check passes everywhere.
  - **Line sacrifice by depth**: the ink goes on the region behind (legend `occluded_by`, then depth, then the
    v2 layer order): fins give to the head, the head to the face ring, arms to their cups. Template and floored
    parts (eyes, visor, face ring, nozzle) never give a pixel.
  - **Read**: the eyes now read like Telly's and Met's (cream boxes in black) at 24, 20 and 16, the clearest of
    the three versions. 32 matches v1's face. 16 shows four separate arms with coral tips, not v2's skirt. The
    costs: below 32 the fins are coral shoulders, not jaunty wings. A variant that keeps the head/fin line at
    every size (`v3/alt_finline/`) makes them striped and noisy at 16. The 32 px arms are busy with cuts. The
    face at 20 and 16 is a big box (visor 8×4 of 14), more Telly than squid. The head red (16) is more saturated
    than the concept's coral. Awaiting Johnny.

- **finding (nano banana, region paint)** (2026-10-23, `regions_nanobanana.py`, `workspace/experiments/regions/nanobanana/`):
  with Johnny's BYOK Google AI Studio key the 429s are gone (`is_byok`, ~$0.04 per paint billed by Google, $0 by
  OpenRouter). On the v2 legend and prompt, Nano Banana **traces** the concept: identity registration (sx = sy = 1,
  t = 0), IoU 0.996/0.995 vs GPT's best 0.943 (which needed ~4–6 % scale and a 14 px shift), all 12 parts present
  in both seeds. Parts agree with GPT's map on 86 % of pixels; it differs most on face housing, undercarriage and
  pods. It is cheaper and more faithful than GPT for region painting, so prefer it.
