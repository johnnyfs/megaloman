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

- **trial** (Johnny, snipe dry run, 2026-10-23): **separate protrusions.** Identify
  parts that are protrusions, weapons or attachments (e.g. the snipe's rifle-beak, which
  ate most of the resolution) and render them as separate pieces, possibly their own
  animation layer or sprite, so the body gets the full resolution budget. Compose them
  back at game time.
- **rule** (Johnny): concept prep must trim whitespace tightly (to the figure's bounding
  box, then square). The snipe crop left too much empty space.

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
  already contrast strongly (Lab dE ≥ 30); that line costs a whole row. *Superseded by
  the separation-lines rule below (Johnny, after v3).*
- **rule** (blessed by Johnny after v3, 2026-10-23): **separation lines are kept at every
  size.** A 1 px line goes between every pair of distinct regions, down to 16 px (v3 lost
  fins | head, face ring/eye surround | head, and head | eyes at ≤ 24). The depth rule
  (the line comes from the region behind) and the region priority decide where the
  pixels come from. Never merge regions to save space.
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
4. **Eyes:** filled *rectangles* (unless the expression demands otherwise), with a clear
   margin around them so they stay distinct from everything else. v1's 32 px won on
   exactly this. *Updated after v3 (Johnny):* "boxy" was over-applied. Rectangular means
   not necessarily square and not necessarily more than 1 px wide. At ≤ 20 px use
   **vertical slits** (1 px wide, 2–3 px tall) with a **smaller surround** (less black
   moat / visor area than v3's box), so the face reads less like Telly. Keep the eyes
   distinct.
5. **Keep like instances distinct as they shrink.** Cut into shapes (gaps, notches,
   little curls) so tentacles don't grow together, and keep a hint of each one's shape.
6. **Line sacrifice by depth.** Where two regions meet, the separating line comes out of
   the region that's underneath or behind.

- **trial** (Johnny, 2026-10-23): clip the corner pixels of boxy shapes (e.g. the face or
  visor surround) the way a pixel artist would, so boxes read as rounded instead of
  blocky.

- **rule** (Johnny, 2026-10-23): at small sizes, keep the suction cups merged with their
  tentacle (no cup edging). The curl and the tentacle's shape matter more than the edge.
- **trial** (Johnny's hand edit of v4 16 px → render_16b.png): treat a coloured
  surround (the face ring) as an *edge* made of colour, and apply the thin-line rule to
  it (each pixel touches at most 2 others): clip its corners. Fill the space that opens
  by growing the neighbouring parts (the fins first). If that leaves a thick black area,
  grow the head into it and reshape the head/ring line to match. Result: less boxy,
  more character.

- **finding** (Johnny's blind picks on E1, 2026-10-23). No pick was v4; none was the
  critic's top pick except where noted.
  - **24: a** = a random survivor (24-c047: fins +0.1 larger, fins win contested cells
    from the head, no fin pay). Reason: "the fins look most like fins".
  - **20: a** = **ring-as-edge alone** (v4 + ring_edge). It narrowly beat d (random:
    ring_edge + bigger fins + thin fin depth + curl 1; "jaunty, but the wide fins are
    slightly too out there") and f (random; "real personality", but the fins are too
    small and the tentacles aren't separated from the face). The extra colour on the
    tentacles gives them an arch.
  - **16: f** = **Johnny's own hand edit (16b)**, beating a (the critic's top 1): pushing
    the darker colour onto the front tentacles gave more volume.
  - Johnny: "honestly almost any of these would be fine in a game."
  - **Takeaways:** ring-as-edge wins at 20 and 16 (trial → candidate rule); fins should
    stay slightly larger and win against the head at 24, but not too wide at 20. The
    critic was no better than random (it picked a only at 16, and lost to the human
    edit). Keep the tentacles separated from the face.

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

- **finding (v4)** (2026-10-23, `ladder_v4.py`, `render_v4.py`, `workspace/experiments/regions/v4/`): Johnny's
  review of v3 applied on top of v3. No new paint and $0 spent (no model calls). 64 and 32 are pixel-identical to v3.
  - **Separation lines at every size**: the v1 contrast exemption is off (`contrast_below = 0`), so fins | head,
    ring | head, ring | fins, ring | pods and nozzle | arms all get a 1 px line down to 16. A check on the final
    pixels (no two distinct non-black regions touching) passes at every size. The pixels come from v3's rules:
    protected template parts (eyes, visor, ring, nozzle) never give; otherwise the region behind gives (so the head
    gives to the ring and the fins give to the head).
  - **Making room for the lines**: the face ring keeps its top row at every size (v3 dropped it below 24). That
    restores head | line | cream ring | black moat | eye, the 32 px structure that separates head from eyes. The
    face block hangs from the concept's ring-top row and grows down, so the two new rows are paid by the arms (lower
    priority), not the head. The fins pay for their head line from the background: each fin pixel that becomes line
    is re-grown one pixel further out (10–14 px a size below 32), so the fins stay ≥ 2 px wide instead of v3
    alt_finline's 1 px stripes. A fin's 1 px top tip is left to the line (a re-grown pixel there was a detached
    speck). Below 32, a head or fin pixel left with no 4-neighbour of its own once the lines are in becomes line
    (2 px at 20: the head's corners beside the ring's top line). The renderer's no-clump clean-up no longer grows
    template parts (it put a 1 px bump on the ring at 16). The nozzle floor now runs after the face, below the
    visor, because the taller face covered it at 20.
  - **Eyes**: 64/32/24 as v3 (3×6, 2×3, 2×3). Vertical slits at 20 (1×3) and 16 (1×2), with a 1 px moat and a
    2 px gap: the visor is 6 px wide instead of 8 (20: 6×5 vs 8×4, the same black pixel count but narrower;
    16: 6×4, 20 black px vs 24). A 1×2 slit at 20 was tried. It reads as dots, not slits, so 20 keeps 1×3.
  - **Inner-arm curl**: v3's 20 px hook is the pattern (a cup run one column in from the arm's outer edge, with arm
    colour on both sides, ending in a tip row that is all cup). It is now drawn explicitly at ≤ 20, with a 2 px run
    at 20 and 1 px at 16, so 16 gets `iii / iji / jj` instead of v3's cup on the outer corner. 20 is unchanged.
  - **Cups vs arms are the one unlined pair below 32** (as in v3). An edge around a 1–3 px cup hook erases it.
    `v4/alt_cupline/` edges them at every size: noisy at 24, and the curl is gone at 20 and 16.
  - **Costs**: lines and the ring top take rows. At 20 the arms have 5 rows (v3: 7) and the head 6 (v3: 7). At 16
    the arms have 4 rows (v3: 5) and the head 4 (v3: 5), and the outer arms are 1 px with a cup tip. Below 32 the
    pods are 1 px coral slivers between lines at 20 and 24, and absent at 16 (as in v3). Under each fin there is a
    black block (the 2 px depth line plus the outline) that looks heavy at 16. Awaiting Johnny.

- **finding (E1, model in the loop)** (2026-10-23, `e1.py`, `workspace/experiments/regions/e1/`, MODEL_IN_LOOP_PLAN
  section 4 + addendum): v4 (A) vs B1 `v4-critic` (knob sweep → hard gate → decision-model critic) vs B2 `v4-nbrefine`
  (Nano Banana refine → snap → repair → gate), at 24 / 20 / 16. $1.00 of the $3 cap (Jev $0.07, Luna $0.12,
  gpt-5.6-sol $0.54, Nano Banana $0.26 billed upstream by Google; OpenRouter reports $0 for BYOK).
  - **Knobs** (defaults reproduce v4 pixel for pixel): fin `k` offset, fin depth 1/2, fin pay on/off, eye slit (20:
    1×3/1×2/2×2, 16: 1×2/2×2), curl run 0/1/2, and the addendum knobs: **ring as edge** per size (the trial rule:
    clip the ring's L corners, fins then head fill the freed space; at 16 with everything else v4 it reproduces
    Johnny's `render_16b.png` exactly), **size yield** head/fins (who keeps contested head|fin cells), and **cup line
    at 24** (the one boundary still free to sacrifice; unlined at 20/16 is blessed). At 24 ring-as-edge is a no-op
    (the ring's corners are already rounded there). No combination broke a blessed check: 64 / 552 / 204 distinct
    gated grids at 24 / 20 / 16.
  - **Flagger** (21 synthetic defects with known locations): yes/no hit rate Jev 57 %, Luna 76 %, both 52 %. Both
    catch a removed eye, a lost line and a black block; neither sees 8 % noise; Jev misses merged arms (p ≈ 0.46, as
    in d1). **Boxes are useless** (Jev only, since Luna refuses box questions): mean IoU 0.03, 52 % even touch the
    defect. On clean v4 they false-flag a lot (Jev 7/24, Luna 14/24: "lost line" at every size, "asymmetric" at 20
    on an exact mirror). The one true flag is the heavy black under the fins at 16. Use: yes/no triggers at best,
    localised deterministically.
  - **Critic** (5 yes/no per sprite, Jev + Luna, no pairwise): it ranks v4 low (#51/64, #375/552, #183/204), and
    **both sentinels fail**: the Johnny-rejected v3 `alt_finline` 16 and v4 `alt_cupline` 16 score above v4. So per the
    plan the run cannot overrule v4: keep v4. Its consistent knob leanings: fin depth 1 everywhere (the heavy block),
    ring as edge at 20 and 16 (implied switch rung 20), no cup line at 24 (agrees with the blessed rule), 1×2 slits at
    16, but 2×2 boxes at 20 (against Johnny's slit rule). Johnny's 16b ranks #52/204, above v4.
  - **Judge** (gpt-5.6-sol vs v4, both orders × 2): critic top 2: 1 win, 4 ties, 1 loss; random survivors: 2 wins,
    3 ties, 1 loss. **The critic is no better than random** in the judge's eye. At 24 every fin-depth-1 variant beats v4
    ("fewer muddy black clumps"). Johnny's 16b vs v4: tie, leaning 16b (3 votes to 1).
  - **Nano Banana at working size: not viable.** Given the 24/20/16 sprite at ~512 px (no grid) and told to keep the
    grid, it redrew a detailed squid at about 2–3× the resolution (spots, shading, new colours). The cells are flat
    (purity 0.85–0.89), but at the wrong scale, so the snap back is mush. 0/6 passed the gate (the eyes broke in all
    6). Its raw drawings are good pixel art at ~48–64 px, which supports I3 (image models at ≥ 64 only).
  - **Awaiting Johnny's blind pick** (`ab-blind`; key in `e1/blind/key.json`): A, critic top 2, 2 random survivors,
    plus the ring-as-edge trial alone at 20 and Johnny's 16b at 16. His picks decide critic vs random and judge vs
    Johnny.

- **finding (snipe dry run)** (2026-10-23, `snipe.py`, `pipeline_gen.py`, `workspace/experiments/regions/snipe/`): the
  whole pipeline on a new, asymmetric enemy. Brief: "a snipe bot for a marsh level, a bird robot that is also a
  sniper, so its beak is recognisably a gun; facing horizontally (not symmetric)". Facing **left**. $0.98 of the $3
  cap, BYOK included (OpenRouter's own `usage.cost` would say $0.83; the $0.15 gap is Nano Banana billed by Google,
  read from `cost_details.upstream_inference_cost`): concept 4 images $0.35 + judge $0.02, legend $0.08, paint
  $0.26 (GPT), plan ×2 $0.08, significance $0.11, filter probes $0.04, read $0.05, Jev $0.0001.
  - **Concept**: 3 Nano Banana + 1 gpt-5.4-image-2, white background removed by a border flood fill, cropped to a
    centred 1024 square. gpt-5.6-sol ranked C (GPT, 9/10) first but picked D (NB, 8/10, "compact") for the sprite.
    I took C. A long horizontal gun sets the width, so D fills only 0.50 of the square's height vs C's 0.76 (about
    4 rows fewer at 16 px), and the NB birds are round three-quarter birds. **Gap: the canvas is always N×N**
    (`ladder_v2.square`, M = N − 2). A horizontal enemy with a long gun wants a wide footprint (24×16, 32×16).
  - **Symmetry**: Jev says none (p = 0.97). Measured: vertical 0.347, horizontal 0.362, so type none and no
    disagreement. **The mirroring code did not honour the spec.** `ladder_v2.symmetric_source` falls back to
    axis 0.5 and mirrors anyway. The v2–v4 rungs always run `mirror_inst`. `render_v4.render` always mirrors the
    final pixels. `render_v4.run` calls `tile_reduce(symmetric=True)`. `ladder_v2.instances` pairs instances by a
    flip, and e1 mirrors too. Fixes: `ladder_v2.prepare` now refuses a spec with no verified vertical axis, and
    `render_v4.render(keep="none")` skips the mirror. `pipeline_gen.prepare` mirrors only on a verified axis,
    otherwise pair = identity and `tile_reduce` runs with `symmetric=False`. Squid v4 still reproduces pixel for
    pixel at all 5 sizes.
  - **Legend** (generic prompt: squid examples swapped for legs, wings and tails, with no cup or undercarriage
    rules): 12 parts, including the occlusion subclasses `leg_back` and `tail_feather_back` (depth back). The feet
    were folded into the legs. 7 colour fixes.
  - **Region paint: Nano Banana refused this concept.** Both v2-prompt paints and a one-line "repaint this toy robot
    bird as a flat colour map" probe came back `PROHIBITED_CONTENT`, and neutralising weapon words in the prompt
    didn't help. The same probe on NB's own candidate D passed, so the filter reacts to the image (a realistic
    rifle-beak, or GPT-made art), not the words. Blocked calls cost $0. Fallback: gpt-5.4-image-2 (the M1 painter).
    IoU went 0.829 raw → 0.962 registered (sy 0.996, ty 12 px) → 0.981 after fill and clip. It left out
    `tail_feather_front` (pruned from the legend), and a 2390 px body island was folded into the wing. So
    "NB paints best, no registration" holds only when NB accepts the image: keep registration in the generic path.
  - **Squid-specific code found** (left in place; generic replacements live in `pipeline_gen`):
    - Part names in `ladder_v3.PALETTE/ASSIGN/PRIORITY` (Johnny's executive calls).
    - The squid tables and structures: `face_geometry`/`face_template` (ring > visor > two eyes on the axis),
      `arm_groups`/`cup_floor`/`separate_arms`/`curl_tips`, `nozzle_floor`, `fin_pay`, `size_yield`
      (cap/cap_side_lobes).
    - Render and naming: `render_v4.ring_as_edge` (face_housing, then fins, then cap), `render_v3.PROTECTED`, the
      `materials` cup and fin heuristics, `render.GRID_CH`, `separation_check`'s cup exemption, `eye_check`'s
      hard-coded 'e'.
    - Significance: `ladder_v2.v3_specs` (reuses the squid's blessed significance JSON via `V3_NAMES`),
      `ladder_sig.OVERRIDES` keyed by the name "eyes", and the significance prompt's art-director notes (fins).
    - Prompts: `regions_v2` legend examples (tentacles, cups, undercarriage, nozzle) and the paint prompt's
      suction-cup line.
    - Constants: the `CONCEPT`/`OUT` paths, and the e1 knobs (all squid parts).
    - `regions.concept_masks`' lavender-shadow heuristic is squid-tuned but harmless here (0 px).
  - **New generic step, the plan** (gpt-5.6-sol, replaces the hand tables): palette first (black + 3 NES colours,
    part → group), priority (eyes forced first), eye parts and eye surround. v1 picked 0B / 1B / 38 and gave the
    scope the eye's gold. At ≤ 32 the gold scope sits 1–2 px above the gold eye, and they read as one gold blob. v2
    adds the rule "the eye is the only part of its colour near it". The scope became dark green and gold is the
    eye's alone.
  - **Significance** (generic prompt, the brief's notes in place of the fin notes): the beak is a stroke,
    exaggerated ×1.8, and the scope a rect ×1.5. Everything else is none, including the legs (significance
    0.72 / 0.62). **The beak can't be exaggerated**: its muzzle touches the crop's left margin, so the frame caps
    k at 1.05–1.1. "Exaggerate into the background" has no background left for a long horizontal part. What it
    needs is the opposite (shorten the beak, grow the rest), which isn't a tool yet.
  - **Thin strokes break up in box coverage**: the 1 px barrel came out dashed at 16, and the muzzle-brake slots
    (see-through in the concept) cut it at 32. v3 adds a **stroke floor**: a part whose essential shape is "stroke"
    and that has more pieces than in the source is redrawn as a 1 px principal-axis line over background and
    lower-priority parts. It fixed the barrel at 32 / 20 / 16. This is the squid's M2 finding ("limbs need stroke
    primitives"), done generically.
  - **Eyes**: the squid face template doesn't apply (one eye, in profile). The generic floor is one filled
    rectangle per eye piece at its scaled centroid, 2×3 down to 24, slits 1×3 / 1×2 at 20 / 16. v2 adds a 1 px moat
    of the plan's eye surround (the black eye housing, playing the squid's visor). At 20 and 16 the moat erased the
    1-row scope. v3 nudges the eye 1 px, but only when staying would erase a character part (predicting which pixels
    the renderer's lines will take). A first nudge that always minimised a cost slid the eye off the head onto the
    neck. All eye checks pass: filled rectangle with a dark margin.
  - **Rules that ran unchanged and generically**: line sacrifice by depth (`behind_fn`: occluded_by, then depth,
    then layer order), separation lines at every size (`separation_check` ok at every size), background-only
    exaggeration, the speck rule, the NES tile rule (0 px changed at every size: 3 colours placed by region, as on
    the squid), and the outline, orphans, holes and colours gates.
  - **N/A here**: face ring as edge (the surround is black, not a coloured ring), the fin depth line (`fin` off: the
    heuristic is squid-shaped), cup rules. Like instances: `leg_front` / `leg_back` are separate parts (back = black
    shadow), and the line rule keeps them apart. At 16 the back leg is 1–2 black px.
  - **Result, v3** (honest read; gpt-5.6-sol's per-size read in `v3/read.json`):
    - 64: good. Bird, rifle-beak with muzzle, eye in its socket, scope, legs, tail.
    - 32: works. Bird and gun read; the eye is a gold 2×3 in a black box, more a lit window than an eye (the model
      says "eye gone").
    - 24 / 20: a long barrel, a black head block with a gold slot, a dark-green wing mass, one leg and a foot.
      "Faceless head". The mid-green body is lost under the wing.
    - 16: a barrel, a scope cap, a gold slit, a body blob and a leg. A thing with a gun; barely a bird.
    - Below 32 the sprite is mostly dark green plus black: low value contrast. 3 colours at 20 / 16.
    - The model's read is noisy: it calls v3's continuous barrel "not a gun" at 20 / 16, but v1's dashed one a gun.
  - **Biggest open problems for a new enemy**: a non-square canvas; de-exaggeration (shrink a dominant part); a
    palette check at the target size (area share per colour, so the mid colour doesn't vanish under one part);
    and a profile face template (eye, socket and beak root) for side views.
