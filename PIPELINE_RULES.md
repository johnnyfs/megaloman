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

## Ladder (shrinking)

- **finding** (M2): compute each rung from the source, not from the previous rung
  (shape IoU 0.88–0.94 vs 0.83–0.89; chaining hides breakage).
- **finding** (M2): down to 64 px the ladder is effectively lossless; shape holds to
  32 px; the 32 → 16 step is where character is lost (tentacles merge, the face
  squeezes).
- **trial** (v2, Johnny): **preserve symmetry.** If the source is symmetric (e.g. across
  a vertical axis), check left vs right at each rung. When it breaks, build two
  candidates (left half mirrored, right half mirrored) and let a judge pick.
- **trial** (v3, Johnny): **preserve significance.** As shapes shrink they lose
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

## Process

- **rule** (Johnny): each alternative to a step is a **version** (ladder v1 / v2 / v3…).
  Versions are compared side by side in the cockpit with a version switcher. Johnny
  **blesses** a winner, and blessed versions are then combined and re-checked together.
