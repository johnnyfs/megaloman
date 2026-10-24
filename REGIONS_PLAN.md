# Region ladder: from concept art to pixel art by tracked shrinking

**Status:** proposal, 2026-10-23. Builds on v0 (`V0_IMPL.md`). It reuses its runner, cache,
judge and NES rules, and the compiler's outline code.

## Why

Asking image models for NES sprites directly hasn't produced usable art. That holds for
`one_shot`, for the blessed stills, and for hand experiments with ChatGPT. Downscaling a
good still with the compiler keeps its pixels but not its *meaning*: eyes shrink into
noise and tentacles turn to mush. Pixel art is partly symbolic. At 16 px a tentacle is a
1–2 px stroke and an eye is a rectangle with a dot, not a smaller version of a painted
eye.

## Hypothesis

Separate **what the parts are** from **what they look like**, and shrink the parts
deliberately:

1. **Concept.** An image model draws cartoon concept art at about 1024².
2. **Regions.** The concept is repainted as a flat *region map*: every part (head, eye
   white, pupil, each tentacle…) is one unique flat colour, with a legend that names it.
3. **Ladder.** Shrink the region map step by step (1024 → 512 → 256 → 128 → 64 → 32 →
   target). At every step, measure each part's health. When a part degrades, *reshape*
   it into its symbolic form: eyes keep a minimum size, a tentacle becomes a 1–2 px
   stroke, merged parts get split apart. A judge confirms the parts are still
   recognisable.
4. **Render.** At the target size, draw the outline procedurally, take each part's colour
   from the concept, snap to the NES palette and the 3-colours-per-tile rule, and add
   minimal shading.

The bet: an agent is good at *one* judgement at a time ("these two tentacles merged: give
them a gap"). It is bad at drawing a whole sprite at once.

## The pipeline (new step types in the v0 runner)

```yaml
# pipelines/region_ladder.yaml (sketch)
steps:
  - {id: concept, step: gen.image, prompt: prompts/concept_cartoon.md, fanout: 2}      # exists
  - {id: legend,  step: regions.legend, input: $concept}     # VLM -> parts list: name, role, colour, min size, priority
  - {id: paint,   step: regions.paint,  input: [$concept, $legend]}   # image-edit model paints flat region colours
  - {id: snap,    step: regions.snap,   input: $paint}       # deterministic: nearest legend colour, drop specks, align check
  - {id: ladder,  step: regions.ladder, input: $snap, params: {rungs: [512, 256, 128, 64, 32, 20]}}
  - {id: render,  step: regions.render, input: [$ladder, $concept]}   # outline + recovered colours + NES rules
  - {id: score,   step: judge, input: $render, suite: convincing_enemy}  # exists
```

- **`regions.legend`.** A vision model looks at the concept and returns 4–10 parts, each
  with a `role` (body / feature / limb / detail), a `priority` (which part wins a
  contested pixel), and a `min_px` per rung (an eye never drops below 2×2). The legend
  is data and is cached like everything else.
- **`regions.paint`.** An image-edit call: "repaint this exactly, each part as one flat
  colour from this table, no outlines, no shading." The model output is never trusted
  directly. `regions.snap` maps every pixel to the nearest legend colour, removes
  islands smaller than N px, and **checks alignment** (silhouette IoU against the
  concept ≥ 0.9, every legend part present). Failing candidates are dropped.
  - Fallback if painting is unreliable: segment deterministically (colour clustering
    plus flood fill on the concept), then let the VLM *name* the segments. That gives
    the same output with cheaper and more stable steps.
- **`regions.ladder`.** Each rung is a deterministic, region-aware resize: per-part area
  coverage, with contested pixels going to the higher-priority part. After each rung,
  **part health** is measured with no model calls:

  | measure | catches |
  |---|---|
  | present / area vs expected | a part vanishing (eyes at 32 px) |
  | connected components | a part splitting, or two tentacles merging into one |
  | minimum thickness | a limb thinning into broken pixels |
  | shape IoU vs a straight resample | drift |
  | adjacency graph (pupil inside eye, eye inside head) | broken structure |

  A part that fails a measure triggers **reshape**: deterministic rules first (enforce
  `min_px`, replace with a primitive such as a rectangle, ellipse or stroke, open a
  1 px gap), then an agent. At ≤ 64 px the agent edits the region grid directly as
  text, which models handle well at that size. Above that it uses an image edit. Every
  reshape is a recorded, cached step with a one-line reason.
- **Judge per rung.** "Here is the region map with its legend. Can you still identify each
  named part? Does it still read as the brief?" This is a short, cheap, structured
  judgement that proves a reshape helped. Each reshape is compared against the
  no-reshape baseline at the same rung.
- **`regions.render`.** The compiler's outline code draws the outer outline and the
  lines between parts. Each part's colour is the median of its region in the concept,
  snapped to the NES palette, with one shade step where the concept's lighting
  justifies it. The output then runs through the existing `checks` and the
  `convincing_enemy` eval.

## MVP: prove the two risky links first

| # | Milestone | Done when | Est. cost |
|---|---|---|---|
| 1 | **Concept + regions** for the fixed squid brief | 2 concepts, each with a snapped region map that passes the alignment check, legend parts all present | ~$1 |
| 2 | **Ladder, observe only** (no reshaping): 1024 → 512 → 256, then on down to 20 | A health table showing *at which rung each part fails*. This tells us where agents are needed. | $0 |
| 3 | **Reshape at the first failing rung** (rules, then agent) | Judge's part-recognition score beats the no-reshape baseline at that rung on both concepts | ~$1–2 |
| 4 | **Carry it to target + render** | A 16–20 px squid next to Telly / Met / Pipi and the compiler v2 squid. You say whether it's usable. | ~$1 |

Milestones 1–2 answer your two questions: does the painting work, and is
1024 → 256 lossless enough that the real fight happens lower down? We stop after
milestone 2 to look at it together before building the reshape agent.

## What you'll see on the Megaloman tab

All images load from run directories via `/api/file`, at the same zoom, pixel-crisp.

1. **Concept → regions.** For each candidate: the concept, the region map, an overlay
   (region map at 50% over the concept), and legend chips (colour plus part name). The
   alignment check is shown pass/fail.
2. **The ladder.** A filmstrip of every rung, all scaled to the same display height, with
   a strip of coloured dots under each rung, one per part: green healthy, amber
   thinning, red merged or vanished.
3. **Part health table.** Parts × rungs, each cell showing area, components and status.
   The first red cell is where reshaping starts.
4. **Reshape diff.** Before/after at that rung, changed pixels in magenta (the existing
   convention), the agent's reason, and the judge's score before → after.
5. **Final.** The target sprite at 1× and 6× beside the authentic baselines and the
   compiler v2 squid.

Panels update in place as each step finishes, and Alba says one line when a milestone
lands.

## Decisions (2026-10-23)

- **Target size is a parameter, not a constant.** `regions.ladder` takes `target` (and
  rungs are derived down to it), so we can sweep 32 → 24 → 20 → 16 and see how far each
  stage of the method holds up. 16×16 is the goal. The sweep separates "the concept is
  flawed" from "the agent gets this far and needs more guidance below it".
- **Concept art keeps its 3-D form and shading.** It should not be flat cartoon art: the
  shading is signal for how to shade at small sizes. Regions describe *parts*, and the
  concept's lighting is sampled per region at render time (a light and a dark tone per
  part where the shading supports it).
- **Squid only.** Milestone 1 starts from Johnny's existing GPT concept (a three-view
  robot squid turnaround), cropped to one view. Generating new concepts comes later, as
  its own comparison.
