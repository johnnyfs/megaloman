# v0: a repeatable, declarative pipeline for one convincing squid

**Scope (pivot, 2026-10-21):** no UI, no voice, no director, no threads. Only:

> Declaratively defined steps that execute model calls and yield the **best candidate image** of a convincing Mega
> Man enemy, plus an **eval** that measures how convincing it is against authentic enemies scored the same way.

To compare approaches, write another pipeline file, run it, and compare the reports. Units (style guidance, more step
types) and any UI come later, one at a time, each justified by a measured difference.

`PLAN.md` stays as the long-term vision. This file is the only thing being built now.

## 1. The baseline: how convincing is authentic art, as the judge measures it?

The question the eval asks: *"Is this image a convincing NES Mega Man enemy that matches this description?"* We ask it
of real Capcom enemies first, each with a neutral description written like one of our briefs.

| Baseline | Source | Size | Why |
|---|---|---|---|
| **Telly** (MM2) | `fusion-man/src/enemy_frames.json` `telly1` | 16x16 | A floating homing enemy; the closest role to the squid |
| **Met, walking** (MM3, hard hat up) | same file, `met_walk` | 20x19 | A small ground enemy with a face; tests that the judge isn't Telly-specific |
| **Pipi** (MM2, the egg-dropping bird; the bird only) | rip from the sprites-inc MM2 sheet | ~16x16 | A flying creature, closest in theme (an animal-as-robot) |

The eval suite runs each subject N=5 times. The calibration report gives:

- **Mean and spread per baseline.** The spread is the judge's noise band; any difference smaller than it is not real.
- **A description control:** Telly judged against *Pipi's* description should score low on fidelity. If it doesn't,
  the judge ignores the description, and the eval is broken.
- **A recolour control:** Telly recoloured judged against its own description. This measures how much the score
  rewards *recognising* a known sprite, which our fresh squid can never benefit from.
- **The pass bar** for a generated squid: mean score ≥ the baseline mean minus one noise band, all deterministic
  checks pass, and the overall verdict is "convincing".

## 2. The squid brief (fixed)

```yaml
# briefs/squid.yaml
id: squid
description: >
  A small floating robot squid, Blooper-like, that swims along at and after Mega Man in pulses
  (tentacles gather, then thrust). It is a swimming mine: near Mega Man its eyes flash red, then it explodes.
deliver: primary frame (glide pose), facing left
size: about 16x16 to 20x20 px
game: Mega Man 2-3 NES
```

## 3. What a pipeline file looks like

```yaml
# pipelines/one_shot.yaml
name: one_shot
version: 1
inputs:
  brief: briefs/squid.yaml
  style_refs: [refs/telly/telly1_4x.png, refs/met/met_walk_4x.png]
steps:
  - id: draw
    step: gen.image
    model: google/gemini-nano-banana-2.1
    prompt: prompts/rules_first_sprite.md        # template; {brief.description} etc. are filled in
    images: [$style_refs]
    fanout: 4                                    # seeds run_seed .. run_seed+3
  - id: snap
    step: gridsnap
    input: $draw                                 # maps over the fanout
    params: {target: [20, 20], palette: nes, max_colours: 4}
  - id: gate
    step: checks
    input: $snap
    params: {checks: [size, palette_tile_3plus_t, orphans, holes, outline]}
  - id: score
    step: judge
    input: $gate.passed
    suite: evals/convincing_enemy.yaml
  - id: best
    step: select
    input: $score
    by: mean_score
output: $best
```

- **Step types (v0):** `gen.image` (OpenRouter, images in and out), `gridsnap`, `palette_lock`, `checks`, `render`
  (1x / 4x / neutral background), `judge`, `select`, and `edit.image` (image-to-image with an instruction, used by
  refine loops).
- **One declarative loop construct**, so pipelines can refine without a director:

  ```yaml
  - id: refine
    repeat: {max: 3, until: "score.mean >= bar or no_gain(2)"}
    steps:
      - {id: fix, step: edit.image, input: $best, instruction_from: $score.issues}
      - {id: resnap, step: gridsnap, input: $fix}
      - {id: rescore, step: judge, input: $resnap, suite: evals/convincing_enemy.yaml}
  ```

- **Prompts live in files** (`prompts/*.md`) and are hashed into the cache key, so editing a prompt is a visible,
  measurable change.

## 4. Repeatability: two different guarantees

1. **Replay (exact).** Every step result is cached by `sha256(step type + step-code version + input hashes +
   canonical params + prompt file hash + model id + seed)`, and every model call's request and response is saved.
   Re-running the same pipeline, brief and seed costs $0 and gives byte-identical outputs.
2. **Reproduce (statistical).** Model calls aren't deterministic, so a pipeline's quality is a **distribution over
   seeds**, not one lucky image. `ml bench pipelines/one_shot.yaml --seeds 4` runs K independent seeds and reports
   the score distribution, pass rate and cost per passing candidate. That is the number we compare pipelines on.
   "A beat B" means the distributions differ by more than the noise band.

## 5. Outputs

```
runs/<pipeline>-<seed>-<hash8>/
  manifest.json        every step: inputs, params, cache key, cost, outputs, timing
  steps/<id>/<k>.png   every intermediate, every fanout branch
  transcripts/         raw model requests and responses
  report.md            scores, checks, cost, the best candidate
  contact.png          one image: each step's candidates in rows, the best one ringed, the baselines alongside at the same zoom
bench/<pipeline>.md    distribution across seeds
```

The `contact.png` sheet is how you (and I) look at a run without a UI.

## 6. Steps

| # | Step | Done when | Est. cost |
|---|---|---|---|
| 1 | **Scaffold**: uv package `megaloman`, `ml` CLI, `.env`, vendored fusion-man tools (`gridsnap`, `gaps`, `pixelize`, NES palette and per-tile rules) | `ml --help`; vendored tools pass a smoke test | $0 |
| 2 | **Refs**: render Telly and Met from fusion-man's grids; rip Pipi; write the three neutral descriptions | `refs/*/` has 1x and 4x PNGs; the checks pass on all three (if authentic art fails a check, the check is wrong) | $0 |
| 3 | **Runner core**: YAML loader and validator, `$ref` resolution, fanout mapping, cache, transcripts, cost ledger with `--budget`, manifest | A pipeline of fake steps runs, and the rerun is 100% cache hits | $0 |
| 4 | **Eval suite + judge step**: `evals/convincing_enemy.yaml` (rubric, presentation, N samples, aggregation) | `ml eval refs/telly --desc …` returns structured scores | ~$0.20 |
| 5 | **Calibration**: the 3 baselines plus the 2 controls, ×5 each | `reports/calibration.md` gives the noise band, the per-baseline means, both controls, and the pass bar | ~$2-3 |
| 6 | **Pipeline A, `one_shot`**: run the squid, then bench 4 seeds | A best candidate plus `bench/one_shot.md` | ~$3-4 |
| 7 | **Pipeline B, `concept_then_sprite`**: a concept image ×4 → select → rules-first sprite from the concept → snap | `bench/concept_then_sprite.md` | ~$4-6 |
| 8 | **Pipeline C, `kitbash_ref`**: ripped parts (Met dome, Telly eye plate) given as image inputs, "build a squid from these" | `bench/kitbash_ref.md` | ~$3-4 |
| 9 | **Refine loop** on the winner (add the `repeat` block) | A paired bench: winner vs winner+refine | ~$4 |
| 10 | **Write-up**: `reports/v0.md`, the best squid at 1x and 4x next to the baselines, and which pipeline wins | You look at it and say whether it's usable | $0 |

That's about $20-25 of the $50 in total. Steps 1-3 need no model calls. I'll stop after step 5 to show you the
calibration, because if the judge can't separate signal from noise, everything after it would need to change.

## 7. Deliberately not in v0

The director, the UI, voice-in-app, threads and the coding-agent bridge, life cycles and animation, globals and
Convex. The pipeline format, cache keys and manifests are designed so that all of these can sit on top later without
changing them.
