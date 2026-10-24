# Model in the loop: A/B plan for the region → render stages

**Status:** speculative plan, 2026-10-23. Nothing here is implemented or run. Baseline: **v4**
(`ladder_v4.py`, `render_v4.py`, `workspace/experiments/regions/v4/`).

**Goal (Johnny):** a repeatable process that reliably gives a really good 24 or 20 px sprite,
and ideally a good 16. Today, region → render is deterministic: a few hundred lines of
squid-specific rules (significance `k`, face template tables, arm gaps/curl, fin pay, line
sides). It does well, but each smaller rung is worse than the last, and every new problem
costs a new rule.

**Principle:** blessed rules are hard constraints. Models decide only where the rules leave
freedom, or where the rule is a *trial*. A model's change that breaks a blessed check
(separation, eyes, exact mirror, holes, ≤ 4 colours) is rejected without being judged.

## What we know about the models

| Model | Role today | Cost / call | Latency | Strength | Weakness |
|---|---|---|---|---|---|
| `typesafe/jev-1.14` (decisions) | symmetry classifier | ~$0.00005 | 0.3 s | defect checks (noise, copied half p ≥ 0.84); near-deterministic | merged parts p ≈ 0.55; **position bias** (mostly B) |
| `openai/gpt-6-luna-decisions` | none | ~$0.00005 | 0.3 s | as Jev; 0 unstable items | **always picks A** on preference; p = 0 on exact mirrors |
| `openai/gpt-5.6-sol` (judge) | legend, judge | $0.002–0.007 measured | 2–6 s | 95 % on known items | 9/29 items unstable; overrode by Johnny once ("too wing-like") |
| `gemini-nano-banana-2.1` (BYOK) | region paint | ~$0.04 (Google) | ~10–30 s (est.) | **traces** (IoU 0.996, identity registration) | untested on pixel grids |
| `openai/gpt-5.4-image-2` | old region paint | ~$0.25 (est.) | ~30–60 s (est.) | clean flat maps | redraws (4–6 % scale, 14 px shift) |

Decision-model images must be 256–1568 px a side, so sprites are nearest-upscaled (`decisions.fit`).
The Decisions API also has a **Box** answer type, which suits "where is the problem?".

## 1. Intervention points

"Per sprite" means one concept through the three target rungs 24/20/16 (64/32 are already good).
Costs are model costs only. The deterministic renders take milliseconds.

| # | Where (function) | Rule today | Model decision | Type | Model class | Calls / sprite | Cost / sprite | Latency (parallel) |
|---|---|---|---|---|---|---|---|---|
| F | after each rung (new) | `L.health` thresholds | **flag** problem areas: part shrunk too far, merged arms, lost line, heavy blob, speck | yes/no per defect + Box | decision (Jev + Luna) | 3 × 2 | $0.0003 | 0.5 s |
| L1 | `ladder_v2.choose_k` | significance schedule + footprint check | which fin exaggeration `k` (3–5 candidates) reads best | Score per candidate | decision (absolute questions) | 15 × 2 | $0.0015 | 1 s |
| L2 | `L.assign` contested cells | fixed `PRIORITY` | which part keeps a contested cluster | Choice (A / B / keep) | decision on a highlighted crop, or LLM grid | 10–30 | $0.001–0.2 | 2–20 s |
| L3 | `L.health` → floor/reshape | area < min_px², merged, 1 px thin | "did part X shrink too far?" → trigger a floor | yes/no | decision (defect check: its strength) | 12 parts × 3 | $0.002 | 1 s |
| L4 | `face_template` | `EYE_FLOOR`, `EYE_SLIT`, `SLIT_MOAT`, ring rows | which face variant (slit 1×2 / 1×3 / box 2×2, moat, ring top) | Choice of K = 4–6 | decision score → judge on top 2 | 18 + 4 | $0.03 | 6 s |
| L5 | `separate_arms`, `curl_tips`, `cup_floor` | gap from arm behind; `NOTCH_W`, `CURL_RUN` | which gap, notch and curl variant per arm | Choice of K = 3–4 | decision score | 12 × 2 | $0.001 | 1 s |
| L6 | `fin_pay` | re-grow from the background below 32 | re-grow or not, and how far | yes/no | decision | 3 × 2 | $0.0003 | 0.5 s |
| L7 | face strays, render specks | nearest non-face part; speck → line | absorb pixel P into A, into B, or make it line (**promote / demote**) | Choice per pixel | LLM grid edit (batched), or decision on a crop | 1–3 LLM | $0.01–0.05 | 5–20 s |
| R1 | `render.nl` | lines everywhere (blessed); cup \| arm unlined below 32 | line or no line on cup \| arm only | yes/no | decision | 3 × 2 | $0.0003 | 0.5 s |
| R2 | `render.gives` | protected > cup > `behind()` | which side gives the line, per boundary segment | Choice (2) | decision (both orders) or LLM grid | 20–40 | $0.002 | 2 s |
| R3 | no-clump clean-up | highest-neighbour pixel first, refill own/neighbour | which clump pixel to un-ink, and with what | Choice | LLM grid | 1–3 | $0.01–0.05 | 5–20 s |
| R4 | fin depth line | `fin_depth = 2` (heavy black block at 16) | 0 / 1 / 2 px | Choice of 3 | decision score | 9 × 2 | $0.001 | 1 s |
| G1 | all constants | hand-tuned | beam over the parameter vector | Score | decision prune → judge gate | 200–500 + 20 | $0.05–0.15 | 1–3 min |
| I1 | final pixels at 24/20/16 | none | **refine at working resolution**: upscale 16–32×, image edit, snap back | pixel edit | image edit (Nano Banana) | 3–6 | $0.12–0.24 | 1–3 min |
| I2 | rung N → N−1 | coverage shrink from the source | **image model does the step-down**, then snap, then re-line | pixel edit | image edit (Nano Banana) | 3–6 | $0.12–0.24 | 1–3 min |
| I3 | region map at 128/256 | none | redraw the regions in iconic form (bigger fins, separate arms), then re-ladder | pixel edit | image edit (Nano Banana) | 2 | $0.08 | 1 min |

The LLM grid edit costs are estimates (32×32 grid ≈ 1 k tokens in, a few hundred out, on
gpt-5.6-sol). None has been measured.

## 2. Candidate designs

### D1. Local critic + candidate generator
Deterministic code proposes K local variants at one point (L1, L4–L7, R2, R4). Examples:
absorb P into A or B, thin or thicken a line, grow a part one pixel, a different slit size. Each
variant is rendered by the existing renderer. Blessed checks reject the bad ones, and a critic
picks from the rest.

- **Pros:** cheap ($0.001–0.03 a sprite). Every change stays local, explainable and
  reversible. It reuses the existing knobs, and the rules still guarantee structure.
- **Cons:** only as good as the generator, so it can't invent a shape no rule proposes.
- **Risks:** position bias (see the table below); scores that barely differ at 16 px (the
  critic can't separate the candidates); and the critic learning nothing beyond the checks.

### D2. Flagger → router (Johnny's 3)
A decision model looks at each rung's render. It answers fixed defect questions (yes/no with
p) and, for each "yes", a Box. Each flag is mapped to the fix for its kind: the
deterministic floor (L3), a D1 local critic in that box, or an I1 image refine of that box.

- **Pros:** these are decision models' proven strength (defect checks), and the flags make the
  work targeted and auditable. Together with the existing `health` it's a second opinion.
- **Cons:** whether Box localisation works on sprites is unknown, so it has to be validated on
  synthetic defects with known locations. Jev is weak on merged parts (p ≈ 0.55).
- **Risks:** false flags lead to needless edits. Mitigation: a flag needs both models (or Jev
  plus `health`), and a fix is kept only if checks pass and the critic doesn't score it lower.

### D3. LLM edits the region grid as text (≤ 32 px)
Give gpt-5.6-sol the legend letters, the rung map (N−2, left half only because of the mirror),
the blessed rules and the flags. It returns JSON edits `{y, x, letter, reason}`, at most E
per call. The edits are applied and mirrored, then `render_v4` redraws the lines and outline,
and the checks gate the result.

- **Pros:** works at the true pixel level, and the reasons are readable. It covers idea (b)
  directly: outline, promote/demote and absorb.
- **Cons:** LLMs count grid columns badly and may break shapes they can't "see".
- **Risks:** off-by-one edits, which the gate and a small E keep cheap to reject; and
  variance, so sample 3–5 and let D1 pick.

### D4. Image-edit refine at working resolution, or the image step-down (Johnny's 1 and 2)
- **I1 refine:** upscale the 24/20/16 render nearest by an integer factor to ~512 px, with an
  optional faint grid. Ask Nano Banana to fix the flagged areas only, "keep the exact grid,
  palette and symmetry". **Snap back:** majority colour per cell, snap to the 4-colour palette,
  mirror the left half. Recover labels: a cell whose colour matches its old group keeps its
  label; otherwise it takes the nearest region of that colour, and black becomes ink. Then run a
  new deterministic `repair` pass on the final pixels: insert a separation line wherever two
  regions touch (from the side `behind()` picks), thin clumps, turn specks to line. Re-check
  everything.
- **I2 step-down:** give it rung N (upscaled) and ask for the same sprite on an (N−1) grid.
  Snap, repair, check. This conflicts with the finding *compute each rung from the source*
  (chaining hides breakage). So always run `health` against the source, and fall back to the
  deterministic rung when it is worse.
- **I3:** refine the region map at 128/256 into iconic form, then run the unchanged ladder.
  This is the safest place for an image model, at a resolution it is good at.
- **Pros:** Nano Banana traces (IoU 0.996), so the grid should survive. It's cheap ($0.04) and
  could produce shapes that no rule proposes.
- **Cons / risks:** Johnny doubts image models at 16–32 px (fair, and untested). Likely
  failures are anti-aliasing, cells off by half a pixel, invented detail and palette drift, which
  the snap hides but doesn't fix. Shared colours (cream = ring, arms, eyes) make label recovery
  ambiguous. A probe of ~$0.25 answers whether it is worth it.

### D5. Judge-gated beam over rule parameters (G1)
The state is the knob vector: `k` per significant part, `EYE_SLIT`, `SLIT_MOAT`, `CURL_RUN`,
`NOTCH_W`, `CUP_FLOOR`, `fin_depth`, `FIN_PAY`, `RING_BOTTOM_MIN`. Search with beam width 4.
Decision models prune with absolute scores, and gpt-5.6-sol picks the final beam in
order-swapped pairs.

- **Pros:** no new drawing code, and it finds good combinations of existing rules.
- **Cons:** it can't escape the rule family.
- **Risks:** Goodhart (the judge has already disagreed with Johnny); tuned on one squid and
  overfit to it.

### Neutralising position bias

| Technique | Use | Cost |
|---|---|---|
| **Avoid pairwise:** score each candidate alone with absolute yes/no questions (several per call) and rank by p | default for decision models | 1 call per candidate |
| Both orders: ask AB and BA, average p(X); count a win only if it holds in both orders, else a tie | all pairwise, every model | 2× |
| K-way: rotate the order through all K positions (Latin square); majority or mean p | K > 2 choices | K× |
| Randomise the labels and the side; log the order shown | always | 0 |
| Sentinels: mix in bake-off pairs with known answers; drop a run that misses them; report the order-flip rate | every batch | ~$0.001 |
| Ties default to the deterministic choice | always | 0 |

## 3. A/B design

| Arm | Description |
|---|---|
| **A (control)** | v4 as committed (`e208ff6`). Frozen v2 region map, palette, legend. 64/32 unchanged. |
| B1 | v4 + D2 flagger + D1 critic on existing knobs (L1, L4, L5, L6, R4) |
| B2 | v4 + D2 flagger + D4/I1 Nano Banana refine at 24/20/16, snap + repair |
| B3 | v4 + D3 LLM grid edits on the flagged areas |
| B4 (later) | D4/I2 image step-down; D5 beam; D4/I3 iconic region map |

Each arm is a pipeline **version** (`v4-critic`, `v4-nbrefine`, `v4-grid`) per the process rule.

**Metrics, in gate order:**
1. **Hard gate (blessed checks):** `separation_check.ok`, `eye_check.ok`, `mirror_exact`,
   `imaging.run_checks` (orphans, holes, outline, ≤ 4 colours), tile changed px. Any failure
   rejects the candidate.
2. **Deterministic health:** `L.health` reds and ambers per part vs A; `health_score`; part
   areas (fins, arms, pods) and the arm count at 16. Reported, not gating.
3. **Judge:** gpt-5.6-sol pairwise vs A, both orders × 2 samples. Report consistent
   win / tie / loss. Decision-model pairwise votes are logged but carry no weight.
4. **Johnny's eye (the arbiter):** a blind pick per size. A and the B candidates are shown in
   random order, without labels, and the mapping is revealed afterwards. We also record his
   agreement with the judge and with the critic. That number decides whether a critic can
   stand in for him later.

**Cockpit:** add each arm to the existing **pipeline-versions** switcher on the megaloman tab.
Each entry shows a 24/20/16 strip at the same zoom, a diff vs v4 (changed pixels in magenta),
the flag boxes, each model decision with its p and reason, the check pass/fail, and the cost.
For the blind pick, a separate panel `ab-blind` shows the shuffled candidates as 1, 2, 3…;
Johnny says "twenty: three", and the answer is recorded with the hidden mapping.

## 4. Recommended first experiment: E1 "flag, then critic or refine" ($3 cap, ~$0.5 expected)

It answers three questions at once: does the flagger localise real problems? Does a cheap
critic agree with Johnny better than chance? And can Nano Banana do anything useful at
24/20/16?

| Step | What | Spend |
|---|---|---|
| 0 | Freeze A: v4 renders and grids at 24/20/16 from `v4/results.json` | $0 |
| 1 | **Plumbing only:** a `params` dict through `ladder_v4.rung` / `render_v4.render` (fin `k` offset −1/0/+1 step, `fin_depth` 1/2, `EYE_SLIT` at 20 {1×3, 1×2, 2×2} and at 16 {1×2, 2×2}, `CURL_RUN` 0/1/2, `FIN_PAY` on/off). Defaults = v4 exactly; assert pixel-identity. | $0 |
| 2 | Enumerate the combinations per size (≤ 108), render, dedupe identical grids, apply the hard gate. Expect 20–60 survivors per size. | $0 |
| 3 | **Validate the flagger:** run the bake-off's synthetic defects (noise, merged, broken eye, copied half: known locations) through Jev and Luna with yes/no + Box. Record hit rate and box IoU. | ~$0.005 |
| 4 | **Flag A:** the same questions on v4 at 24/20/16 (defects: fin too small, arms merged, lost line, heavy black blob, speck, eyes indistinct). Show the boxes in the cockpit. | ~$0.001 |
| 5 | **Critic (arm B1):** score each survivor alone, 5 yes/no questions in one call, Jev and Luna → mean p(yes). No pairwise, so no position bias. Sentinels: v3 `alt_finline` 16 and v4 `alt_cupline` 16 must rank below A. | ~$0.01 |
| 6 | **Refine probe (arm B2):** Nano Banana on v4 24/20/16, upscaled to ~512 with a faint grid, prompt names the flagged boxes. 2 seeds per size. Snap back (cell majority, palette, mirror), label recovery, `repair`, hard gate. Log how many cells changed and why each was rejected. | ~$0.24 |
| 7 | **Judge:** for each size, the critic's top 2 and the best gated B2 vs A, gpt-5.6-sol, both orders × 2 samples. | ~$0.15 |
| 8 | **Johnny blind** (`ab-blind` panel): per size, A + critic top 2 + 2 random survivors (the control) + the B2 result, shuffled. He picks the best, and optionally ranks the rest. | $0 |
| 9 | Write a finding in `PIPELINE_RULES.md`: critic vs random, judge vs Johnny, flagger hit rate, whether NB is viable at working size. | $0 |

Budget enforced with `models.Ledger(3.0)`; everything is cached by `store`.

**Expected outcomes, and what each means:**

| Outcome | Likely? | Then |
|---|---|---|
| Flagger hits synthetic defects (p ≥ 0.8) but boxes are loose | likely | use the flags as yes/no triggers and localise deterministically (`health`, a diff to the source) |
| Critic picks `fin_depth 1` at 16 (v4's "heavy black block") and different slit sizes | likely | promote those knobs to critic-chosen; next, E2 = pixel-level absorb/promote (L7, R2) by the same critic |
| Critic scores cluster (spread < 0.1) and its picks ≈ the random control in Johnny's eye | plausible | decision models stay defect screens only; go to D5 with gpt-5.6-sol, or D3 |
| Nano Banana's refine survives snap + repair and Johnny picks it at 24 or 20 | uncertain (Johnny doubts it) | E2 = I1 on flagged boxes only, then the I2 step-down 24 → 20 → 16 |
| NB output is blurred or off-grid, so the snap reverts most changes or breaks lines | plausible | keep image models at ≥ 64 (I3) |

**Success for E1:** Johnny prefers a B pick over A at 24 or 20, and the critic's top 2 beat the
random control in his ranking.

## 5. Open questions for Johnny

1. Which sizes count? This plan targets 24/20 (must be good) and 16 (stretch). Is 32 frozen?
2. May a model override a **blessed** rule (e.g. drop a separation line), or only choose where
   the rules leave freedom (the plan's assumption)?
3. Are you willing to do a short blind ranking session (about 6 sprites × 3 sizes)? Your picks
   are the ground truth that calibrates the critic and the judge.
4. Do we need a second creature soon? The rules hard-code squid parts, so a model in the loop
   only pays off if the process generalises.
5. For the Nano Banana probe: a faint grid overlay or none? Should it fix only the flagged
   boxes, or be free to refine the whole sprite?
6. When the critic sees no clear winner, should it keep v4 (the conservative default here) or
   pick the best-scoring candidate anyway?
7. Latency: is a pipeline of ~1–3 minutes per sprite fine (offline), or should the critic be
   interactive?
8. Budget after E1: is ~$5–10 acceptable for E2 (pixel-level critic + NB step-down)?
