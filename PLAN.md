# Megaloman: a reproducible pipeline for Mega Man stage assets

Product plan, v0 → v1. Scope: **basic enemies**. Fixed test case: **the robot squid** (described below), benchmarked
against **Telly** (MM2).

> **Status 2026-10-21:** scope pivoted. We are building only the repeatable declarative pipeline in `V0_IMPL.md`.
> This document is the long-term vision; its UI, voice, director and thread features are deferred.

---

## 0. Where we start from (what fusion-man taught us, and what it didn't)

`johnnyfs/fusion-man` (main + the unmerged `stage/sonar` branch) is a complete fan stage built by Claude Code agents
with human direction. It is our **evidence base, not our template**.

What it gives us:

- **Working tools**: `judge.py` (vision judge via OpenRouter with rubrics), `imagegen.py` (image model calls),
  `gridsnap.py` (recover a true 1x grid in exact palette colours), `rip.py` / `segment.py` (rip reference sheets),
  `work/fusion/gaps.py` (hole check), `assets.py` (NES palette table plus the 3-colours-per-8x8 rule check), a
  headless game engine with proofs.
- **Hard-won findings** (`docs/LEARNINGS.md`):
  - the judge's noise (a single sample swings by up to 35 points)
  - its low ceiling (authentic Telly scores only about 72)
  - its recognition and shading biases
  - "rip series-standard things, never draw them"
  - "refine the thumbnail, don't redraw it"
  - "frame 0 equals the static art"
  - cheap NES animation tricks
- **A prior human-approved answer to our exact test case**: sonar's `g` robot squid (a Met-dome mantle kitbashed with
  a Telly eye plate, 2 frames) and the `b` depth-charge mine (proximity, a 36 f flash, then a burst). This is our
  **human baseline** to compare against.

Why it isn't good enough:

- The progress was **unreproducible**. Coding agents improvised in free-form sessions, so we can't say which change
  (a style rule, a prompt pattern, a model, a tool) actually moved quality, and which was noise or luck.
- Guidance piled up into one big doc that every worker read in full, whatever its task.
- The human's feedback went through chat and was lost between sessions.

**So the core goal is to isolate what works.** Every run is reproducible and every variable is explicit, so we
can run ablations: the same brief and seed, with one style unit off, one model swapped, or one step removed, then
compare. The LEARNINGS claims come in as **hypotheses with provenance `fusion-man, unverified`**, and we promote
or demote them by measurement.

---

## 1. Principles (the opinionated part)

1. **The production loop is not a coding agent.** A *director* model is called with an assembled prompt and a fixed
   tool or JSON schema. Its context is built deterministically from files on disk, and every call is recorded. The
   coding agent (Claude Code) is a *separate* collaborator that maintains the workspace. It is never in the loop
   that makes art.
2. **Determinism means replayability, not bit-identical model outputs.** Every step is content-addressed:
   `hash(step type, inputs, params, seed, model id, guidance snapshot)`. A re-run hits the cache and is free and
   identical. Changing one variable invalidates exactly the downstream steps. Model nondeterminism is contained:
   it happens once per cache key, then it is frozen.
3. **Plans are data.** A plan is a small DAG of registry steps with parameters and fanout. It is not code. The
   director picks a plan template and fills in its parameters, or it composes a new plan from registered steps.
   It can never call anything that isn't in the registry; it *asks* for new tools instead (see F14).
4. **Score relative to a benchmark, never absolutely.** The judge scores the candidate and the benchmark (Telly)
   under the same rubric, with N≥3 samples. The headline number is **parity** (candidate ÷ benchmark). We prefer
   **pairwise** comparisons for fanout selection, and deterministic checks act as hard gates.
5. **The human outranks the judge.** Approval is a first-class state, and approved artifacts are frozen. A
   regression against the last approved version blocks promotion: we ran into that twice in fusion-man.
6. **The primary frame anchors everything.** Animation frames are derived *from* the approved primary frame
   ("same sprite, only X changed"), not regenerated.
7. **Every UI action is a tool call** on one command bus, and the agent and the human use the same verbs. The UI is
   just a client that renders the event log.
8. **Voice is deictic.** You speak while pointing (focus or hover), so "this is too dark" binds to the element under
   the cursor. Voice is a way to make comments and commands, not a separate chat.

---

## 2. Core objects

```
Stage ─┬─ Asset (enemy "robot squid")
       │    ├─ Brief: prompt + seed + benchmark refs + kind (small_enemy)
       │    ├─ Tracks: design ▸ lifecycle ▸ states ▸ animation map ▸ animations
       │    └─ Versions v1..vN   (one Version = one full pass)
       │         ├─ Plan (DAG + director rationale + guidance snapshot hash)
       │         ├─ Steps → fanout candidates → selected artifact (thumbnails at each step)
       │         ├─ Score (judge samples, parity vs benchmark, deterministic checks)
       │         └─ Transition → next version: {decision: retry|accept|escalate, reasoning,
       │                                         human feedback consumed, what to change}
       └─ Global library refs (explode, explode_small, splash, …)
```

The **Transition** record is the historical mapping from one pass to the next, *with reasoning*. It is the
director's memory: it reads the chain of transitions, not a raw chat history.

On disk, the run store is plain files (JSON + PNG, content-addressed blobs). You can inspect it with `ls`, and the
coding agent can grep it.

---

## 3. The test case

**Brief (fixed):** *"A floating, Blooper-like robot squid that swims along at and after Mega Man. It's a swimming
mine: when it gets close, its eyes flash red at an increasing rate, then it blows up."*

- **Benchmark:** Telly (MM2): a small floating homing enemy, 16x16, with 6 spin frames. It's the closest authentic
  analogue in size, role and movement. Secondary references: M-445 (MM2 jellyfish swimmer) and the MM4 sea mine
  (for the proximity-flash convention).
- **Human baseline:** the sonar-branch squid plus the mine's flash and burst.
- **Pass bar for v0:** median parity ≥ 0.95 vs Telly over 3 judge samples, all deterministic checks pass, and in a
  pairwise test "which one is authentic Capcom?" the judge picks ours vs Telly at least 30% of the time. The human
  decides whether it beats the sonar baseline.

The four layers this one enemy has to get right (and that the system must keep separate):

| Layer | For the squid |
|---|---|
| **Primary frame** (representative image) | The glide pose: mantle, eye plate, tentacles together. ~16-20 px. |
| **Life cycle / attack pattern** | Spawn → pulse toward Mega Man (gather, then stroke) → within range: stop, **arm** (the eyes flash red, accelerating) → burst. If shot: pops (small explosion), no burst. |
| **States** (derived from the life cycle) | `glide`, `gather` (the tell), `stroke`, `arming`, `burst`, `popped` |
| **State → animation map** | `glide` = frame A; `gather` = frame B (tentacles spread, held); `stroke` = B→A snap; `arming` = frame A + **eye palette cycle** (red/off, period 8f→4f→2f): **no new frames**; `burst` = **global** `explode` arranged as 1 big + 3 small around the centre; `popped` = global `explode_small`. |

The map makes the economy visible: the squid needs only **2 drawn frames** plus a palette cycle. The rest is global.
A good director should find that itself; the style unit "prefer palette cycles for warning lights" is how we test
whether guidance changes its plans.

---

## 4. The workspace on disk (hot-reloaded)

A nested layout where depth means specificity. Context assembly walks from the root to the leaf, and more specific
layers can override or extend.

```
workspace/
  guidance/
    pixel-art/            units/*.md   (general pixel-art rules)
    nes/                  units/*.md   (hardware-ish: 3 colours + transparent per 8x8, NES palette)
    megaman/              units/*.md   refs/   rubrics/
      small-enemies/      units/*.md   refs/telly/ …  plan-templates/*.yaml
      robot-masters/ …
    stages/sonar/         units/*.md   (stage palette, theme)
  skills/                 *.md         (how-to docs for steps: "derive a frame", "kitbash from a rip")
  tools/                  *.yaml       (registry manifests: inputs, outputs, cost, model)
  globals/                explode/, explode_small/, splash/  (shared assets + arrangement params)
  inbox/                  escalations from the running system (see F14)
  runs/                   the run store
```

- A **style unit** is one small rule with front matter: `id`, `scope`, `applies_to` (planner | generator prompt |
  judge rubric | check), `provenance` (`fusion-man/LEARNINGS§3`, `human 2026-10-21`, …), `status` (hypothesis |
  validated | rejected), and an `evidence` link to the ablation runs.
- **Hot reload:** a file watcher reloads guidance. Assets whose guidance-snapshot hash changed show a **stale** badge.
  Nothing re-runs automatically; you choose to re-run.

---

## 5. Features

Each feature is a discrete unit. `→` lists its dependencies. *Proof example* names the concrete thing we'd use first.

### Foundation

- **F1 Workspace + context assembly.** The nested guidance layers and style units; deterministic prompt assembly per
  role, with a snapshot hash. First content: migrate fusion-man's LEARNINGS into units, all marked `hypothesis`.
  → none
- **F2 Step registry + executors.** Manifested step types:
  - `gen.pixel` (specialized text-to-pixel). *Proof: PixelLab or Retro Diffusion APIs. Both offer reference-image
    conditioning, and PixelLab offers skeleton animation.*
  - `gen.image` (general image model / edit). *Proof: the nano-banana or gpt-image models via OpenRouter, already
    wired in `imagegen.py`.*
  - `gridsnap` (from fusion-man)
  - `palette_lock` (NES table + per-tile rule)
  - `kitbash` (compose from ripped parts)
  - `checks.*` (palette/8x8, orphans, enclosed holes: port `gaps.py`, outline continuity, canvas/anchor)
  - `render` (1x / 4x / on-background / beside Mega Man)

  → F1
- **F3 Run store.** Content-addressed artifacts and cached steps, versions, transitions, and a recorded transcript
  for every model call. → none
- **F4 Director.** One model call per decision, using structured output: (a) **plan** from the brief, history and
  guidance; (b) **transition** after scoring (retry, accept or escalate, with reasoning and the delta to try).
  *Proof: Claude Opus 5.5 via the API, with a JSON schema and fixed temperature.* → F1, F3
- **F5 Executor with fanout.** Runs the DAG. Any step can fan out ("4 passes, seeds s..s+3") with a selector:
  `best_by_judge_pairwise`, `best_by_checks`, `all` (pass-through), or `human_pick`. → F2, F3
- **F6 Judge.** A vision rubric for each asset kind; N samples; the benchmark is scored in the same batch; parity,
  plus pairwise, plus the deterministic checks as gates. The judge returns **actionable issues bound to regions or
  steps**, not just a number. *Proof: a different model family from the director to avoid self-preference
  (gpt-5.6-sol, already calibrated in fusion-man), with a Claude second opinion as an ensemble.* → F2
- **F7 Retry loop.** Brief → plan → execute → score → transition, until it passes the bar, the budget runs out, or
  it plateaus (two passes with no gain: fusion-man's stop rule, now measurable). → F4, F5, F6

### Watching and directing

- **F8 Command bus + event log.** Every action (agent or human) is a typed command: `comment`, `approve`, `reject`,
  `retry`, `pin_seed`, `fork_version`, `compare`, `set_benchmark`, `promote_guidance`, … The log is append-only and
  is streamed to clients. → F3
- **F9 Asset card UI.**
  - *Collapsed:* the animated thumbnail (or the latest mock, or "?"), version, parity score, status (running,
    awaiting you, stale, approved).
  - *Expanded:* the plan for the latest version as a horizontal step strip, with fanout thumbnails under each step
    and the winner highlighted. The director's rationale is truncated, with hover to expand. Score breakdown,
    benchmark beside it.

  → F8
- **F10 Versions: history and compare.** Step back through versions. Pin any version side by side (same zoom, same
  background, benchmark third). A "results only" filmstrip across all versions. → F9
- **F11 Targeted feedback.** Comment on *a step*, *a candidate*, or *a region of an image* (drag a box). The
  director's next transition must cite which feedback it consumed. → F8, F9
- **F12 Voice.** Push-to-talk (hold a key) with streaming speech-to-text. The live transcript draws as a chip on the
  focused element, and on release it posts as a comment there. A small router maps imperative phrases ("approve",
  "retry with four seeds", "compare with v2") to commands. Everything else stays a comment. Optionally the director
  asks short questions by text-to-speech. *Proof: a streaming transcription API (OpenAI realtime transcription or
  Deepgram), or local whisper.cpp; the browser Web Speech API as the zero-cost first cut.* → F11
- **F13 Human-action digest.** Before each director call, the human's actions since its last turn are summarized
  into a structured digest (what was looked at, compared, commented, approved). The UI shows that digest, so you
  see exactly what the agent was told. → F8, F4

### The system improves itself

- **F14 Escalation inbox + coding-agent bridge.** The director can file a typed request:
  - `need_tool` ("I need a skeleton-animation step")
  - `propose_guidance` ("this insight belongs at scope `megaman/small-enemies`")
  - `blocked`
  - `question`

  Requests land in `inbox/` and in a UI lane. Claude Code reads them through a small MCP server (or a skill), makes
  the workspace change, and replies. Hot reload picks the change up, and the affected cards go stale. It's a
  two-way dialog: either side can reply in a thread. → F1, F8
- **F15 Guidance promotion with scope.** A proposed unit shows as a diff at a chosen depth (pixel-art / nes /
  megaman / small-enemies / stage / asset). You approve or rescope it with one click or by voice. → F14
- **F16 Ablation runner.** "Re-run vN with change X": toggle a unit, swap a model, drop a step, or change the plan
  template. It reuses the cache for everything upstream of the change, then shows a paired score diff across K
  seeds. A unit's `status` moves from hypothesis to validated or rejected based on this evidence. **This is the
  feature that isolates what works.** → F3, F7

### It moves

- **F17 Life-cycle designer.** From the brief and the primary frame, the director drafts the behaviour as a timeline
  table (in the format of fusion-man's ENEMIES.md: tells, frames, speeds, avoidance). You edit it as text or by
  voice. → F4, F9
- **F18 States + animation map.** States are derived from the life cycle. Each state maps to one of: drawn frames,
  derived frames, a palette cycle, a transform (bob or flip), or a **global** ref. The map is shown as a grid on the
  card. → F17
- **F19 Animation production.** Derive frames from the approved primary frame. *Proof: an image-edit model with "same
  sprite, only X changed" plus gridsnap and a region transplant (the method proven in fusion-man); or PixelLab
  skeleton animation.* Palette cycles are rendered, and the collapsed card shows the animated GIF. The judge gets an
  animation rubric (frame consistency, anchor stability). → F18, F5, F6
- **F20 Global library + arrangement.** Shared assets (explosions, death orbs, splash) are ripped and frozen. An
  asset refers to one with arrangement params (count, offsets, timing), and changing a global marks its users stale.
  → F18

### After v1

- **F21 Engine sandbox:** drop the enemy and its life cycle into a fusion-man test room. A bot plays it, and checks
  that it is dodgeable and readable (the existing harness is reusable).
- **F22 Stage board:** stage cards and "+ theme".
- **F23 Family consistency:** judge "same artist?" across all of a stage's enemies.
- **F24 Style adapter:** train on 100-300 approved assets.
- **F25 Cost and budget dashboard.**

---

## 6. Milestones

### v0: "one loop you can watch" (primary frame only)

F1 (minimal), F2 (one `gen.pixel`, one `gen.image`, gridsnap, palette_lock, checks, render), F3, F4, F5, F6, F7,
F8 (log only), F9 read-only (a single expanded asset page that updates live), and **minimal F14** (threads:
`feature_request`, `blocker` with a scoped halt, and `resolution`, over MCP; see §9).

We ship **three plan templates** so the director has a real choice. They are also the three-way bake-off from the
ChatGPT notes, free as plan-level fanout:

1. `one_shot_pixel`: `gen.pixel` ×4 seeds → palette_lock → checks → pick best.
2. `concept_then_snap`: `gen.image` concept ×4 → pick → `gen.image` "rules-first NES sprite" → gridsnap → checks.
3. `kitbash_rip`: choose ripped parts → compose → recolour → checks (what made the sonar squid).

**Done when** `megaloman run squid --seed 7` runs unattended to the pass bar or the budget, you can watch every step
in the browser, the same command re-runs from cache identically, and the report shows parity against Telly and
against the sonar baseline.

### M1: "direct it"

F8 (full), F10, F11, F12, F13. You can redirect the squid by voice while pointing at the step that went wrong, and
see the director's next plan cite your note.

### M2: "it improves itself"

F14 (full), F15, F16. At least one escalation goes round trip through Claude Code with hot reload, and at least **three
LEARNINGS hypotheses are measured by ablation** (suggested: rules-first prompting; "refine the thumbnail, don't
redraw it"; "one highlight colour vs a shaded variant" against the judge's known shading bias).

### M3 → v1: "it moves"

F17, F18, F19, F20.

**v1 =** the squid from brief to an **approved primary frame + life cycle + state map + animations + global
burst**. It is scored against Telly and the sonar baseline, revised at least once by voice, with at least one
guidance promotion made through the coding agent, and every result is reproducible from the run store.

```
F1 ─┬─ F2 ─┬─ F5 ─┐
    │      └─ F6 ─┼─ F7 ── v0 ── F11 ─ F12 ── M1
F3 ─┼─ F4 ────────┘       F8 ─ F9 ─ F10 ─ F13
    └───────────── F14 ─ F15, F16 ── M2
                   F17 ─ F18 ─ F19, F20 ── v1
```

---

## 7. Tensions I resolved (tell me if you disagree)

- **"Deterministic" vs model calls:** resolved as replayable (a cache plus recorded transcripts). We don't aim for
  bit-identical regeneration.
- **"Not a coding agent" vs "dialog with the coding agent":** two separate loops. The director makes art inside a
  sandbox of registered steps; Claude Code changes the sandbox. They talk only through the inbox.
- **A minimal v1 vs the full feature list:** v0 is your minimal loop (brief → plan → retry → benchmark). v1 adds
  direction, voice, self-improvement and animation, because those were most of what you described.
- **Absolute score vs "the human wins":** the score only ranks candidates and decides retries. Only you approve.

## 8. Decisions (2026-10-21)

- **Repo:** `megaloman` (this directory). fusion-man's tools get vendored in as step executors.
- **Models:** everything goes through the local OpenRouter key: director, router, judge, and the general image
  models. The specialized pixel models (PixelLab, Retro Diffusion) are added as a step later, through a feature
  request, if the bake-off justifies it.
- **Stack, the "Modal for React":** **Convex** as the reactive state layer.
  - Its live queries push discrete updates to exactly the components that read the changed rows.
  - `npx convex dev` hot-deploys backend functions on save.
  - The same code runs in the cloud, or self-hosted locally.

  The front end is **Vite + React** (hot module reload). Heavy steps (model calls, gridsnap) run as **Python
  workers** that pull jobs from Convex. These are stateless and blob-addressed, so they can later move to Modal
  as they are.
  - **Split of truth:** the *workspace* (guidance, skills, tools, code) lives in git on disk. It is the coding
    agent's domain, and a sync worker mirrors snapshots and their hashes into Convex.
  - *Runs, events and threads* live in Convex. Blobs are content-addressed behind a storage interface (local
    directory now, S3 or R2 later).
- **Cloud discipline from day one:** no absolute paths in data; secrets only from env; workers are idempotent jobs
  keyed by cache hash; the coding-agent bridge is MCP over HTTP, not stdio only.

## 9. Three parties, one thread system (nail this in v0)

The **human**, the **harnessed agents** (director, router, judge), and the **coding agent** (Claude Code) all post
into the same typed **threads**. Each thread has a target (an asset, a step, a version, or the workspace) and a
type:

| Type | From → to | Effect |
|---|---|---|
| `comment` | human → system | Becomes router input (see §10) |
| `feature_request` | harnessed → coding | "Need step X / check Y / tool Z": the asset waits on it, other work continues |
| `blocker` | harnessed → coding | **Halts at a scope** (asset / stage / global) until resolved. Use it when it truly can't proceed, e.g. the judge's quota is gone or a step is crashing |
| `guidance_proposal` | harnessed → human (+ coding) | A scoped unit diff; the human approves, and the coding agent commits |
| `question` | any → any | Asked as a multiple-choice question when possible, so a voice answer can be "the second one" |
| `resolution` | coding → thread | "Done in commit abc; the registry now has `check.facing`." Hot reload follows, and waiting assets re-plan |

- **The coding-agent side:** an MCP server with `threads.list / claim / reply / resolve`, plus a push
  notification, so an idle Claude Code session wakes up on a new request or blocker.
- **The UI:** a **Requests lane** that shows open threads, who holds each one, and any active halts (red banner
  with its scope).
- **The voice toggle** (`→ System` | `→ Coding agent`) only changes the recipient. Speech sent to the coding agent
  becomes a thread with author `human` and the focused element attached. The toggle stays manual for now; later
  the router can classify the recipient.

This moves F14 into v0, in a minimal form: `feature_request`, `blocker` and `resolution`, with the MCP tools.

## 10. M1 directed revision: how a spoken note becomes a version

You say, pointing at the `gen.image` step of v3:
*"Graphics were off here. I think it's the orientation of the base model you're working from."*

1. **The router** (a fast model) gets the transcript plus the deictic context (the focused step, its inputs, and
   the version) and returns a **command bundle**. It doesn't execute anything; it proposes. Each bundle comes with
   a confidence and any alternative readings.
   - `fork_plan(v3 → draft v4)`
   - `add_guidance(scope=asset, step=gen.image, "the base reference faces right; the enemy must face left. Mirror
     the reference first, or state the facing in the prompt")`
   - `add_expectation(eval, "sprite faces left (eye toward -x)", check=check.facing | judge_question)`
2. **The expectation is the key addition.** Your note becomes a *falsifiable hypothesis* attached to the eval step.
   The next version must show the expected difference, not just score higher. Use a deterministic check if one
   exists; otherwise ask the judge a yes/no question.
3. **Run the draft.** It is **blessed** as v4 automatically if:
   - the expectation passes,
   - the parity score is **not worse beyond noise** (the judge's 3-sample noise is about ±3-7, so "went up" alone
     isn't a reliable signal), and
   - a pairwise comparison against the last *approved* version doesn't regress.

   Otherwise it goes back to the director with the failure.
4. **Ambiguity** (low confidence or two readings) → one `question` thread with options, answerable by voice.
5. **A missing capability** (e.g. there is no `check.facing`) → a `feature_request` to the coding agent. Meanwhile the
   eval falls back to the judge question and marks the result `provisional`.
6. **When to ask the human** comes from a **deterministic policy table**, not from the model's own sense of "drastic":

   | Change | Policy |
   |---|---|
   | Asset-scoped guidance or expectation, same plan template, within budget | auto |
   | Plan template switch, stage-scoped guidance, over 2x the usual cost | confirm (one tap or "yes") |
   | Guidance promotion above stage scope, marking art `approved`, editing a global asset | always the human |

So yes, the flow you described feels right. The three refinements are: (a) make the expected difference an explicit
check, (b) define blessing as "expectation passed + no regression" rather than "score went up", and (c) put the
"drastic" threshold in a policy table you can edit.

## 11. Built to generalize (without building it yet)

- **The game is the top of the guidance tree:** `pixel-art/ → nes/ → games/megaman/ → kinds/small-enemy/ →
  stages/sonar/ → assets/squid/`. Kid Icarus becomes `games/kid-icarus/` and reuses `pixel-art/` and `nes/` as
  they are. Making that reuse work is a good test of the layering.
- **Asset kind is data:** each kind declares its **tracks**, rubric, benchmark set, and plan templates.
  - Small enemy: frame, life cycle, states, animations.
  - Big robot / roomstopper: adds the arena and the weak point.
  - Robot master: adds the portrait and the weapon.
  - Tiles: frame (metatile set) and autotile rules; no life cycle.
  - Backgrounds: parallax bands.
  - Hazards, allies: their own track sets.

  Nothing in the code says "enemy".
- **Asset chains:** an asset can depend on another (the puffer's mines are the `b` mine; the weapon-get palette
  depends on the boss). Dependencies drive stale-marking, exactly like globals do.
