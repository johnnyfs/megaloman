# Pixel transformer scaffold report — 2026-10-24

**Implementation ready for an approved-data pilot; artistic success unassessed.**
No approved sprite corpus was ingested or cleaned, no real training pilot was run,
and no Modal functions were launched. Remote spend: **$0**.

## Reproduction and artifacts

Follow [the run instructions](pixel-transformer.md). The local smoke used
`configs/pixel_model/smoke.yaml`, PyTorch 2.10.0+cpu, FP32, two CPU threads,
width 32, two layers, four heads, MLP ratio 2, no dropout, 18,692 parameters,
batch 2, no accumulation, eight AdamW updates, one warmup step, and four sampling
iterations. Training deliberately stopped at step 4 and resumed through step 8
without changing the scheduler horizon.

The synthetic snapshot has 24 frames in 12 families: 16/4/4 train/val/test frames,
8/2/2 families, native 16×16, four classes. Its snapshot SHA256 is
`6fa825a8890ba33c50fdbb70f7e49e3277ba78907662bda9b79fdc1619500d80`.
This is a fixture, not a blessed library cohort. Test examples were not used for
model selection or this development evaluation.

Artifacts are intentionally ignored by Git under `workspace/pixel_model/`:

- `datasets/tiny/{manifest.json,summary.json,sheet.png,frames/}`
- `runs/smoke/{config.json,dataset.json,environment.json,metrics.jsonl,result.json}`
- `runs/smoke/checkpoints/{step-00000004,step-00000008,latest.json}`
- `runs/smoke/{generated,reconstructed,inpainted}/` (PNG/JSON/sheets and optional traces)
- `runs/smoke/evaluation/report.json`, plus 64 unselected samples, blind sheet,
  rating CSV, resampling control and nearest-training panels under `evaluation/16/`
- `runs/smoke/overfit-diagnostics/overfit.json`
- `runs/doctor-{smoke,sprite16}.json`

## Observations

| Check | Result |
| --- | --- |
| Smoke updates | 8 completed, including stop/resume at step 4 |
| Training loss (different sampled masks) | 0.8164 at step 1; 0.6863 at step 8 |
| Training segments, including validation/sampling/checkpoints | Approximately 0.58 s + 0.69 s, excluding process startup |
| Smoke model warmed optimizer step | Approximately 7–10 ms at batch 2 across two measurements; about 52–72k pixel tokens/s |
| Default 192×6 model | 2,709,508 parameters; approximately 0.69 s/optimizer step, batch 8, CPU FP32 |
| Observed attention backend | CPU flash SDPA for the no-dropout smoke; math SDPA for default dropout training |
| Resume regression | Interrupted vs continuous four-step training matches every model tensor exactly, with dropout and accumulation enabled |
| Inpainting | All 220 locked IDs preserved; 36 editable cells completed |
| Test suite | 16 tests pass, including existing client tests, CPU model/masks/round-trip, integration/resume, and offline Modal construction |
| Modal | Dry-run and SDK construction only; no image build, upload, remote invocation or spend |

The NVIDIA driver could not communicate with the GPU, so the known RTX 4070
Laptop's 8 GB budget was not profiled. CPU timings are measurements of this
machine/configuration, not GPU throughput predictions.

The optional eight-update diagnostics decreased fixed-mask loss from **0.6959 to
0.5383** on one full-mask sprite, and from **0.7515 to 0.5895** on four partial-mask
sprites. These short checks establish optimization activity. They do not establish
memorization; foreground performance and substantially longer correctness checks
remain necessary before a real pilot.

Validation uses fixed random and contiguous masks at all five planned ratios,
with identities weighted equally. Random-mask results on the two fixture
validation families are:

| Mask ratio | Model conditional NLL | Position baseline NLL |
| --- | ---: | ---: |
| 50% | 0.7655 | 0.5049 |
| 75% | 0.7480 | 0.4914 |
| 100% | 0.7625 | 0.5009 |

The provisional baseline gate is **not met**. Masked argmax predictions are
background-dominated, with zero foreground accuracy in this smoke. Inspecting
the generated sheet shows scattered noisy pixels rather than coherent silhouettes.
Across 64 unselected samples, approximately 87.4% of pixels are transparent,
5.4% ink, 3.7% tone 1 and 3.5% tone 2. All 64 grids are unique, none is exactly a
training frame, and none crosses the uncalibrated 0.10 foreground-distance
near-copy threshold. These novelty figures do not establish quality. Human
rating cells are empty and noisy-sample rate is deliberately unassessed.

## Remaining decisions and gates

1. Which externally approved source/style cohort, canonical families and frozen
   snapshot should the first real experiment use? Upstream prep must establish
   role consistency, native placement, provenance, approval binding and leakage
   checks. This scaffold does not implement that workflow.
2. Calibrate near-copy distance and predeclare acceptable copying/diversity
   limits. Confirm the provisional human useful-sample threshold (50%) and
   silhouette/coherence/readability/usefulness rubric with real controls.
3. Repair the NVIDIA driver and measure local 8 GB feasibility with the default
   model. Run full single-image and few-image overfit checks, then a bounded
   approved-data pilot and baseline comparison before any scale-up.
4. A future authorized Modal integration should refresh pricing, test image
   construction, upload once, deliberately interrupt/resume, download, and sample
   locally. Concurrency/retry/time/step guards exist; remote durability and GPU
   behavior have not been demonstrated. Multi-seed experiments remain future work.
5. Keep transparent + three colors. The position-provider boundary accepts depth
   metadata for a future versioned axis; larger palettes and new native cohorts
   require their own data/resource/evaluation gates.

Go/no-go: **go for scaffold handoff; no artistic/generalization conclusion.**
The next experiment is a faithful approved 16×16 cohort plus correctness
overfitting and a measured local pilot, after data preparation is ready.

## Commit status

The worktree is on `feat/pixel-transformer-v0`. The attempted first `git add` failed
because shared metadata at
`/home/johnnyfs/Projects/megaloman/.git/worktrees/megaloman-ptv0` is mounted read-only
in this session. No commits or pushes were made. Intended logical commit messages
(each with the final trailer `Co-Authored-By: Codex`) are:

1. `feat(pixel): add approved dataset contract and masked transformer core`
2. `feat(pixel): add reproducible training, decoding, evaluation and Modal adapter`
3. `docs(pixel): document scaffold workflow and local smoke evidence`
