# Review and recommended plan: masked-pixel transformer v0

*Read-only review, October 24, 2026. No files modified.*

## 1. Critical review

**Recommendation: proceed with a bounded experiment, after tightening the data, representation, and evaluation contracts.** The proposed architecture is plausible and appropriately small. The main uncertainty is whether it can learn coherent, novel sprites from the relevant approved subset.

The strongest parts of [the current plan](/home/johnnyfs/Projects/megaloman/plans/pixel-transformer-v0.md) are its categorical output, explicit treatment of transparency and masking, local/Modal code sharing, incremental implementation, and separation of unconditional generation from future concept conditioning. The [diffusion trials](/home/johnnyfs/Projects/megaloman/plans/model-trials-status.md) justify exploring a dedicated model. They do not establish which dedicated architecture will work best.

### What the actual library changes

A read-only inventory of [the sprite library](/home/johnnyfs/datasets/sprites), excluding `_sources` and `_tools` and following frame entries in `animation.json`, found:

| Observation | Count | Implication |
|---|---:|---|
| Sprite folders / animation files | 2,703 / 9,490 | Folder counts substantially overstate independent identities. |
| Referenced frame PNGs | 38,875 | Raw candidate inventory; approval status was not established. |
| Exactly 16×16 frames | 7,490 | Many belong to tiles, gimmicks, and other categories. |
| Frames with more than three opaque RGB colors | 16,848, about 43% | The proposed representation would reduce or exclude substantial data. |
| Frames containing partial alpha | 651 | Alpha thresholding can alter source artwork. |
| Enemy frames fitting 16×16 opaque bounds, with binary alpha and ≤3 opaque colors | 1,290 across 266 folder identities | A plausible initial candidate pool, before approval and semantic deduplication. |
| Exact rendered-image duplicate groups spanning different identity folders | 503 | Folder-based splitting alone will leak duplicates. |

These are observations of the live candidate tree, not statistics for blessed training data. The [library design](/home/johnnyfs/Projects/cockpit/plans/sprite-library.md) and [capture status](/home/johnnyfs/datasets/sprites/_tools/CAPTURE_STATUS.md) remain useful provenance context, but their inventory is older than the current tree.

### Critical findings

1. **The four-class palette is a substantive research restriction.**  
   [The palette contract](/home/johnnyfs/Projects/megaloman/plans/pixel-transformer-v0.md:60) assumes that three luminance-ordered colors correspond to outline, dark, and light. Dark pixels also represent eyes, mouths, markings, and solid fills. Missing colors can make a per-frame ranking assign inconsistent roles across an animation.

   NES subpalettes also do not imply three opaque colors for an entire assembled character. The inspected Mega Man run frame contains five opaque RGB colors. Indexed PNG IDs likewise differ between sources and cannot serve directly as universal model classes. Start with a faithful restricted subset; make destructive reduction a separate experiment.

2. **The ingestion contract should exploit the existing library.**  
   An arbitrary recursive PNG glob would include source sheets, layers, tiles, and incomplete components. Read `sprite.json`, `animation.json`, frame entries, and provenance at the appropriate level. Mixed Maker/capture folders require animation-level provenance. The documented collision repair is already complete; preserve that work.

   Human approval addresses source-quality concerns. The missing training contract is how approval binds to exact content and preprocessing so a subsequent extraction cannot silently change an approved dataset.

3. **Split by canonical identity or design family, before augmentation.**  
   [“Character/animation group”](/home/johnnyfs/Projects/megaloman/plans/pixel-transformer-v0.md:44) is ambiguous: animation-level splitting permits different poses of the same character on both sides. Related assets also span games, capture sources, palette variants, and fan recreations. Detect duplicates again after normalization and categorical conversion. Balance sampling by identity and animation so long sequences and palette variants do not dominate.

4. **Reconstruction success is an insufficient generation gate.**  
   A model can exploit visible pixels and still generate incoherent sprites from an empty canvas. At full masking, distinct sprites with identical metadata provide ambiguous targets; low reconstruction error on several such examples is not a valid memorization requirement. Test single-image full-mask overfitting separately from multi-image partial-mask overfitting.

   Masked cross-entropy should be called **masked conditional NLL**. It does not directly supply an image-level joint likelihood. Include a position-dependent class-frequency baseline, which captures alignment and background occupancy more strongly than a global frequency baseline.

5. **The sampler is reasonable, but underspecified.**  
   Preserving committed pixels is consistent with the [reference MaskGIT implementation](https://github.com/google-research/maskgit/blob/main/maskgit/libml/parallel_decode.py). The important missing details are the exact cosine schedule, confidence of the sampled class, tie handling, separate sampling/ranking temperatures, and scheduling relative to the initially editable cells. Pixel-level tokens and a four-class vocabulary remain an unvalidated adaptation of [MaskGIT](https://arxiv.org/abs/2202.04200).

6. **The evaluation needs declared success and failure criteria.**  
   Sixty-four samples are useful for development sheets. Final assessment needs several training seeds, unselected outputs, calibrated duplicate detection, and identity-level uncertainty estimates. Separate readability, novelty, and usefulness. A sharp lattice is guaranteed by representation; artistic quality must be measured independently. Inpainting also needs contiguous-hole tests, since random missing pixels are an easier task.

7. **Hardware and operational assumptions need updating.**  
   The local budget is known: RTX 4070 Laptop, 8 GB. The proposed six-layer, width-192 model has roughly 2.7 million parameters; activations and attention implementation deserve more attention than parameter storage. Actual feasibility still requires measurement. This review could not benchmark the GPU because `nvidia-smi` could not communicate with the driver.

   Use the available H100 for bounded paid runs. Select hardware using measured cost per comparable training workload. SDPA can fall back to other implementations, so verify the executed backend; evaluation must explicitly disable its attention dropout. [PyTorch SDPA documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)

8. **Future conditioning is a separate data program.**  
   The library provides categorical artwork, poses, palettes, and some source structure. It does not automatically provide aligned high-resolution concepts, semantic part maps, or examples separating reference style from identity. Preserve useful metadata now; defer those encoders until a paired-data pilot exists. Single-frame v0 also provides no evidence of animation consistency.

## 2. Recommended final plan

### Objective and scope

Build a reproducible categorical sprite-prior experiment using approved library data. It must support training, checkpoint resume, unconditional generation, reconstruction, and inpainting through the same core code locally and on Modal.

Start with complete, outlined enemy sprites fitting a native 16×16 canvas. Use four output classes: `TRANSPARENT`, `INK`, `TONE_1`, and `TONE_2`. `INK` denotes a palette role that may include outlines and interior details.

The required v0 ends with a reproducible experiment and an evidence-based go/no-go decision. Native 24×24, 32×32, and future conditioning are gated extensions.

### Phase 1 — Define and freeze the experiment

1. **Declare the target distribution and success criteria.**

   Choose the initial source/style cohort, category allowlist, completeness requirements, orientation policy, and palette eligibility. Default to one coherent enemy-sprite cohort with binary transparency and faithful conversion to the four classes. Keep source types distinguishable throughout evaluation.

   Define a short human rubric for silhouette, coherence, readability, and usefulness as a starting asset. Record quality thresholds and the total paid-run envelope before training.

   **Deliverable:** Experiment specification, initial config, rating rubric, and run budget.

   **Exit criterion:** Dataset scope, permitted transformations, evaluation protocol, and spending limits are explicit. Remaining choices use documented provisional defaults.

2. **Implement a library-aware, read-only importer.**

   Default the source root to `~/datasets/sprites`. Enumerate frame files through `sprite.json` and `animation.json`; exclude `_sources`, `_tools`, sheets, layers, and unselected categories.

   Record franchise, game, category, folder identity, canonical identity when available, state, frame, dimensions, alpha statistics, palette, origin/offset, source provenance, and uncertainty notes. Resolve provenance from frame/animation records before broader sprite/game defaults. Preserve available credit and source fields.

   Report missing files, unreferenced images, schema differences, and uncertain grouping. Generic PNG import may remain an explicit secondary mode.

   **Deliverable:** Candidate manifest, inventory report, and representative contact sheets.

   **Exit criterion:** Every candidate has an identifiable source frame and identity group; exclusions and unresolved metadata are visible.

3. **Define faithful normalization and palette mapping.**

   Preserve native pixels. Crop transparent margins and integer-pad fitting sprites onto 16×16 canvases using one documented placement policy. Record the crop, placement, and original anchor. Reject oversized content from this cohort.

   Reject partial alpha initially. Map approved opaque colors to verified ink and ordered tone roles, using palette metadata or an approved mapping consistently across related frames. Leave absent roles unused. Keep transparent black distinct from opaque black.

   Show source and normalized categorical images together during review. Treat nearest-neighbor shrinking and color merging as separately named transformations requiring inclusion in the approved preprocessing policy.

   **Deliverable:** Versioned normalization/encoding policy, mappings, transformation records, and comparison sheets.

   **Exit criterion:** Accepted examples preserve geometry and intended color distinctions; every conversion is reproducible.

4. **Create an immutable approved snapshot and leakage-resistant splits.**

   Consume the human-review workflow’s approval records. Bind inclusion to source content hashes and the preprocessing-policy version; changed content becomes a new candidate.

   Establish canonical identity/design-family groups across animations, games, sources, mirrors, and palette variants. Detect exact and near duplicates in decoded images and final categorical grids. Review ambiguous matches rather than merging identities merely because they share a simple component.

   Split whole groups deterministically, targeting 80/10/10 while preserving useful cohort coverage. Keep test groups closed during model selection. Use identity-balanced sampling, then animation/frame sampling; ignore capture durations as training weights.

   **Deliverable:** Hashed approved snapshot, frozen split manifest, duplicate report, and sampling policy.

   **Exit criterion:** No known identity-family or normalized-duplicate leakage. As a provisional planning floor, seek at least 100 canonical families overall and 20 each in validation and test. Smaller datasets support a feasibility pilot with explicitly limited generalization claims.

### Phase 2 — Implement the minimal model and sampler

5. **Build the shared PyTorch core and configuration layer.**

   Keep the proposed starting model: six pre-LN blocks, width 192, six heads, MLP ratio 4, dropout 0.1, bidirectional SDPA, and axial 2D RoPE. Use exactly `H×W` pixel tokens plus a resolution prefix; output four logits per pixel. Reserve `MASK` and `PAD` for inputs.

   Define zero-based X/Y coordinates, rotary channel allocation, numerical precision, and unrotated metadata prefixes. Validate head dimensions. Keep category conditioning disabled for the first cohort.

   Add optional training and Modal dependencies, lazy imports, validated configs, and the `ml pixel` CLI namespace.

   **Deliverable:** Importable model/config modules, documented tensor interface, and dependency setup.

   **Exit criterion:** CPU forward/backward succeeds; four-class output shapes are correct; unrelated CLI commands work without Torch or Modal installed.

6. **Implement a fully specified masked objective.**

   Use a configurable initial mixture:

   - 20% fully masked examples.
   - Otherwise, draw a cosine-distributed mask fraction spanning nearly zero through nearly one.
   - Use random-cell masks for most examples and contiguous masks for a smaller inpainting component.

   Ensure at least one valid masked cell. Average loss over masked cells within each example, then across examples. Exclude PAD from attention and loss; preserve transparency as a valid predicted class. The [reference mask schedule](https://github.com/google-research/maskgit/blob/main/maskgit/libml/mask_schedule.py) provides the cosine definition.

   Log actual mask ratios, loss by ratio, per-class metrics, foreground precision/recall, and class balance. Begin without class weighting so the baseline objective remains interpretable.

   **Deliverable:** Mask generators, loss implementation, diagnostics, and focused correctness tests.

   **Exit criterion:** Tests prove mask counts, loss exclusions, target isolation, and finite gradients. Full-mask and contiguous-mask examples occur at the configured frequencies.

7. **Implement deterministic, complete iterative decoding.**

   Start with 16 iterations, categorical sampling temperature 1, and configurable confidence noise. For `M₀` initially editable cells, target this cumulative committed count at iteration `t`:

   \[
   C_t=M_0-\left\lfloor M_0\cos\left(\frac{\pi t}{2T}\right)\right\rfloor
   \]

   Rank candidates using the probability of each **sampled class**, with independently configured seeded ranking noise. Define ties and rounding. Preserve committed and user-locked cells; force completion at the final iteration.

   Expose generation, reconstruction, and inpainting. Use an unambiguous edit mask, such as white = editable. Preserve locked categorical IDs exactly; reject incompatible input palettes rather than silently recoloring locked pixels.

   **Deliverable:** Sampler, categorical JSON, native/indexed PNGs, enlarged PNGs, contact sheets, and optional traces.

   **Exit criterion:** Outputs contain only valid classes, terminate for small editable regions, reproduce within the declared backend, and preserve locks exactly.

### Phase 3 — Prove correctness and run the local pilot

8. **Establish baselines, overfitting checks, and reliable resume.**

   Fit training-only global and position-dependent class-frequency baselines with smoothing. Add visible-pixel nearest-neighbor retrieval as a reconstruction baseline and training-image resampling as a memorization control.

   Overfit one sprite with full masking. Separately overfit 2–8 sprites with partial masks. Verify that fully masked batches contain no unintended identity information.

   Save atomic checkpoint bundles containing model weights, optimizer/scheduler state, precision scaler where applicable, RNG states, sampler position, step, config, encoding policy, and dataset/split hashes. Record dependency versions and a source hash as well as git state.

   **Deliverable:** Baseline report, overfitting artifacts, and interrupted-versus-continuous training comparison.

   **Exit criterion:** The single-image and partial-mask checks succeed; restart restores training state on the same backend within declared tolerance. Dataset or encoding mismatches cannot silently resume.

9. **Profile and train a bounded local pilot.**

   Implement `doctor` to report the known GPU, available VRAM, CUDA/BF16 support, and attention backend. Benchmark warmed-up forward/backward **and optimizer** steps at 16×16, including reserved memory. Choose batch size from measurements and accumulate toward an effective batch of roughly 32.

   Run an initial 1,000–2,000 optimizer updates. Use AdamW, learning rate `3e-4`, weight decay `0.05`, gradient clipping at 1, and warmup/cosine scheduling scaled to the pilot length. Log effective dataset passes, throughput, memory, validation results, and fixed-seed sample sheets.

   **Deliverable:** Hardware profile, pilot checkpoint, learning curves, sample sheets, and validation comparison.

   **Exit criterion:** Stable training fits the measured 8 GB budget with headroom, beats the position-dependent baseline on held-out partial masks, and produces some coherent unconditional samples. A provisional numerical gate is ≥5% lower masked NLL at 50% and 75% masking; report foreground performance and identity-level uncertainty alongside it.

### Phase 4 — Validate Modal execution and bounded paid training

10. **Add a thin Modal adapter with durable artifacts.**

    Import the same core training and sampling functions. Pin dependencies and code; upload each approved dataset snapshot once under its content hash. Upload the selected frame tensors and manifests rather than raw capture inputs.

    Use durable storage for checkpoints, configs, logs, and reports. Flush and commit complete checkpoint bundles before updating the “latest” pointer. Verify downloading and local loading. Modal’s [Volume documentation](https://modal.com/docs/guide/volumes) describes persistence and commit behavior.

    **Deliverable:** Explicit dataset upload, remote training/resume/sampling, and artifact download commands.

    **Exit criterion:** A deliberately interrupted short remote run resumes from durable state, and its checkpoint samples locally.

11. **Run capped H100 confirmation experiments.**

    Begin with a short throughput/cost pilot on one H100. Compare workloads at equal effective batch size and training progress. Benchmark another GPU only when it can materially inform the spending decision.

    Declare maximum steps, wall time, retries, concurrency, and checkpoint cadence. Initially use one concurrent run and avoid automatic retry loops. Extend training in stages when validation and sample quality justify it; run the selected configuration with three training seeds.

    Current listed H100 GPU cost is approximately **$3.95/hour**, so 30 minutes is about **$1.97 GPU-only**, before other resources. Refresh rates at execution time. [Modal pricing](https://modal.com/pricing)

    **Deliverable:** Comparable seed runs, measured throughput, projected completion cost, and actual resource usage.

    **Exit criterion:** The selected configuration has reproducible behavior across seeds and stays within the declared run envelope. Guards trigger checkpointing and shutdown; billing estimates are labeled as estimates.

### Phase 5 — Evaluate and conclude

12. **Run the frozen quantitative and human evaluation.**

    Select model and sampler settings using validation only. Evaluate the frozen configuration on held-out groups with fixed random masks at 25%, 50%, 75%, 95%, and 100%, plus contiguous inpainting masks.

    Report masked conditional NLL, per-class accuracy, foreground precision/recall, and reconstruction foreground IoU. Aggregate by identity and resolution, with uncertainty over identities.

    Generate 64 unselected unconditional samples per training seed. Blindly mix them with approved authentic controls using identical rendering. Rate silhouette, coherence, readability, and usefulness. Report blank/noisy samples, diversity, exact copies, and calibrated near copies; inspect nearest training matches using foreground-sensitive distances.

    **Deliverable:** Machine-readable evaluation, blind sheets, human ratings, and nearest-neighbor panels.

    **Exit criterion:** All declared metrics and failures are reported. A provisional artistic gate is that at least half of unselected samples are readable and useful starting points without structural redraw, with accepted copying/diversity limits chosen beforehand.

13. **Deliver the reproducible v0 and a go/no-go report.**

    Provide working `doctor`, `prepare`, `train --resume`, `sample`, `reconstruct`, `inpaint`, and `evaluate` commands, plus Modal upload/train/resume/sample/download workflows.

    Document setup, dataset acquisition, configs, hashes, checkpoints, samples, human results, memory, elapsed time, costs, and failure cases. Include CPU end-to-end checks and focused tests for approval filtering, splitting, encoding, masking, locks, and resume. CPU checks require no cloud credentials.

    Distinguish implementation completion from artistic success. Diagnose a no-go in terms of data coverage, representation damage, optimization, sampling, or memorization, and recommend one next experiment.

    **Deliverable:** Runnable pipeline, documented artifacts, and `REPORT.md`.

    **Exit criterion:** A fresh environment with the approved snapshot can reproduce preparation, resume training, and generate evaluation artifacts. The report clearly states whether the model met the declared quality gate.

### Phase 6 — Gated extensions

14. **Add native 24×24 and 32×32 cohorts after the 16×16 result.**

    Preserve native pixels and frozen identity splits across sizes. Bucket batches by resolution and define sampling weights explicitly. Evaluate each size independently before mixing.

    Treat resolution metadata as conditioning on supported training sizes; require an explicit error for unsupported sizes. Benchmark 64×64 separately: its pixel-attention pair count is 256 times that of 16×16, although actual runtime and memory depend on the implementation.

    **Deliverable:** Per-size approved manifests, profiles, configs, and evaluation reports.

    **Exit criterion:** Each advertised size independently meets its resource and quality gates. Optional 64×64 support remains gated by measured feasibility.

15. **Prepare a small future-conditioning dataset pilot.**

    Preserve anchors, poses, source palettes, and identity groups now. Later curate a small set of aligned concept/region-map/sprite examples and reference-style sets that avoid identity leakage.

    Label whether examples are authentic pairs, synthetic upscales, or derived annotations. Nearest-neighbor enlargement alone adds no independent high-resolution design information. Assess alignment and label usefulness before implementing conditioning encoders.

    **Deliverable:** Paired-data schema and a small reviewed example set.

    **Exit criterion:** The intended mapping and its evaluation are credible on inspected pairs. Architecture work follows that evidence.

### Open decisions for Johnny

| Decision | Recommended default |
|---|---|
| Initial artistic scope | Complete, outlined enemy sprites from one coherent source/style cohort. |
| Four-class representation | Faithful subset first; richer palettes and destructive reductions become separate experiments. |
| Source breadth | Preserve original, Maker, demake, and other provenance distinctions; expand through labeled cohorts. |
| Placement | Deterministic native crop/pad for the single-frame prior; retain original anchors and offsets. |
| Approval integration | Reuse the existing review workflow through a content-hashed approval manifest. |
| Insufficient independent data | Run a labeled feasibility pilot and collect more families before making generalization claims. |
| Paid-run envelope | Start with a short H100 pilot; provisionally allow $10 total for confirmation runs, subject to measured projections. |
| Artistic success | Predeclare the useful-sample fraction and acceptable exact/near-copy rates; suggested starting useful-sample threshold is 50%. |
| First extension after success | Native 24×24 coverage, followed by 32×32; conditioning follows a reviewed paired-data pilot. |
## Johnny's decision on palette (2026-10-24)
- Stay 4-color (NES, transparent + 3) for now. Frames with more colors are mostly two overlaid sprites with different palettes; reducing to a 4-color (or reasonable monochrome) palette is part of the human-reviewed cleaning step, not a reason to widen the model.
- Pinned for later: extend to 4–16 colors (and other resolutions) by adding a color-depth/resolution axis to the token positional encoding. Only worth it with a much larger dataset (SNES and beyond), which may be qualitatively different from the iconic NES target. Johnny is still thinking about it.
