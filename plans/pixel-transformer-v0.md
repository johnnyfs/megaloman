# Megaloman: masked-pixel transformer v0

## Objective
Build a configurable PyTorch training and inference pipeline for **categorical, low-resolution pixel art**. It must run locally (RTX 4070-class GPU, actual VRAM unknown) or on **Modal**, using identical model/training code. It should learn a prior from authentic sprite images and generate samples from a checkpoint. **Do not attempt concept-to-sprite translation in v0**: that requires paired data and conditioning not yet validated.

## Decisions
- One token per pixel; output classes `BACKGROUND`, `OUTLINE`, `DARK`, `LIGHT`. Input-only `[MASK]` and `[PAD]` tokens.
- Bidirectional masked-token transformer, iterative parallel generation (MaskGIT-like), **2D axial RoPE** for pixel positions.
- Explicit resolution metadata token; optional object-category token if labeled data exists.
- Train initially at **16×16**; then 24×24 and 32×32. Support 64×64 in interfaces but treat its memory cost as an experimental gate. Exclude 128×128.
- No diffusion, autoencoder, learned style encoder, region-map encoder, or preference model in v0. Design extension points, not speculative implementations.
- Keep existing Megaloman deterministic pipeline intact. Inspect repository conventions and reuse `uv`, CLI, workspace/manifests where appropriate.

## Deliverables
A coding agent should implement and document these commands (adapt naming to existing CLI):

```bash
uv run ml pixel doctor
uv run ml pixel prepare -c configs/pixel_model/sprite16.yaml
uv run ml pixel train -c configs/pixel_model/sprite16.yaml
uv run ml pixel train -c ... --resume path/to/checkpoint
uv run ml pixel sample --checkpoint path --size 16 --count 32 --seed 42
uv run ml pixel reconstruct --checkpoint path --image input.png --mask-ratio .75
uv run ml pixel inpaint --checkpoint path --image input.png --mask-mask mask.png
uv run ml pixel evaluate --checkpoint path
uv run ml pixel modal train -c configs/pixel_model/sprite16.yaml
uv run ml pixel modal sample --checkpoint <remote-path> --size 16
```

Done means a fresh checkout can prepare a configurable dataset, train and resume, run local or Modal inference, and produce PNGs/contact sheets plus an evaluation report. CPU tests must not require Modal or GPU credentials.

## Files

```text
src/megaloman/pixel_model/
  config.py data.py palette.py rope2d.py model.py masking.py
  train.py sample.py evaluate.py render.py cli.py modal_app.py
configs/pixel_model/{smoke,sprite16,mixed}.yaml
tests/pixel_model/
workspace/pixel_model/{datasets,runs}/
```

## Dataset contract
Accept arbitrary user-configured PNG roots/globs and optional CSV/JSONL metadata (`path`, `group_id`, `object_label`, `style_label`, `source`, `license`). No metadata must work. Produce manifest with SHA256, original size, palette mapping, preprocessing actions, group ID, split, warnings. Deduplicate and split by *character/animation group* when possible to avoid leakage from related frames. Without metadata, warn that leakage may remain.

```yaml
dataset:
  roots: [data/sprites]
  glob: '**/*.png'
  metadata: null
  resolutions: [16]
  normalization: reject    # reject | crop_pad | nearest_resize
  alpha_threshold: 128
  split: {train: 0.90, val: 0.05, test: 0.05}
  seed: 1729
```

Default to rejecting nonconforming dimensions. Any resizing must be explicit, use nearest-neighbor for categorical data, and be recorded. `crop_pad` centers opaque bounds on a transparent canvas with configurable margins. Do not blur or anti-alias categorical labels. Horizontal flip optional; no arbitrary rotation or fractional translation.

**Palette:** input IDs `0=background/transparent`, `1=outline`, `2=dark`, `3=light`, `4=MASK`, `5=PAD`; prediction head has **four** logits only. Deterministically reduce each sprite's opaque colors to three luminance-ordered roles, storing original palette and mapping. Flag destructive reductions. This restricted representation intentionally sacrifices hue; later compare richer palettes rather than silently expanding the vocabulary. Inference maps roles to a configurable display palette.

Report counts by size, colors, alpha fraction, duplicates, split, and rejected images; render a normalized dataset contact sheet. Stop if too few examples exist for valid held-out splits.

## Model
For a sprite of `H×W`, create **H×W pixel tokens**, not a padded 64×64 array. Bucket training batches by resolution. If batching needs padding, exclude PAD from attention and loss.

- Pixel embedding: `E(class_or_MASK) + E(PIXEL_TYPE)`.
- Prefix token: `[RESOLUTION]` with type embedding plus small MLP encoding `(log H, log W)`.
- Optional prefix: object category ID with learned embedding, using an explicit null/unknown ID.
- Apply **axial 2D RoPE** to each pixel token's attention Q/K: split rotary channels into X/Y groups; rotate scalar pairs by `x*omega_i` and `y*omega_i`. Do not rotate V. Use a documented neutral convention for metadata-token positions.
- Full **bidirectional** self-attention, no causal mask; all visible pixels and metadata are context for masked pixels.

Start configuration:

```yaml
model:
  d_model: 192
  layers: 6
  heads: 6
  mlp_ratio: 4
  dropout: 0.1
  attention: sdpa
  rope: axial_2d
  num_colors: 4
  use_object_label: false
```

Pre-LN blocks, residual attention/MLP, GELU or SwiGLU, final LN, categorical head. Use PyTorch `scaled_dot_product_attention` when possible; never materialize full attention matrices for routine logging. Expected interface:

```python
logits = model(pixels, height=H, width=W, object_ids=None, valid_mask=None)
# pixels: LongTensor[B,H,W]; logits: FloatTensor[B,H,W,4]
```

## Training
For every example, independently sample a mask fraction broadly in `[0.1,1.0]`, with some explicit probability at **1.0**. Replace those positions with MASK; ensure at least one masked cell. Cross-entropy is calculated **only on masked valid positions**:

`loss = mean(-log p(true_class_i | visible_pixels, resolution, metadata))` for masked `i`.

100%-masked examples matter because unconditional inference starts fully masked. They do not guarantee artistic output.

```yaml
train:
  seed: 1729
  batch_size: 32
  grad_accum_steps: 1
  max_steps: 20000
  lr: 0.0003
  min_lr: 0.00001
  warmup_steps: 500
  optimizer: adamw
  weight_decay: 0.05
  grad_clip_norm: 1.0
  precision: auto
  eval_every: 500
  sample_every: 500
  checkpoint_every: 1000
```

Use warmup + cosine schedule, bf16 if supported, else fp16/fp32. Log loss, validation NLL, masked accuracy, foreground accuracy, tokens/sec, time/step, peak GPU memory, generated sample sheets. Save atomic checkpoints containing model, optimizer, scheduler, RNG state, step, config and dataset hashes; resume must work. Use safetensors for model weights; never load untrusted optimizer pickle files. Stop on NaN/Inf. Record git SHA and package versions.

**Training gates:** overfit 2–8 examples on CPU/GPU first; verify gradients and decreasing loss; beat a marginal class-frequency baseline on held-out partial-mask NLL before any large run.

## Inference algorithm
Modes: (1) generate from all MASK; (2) reconstruct by masking a fraction of an existing sprite; (3) inpaint while preserving user-locked cells.

For `T` steps:
1. Predict logits for **all currently masked positions in parallel**.
2. Sample or argmax class per masked cell (configurable temperature/top-k/seed).
3. Rank candidate predictions by confidence, optionally with seeded Gumbel perturbation.
4. Commit the top confident cells until the **scheduled cumulative filled count** is reached; leave remaining cells MASK.
5. Repeat until all are filled. Ensure final step commits all remaining positions.

Default to a cosine commit schedule. **Do not revise committed pixels in v0**; remasking is a separate later experiment. A fixed probability threshold is insufficient because it may stall. Preserve locked pixels exactly. Export indexed JSON, PNG, nearest-neighbor enlarged PNG, contact sheet, and optional per-step trace. Log class balance to diagnose early background domination.

Without labeled object conditioning, `generate` cannot obey a prompt such as “squid”; it samples from the learned distribution. Never describe it as text-to-sprite or concept-to-sprite.

## Evaluation
Use fixed held-out examples and masks at **25%, 50%, 75%, 95%, 100%**. Report masked NLL and accuracy by resolution and mask ratio, including foreground-only metrics. At 100% mask, per-pixel accuracy is *not* a measure of sample quality.

Generate at least 64 unconditional samples per tested size; create side-by-side blind sheets with held-out authentic sprites. Report exact/near-duplicate rate against training images and diversity among samples. Human ratings of silhouette, coherence, readability and stylistic quality remain essential. Do not trust an uncalibrated LLM aesthetic judge as the optimization target.

## Local and Modal execution
**Local:** implement `doctor` to detect NVIDIA GPU model, actual free/total VRAM, CUDA, bf16, and benchmark one forward/backward step at 16, 32 and 64. Use local GPU for smoke runs. If OOM, suggest lower batch size, gradient accumulation, mixed precision or Modal; never silently reduce sprite size.

**Modal:** thin adapter importing the *same* `train()` and `sample()` code. Configurable GPU type, image, time limit, dataset location and durable checkpoint storage. Use a **Modal Volume or object storage**; never rely on ephemeral filesystem. Provide explicit dataset sync/upload and report download; do not reupload data every run. Start with **L4** where available, try **A10G/A100** only when measured throughput/memory warrants. Verify current GPU offerings and rates at execution time. Provide runtime/step budget guard and estimated cost, but don't claim this guarantees a billing cap. Store secrets outside source control. Support checkpoint resume after remote interruption.

**Performance:** full attention pair count scales as `(H*W)^2`. 64×64 has 256× as many attention pairs as 16×16. Use SDPA/FlashAttention kernels, mixed precision, smaller batches and optional activation checkpointing. Benchmark before promising 64×64 training. If impractical, explicitly mark 64 as unsupported for the chosen hardware/config, not as a failed v0.

## Tests
- Palette mapping deterministic; alpha handling; rejection of bad sizes; split grouping prevents known leakage.
- Mask fractions, PAD exclusion, at least one mask, loss only on masked cells.
- 2D RoPE shapes, X/Y separation and relative-displacement invariance.
- Forward shapes, finite gradients, CPU training step at 16 and 24.
- Generation completes all pixels, respects seed and locked cells; checkpoint save/load and resume.
- End-to-end test with synthetic tiny sprites, no GPU or Modal connection.
- Modal adapter imports and config-validates without running remotely.

## Implementation order
1. **Dataset**: ingestion, normalization, manifest, contact sheet. Inspect real samples before training.
2. **Model**: embedding, RoPE, attention, masked objective; CPU tests; overfit tiny dataset.
3. **Local v0**: 16×16 training, evaluation, checkpoints, unconditional sampling.
4. **Modal**: durable storage, remote training/resume/inference with same core code.
5. **Variable sizes**: 24 and 32, then optional 64 after memory benchmarks.
6. **Report**: `REPORT.md` with configuration, dataset stats, plots/samples, elapsed time, cost, failures, and a go/no-go recommendation.

## Future conditioning — design extension points, don't implement yet
Once v0 works, add (a) global object-label and per-part semantic tokens (`eye`, `head`, `tentacle`, etc.), (b) spatial **region-coverage hints**: for each output cell, calculate coverage fractions of high-resolution semantic regions and form `h_xy = sum_r coverage_xy,r * E(part_label_r)` plus `REGION_HINT` type and 2D position, (c) one global token per region instance to preserve tiny features, and (d) style-reference sprites encoded as pixel tokens with a `STYLE_REFERENCE` type embedding and reference-ID/2D positions. Use cross-attention or carefully masked joint attention. The goal is a **pose-matched high-resolution concept + semantic region map + reference style sprites → categorical low-resolution sprite**. It requires paired training examples; never claim that an unconditional v0 has learned this mapping.

## Coding-agent working agreement
Implement one milestone at a time, run tests, inspect artifacts, and record decisions. Avoid building an elaborate UI, agent director, preference loop or new model infrastructure before producing real v0 samples. Don't silently modify blessed rules or existing pipelines. If dataset size/quality is insufficient, report that explicitly rather than claiming the architecture failed. Optimize for a falsifiable experiment and a maintainable training/inference system, not a demonstration image.
