# Pixel transformer scaffold

This implements the shared categorical sprite-prior machinery from
[v0](../plans/pixel-transformer-v0.md) and the
[review's phased plan](../plans/pixel-transformer-v1-review.md). It is ready for
an approved dataset and a bounded experiment. The included synthetic blobs prove
execution only; no artistic or generalization result is claimed. It is an
unconditional sprite prior, with no text/concept conditioning.

The phase boundaries for this delivery are:

| Review phase | Scaffold implementation |
| --- | --- |
| 1: freeze data/experiment | Read-only approved snapshot contract and explicit synthetic fixture. Import, cleaning, approval, identity resolution and splitting happen separately. |
| 2: model/sampler | PyTorch core, per-example masked objective, deterministic complete decoding. |
| 3: local correctness/pilot | CPU smoke, exact resume test, overfit diagnostic hook, hardware profiler. A real 1,000–2,000-update pilot remains future work. |
| 4: Modal | Lazy adapter, explicit upload/download, durable bundles and bounded execution settings. Remote execution is untested and was not launched. |
| 5: evaluation | Quantitative controls, fixed masks, duplicate/diversity panels, blind sheets and rating CSV. Human evaluation remains pending. |
| 6: extensions | Native size buckets are available for interface exercises. Approved 24/32 cohorts and experimental 64 resource measurements remain gated. |

## Local setup and smoke

Python 3.12+, `uv`, and no cloud credentials are required. Torch/safetensors are
optional extras; existing `ml` commands and help do not import them. The lockfile
defaults to CPU Torch on Linux. Use a writable `UV_CACHE_DIR` if your normal cache
is mounted read-only (this run used `/tmp/megaloman-uv-cache`).

```sh
uv sync --extra pixel
uv run --extra pixel ml pixel doctor -c configs/pixel_model/smoke.yaml --benchmark
uv run --extra pixel ml pixel prepare -c configs/pixel_model/smoke.yaml --fixture
uv run --extra pixel ml pixel train -c configs/pixel_model/smoke.yaml --stop-after 4
uv run --extra pixel ml pixel train -c configs/pixel_model/smoke.yaml \
  --resume workspace/pixel_model/runs/smoke/checkpoints/latest.json
uv run --extra pixel ml pixel sample \
  --checkpoint workspace/pixel_model/runs/smoke/checkpoints/latest.json \
  --size 16 --count 8 --seed 42 --trace \
  --output workspace/pixel_model/runs/smoke/generated
uv run --extra pixel ml pixel evaluate \
  --checkpoint workspace/pixel_model/runs/smoke/checkpoints/latest.json \
  --count 64 --output workspace/pixel_model/runs/smoke/evaluation
uv run --extra pixel python -m unittest discover -s tests -v
```

`prepare --fixture` creates 24 indexed frames only when the manifest is absent;
otherwise it validates the existing snapshot. `prepare` without that flag only
validates and renders a sheet. Nothing reads or modifies the sprite library.
The fixture contains 8 train, 2 validation, and 2 test identities, with two
frames each. These held-out counts are deliberately inadequate for research
claims. Paths in YAML resolve from the working directory; frame paths resolve
from the manifest directory. Run commands at the repository root.

Runs refuse to overwrite an existing training log. To repeat, choose a new
`train.run_dir` in a copied YAML config. `--stop-after` is an absolute optimizer
step and preserves the configured scheduler horizon. Changing `max_steps` on
resume is an error. Zero cadences disable validation/sample/checkpoint intervals;
the final completed step is still checkpointed. A hard interruption resumes the
latest complete bundle; partial gradients are never saved. Wall-time guards act
between optimizer updates. Resume can discard log lines beyond the durable step;
resuming an older checkpoint into a directory with newer checkpoints is rejected.

For a working NVIDIA driver, install CUDA Torch with:

```sh
uv sync --extra pixel --no-sources
# Keep --no-sources when uv syncs, or use --no-sync after that installation.
uv run --no-sync ml pixel doctor -c configs/pixel_model/sprite16.yaml --benchmark
```

`sprite16.yaml` uses device/precision `auto`: BF16 if CUDA supports it, otherwise
FP16 on CUDA or FP32 on CPU. Explicit unavailable CUDA is an error. The smoke YAML
deliberately selects CPU. Profile with `doctor --benchmark --sizes 16 32 64` only
when you intend to measure those sizes; 64 is experimental. The profiler warms up
and measures forward, backward, **and AdamW**, reports actual attention operator
names, and records allocated/reserved CUDA memory. It does not infer FlashAttention
from an enabled-backend flag. No automatic resolution reduction or cloud fallback
occurs.

## Approved dataset contract

External preparation supplies one JSON snapshot, an external `approval_id`, frozen
identity/design-family splits, and native P-mode PNGs. This reader does not decide
approval or attempt to repair data. The upstream workflow must bind approval to
the exact frame content and encoding policy, assign consistent roles across
related frames, and handle near duplicates and provenance.

```json
{
  "schema_version": 1,
  "approval_id": "external-review-snapshot-id",
  "encoding": {
    "version": "megaloman-indexed-v1",
    "num_colors": 4,
    "roles": ["TRANSPARENT", "INK", "TONE_1", "TONE_2"],
    "transparent_index": 0,
    "geometry": "native-approved",
    "palette_mapping": "approved-role-ids"
  },
  "records": [
    {
      "id": "enemy-a-idle-0",
      "path": "frames/enemy-a-idle-0.png",
      "sha256": "<sha256 of exact PNG bytes>",
      "identity": "canonical-design-family-a",
      "animation": "idle",
      "split": "train",
      "size": 16
    }
  ]
}
```

Supply at least one frame in **each** train/val/test split for every configured
size; the abbreviated example alone fails validation. Index 0 must be fully
transparent; indices 1–3 must be fully opaque. Index 1 is **INK**, including
interior details. Missing roles may be unused. Transparent black and opaque black
remain different classes. Palette RGB values are display colors, not automatically
ranked or quantized by the model. Approved palette conversion is an upstream job.
The reader rejects RGBA/RGB images, partial alpha, extra IDs, wrong native sizes,
content-hash mismatches, escaping paths, repeated IDs, and declared identity or
exact categorical-duplicate leakage across splits. It never crops, resizes,
merges colors, splits identities, or changes approval state.

Extra per-record fields (source/credit/license, anchors, offsets, mappings,
uncertainty, etc.) are preserved in the snapshot fingerprint. The fingerprint is
SHA256 of canonical JSON; PNG byte hashes bind frame content. Separate split and
encoding hashes are checkpointed. Relocating the identical directory is allowed;
changing even approval metadata produces a different snapshot. PNG hashes alone
do not prove approval: trust in the external approval issuer remains necessary.

Sampling chooses a size uniformly, then an identity uniformly, an animation
uniformly, and a frame uniformly, with replacement. Capture durations do not
affect weights. Native sizes are bucketed rather than padded to a maximum canvas.
`mixed.yaml` plus `prepare --fixture` exercises native 16/24/32 interfaces; it is
not evidence of a trained multi-size model.

## Model, masking, and checkpoints

`PixelTransformer(ModelConfig(...), resolutions=(16,))` accepts
`LongTensor[B,H,W]` and returns `FloatTensor[B,H,W,4]`. Optional `height`/`width`
metadata must agree with the input. `valid_mask` excludes padding from attention
keys and the loss, while transparency remains a valid prediction. Object IDs are
explicitly rejected in v0. Unsupported sampling/forward sizes raise an error.

The default is six pre-LN blocks, width 192, six heads, GELU MLP ratio 4, dropout
0.1, and bidirectional
[PyTorch SDPA](https://docs.pytorch.org/docs/2.10/generated/torch.nn.functional.scaled_dot_product_attention.html).
Each image has exactly H×W pixels plus one resolution metadata token. Pixel
embeddings include a type vector; the prefix has a type vector and MLP of
`(log H, log W)`. Zero-based X/Y RoPE divides each attention head equally between
axes, rotates adjacent scalar pairs using FP32 trigonometry, leaves the prefix
unrotated, and never rotates V. Head dimension must be divisible by four.

The four-class encoding is versioned separately from positions. The
`pixel_coordinates(..., num_colors=4)` position-provider boundary explicitly
receives palette depth, and resolution conditioning is isolated from token IDs.
A later depth axis can extend the position provider, rotary channel allocation,
and metadata under a new encoding/checkpoint version. No depth axis or larger
vocabulary is active now; changing `num_colors` is rejected.

The mask mixture explicitly chooses 20% full masks. Other fractions use
`cos(pi*u/2)`, with 20% of non-full examples assigned a contiguous-hole mask.
Ceiling to integer cells may make additional near-full masks become full.
Contiguous masks expand from a seeded valid center using Chebyshev distance and
row-major ties. Every example masks at least one valid cell. Conditional
cross-entropy averages masked valid cells **within each example**, then examples;
visible/PAD target values contribute no loss. No identity or frame metadata enters
the model. Training logs include mask kinds/ratios, loss bands, per-class/foreground
metrics, predicted class balance, throughput, effective passes, and CUDA memory.

Training uses AdamW, warmup/cosine LR, accumulation, gradient clipping, finite-loss
and finite-gradient checks, and AMP/scaler where applicable. Bundles contain model
and optimizer tensors in **safetensors**, structured state in JSON, scheduler,
scaler, Python/NumPy/Torch/CUDA RNG, masking RNG, balanced-stream position, config,
encoding, dataset/split hashes, package versions, source hash, and Git state. There
is no optimizer pickle loader. A complete bundle is renamed atomically before the
latest pointer changes. Resume requires the same dataset, encoding, optimizer and
scheduler configuration, backend, precision and thread count. CPU tests compare
interrupted and continuous weights with zero tolerance. Different backends are
supported for weight-only inference, without cross-backend seed guarantees.

## Sampling, reconstruction, and inpainting

The default decoder uses 16 iterations, categorical temperature 1, optional top-k,
and a separate ranking-noise temperature (default 0). At iteration `t=1..T`, it
targets `M0 - floor(M0*cos(pi*t/(2*T)))` cumulative commits, where M0 is the
**initially editable** count. Confidence is log probability of the sampled class.
Optional Gumbel noise has a separate seeded generator. Ties use row-major order
(class-ID order for top-k). Committed cells are never revised; the final iteration
fills everything. Temperature 0 selects argmax. Identical seeds reproduce on the
same backend; multi-sample CLI/evaluation calls use seed+i and sequential inference
to bound memory.

```sh
uv run --extra pixel ml pixel reconstruct \
  --checkpoint workspace/pixel_model/runs/smoke/checkpoints/latest.json \
  --image workspace/pixel_model/datasets/tiny/frames/16-8-0.png \
  --mask-ratio .75 --seed 42 --output workspace/pixel_model/runs/smoke/reconstructed

uv run --extra pixel ml pixel inpaint \
  --checkpoint workspace/pixel_model/runs/smoke/checkpoints/latest.json \
  --image workspace/pixel_model/datasets/tiny/frames/16-8-0.png \
  --mask-mask editable.png --seed 42 --output workspace/pixel_model/runs/smoke/inpainted
```

`editable.png` must be native-size binary grayscale: **white=editable,
black=locked**. Source images must already satisfy the indexed role contract;
incompatible images are rejected. Locks preserve categorical IDs and the source
display palette. Exports include native indexed PNGs, nearest-neighbor enlarged
PNGs, categorical `pixels.json`, a checkerboard contact sheet, and optional JSON
traces (unfinished cells retain MASK=4 in traces).

## Evaluation and gates

`evaluate` defaults to validation, keeping test closed until a deliberate
`--split test`. Fixed masks cover 25/50/75/95/100%, both random cells and contiguous
holes, at every configured size. Metrics include masked **conditional** NLL,
masked/per-class accuracy, foreground NLL/accuracy, foreground precision/recall,
and reconstruction foreground IoU. Reports aggregate frames within identities
then identities equally, with descriptive identity bootstrap intervals.

Training-only baselines include smoothed global and position-dependent class
frequencies, visible-pixel nearest-neighbor reconstruction (its reported NLL is an
explicit .01-smoothed surrogate), and training-image resampling as a copying
control. Generation produces 64 unselected samples per size by default, exact
training-copy rates, an **uncalibrated** near-copy threshold, foreground-sensitive
nearest-neighbor panels, pairwise diversity, unique fraction, class balance and
blank fraction. Noisy-sample rate remains null until rated by humans.

`blind-sheet.png` mixes generated examples and held-out controls using identical
rendering. Controls may repeat in tiny fixtures. Keep `blind-key.json` away from
raters. `human-ratings.csv` uses zero-based row-major sheet indices and empty
silhouette/coherence/readability/usefulness ratings (1–5), a useful-without-redraw
decision, and blank/noisy flag. Human results are not fabricated or automatically
scored. The provisional future gates are ≥5% lower position-baseline NLL at 50%
and 75% random masking and ≥50% useful unselected samples without structural
redraw. Near-copy/diversity limits must be chosen and calibrated before real
evaluation. Full-mask per-pixel accuracy does not measure artistic quality.

An optional diagnostic hook exercises single-image full masking separately from
four-image partial masking:

```sh
uv run --extra pixel ml pixel overfit-check -c configs/pixel_model/smoke.yaml \
  --steps 8 --output workspace/pixel_model/runs/smoke/overfit-diagnostics
```

This short run checks gradients/loss movement; it does not claim memorization.
Real overfit gates, position-baseline improvement, multi-seed training and human
assessment remain experiments to run after receiving approved data.

## Modal entry point (not launched for this delivery)

```sh
uv sync --extra pixel --extra pixel-modal
uv run --extra pixel --extra pixel-modal ml pixel modal train \
  -c configs/pixel_model/sprite16.yaml --run-id pilot --dry-run
```

Dry-run imports neither Torch nor Modal, authenticates nowhere, and prints the
execution envelope. The adapter factory can also be constructed offline. Actual
remote calls are reachable only through explicit commands without `--dry-run`.

Before any later launch, configure `modal.gpu` (review default H100), `image`,
`volume`, `timeout`, `estimated_hourly_usd` from refreshed
[pricing](https://modal.com/pricing), and `max_estimated_usd`. There is deliberately
no stale hourly-rate default. The full-timeout GPU estimate must fit the configured
estimated budget. Timeout must leave at least 60 seconds beyond the training time
guard. These controls are estimates/guards, **not a billing cap**, and exclude
CPU/storage/build charges. Each function has one maximum container, zero retries,
and a short idle window. Use one run at a time and a unique run ID.

The following commands are documented for future use; none was run remotely:

```sh
# Once per approved content hash; uploads only manifest + selected indexed frames.
uv run --extra pixel --extra pixel-modal ml pixel modal upload -c configs/pixel_model/sprite16.yaml
uv run --extra pixel --extra pixel-modal ml pixel modal train \
  -c configs/pixel_model/sprite16.yaml --run-id pilot
uv run --extra pixel --extra pixel-modal ml pixel modal train \
  -c configs/pixel_model/sprite16.yaml --run-id pilot \
  --resume runs/pilot/checkpoints/latest.json
uv run --extra pixel --extra pixel-modal ml pixel modal sample \
  -c configs/pixel_model/sprite16.yaml --run-id pilot \
  --checkpoint runs/pilot/checkpoints/latest.json --size 16 --count 32
uv run --extra pixel --extra pixel-modal ml pixel modal download \
  -c configs/pixel_model/sprite16.yaml --remote-path runs/pilot \
  --output workspace/pixel_model/runs/downloaded-pilot
```

The pinned image copies the same `megaloman` Python source, invokes the same
`train()` / `sample()`, and stores data under `/pixel/datasets/<snapshot-hash>` and
artifacts under `/pixel/runs/<run-id>` on a
[Modal Volume](https://modal.com/docs/guide/volumes). Training never uploads data.
It verifies the uploaded hash, and commits each complete checkpoint bundle
**before** updating/committing the latest pointer. Submitted local provenance is
also persisted. Downloaded weights sample locally with `--device cpu`; evaluation
can relocate an identical snapshot with `--manifest`. Hard remote interruption,
durability, image build, GPU execution and download round-trip still require a
future deliberately bounded remote integration run.

See [the scaffold report](pixel-transformer-report.md) for measured local results
and open experiment decisions.
