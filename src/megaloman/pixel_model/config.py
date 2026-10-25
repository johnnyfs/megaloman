"""Small, strict configuration shared by local and remote entry points."""
from dataclasses import asdict, dataclass, field, fields
import json
import math
from pathlib import Path
import types
from typing import get_args, get_origin

import yaml

# Square canvas sizes: every multiple of 8 from 16 to 64 (56/64 are thin; stage them in last).
ALLOWED_SIZES = tuple(range(16, 65, 8))
LOSSES = ("ce", "fg_bg_balanced")
BUCKET_SAMPLING = ("uniform", "sqrt_count")
SELECT_METRICS = ("val_loss", "val_balanced_nll")

@dataclass
class DatasetConfig:
    manifest: str = "workspace/pixel_model/datasets/tiny/manifest.json"
    resolutions: list[int] = field(default_factory=lambda: [16])
    # Records at sizes outside `resolutions` are skipped instead of rejected (staged runs).
    skip_other_sizes: bool = False
    # ce: plain masked CE. fg_bg_balanced: .5 fg CE + .5 bg CE per example (plain CE still logged).
    loss: str = "ce"
    # uniform: every size bucket equally likely; sqrt_count: weight ~ sqrt(train frames).
    bucket_sampling: str = "uniform"
    # Training-only horizontal shift of up to this many pixels, only when the foreground stays in bounds.
    hshift: int = 0
    hflip: bool = False  # training-only random horizontal mirror (p=.5)


@dataclass
class ModelConfig:
    d_model: int = 192
    layers: int = 6
    heads: int = 6
    mlp_ratio: int = 4
    dropout: float = 0.1  # residual + MLP
    attention_dropout: float | None = None  # None: same as dropout (run-1 behaviour)
    num_colors: int = 4
    conv_stem: int = 0      # residual stride-1 3x3 conv layers on (already masked) pixel embeddings
    conv_decoder: int = 0   # residual stride-1 3x3 conv layers on features before the logits head
    rel_bias: bool = False  # learned per-head 2D relative-offset attention bias, added to RoPE


@dataclass
class MaskConfig:
    full_probability: float = 0.2
    contiguous_probability: float = 0.2  # conditional on not fully masked


@dataclass
class TrainConfig:
    seed: int = 1729
    batch_size: int = 8
    grad_accum_steps: int = 4
    max_steps: int = 2000
    lr: float = 3e-4
    min_lr: float = 1e-5
    warmup_steps: int = 100
    weight_decay: float = 0.05
    grad_clip_norm: float = 1.0
    precision: str = "auto"
    device: str = "auto"
    cpu_threads: int = 2
    eval_every: int = 100
    sample_every: int = 100
    checkpoint_every: int = 100
    max_seconds: float = 1800
    # Early stop: halt after this many evaluations without val-loss improvement (0 = off).
    patience: int = 0
    min_delta: float = 0.0
    # Validation scalar that drives early stopping and the saved best/ weights.
    select_metric: str = "val_loss"
    run_dir: str = "workspace/pixel_model/runs/sprite16"
    ema: float = 0.0  # >0: evaluate, select and save best with EMA weights of this decay


@dataclass
class SampleConfig:
    steps: int = 16
    temperature: float = 1.0
    ranking_temperature: float = 0.0
    # constant, or linear: amplitude decays from ranking_temperature to ~0 over the steps.
    ranking_schedule: str = "constant"
    top_k: int = 0
    # Revisable decoding: after each step's commits, re-mask this fraction of earlier
    # commits with the lowest commit-time confidence, while t <= remask_until * steps.
    remask: float = 0.0
    remask_until: float = 0.75


@dataclass
class ModalConfig:
    gpu: str = "H100"
    image: str = "debian:bookworm-slim"
    volume: str = "megaloman-pixel-transformer"
    timeout: int = 3600
    # Deliberately no stale pricing default. Supply a refreshed estimate before launch.
    estimated_hourly_usd: float | None = None
    max_estimated_usd: float = 2.0


@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    masking: MaskConfig = field(default_factory=MaskConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sample: SampleConfig = field(default_factory=SampleConfig)
    modal: ModalConfig = field(default_factory=ModalConfig)

    def validate(self):
        def matches(value, expected):
            origin, args = get_origin(expected), get_args(expected)
            if origin is types.UnionType:
                return any(matches(value, option) for option in args)
            if origin is list:
                return isinstance(value, list) and all(matches(v, args[0]) for v in value)
            if expected is float:
                return type(value) in (int, float)
            return type(value) is expected
        for section in fields(self):
            value = getattr(self, section.name)
            if not isinstance(value, section.type):
                raise ValueError(f"{section.name} must be {section.type.__name__}")
            for item in fields(value):
                if not matches(getattr(value, item.name), item.type):
                    raise ValueError(f"invalid type for {section.name}.{item.name}")
        m, t, s = self.model, self.train, self.sample
        if m.num_colors != 4:
            raise ValueError("v0 requires transparent + 3 colors (num_colors=4)")
        if m.heads < 1 or m.d_model < 1 or m.d_model % m.heads or (m.d_model // m.heads) % 4:
            raise ValueError("d_model / heads must be a positive multiple of 4 for axial RoPE")
        if min(m.layers, m.mlp_ratio) < 1 or not 0 <= m.dropout < 1 or not (
                m.attention_dropout is None or 0 <= m.attention_dropout < 1):
            raise ValueError("invalid model depth, MLP ratio, or dropout")
        d = self.dataset
        if not d.resolutions or any(n not in ALLOWED_SIZES for n in d.resolutions):
            raise ValueError(f"resolutions must be a nonempty subset of {ALLOWED_SIZES}")
        if d.loss not in LOSSES or d.bucket_sampling not in BUCKET_SAMPLING or not 0 <= d.hshift <= 4:
            raise ValueError(f"dataset.loss in {LOSSES}, bucket_sampling in {BUCKET_SAMPLING}, hshift in 0..4")
        if t.select_metric not in SELECT_METRICS or s.ranking_schedule not in ("constant", "linear"):
            raise ValueError(f"train.select_metric in {SELECT_METRICS}; sample.ranking_schedule constant|linear")
        if len(set(self.dataset.resolutions)) != len(self.dataset.resolutions):
            raise ValueError("duplicate resolutions")
        for p in (self.masking.full_probability, self.masking.contiguous_probability):
            if not 0 <= p <= 1:
                raise ValueError("mask probabilities must be in [0,1]")
        if min(t.batch_size, t.grad_accum_steps, t.max_steps, t.cpu_threads) < 1:
            raise ValueError("batch, accumulation, steps, and CPU threads must be positive")
        if not 0 <= t.warmup_steps < t.max_steps:
            raise ValueError("warmup_steps must be in [0,max_steps)")
        if not 0 <= t.min_lr <= t.lr or t.lr <= 0 or t.weight_decay < 0 or t.grad_clip_norm <= 0:
            raise ValueError("invalid optimizer settings")
        if t.precision not in ("auto", "fp32", "bf16", "fp16") or t.device not in ("auto", "cpu", "cuda"):
            raise ValueError("unsupported device or precision")
        if any(n < 0 for n in (t.eval_every, t.sample_every, t.checkpoint_every)) or t.max_seconds <= 0:
            raise ValueError("cadences must be nonnegative and time budget positive")
        if t.patience < 0 or t.min_delta < 0 or (t.patience and not t.eval_every):
            raise ValueError("patience needs eval_every; patience and min_delta must be nonnegative")
        if s.steps < 1 or s.temperature < 0 or s.ranking_temperature < 0 or not 0 <= s.top_k <= 4:
            raise ValueError("invalid sampler settings")
        if self.modal.timeout < t.max_seconds + 60:
            raise ValueError("Modal timeout must leave >=60 seconds beyond the training time guard")
        if self.modal.max_estimated_usd <= 0:
            raise ValueError("Modal estimated budget must be positive")
        if self.modal.estimated_hourly_usd is not None and self.modal.estimated_hourly_usd <= 0:
            raise ValueError("Modal hourly estimate must be positive when supplied")
        # YAML also accepts .nan and .inf; reject before they reach optimizers/budgets.
        def check(value):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("config numbers must be finite")
            if isinstance(value, dict):
                for v in value.values():
                    check(v)
        check(self.to_dict())
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, raw):
        if not isinstance(raw, dict):
            raise ValueError("config must be a mapping")
        types = {"dataset": DatasetConfig, "model": ModelConfig, "masking": MaskConfig,
                 "train": TrainConfig, "sample": SampleConfig, "modal": ModalConfig}
        unknown = raw.keys() - types.keys()
        if unknown:
            raise ValueError(f"unknown config sections: {sorted(unknown)}")
        if not all(isinstance(value, dict) for value in raw.values()):
            raise ValueError("config sections must be mappings")
        try:
            return cls(**{k: types[k](**v) for k, v in raw.items()}).validate()
        except TypeError as exc:
            raise ValueError(f"invalid config: {exc}") from exc


def load_config(path):
    return Config.from_dict(yaml.safe_load(Path(path).read_text()) or {})


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
