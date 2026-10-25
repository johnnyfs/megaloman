"""Small, strict configuration shared by local and remote entry points."""
from dataclasses import asdict, dataclass, field, fields
import json
import math
from pathlib import Path
import types
from typing import get_args, get_origin

import yaml


@dataclass
class DatasetConfig:
    manifest: str = "workspace/pixel_model/datasets/tiny/manifest.json"
    resolutions: list[int] = field(default_factory=lambda: [16])


@dataclass
class ModelConfig:
    d_model: int = 192
    layers: int = 6
    heads: int = 6
    mlp_ratio: int = 4
    dropout: float = 0.1
    num_colors: int = 4


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
    run_dir: str = "workspace/pixel_model/runs/sprite16"


@dataclass
class SampleConfig:
    steps: int = 16
    temperature: float = 1.0
    ranking_temperature: float = 0.0
    top_k: int = 0


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
        if min(m.layers, m.mlp_ratio) < 1 or not 0 <= m.dropout < 1:
            raise ValueError("invalid model depth, MLP ratio, or dropout")
        if not self.dataset.resolutions or any(n not in (16, 24, 32, 64) for n in self.dataset.resolutions):
            raise ValueError("resolutions must be a nonempty subset of 16, 24, 32, 64; 64 is experimental")
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
