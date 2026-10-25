"""Atomic bundles of JSON and safetensors only: no Python pickle loader."""
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import uuid

import numpy as np
from safetensors.torch import load_file, save_file
import torch

from .config import Config, canonical_json
from .model import PixelTransformer
from .palette import ENCODING


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as f:
        f.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def provenance():
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"
    digest = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(p.name.encode()); digest.update(p.read_bytes())
    packages = {}
    for name in ("torch", "numpy", "pillow", "safetensors", "megaloman", "modal"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    return {"git_sha": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"),
            "source_sha256": digest.hexdigest(), "packages": packages, "python": sys.version}


def rng_state():
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": (state[0], torch.tensor(state[1].astype(np.int64)), *state[2:]),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], n[1].numpy().astype(np.uint32), *n[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def _encode(value, tensors):
    if isinstance(value, torch.Tensor):
        key = f"tensor_{len(tensors)}"
        tensors[key] = value.detach().cpu().contiguous().clone()
        return {"tensor": key}
    if isinstance(value, dict):
        return {"dict": [[_encode(k, tensors), _encode(v, tensors)] for k, v in value.items()]}
    if isinstance(value, (tuple, list)):
        return {"tuple" if isinstance(value, tuple) else "list": [_encode(v, tensors) for v in value]}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported checkpoint state type: {type(value)}")


def _decode(value, tensors):
    if not isinstance(value, dict):
        return value
    if "tensor" in value:
        return tensors[value["tensor"]]
    if "dict" in value:
        return {_decode(k, tensors): _decode(v, tensors) for k, v in value["dict"]}
    if "tuple" in value:
        return tuple(_decode(v, tensors) for v in value["tuple"])
    return [_decode(v, tensors) for v in value["list"]]


def save_checkpoint(run_dir, *, model, optimizer, scheduler, scaler, stream, mask_generator,
                    step, cfg, dataset, runtime, durable_commit=None):
    root = Path(run_dir) / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    final = root / f"step-{step:08d}"
    if final.exists():
        raise ValueError(f"checkpoint already exists: {final}")
    temp = root / f".pending-{uuid.uuid4().hex}"
    temp.mkdir()
    try:
        save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(temp / "model.safetensors"))
        tensors = {}
        tree = _encode({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(), "stream": stream.state_dict(),
                        "mask_generator": mask_generator.get_state(), "rng": rng_state()}, tensors)
        save_file(tensors, str(temp / "state.safetensors"))
        atomic_json(temp / "state.json", tree)
        meta = {"schema_version": 1, "step": step, "config": cfg.to_dict(), "encoding": ENCODING,
                "dataset_sha256": dataset.fingerprint, "split_sha256": dataset.split_hash,
                "encoding_sha256": dataset.encoding_hash, "runtime": runtime, "provenance": provenance()}
        atomic_json(temp / "metadata.json", meta)
        temp.rename(final)
        if durable_commit:
            durable_commit()  # Complete bundle first, then the small latest pointer.
        atomic_json(root / "latest.json", {"checkpoint": final.name, "step": step})
        if durable_commit:
            durable_commit()
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return final


def checkpoint_path(path):
    path = Path(path)
    if path.is_file() and path.name == "latest.json":
        name = json.loads(path.read_text())["checkpoint"]
        if Path(name).name != name:
            raise ValueError("invalid checkpoint pointer")
        path = path.parent / name
    return path


def load_model(path, device="cpu"):
    path = checkpoint_path(path)
    meta = json.loads((path / "metadata.json").read_text())
    if meta.get("schema_version") != 1 or meta.get("encoding") != ENCODING:
        raise ValueError("incompatible checkpoint format or encoding")
    cfg = Config.from_dict(meta["config"])
    model = PixelTransformer(cfg.model, cfg.dataset.resolutions).to(device)
    model.load_state_dict(load_file(str(path / "model.safetensors")))
    return model, cfg, meta


def resume_signature(cfg):
    raw = cfg.to_dict()
    for key in ("run_dir", "max_seconds", "device", "cpu_threads"):
        raw["train"].pop(key)
    raw.pop("modal")
    raw.pop("sample")
    raw["dataset"].pop("manifest")
    return canonical_json(raw)


def restore_training(path, *, cfg, dataset, runtime, model, optimizer, scheduler, scaler, stream, mask_generator):
    path = checkpoint_path(path)
    meta = json.loads((path / "metadata.json").read_text())
    if meta.get("schema_version") != 1 or meta.get("encoding") != ENCODING:
        raise ValueError("incompatible checkpoint encoding")
    if (meta["dataset_sha256"], meta["split_sha256"], meta["encoding_sha256"]) != (
            dataset.fingerprint, dataset.split_hash, dataset.encoding_hash):
        raise ValueError("resume dataset/split/encoding mismatch")
    if resume_signature(Config.from_dict(meta["config"])) != resume_signature(cfg):
        raise ValueError("resume training config mismatch (including fixed scheduler horizon)")
    if meta["runtime"] != runtime:
        raise ValueError("resume backend/precision mismatch; reproducibility requires the same runtime")
    model.load_state_dict(load_file(str(path / "model.safetensors")))
    state = _decode(json.loads((path / "state.json").read_text()), load_file(str(path / "state.safetensors")))
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    stream.load_state_dict(state["stream"])
    mask_generator.set_state(state["mask_generator"])
    restore_rng(state["rng"])
    return meta["step"]
