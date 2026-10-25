"""Lazy Modal adapter. Import and --dry-run never authenticate, build, or launch.

The factory uses the installed pinned SDK only when explicitly requested. The
remote bodies call the same local train/sample functions; no duplicate model.
"""
from copy import deepcopy
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile

from .config import Config, load_config

MOUNT = "/pixel"
PINNED_DEPENDENCIES = ("torch==2.10.0", "safetensors==0.6.2", "packaging==25.0",
                       "numpy==2.5.4", "pillow==13.0.0", "pyyaml==6.0.3")


def safe_remote_path(value):
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("remote paths must be relative to the durable volume root")
    return str(path)


def execution_plan(cfg, operation, run_id="pilot"):
    cfg.validate()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", run_id):
        raise ValueError("run-id may contain only letters, numbers, underscores, and hyphens")
    estimate = (cfg.modal.timeout / 3600 * cfg.modal.estimated_hourly_usd
                if cfg.modal.estimated_hourly_usd is not None else None)
    return {"operation": operation, "gpu": cfg.modal.gpu, "image": cfg.modal.image,
            "volume": cfg.modal.volume, "run_id": run_id, "max_steps": cfg.train.max_steps,
            "training_wall_seconds": cfg.train.max_seconds, "function_timeout_seconds": cfg.modal.timeout,
            "retries": 0, "max_containers": 1, "estimated_gpu_usd_at_full_timeout": estimate,
            "max_estimated_usd": cfg.modal.max_estimated_usd,
            "note": "Estimate only; refresh pricing before execution. Includes no CPU/storage/build costs. Not a billing cap."}


def require_budget(plan):
    estimate = plan["estimated_gpu_usd_at_full_timeout"]
    if estimate is None:
        raise ValueError("set modal.estimated_hourly_usd from refreshed pricing before a remote launch")
    if estimate > plan["max_estimated_usd"]:
        raise ValueError("estimated full-timeout GPU cost exceeds modal.max_estimated_usd")


def create_app(cfg):
    """Creates only lazy SDK handles. No .run(), .remote(), or volume hydration."""
    import modal
    cfg.validate()
    volume = modal.Volume.from_name(cfg.modal.volume, create_if_missing=True)
    image = (modal.Image.from_registry(cfg.modal.image, add_python="3.12")
             .pip_install(*PINNED_DEPENDENCIES).add_local_python_source("megaloman"))
    app = modal.App("megaloman-pixel-transformer")
    settings = dict(image=image, gpu=cfg.modal.gpu, volumes={MOUNT: volume}, timeout=cfg.modal.timeout,
                    max_containers=1, retries=0, scaledown_window=2, serialized=True)

    @app.function(name="train", **settings)
    def remote_train(raw, snapshot_id, run_id, resume_path=None, source_provenance=None):
        from megaloman.pixel_model.checkpoint import atomic_json
        from megaloman.pixel_model.config import Config
        from megaloman.pixel_model.data import ApprovedDataset
        from megaloman.pixel_model.train import train
        volume.reload()
        remote_cfg = Config.from_dict(raw)
        remote_cfg.dataset.manifest = f"{MOUNT}/datasets/{snapshot_id}/manifest.json"
        remote_cfg.train.run_dir = f"{MOUNT}/runs/{run_id}"
        remote_cfg.train.device = "cuda"
        dataset = ApprovedDataset(remote_cfg.dataset.manifest, remote_cfg.dataset.resolutions)
        if dataset.fingerprint != snapshot_id:
            raise ValueError("uploaded snapshot fingerprint mismatch")
        run_path = Path(remote_cfg.train.run_dir)
        run_path.mkdir(parents=True, exist_ok=True)
        atomic_json(run_path / "submitted-source.json", source_provenance)
        return train(remote_cfg, resume=f"{MOUNT}/{safe_remote_path(resume_path)}" if resume_path else None,
                     durable_commit=volume.commit)

    @app.function(name="sample", **settings)
    def remote_sample(checkpoint, output, size, count, seed):
        import torch
        from megaloman.pixel_model.checkpoint import load_model
        from megaloman.pixel_model.render import export_samples
        from megaloman.pixel_model.sample import sample
        volume.reload()
        model, remote_cfg, meta = load_model(f"{MOUNT}/{safe_remote_path(checkpoint)}", "cuda")
        torch.set_num_threads(remote_cfg.train.cpu_threads)
        grids = torch.cat([sample(model, size=size, seed=seed + i, cfg=remote_cfg.sample).cpu() for i in range(count)])
        target = f"{MOUNT}/{safe_remote_path(output)}"
        export_samples(grids, target, metadata={"checkpoint_step": meta["step"], "seed": seed})
        volume.commit()
        return output

    return app, remote_train, remote_sample


def upload_snapshot(cfg):
    import modal
    from .data import ApprovedDataset
    dataset = ApprovedDataset(cfg.dataset.manifest, cfg.dataset.resolutions)
    remote = f"/datasets/{dataset.fingerprint}"
    volume = modal.Volume.from_name(cfg.modal.volume, create_if_missing=True)
    # This command is the only place that uploads data. Train never reuploads it.
    # Repeating an upload uses the same immutable name and is rejected by Modal
    # unless the complete identical snapshot was already present.
    try:
        entries = volume.listdir(remote)
    except FileNotFoundError:
        entries = []
    if any(PurePosixPath(e.path).name == "manifest.json" for e in entries):
        print(f"Snapshot already present: {remote}")
        return dataset.fingerprint
    with tempfile.TemporaryDirectory() as directory:
        staging = Path(directory)
        for record in dataset.records:
            dest = staging / record["path"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(dataset.path.parent / record["path"], dest)
        (staging / "manifest.json").write_text(json.dumps(dataset.raw, indent=2) + "\n")
        with volume.batch_upload() as batch:
            batch.put_directory(staging, remote)
    print(f"Uploaded {remote}")
    return dataset.fingerprint


def download_artifacts(cfg, remote_path, output):
    import modal
    remote_path = safe_remote_path(remote_path)
    volume = modal.Volume.from_name(cfg.modal.volume)
    output = Path(output)
    # Download the whole run tree (including checkpoint bundles and latest.json).
    entries = volume.listdir(remote_path, recursive=True)
    for entry in entries:
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        relative = PurePosixPath(entry.path.lstrip("/")).relative_to(remote_path)
        target = output / str(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as f:
            volume.read_file_into_fileobj(entry.path, f)
    print(f"Downloaded {remote_path} to {output}")


def dispatch(args):
    cfg = load_config(args.config)
    plan = execution_plan(cfg, args.operation, args.run_id)
    for key in ("resume", "checkpoint", "remote_path"):
        value = getattr(args, key)
        if value:
            safe_remote_path(value)
    if args.dry_run:
        print(json.dumps({**plan, "dry_run": True, "remote_calls": 0}, indent=2))
        return
    if args.operation == "upload":
        return upload_snapshot(cfg)
    if args.operation == "download":
        if not args.remote_path or not args.output:
            raise ValueError("download requires --remote-path and --output")
        return download_artifacts(cfg, args.remote_path, args.output)
    require_budget(plan)
    if args.operation == "sample" and (args.count < 1 or args.size not in cfg.dataset.resolutions):
        raise ValueError("sample count must be positive and size must be configured")
    if args.operation == "sample" and not args.checkpoint:
        raise ValueError("remote sample requires --checkpoint relative to the volume")
    from .checkpoint import provenance
    from .data import ApprovedDataset
    snapshot = ApprovedDataset(cfg.dataset.manifest, cfg.dataset.resolutions).fingerprint if args.operation == "train" else None
    app, remote_train, remote_sample = create_app(cfg)
    import modal
    with modal.enable_output(), app.run():
        if args.operation == "train":
            result = remote_train.remote(deepcopy(cfg.to_dict()), snapshot, args.run_id, args.resume, provenance())
        else:
            result = remote_sample.remote(args.checkpoint, f"runs/{args.run_id}/remote-samples", args.size, args.count, args.seed)
    print(json.dumps(result, indent=2))
