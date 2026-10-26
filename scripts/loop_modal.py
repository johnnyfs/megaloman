"""Run one loop experiment on a Modal L4: train + loop_score remotely, then download the
run into the local runs/<name> layout. Appends billed-time estimates to runs/modal-spend.jsonl.

usage: PYTHONPATH=src .venv-cuda/bin/python scripts/loop_modal.py NAME [loop_score args...]
Refuses to launch when the recorded spend is >= $18 (hard budget $20). L4 only.
"""
import json
from pathlib import Path
import sys
import time

import modal

from megaloman.pixel_model.config import load_config
from megaloman.pixel_model.data import ApprovedDataset
from megaloman.pixel_model.modal_app import MOUNT, PINNED_DEPENDENCIES, download_artifacts, upload_snapshot

RUNS = Path.home() / "Projects/megaloman-ptv0/workspace/pixel_model/runs"
SPEND = RUNS / "modal-spend.jsonl"
# L4 $0.7992/h + 4 cores (~$0.047/core-h) + memory, times a margin
RATE, MARGIN, STOP, BUDGET = 0.7992 + 4 * 0.0473 + 0.05, 1.15, 18.0, 20.0
TIMEOUT = 1800


def spent():
    if not SPEND.exists():
        return 0.0
    return sum(json.loads(l)["usd"] for l in SPEND.read_text().splitlines() if l.strip())


def main():
    name, *score_args = sys.argv[1:]
    if spent() >= STOP:
        sys.exit(f"Modal spend {spent():.2f} >= {STOP}: not launching")
    if spent() + TIMEOUT / 3600 * RATE * MARGIN > BUDGET:
        sys.exit("a worst-case run would exceed the $20 budget")
    cfg = load_config(f"configs/pixel_model/loop/{name}.yaml")
    cfg.modal.gpu = "L4"
    snapshot = ApprovedDataset(cfg.dataset.manifest, cfg.dataset.resolutions, cfg.dataset.skip_other_sizes).fingerprint
    upload_snapshot(cfg)
    volume = modal.Volume.from_name(cfg.modal.volume, create_if_missing=True)
    image = (modal.Image.from_registry("debian:bookworm-slim", add_python="3.12")
             .pip_install(*PINNED_DEPENDENCIES).add_local_file("scripts/loop_score.py", "/root/loop_score.py")
             .add_local_file("scripts/ar_score.py", "/root/ar_score.py")
             .add_local_python_source("megaloman"))
    app = modal.App(f"pixel-loop-{name}")

    @app.function(image=image, gpu="L4", volumes={MOUNT: volume}, timeout=TIMEOUT, retries=0, cpu=4.0,
                  max_containers=1, scaledown_window=2, serialized=True)
    def run(raw, snapshot_id, name, score_args):
        import subprocess
        import time as _t
        from megaloman.pixel_model.config import Config
        from megaloman.pixel_model.train import train
        start = _t.time()
        volume.reload()
        c = Config.from_dict(raw)
        c.dataset.manifest = f"{MOUNT}/datasets/{snapshot_id}/manifest.json"
        c.train.run_dir = f"{MOUNT}/runs/{name}"
        c.train.device = "cuda"
        import shutil
        shutil.rmtree(c.train.run_dir, ignore_errors=True)  # a relaunch replaces an aborted attempt
        train(c)
        target = f"{c.train.run_dir}/best"
        if c.train.patience == 0 or c.model.causal:  # final-weights protocol: score the last checkpoint
            from megaloman.pixel_model.checkpoint import checkpoint_path
            target = str(checkpoint_path(f"{c.train.run_dir}/checkpoints/latest.json"))
            shutil.copytree(target, f"{c.train.run_dir}/final")
        subprocess.run(["python", "/root/ar_score.py" if c.model.causal else "/root/loop_score.py", target, f"{c.train.run_dir}/score",
                        "--manifest", c.dataset.manifest, *score_args], check=True)
        shutil.rmtree(f"{c.train.run_dir}/checkpoints", ignore_errors=True)  # keep best (+ final) only
        volume.commit()
        return _t.time() - start

    t0 = time.time()
    seconds = None
    try:
        with app.run():
            seconds = run.remote(cfg.to_dict(), snapshot, name, score_args)
    finally:
        wall = time.time() - t0
        billed = max(seconds or 0, wall)  # wall includes cold start; conservative
        with SPEND.open("a") as f:
            f.write(json.dumps({"name": name, "remote_seconds": seconds, "wall_seconds": wall,
                                "usd": billed / 3600 * RATE * MARGIN, "time": time.time()}) + "\n")
    download_artifacts(cfg, f"runs/{name}", RUNS / name)
    print(json.dumps({"name": name, "remote_seconds": seconds, "wall_seconds": wall, "spent": spent()}))


if __name__ == "__main__":
    main()
