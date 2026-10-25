"""Argparse registration stays usable without Torch or Modal installed."""
import json
from pathlib import Path

from .config import Config, load_config


def register(subparsers):
    pixel = subparsers.add_parser("pixel", help="four-class masked pixel transformer")
    commands = pixel.add_subparsers(dest="pixel_command", required=True)

    p = commands.add_parser("doctor", help="report hardware; optionally benchmark complete optimizer steps")
    p.add_argument("-c", "--config")
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--sizes", nargs="+", type=int, default=[16])
    p.add_argument("--output")

    p = commands.add_parser("prepare", help="validate approved manifest; optionally create synthetic fixture")
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--fixture", action="store_true", help="explicitly synthesize fixture if manifest does not exist")

    p = commands.add_parser("train", help="bounded local training (no automatic cloud fallback)")
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--resume")
    p.add_argument("--stop-after", type=int, help="stop at this absolute step without changing scheduler horizon")

    for mode in ("sample", "reconstruct", "inpaint"):
        p = commands.add_parser(mode)
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--steps", type=int)
        p.add_argument("--temperature", type=float)
        p.add_argument("--ranking-temperature", type=float)
        p.add_argument("--top-k", type=int)
        p.add_argument("--trace", action="store_true")
        if mode == "sample":
            p.add_argument("--size", type=int, default=16)
            p.add_argument("--count", type=int, default=32)
        else:
            p.add_argument("--image", required=True, help="approved indexed PNG, role IDs 0..3")
            if mode == "reconstruct":
                p.add_argument("--mask-ratio", type=float, default=.75)
            else:
                p.add_argument("--mask-mask", required=True, help="binary grayscale PNG: white=editable, black=locked")

    p = commands.add_parser("evaluate", help="fixed conditional metrics and unselected unconditional samples")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", help="relocate the identical content-hashed snapshot")
    p.add_argument("--output", required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--count", type=int, default=64)
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--near-threshold", type=float, default=.10, help="uncalibrated development threshold")

    p = commands.add_parser("overfit-check", help="optional single-full / few-partial correctness experiment")
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--output", required=True)

    p = commands.add_parser("modal", help="explicit remote adapter; dry-run never imports or contacts Modal")
    p.add_argument("operation", choices=["upload", "train", "sample", "download"])
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume")
    p.add_argument("--checkpoint")
    p.add_argument("--run-id", default="pilot")
    p.add_argument("--remote-path")
    p.add_argument("--output")
    p.add_argument("--size", type=int, default=16)
    p.add_argument("--count", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)


def run(args):
    cmd = args.pixel_command
    if cmd == "modal":
        from .modal_app import dispatch
        return dispatch(args)
    if cmd in ("doctor", "prepare", "train", "overfit-check"):
        cfg = load_config(args.config) if args.config else Config().validate()
        if cmd == "doctor":
            from .doctor import doctor
            report = doctor(cfg, benchmark=args.benchmark, sizes=args.sizes)
            if args.output:
                from .checkpoint import atomic_json
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                atomic_json(args.output, report)
            print(json.dumps(report, indent=2))
        elif cmd == "prepare":
            from .data import ApprovedDataset, create_fixture
            from .render import contact_sheet
            manifest = Path(cfg.dataset.manifest)
            if args.fixture and not manifest.exists():
                if manifest.name != "manifest.json":
                    raise ValueError("fixture expects a manifest.json filename")
                create_fixture(manifest.parent, cfg.dataset.resolutions)
            ds = ApprovedDataset(manifest, cfg.dataset.resolutions)
            report = ds.summary()
            (manifest.parent / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
            contact_sheet(ds.grids, manifest.parent / "sheet.png")
            print(json.dumps(report, indent=2))
        elif cmd == "train":
            from .train import train
            print(json.dumps(train(cfg, resume=args.resume, stop_after=args.stop_after), indent=2))
        else:
            from .overfit import overfit_checks
            print(json.dumps(overfit_checks(cfg, steps=args.steps, output=args.output), indent=2))
        return
    import numpy as np
    from PIL import Image
    import torch
    from .checkpoint import load_model
    model, cfg, meta = load_model(args.checkpoint, args.device)
    torch.set_num_threads(cfg.train.cpu_threads)
    if cmd == "evaluate":
        from .data import ApprovedDataset
        from .evaluate import evaluate
        if args.manifest:
            cfg.dataset.manifest = args.manifest
        if ApprovedDataset(cfg.dataset.manifest, cfg.dataset.resolutions).fingerprint != meta["dataset_sha256"]:
            raise ValueError("evaluation snapshot differs from the checkpoint dataset")
        evaluate(model, cfg, output=args.output, split=args.split, count=args.count,
                 seed=args.seed, near_threshold=args.near_threshold)
        print(f"Wrote {Path(args.output) / 'report.json'}")
        return
    from .masking import make_mask
    from .palette import DISPLAY_PALETTE, read_indexed
    from .render import export_samples
    from .sample import sample
    for name in ("steps", "temperature", "ranking_temperature", "top_k"):
        if getattr(args, name) is not None:
            setattr(cfg.sample, name, getattr(args, name))
    cfg.validate()
    initial = editable = None
    palette = DISPLAY_PALETTE
    if cmd == "sample":
        size, count = args.size, args.count
        if count < 1:
            raise ValueError("sample count must be positive")
    else:
        grid = read_indexed(args.image)
        if grid.shape[0] != grid.shape[1]:
            raise ValueError("input must be square")
        size, count = grid.shape[0], 1
        initial = torch.from_numpy(grid)[None]
        # Preserve the source display palette as well as the source class IDs.
        with Image.open(args.image) as image:
            colors = image.getpalette()
            palette = tuple(tuple(colors[3 * i:3 * i + 3]) for i in range(4))
        if cmd == "reconstruct":
            editable = make_mask(torch.ones_like(initial, dtype=torch.bool), ratio=args.mask_ratio,
                                 generator=torch.Generator().manual_seed(args.seed))
        else:
            with Image.open(args.mask_mask) as mask_image:
                if mask_image.mode not in ("1", "L") or mask_image.size != (size, size):
                    raise ValueError("edit mask must be native-size binary grayscale PNG")
                values = np.array(mask_image.convert("L"))
                if not np.isin(values, [0, 255]).all():
                    raise ValueError("edit mask must contain only 0 (locked) or 255 (editable)")
                editable = torch.from_numpy(values == 255)[None]
    # Sequential generation keeps sample count from multiplying attention memory.
    outputs, traces = [], []
    for i in range(count):
        result = sample(model, size=size, count=1, seed=args.seed + i, cfg=cfg.sample,
                        initial=initial, editable=editable, return_trace=args.trace)
        if args.trace:
            result, trace = result
            traces.append(trace)
        outputs.append(result.cpu())
    grids = torch.cat(outputs)
    trace = [torch.cat([row[t] for row in traces]) for t in range(len(traces[0]))] if args.trace and traces else None
    export_samples(grids, args.output, palette=palette, trace=trace,
                   metadata={"mode": cmd, "seed": args.seed, "checkpoint_step": meta["step"],
                             "sample_config": cfg.to_dict()["sample"],
                             "class_balance": torch.bincount(grids.flatten(), minlength=4).float().div(grids.numel()).tolist()})
    print(f"Wrote {args.output}")
