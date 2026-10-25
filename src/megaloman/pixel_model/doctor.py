"""Hardware report and opt-in warmed-up forward/backward/optimizer profiling."""
import subprocess
import time

import torch

from .masking import corrupt, masked_loss
from .model import PixelTransformer
from .train import autocast_context, resolve_runtime


def doctor(cfg, *, benchmark=False, sizes=(16,)):
    runtime = resolve_runtime(cfg.train)
    torch.set_num_threads(cfg.train.cpu_threads)
    cuda = runtime["device"] == "cuda"
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv"],
                             capture_output=True, text=True, timeout=10)
        smi_report = smi.stdout.strip() if smi.returncode == 0 else (smi.stderr + smi.stdout).strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        smi_report = str(exc)
    report = {"runtime": runtime, "cuda_available": torch.cuda.is_available(), "torch_cuda_version": torch.version.cuda,
              "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
              "nvidia_smi": smi_report, "benchmarks": [], "attention": "SDPA; see profiler operator names per benchmark"}
    if cuda:
        free, total = torch.cuda.mem_get_info()
        report["vram"] = {"free_bytes": free, "total_bytes": total}
    if not benchmark:
        return report
    for size in sizes:
        if size not in (16, 24, 32, 64):
            raise ValueError("benchmark sizes must be 16,24,32,64; 64 is experimental")
        model = optimizer = None
        try:
            if cuda:
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            model = PixelTransformer(cfg.model, (size,)).to(runtime["device"])
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
            scaler = torch.amp.GradScaler("cuda", enabled=runtime["precision"] == "fp16")
            target = torch.randint(4, (cfg.train.batch_size, size, size), device=runtime["device"])
            mask = torch.ones_like(target, dtype=torch.bool)

            def update():
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(runtime):
                    loss = masked_loss(model(corrupt(target, mask)).float(), target, mask)
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()
                if cuda:
                    torch.cuda.synchronize()

            for _ in range(2):
                update()
            start = time.monotonic()
            for _ in range(3):
                update()
            elapsed = (time.monotonic() - start) / 3
            activities = [torch.profiler.ProfilerActivity.CPU]
            if cuda:
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(activities=activities) as prof:
                update()
            operators = sorted({event.key for event in prof.key_averages()
                                if "attention" in event.key.lower()})
            report["benchmarks"].append({"size": size, "batch_size": cfg.train.batch_size,
                "parameters": sum(p.numel() for p in model.parameters()), "seconds_per_optimizer_step": elapsed,
                "tokens_per_second": cfg.train.batch_size * size * size / elapsed,
                "attention_operators": operators,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated() if cuda else 0,
                "peak_reserved_bytes": torch.cuda.max_memory_reserved() if cuda else 0})
        except torch.cuda.OutOfMemoryError:
            report["benchmarks"].append({"size": size, "status": "OOM: unsupported at this batch/config",
                "suggestion": "reduce batch, accumulate gradients, use mixed precision, or benchmark another device"})
        finally:
            del model, optimizer
            if cuda:
                torch.cuda.empty_cache()
    return report
