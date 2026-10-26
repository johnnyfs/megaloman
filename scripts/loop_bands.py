"""Val NLL by random-mask band for checkpoints (mean over sizes; frozen masks).

usage: PYTHONPATH=src python scripts/loop_bands.py CKPT [CKPT ...]
"""
import json
import sys
from pathlib import Path

from megaloman.pixel_model.checkpoint import load_model
from megaloman.pixel_model.data import ApprovedDataset
from megaloman.pixel_model.evaluate import conditional_report

V2 = Path.home() / "Projects/megaloman-ptv0/workspace/pixel_model/datasets/mm-rm-enemies-v2/manifest.json"
RATIOS = (.5, .75, .9, .95, 1.0)
out = {}
for ckpt in sys.argv[1:]:
    model, cfg, _ = load_model(ckpt, "cuda")
    ds = ApprovedDataset(V2, cfg.dataset.resolutions, True)
    rows = conditional_report(model, ds, ratios=RATIOS, seed=1829, include_contiguous=False)["rows"]
    out[ckpt] = {str(q): sum(r["model"]["metrics"]["masked_conditional_nll"] for r in rows if r["mask_ratio"] == q) / 3
                 for q in RATIOS}
    print(ckpt.split("runs/")[-1], {k: round(v, 4) for k, v in out[ckpt].items()}, flush=True)
