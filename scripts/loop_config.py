"""Write a loop experiment config: base yaml + dotted overrides, run_dir runs/<name>.

usage: python scripts/loop_config.py BASE.yaml NAME key.sub=value ...  -> configs/pixel_model/loop/NAME.yaml
"""
import sys
from pathlib import Path

import yaml

RUNS = Path.home() / "Projects/megaloman-ptv0/workspace/pixel_model/runs"
base, name, *overrides = sys.argv[1:]
cfg = yaml.safe_load(Path(base).read_text())
cfg["train"]["run_dir"] = str(RUNS / name)
for item in overrides:
    key, value = item.split("=", 1)
    section, field = key.split(".")
    cfg.setdefault(section, {})[field] = yaml.safe_load(value)
out = Path("configs/pixel_model/loop") / f"{name}.yaml"
out.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(out)
