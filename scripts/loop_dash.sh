#!/bin/sh
# Point the single loop dashboard (panel prefix loop) at a run: scripts/loop_dash.sh NAME "Loop #NN"
systemctl --user stop loop-dash 2>/dev/null
d=$(cd "$(dirname "$0")/.." && pwd)
systemd-run --user --unit=loop-dash --collect "$d/.venv-cuda/bin/python" "$d/scripts/cockpit_training_dashboard.py" \
  "$HOME/Projects/megaloman-ptv0/workspace/pixel_model/runs/$1" --tab training --prefix loop --label "$2 ·"
