#!/bin/sh
# Train one loop experiment config, then score its best checkpoint into RUN/score.
# usage: scripts/loop_run.sh NAME [extra loop_score args]
set -e
cd "$(dirname "$0")/.."
name=$1; shift
run=$HOME/Projects/megaloman-ptv0/workspace/pixel_model/runs/$name
export PYTHONPATH=src
.venv-cuda/bin/python -m megaloman.cli pixel train -c configs/pixel_model/loop/$name.yaml > /tmp/claude-1000/loop-$name.log 2>&1
target=$run/best
grep -q "patience: 0" configs/pixel_model/loop/$name.yaml && target=$run/checkpoints/latest.json  # final-weights protocol
.venv-cuda/bin/python scripts/loop_score.py $target $run/score "$@"
