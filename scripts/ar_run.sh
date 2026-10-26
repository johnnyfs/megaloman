#!/bin/sh
# Train one AR config (final-weights protocol), then score its last checkpoint.
# usage: scripts/ar_run.sh NAME [ar_score args]
set -e
cd "$(dirname "$0")/.."
name=$1; shift
run=$HOME/Projects/megaloman-ptv0/workspace/pixel_model/runs/$name
export PYTHONPATH=src
.venv-cuda/bin/python -m megaloman.cli pixel train -c configs/pixel_model/loop/$name.yaml > /tmp/claude-1000/ar-$name.log 2>&1
.venv-cuda/bin/python scripts/ar_score.py $run/checkpoints/latest.json $run/score "$@"
