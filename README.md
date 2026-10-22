# megaloman

Repeatable, declarative pipelines for NES Mega Man-style game assets. See `PLAN.md` (vision) and `V0_IMPL.md` (current scope).

- `uv run ml --help`: commands (benchmarks, run, bench, version, bless, feedback, promote, status)
- `workspace/`: definitions, benchmarks, pipeline versions, runs, feedback: the full data trail (layout in `src/megaloman/layout.py`)
- `cache/`: content-addressed blobs and raw model-call transcripts (keyed from the JSON in workspace/)
- External reference: `fusion-man/` is a clone of johnnyfs/fusion-man (not tracked here; benchmark rips come from its `main` at 65624f5)
