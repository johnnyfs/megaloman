# Benchmarks: animation (eval `convincing_anim@v2` #aa2d1dec68, judge `openai/gpt-5.6-sol`)

| benchmark | kind | mean | sd | range | fidelity | convincing | cost |
|---|---|---|---|---|---|---|---|
| met_walk_anim | authentic | 55.6 | 4.34 | 52-62 | None | 0.0 | $0.228 |
| telly_spin | authentic | 85 | 9.17 | 74-92 | None | 0.6 | $0.262 |
| telly_static | static | 61 | 16.22 | 32-69 | None | 0.0 | $0.199 |

- authentic mean **70.3**, noise (mean sd) **6.75**, recognition bonus **0.0**
- pass bar for a fresh design: **63.5** (mean >= pass_bar AND all gate checks pass (verdict ignored: authentic benchmarks fail it))
- control `telly_static` (static): score -24 vs its reference, fidelity +0
