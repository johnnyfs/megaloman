# Benchmarks: animation (eval `convincing_anim@v1` #f0b91a7ff0, judge `openai/gpt-5.6-sol`)

| benchmark | kind | mean | sd | range | fidelity | convincing | cost |
|---|---|---|---|---|---|---|---|
| met_walk_anim | authentic | 83.2 | 10.85 | 69-92 | None | 0.6 | $0.119 |
| telly_spin | authentic | 88 | 11.2 | 68-94 | None | 0.8 | $0.222 |
| telly_static | static | 79.4 | 10.9 | 68-92 | None | 0.4 | $0.232 |

- authentic mean **85.6**, noise (mean sd) **11.02**, recognition bonus **0.0**
- pass bar for a fresh design: **74.6** (mean >= pass_bar AND convincing_rate >= 0.5 AND all gate checks pass)
- control `telly_static` (static): score -8.6 vs its reference, fidelity +0
