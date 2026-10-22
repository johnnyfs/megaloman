# Benchmarks: animation (eval `convincing_anim_timed@561e641e45`, judge `openai/gpt-5.6-sol`)

| benchmark | kind | mean | sd | range | fidelity | convincing | cost |
|---|---|---|---|---|---|---|---|
| met_walk_anim | authentic | 66 | 3.94 | 59-68 | None | 0.0 | $0.236 |
| telly_spin | authentic | 91 | 2.35 | 87-93 | None | 1.0 | $0.255 |
| telly_static | static | 69.2 | 26.79 | 40-93 | None | 0.6 | $0.266 |

- authentic mean **78.5**, noise (mean sd) **3.15**, recognition bonus **0.0**
- pass bar for a fresh design: **75.3** (mean >= pass_bar AND all gate checks pass (verdict ignored: authentic benchmarks fail it))
- control `telly_static` (static): score -21.8 vs its reference, fidelity +0
