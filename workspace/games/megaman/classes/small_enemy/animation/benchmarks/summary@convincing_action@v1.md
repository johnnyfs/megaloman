# Benchmarks: animation (eval `convincing_action@v1` #4744049bfe, judge `openai/gpt-5.6-sol`)

| benchmark | kind | mean | sd | range | fidelity | convincing | cost |
|---|---|---|---|---|---|---|---|
| met_peek | authentic | 82.2 | 10.64 | 68-93 | None | 0.6 | $0.218 |
| met_peek_static | static | 34.2 | 3.42 | 30-39 | None | 0.0 | $0.213 |

- authentic mean **82.2**, noise (mean sd) **10.64**, recognition bonus **0.0**
- pass bar for a fresh design: **71.6** (mean >= pass_bar AND convincing_rate >= 0.5 AND all gate checks pass)
- control `met_peek_static` (static): score -48.0 vs its reference, fidelity +0
