# Benchmarks: keyframe (eval `convincing_enemy@v1` #4aa584a9ac, judge `openai/gpt-5.6-sol`)

| benchmark | kind | mean | sd | range | fidelity | convincing | cost |
|---|---|---|---|---|---|---|---|
| met_walk | authentic | 64.4 | 3.58 | 60-68 | 7.4 | 0.0 | $0.142 |
| pipi | authentic | 85.6 | 10.45 | 68-93 | 8.9 | 0.8 | $0.172 |
| telly | authentic | 66 | 4.12 | 62-73 | 8.4 | 0.0 | $0.190 |
| telly_as_pipi | wrong_description | 33.6 | 4.83 | 28-39 | 1.6 | 0.0 | $0.093 |
| telly_recolor | recolor | 61.8 | 6.38 | 52-67 | 6.2 | 0.0 | $0.141 |

- authentic mean **72.0**, noise (mean sd) **6.05**, recognition bonus **4.2**
- pass bar for a fresh design: **61.8** (mean >= pass_bar AND all gate checks pass (verdict ignored: authentic benchmarks fail it))
- control `telly_as_pipi` (wrong_description): score -32.4 vs its reference, fidelity -6.8
- control `telly_recolor` (recolor): score -4.2 vs its reference, fidelity -2.2
