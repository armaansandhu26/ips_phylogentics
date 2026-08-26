# Table 2 — Matched-transform Pearson correlations

Pathwise terminal probability vs reward on 1M self-sampled trajectories.
Both methods report **linear** \(P(x)\) vs \(R(x)\) and **log-log** \(\log P(x)\) vs \(\log R(x)\) under the same transform within each column.

| Taxa | Method | Linear \(r\) | Log-log \(r\) | ESS | Unique sig. / 1M |
|-----:|--------|------------:|-------------:|----:|-----------------:|
| 5 | GFlowNet | 0.982 | — | — | 960,850 |
| 5 | MIPS-GRPO | 0.994 | 0.994 | 1.000 | 951,175 |
| 10 | GFlowNet | 0.881 | 0.698 | 0.977 | 1,000,000 |
| 10 | MIPS-GRPO | 0.976 | 0.835 | 0.995 | 999,986 |
| 27 | GFlowNet | 0.002 | 0.024 | 0.847 | 1,000,000 |
| 27 | MIPS-GRPO | 0.977 | 0.977 | 0.999 | 999,987 |

## Caption draft

Pearson correlation between pathwise implied terminal probability and terminal reward on 1M forward samples. Linear and log-log correlations are reported for both MIPS-GRPO and GFlowNet under matched transforms. At 27 taxa, linear axes compress dynamic range; log-log panels are the primary visual comparison.
