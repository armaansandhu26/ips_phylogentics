# Final sEH comparison (2,500 updates, 50,000 samples)

Values are mean ± sample standard deviation over three independent seeds. Modes follow the RGFN protocol: sEH proxy > 7 and maximum pairwise Tanimoto similarity ≤ 0.5 under greedy leader selection.

| Method | Valid (%) | Unique (%) | Unique molecules | Mean proxy | Modes ↑ | Scaffolds >7 ↑ | Top-mode proxy ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| GRPO | 100.00 ± 0.00 | 0.011 ± 0.010 | 5.7 ± 5.0 | 4.261 ± 0.424 | 0.0 ± 0.0 | 0.0 ± 0.0 | — |
| Count IPS-GRPO | 100.00 ± 0.00 | 0.173 ± 0.286 | 86.7 ± 143.2 | 3.816 ± 1.374 | 0.0 ± 0.0 | 0.0 ± 0.0 | — |
| MIPS-GRPO | 99.64 ± 0.14 | 93.591 ± 2.427 | 46629.7 ± 1223.1 | 4.281 ± 0.177 | 22.3 ± 17.0 | 32.0 ± 25.5 | 7.210 ± 0.061 |
| RGFN | 99.94 ± 0.09 | 74.540 ± 3.994 | 37247.0 ± 1959.9 | 4.227 ± 0.421 | 14.0 ± 9.2 | 32.3 ± 24.2 | 7.191 ± 0.076 |

Top-mode proxy is undefined for GRPO and Count IPS-GRPO because none of their final samples met the mode threshold in any seed.
