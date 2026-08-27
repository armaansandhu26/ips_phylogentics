# Table 1. Full-space sEH at 500 updates (seed 0)

All methods use 50,000 training oracle calls and 5,000 independent final-policy samples. This is an interim, single-seed result.

| Method | Valid | Unique | Unique fraction | Mean proxy | Candidates >7 | Leader modes >7 | Top-1 mass | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| GRPO | 100.00% | 3 | 0.060% | 4.150 | 0 | 0 | 98.6% | collapsed |
| Count IPS-GRPO | 100.00% | 2 | 0.040% | 6.040 | 0 | 0 | 99.96% | collapsed |
| MIPS-GRPO | 99.1% | 4,923 | 99.4% | 4.110 | 4 | 3 | 0.081% | diverse |
| RGFN | 100.00% | 3,965 | 79.3% | 5.149 | 2 | 2 | 0.96% | diverse |

Leader modes use proxy > 7 and greedy Morgan-fingerprint leader clustering with maximum Tanimoto similarity ≤ 0.5.
GRPO and Count IPS-GRPO checkpoint values come from the frozen 500-update results record because their live summaries were later replaced by 2,000-update evaluations.
