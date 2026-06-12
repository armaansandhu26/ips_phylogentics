# Evals Results

This file records the Pearson-eval results collected so far for the DS1 / r64 pooled benchmark.

## Benchmark setup

- Benchmark artifact:
  - `grpo_experiments/eval_benchmarks/signature_pooled_ds1_r64/benchmark.json`
- Candidate pool sources:
  - `grpo_experiments/runs/ips_replay_ablation/topo/20260603_112929_ablation_phylgfn_r64_phylgfn`
  - `grpo_experiments/runs/ips_replay_ablation/topo/20260603_115451_ablation_hyb_ips_pfloor_005_hybrid_ips_grpo`
  - `grpo_experiments/runs/ips_replay_ablation/topo/20260603_115912_ablation_hyb_ips_pfloor_002_hybrid_ips_grpo`
- Dedup used for the main benchmark:
  - `signature`
- Frozen benchmark size:
  - `300` trees total
- `log q(tree)` estimation:
  - `200` backward trajectories per tree

## 1. Signature-level Pearson

Source:
- `grpo_experiments/eval_benchmarks/signature_pooled_ds1_r64/pearson_eval/summary.json`

| Run | Overall | Low | Medium | High |
| --- | ---: | ---: | ---: | ---: |
| `phylgfn_r64` | `0.9103` | `0.8891` | `0.7520` | `0.9704` |
| `hyb_ips_p005` | `0.0144` | `-0.2709` | `0.0265` | `0.5580` |
| `hyb_ips_p002` | `0.0175` | `-0.2760` | `0.0666` | `0.5588` |

### Quick read

- `PhyloGFN` is strongly aligned with posterior quality on the frozen benchmark.
- Both `hyb_ips` checkpoints are much weaker overall.
- Both `hyb_ips` runs are better in the `high` band than overall, which is consistent with being able to reach good trees without matching posterior-relative probabilities well across the full benchmark.

## 2. Topology-collapsed Pearson

Source:
- `grpo_experiments/eval_benchmarks/signature_pooled_ds1_r64/pearson_eval/topology_collapsed/summary.json`

Collapse used:
- collapse `estimated_log_q` by topology using `mean`
- collapse `true_log_score` by topology using `mean`

| Run | Overall | Low | Medium | High | Topology count |
| --- | ---: | ---: | ---: | ---: | ---: |
| `phylgfn_r64` | `0.9811` | `0.9794` | `n/a` | `0.9863` | `20` |
| `hyb_ips_p005` | `-0.5198` | `-0.5859` | `n/a` | `0.4422` | `20` |
| `hyb_ips_p002` | `-0.4626` | `-0.5027` | `n/a` | `0.4262` | `20` |

### Quick read

- This does **not** look like only a signature-level artifact on DS1.
- Even after collapsing to topology level, `PhyloGFN` remains strongly aligned.
- The two `hyb_ips` checkpoints remain misaligned overall.
- The `medium` band is `n/a` here because after topology collapse on DS1 there were not enough medium-band topology points for a meaningful Pearson estimate.

## 3. Source-conditioned analysis

Question:
- does each evaluator assign relatively higher probability to trees that originally came from itself?

Result:
- yes, both `hyb_ips` evaluators strongly prefer trees sourced from themselves
- `PhyloGFN` is much more even across source groups

### 3.1 `phylgfn_r64` evaluator

| Source label | Count | Mean estimated `log q(tree)` | Median estimated `log q(tree)` | Mean true log score |
| --- | ---: | ---: | ---: | ---: |
| `hyb_ips_p002` | `44` | `-17.229` | `-16.168` | `-3219.149` |
| `hyb_ips_p005` | `88` | `-17.172` | `-16.177` | `-3219.768` |
| `phylgfn_r64` | `168` | `-17.125` | `-16.783` | `-3219.380` |
what d
By band:

| Band | Source label | Count | Mean estimated `log q(tree)` | Median estimated `log q(tree)` |
| --- | --- | ---: | ---: | ---: |
| `low` | `hyb_ips_p002` | `9` | `-22.251` | `-19.297` |
| `low` | `hyb_ips_p005` | `23` | `-22.462` | `-19.281` |
| `low` | `phylgfn_r64` | `68` | `-18.884` | `-18.571` |
| `medium` | `hyb_ips_p002` | `20` | `-16.870` | `-16.447` |
| `medium` | `hyb_ips_p005` | `26` | `-16.656` | `-16.610` |
| `medium` | `phylgfn_r64` | `54` | `-16.570` | `-16.534` |
| `high` | `hyb_ips_p002` | `15` | `-14.694` | `-14.597` |
| `high` | `hyb_ips_p005` | `39` | `-14.397` | `-14.388` |
| `high` | `phylgfn_r64` | `46` | `-15.177` | `-15.185` |

### 3.2 `hyb_ips_p005` evaluator

| Source label | Count | Mean estimated `log q(tree)` | Median estimated `log q(tree)` | Mean true log score |
| --- | ---: | ---: | ---: | ---: |
| `hyb_ips_p002` | `44` | `-551.266` | `-335.280` | `-3219.149` |
| `hyb_ips_p005` | `88` | `-7.727` | `-7.560` | `-3219.768` |
| `phylgfn_r64` | `168` | `-3255.726` | `-3222.697` | `-3219.380` |

By band:

| Band | Source label | Count | Mean estimated `log q(tree)` | Median estimated `log q(tree)` |
| --- | --- | ---: | ---: | ---: |
| `low` | `hyb_ips_p002` | `9` | `-618.823` | `-394.497` |
| `low` | `hyb_ips_p005` | `23` | `-9.379` | `-8.930` |
| `low` | `phylgfn_r64` | `68` | `-3576.049` | `-3718.861` |
| `medium` | `hyb_ips_p002` | `20` | `-457.080` | `-256.027` |
| `medium` | `hyb_ips_p005` | `26` | `-7.924` | `-7.475` |
| `medium` | `phylgfn_r64` | `54` | `-3452.625` | `-3478.503` |
| `high` | `hyb_ips_p002` | `15` | `-636.313` | `-335.937` |
| `high` | `hyb_ips_p005` | `39` | `-6.621` | `-6.466` |
| `high` | `phylgfn_r64` | `46` | `-2551.064` | `-2550.413` |

### 3.3 `hyb_ips_p002` evaluator

| Source label | Count | Mean estimated `log q(tree)` | Median estimated `log q(tree)` | Mean true log score |
| --- | ---: | ---: | ---: | ---: |
| `hyb_ips_p002` | `44` | `-7.359` | `-6.860` | `-3219.149` |
| `hyb_ips_p005` | `88` | `-403.258` | `-52.310` | `-3219.768` |
| `phylgfn_r64` | `168` | `-6098.172` | `-5678.422` | `-3219.380` |

By band:

| Band | Source label | Count | Mean estimated `log q(tree)` | Median estimated `log q(tree)` |
| --- | --- | ---: | ---: | ---: |
| `low` | `hyb_ips_p002` | `9` | `-9.772` | `-9.660` |
| `low` | `hyb_ips_p005` | `23` | `-836.333` | `-1146.544` |
| `low` | `phylgfn_r64` | `68` | `-6710.099` | `-6996.948` |
| `medium` | `hyb_ips_p002` | `20` | `-7.019` | `-6.803` |
| `medium` | `hyb_ips_p005` | `26` | `-197.786` | `-54.160` |
| `medium` | `phylgfn_r64` | `54` | `-6441.579` | `-6400.825` |
| `high` | `hyb_ips_p002` | `15` | `-6.364` | `-6.273` |
| `high` | `hyb_ips_p005` | `39` | `-284.836` | `-37.449` |
| `high` | `phylgfn_r64` | `46` | `-4790.455` | `-4860.820` |

### Quick read

- `hyb_ips_p005` strongly prefers trees sourced from `hyb_ips_p005`.
- `hyb_ips_p002` strongly prefers trees sourced from `hyb_ips_p002`.
- So the weak Pearson result is **not** because IPS checkpoints give uniformly low scores to everything.
- Instead, each IPS checkpoint seems to strongly like its own sampled region, but that self-preference does not translate into posterior-aligned ranking across the shared benchmark.

## Current overall interpretation

- `PhyloGFN` looks much better as a posterior-aligned density model on this DS1 benchmark.
- `hyb_ips_p005` and `hyb_ips_p002` can still sample good trees, but their assigned probability mass is not aligning with posterior quality the way `PhyloGFN` does.
- The weak IPS result is not only a signature-level artifact, because topology collapse does not rescue it.
- The source-conditioned analysis suggests the IPS checkpoints are at least self-consistent in preferring their own sampled region.
