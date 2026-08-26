# Completed run results — Aug 24–26 batch

Compiled snapshot of runs started **Aug 24–25, 2026** that have finished training and post-train evaluation as of **Aug 26, 2026**. Paths are relative to the repo root unless noted.

**Still in progress (not included below):** 27-taxa LR-IPS seeds 0–2, 27-taxa PhyloGFN seeds 0–2, 27-taxa seed0 GRPO (failed OOM — not restarted).

---

## Summary tables

### 5-taxa noreplay suite (`5taxa_noreplay`)

10k epochs, full signature model, no replay, shift 3600, 1M eval samples.

| Method | Final epoch | Train mean log reward | Eval ESS frac | Log-log P vs R (Pearson) | Unique signatures (1M) |
|--------|------------:|----------------------:|--------------:|-------------------------:|-------------------------:|
| GRPO | 9999 | 5.89 | — | — (collapsed to 1 sig) | 1 |
| Count-IPS | 9999 | 5.91 | — | 0.565 (empirical freq fit) | 66,268 |
| Learned-Reverse IPS | 9999 | 5.58 | **0.9998** | **0.992** | 951,180 |
| PhyloGFN (TB) | 9999 | — | 0.129 | **0.991** (log P vs log L) | 100,508 |

Run dirs under `final/runs/5taxa_noreplay/{grpo,count_ips,learned_reverse,phylgfn}/20260825_*`.

### 10-taxa uniform q\_φ ablation (`10taxa_uniform_reverse_ablation`)

Frozen uniform backward policy (`reverse_policy=uniform`, `reverse_train_epochs=0`) vs fitted q\_φ baseline in P1_RESULTS.md.

| Field | Value |
|-------|------:|
| Final epoch | 9999 |
| Train mean log reward | 5.76 |
| Eval ESS fraction | **0.995** |
| Log-log P vs R (Pearson vs ideal) | **0.969** |
| Unique signatures (1M) | 999,974 |
| Run dir | `final/runs/10taxa_uniform_reverse_ablation/learned_reverse/20260825_235540_*` |

### Hyper-Grid 64 (`hypergrid_64`) — 25k epochs

| Method | Recovery | Modes found | L1 to GT | Peak mode mass | Unique terminals (50k eval) |
|--------|:--------:|:-----------:|---------:|---------------:|------------------------------:|
| GRPO | 25% | 1/4 | 1.97 | 1.00 | 15 |
| Count-IPS | 25% | 1/4 | 1.84 | 0.99 | 44 |
| Learned-Reverse IPS | **100%** | **4/4** | **0.29** | 0.28 | 4034 |
| GFlowNet TB | **100%** | **4/4** | 0.36 | 0.26 | 4056 |

Source: `final/runs/hypergrid_64/plots/epoch_9999/recovery_summary.json`.

### Hyper-Grid 4096 (`hypergrid_4096`) — 100 epochs (smoke scale)

| Method | Recovery | Modes found | L1 to GT |
|--------|:--------:|:-----------:|---------:|
| GRPO | 0% | 0/4 | 2.00 |
| Count-IPS | 0% | 0/4 | 2.00 |

Short run for pipeline validation; not comparable to 64-grid headline results.

---

## 1. Five-taxa paper comparison suite

### 1.1 GRPO

**Run:** `final/runs/5taxa_noreplay/grpo/20260825_150308_grpo_5taxa_noreplay_grpo`

Training collapsed to a single observed signature at 1M samples (mode collapse).

#### Training curves

![5taxa GRPO training curves](../runs/5taxa_noreplay/grpo/20260825_150308_grpo_5taxa_noreplay_grpo/plots/training_curves.png)

#### Sampling — model P vs reward

![5taxa GRPO P vs R](../runs/5taxa_noreplay/grpo/20260825_150308_grpo_5taxa_noreplay_grpo/plots/mlp_shifted_linear_reference_1000k/model_probability_vs_reward.png)

<details>
<summary>Additional GRPO plots</summary>

![log P vs log R](../runs/5taxa_noreplay/grpo/20260825_150308_grpo_5taxa_noreplay_grpo/plots/mlp_shifted_linear_reference_1000k/log_model_probability_vs_log_reward.png)

![topology vs reward ref](../runs/5taxa_noreplay/grpo/20260825_150308_grpo_5taxa_noreplay_grpo/plots/mlp_shifted_linear_reference_1000k/topology_checkpoint_vs_reward_reference.png)

![signature top-k](../runs/5taxa_noreplay/grpo/20260825_150308_grpo_5taxa_noreplay_grpo/plots/mlp_shifted_linear_reference_1000k/signature_checkpoint_vs_reward_reference_topk.png)

</details>

---

### 1.2 Count-IPS-GRPO

**Run:** `final/runs/5taxa_noreplay/count_ips/20260825_152552_count_ips_5taxa_noreplay_ips_grpo`

#### Training curves

![5taxa Count-IPS training curves](../runs/5taxa_noreplay/count_ips/20260825_152552_count_ips_5taxa_noreplay_ips_grpo/plots/training_curves.png)

#### Sampling — model P vs reward

![5taxa Count-IPS P vs R](../runs/5taxa_noreplay/count_ips/20260825_152552_count_ips_5taxa_noreplay_ips_grpo/plots/mlp_shifted_linear_reference_1000k/model_probability_vs_reward.png)

![5taxa Count-IPS log P vs log R](../runs/5taxa_noreplay/count_ips/20260825_152552_count_ips_5taxa_noreplay_ips_grpo/plots/mlp_shifted_linear_reference_1000k/log_model_probability_vs_log_reward.png)

<details>
<summary>Additional Count-IPS plots</summary>

![topology vs reward ref](../runs/5taxa_noreplay/count_ips/20260825_152552_count_ips_5taxa_noreplay_ips_grpo/plots/mlp_shifted_linear_reference_1000k/topology_checkpoint_vs_reward_reference.png)

![signature qhat vs loglikelihood](../runs/5taxa_noreplay/count_ips/20260825_152552_count_ips_5taxa_noreplay_ips_grpo/plots/mlp_shifted_linear_reference_1000k/signature_qhat_vs_loglikelihood_1000k.png)

![partition calibrated P vs R](../runs/5taxa_noreplay/count_ips/20260825_152552_count_ips_5taxa_noreplay_ips_grpo/plots/mlp_shifted_linear_reference_1000k/partition_calibrated_model_probability_vs_reward.png)

</details>

---

### 1.3 Learned-Reverse IPS-GRPO

**Run:** `final/runs/5taxa_noreplay/learned_reverse/20260825_164705_learned_reverse_5taxa_noreplay_learned_reverse_ips_grpo`

Best diversity and IPS ESS among GRPO-family methods on 5 taxa.

#### Training curves

![5taxa LR-IPS training curves](../runs/5taxa_noreplay/learned_reverse/20260825_164705_learned_reverse_5taxa_noreplay_learned_reverse_ips_grpo/plots/training_curves.png)

#### Sampling — model P vs reward

![5taxa LR-IPS P vs R](../runs/5taxa_noreplay/learned_reverse/20260825_164705_learned_reverse_5taxa_noreplay_learned_reverse_ips_grpo/plots/mlp_shifted_linear_reference_1000k/model_probability_vs_reward.png)

![5taxa LR-IPS log P vs log R](../runs/5taxa_noreplay/learned_reverse/20260825_164705_learned_reverse_5taxa_noreplay_learned_reverse_ips_grpo/plots/mlp_shifted_linear_reference_1000k/log_model_probability_vs_log_reward.png)

<details>
<summary>Additional LR-IPS plots</summary>

![topology vs reward ref](../runs/5taxa_noreplay/learned_reverse/20260825_164705_learned_reverse_5taxa_noreplay_learned_reverse_ips_grpo/plots/mlp_shifted_linear_reference_1000k/topology_checkpoint_vs_reward_reference.png)

![signature top-k](../runs/5taxa_noreplay/learned_reverse/20260825_164705_learned_reverse_5taxa_noreplay_learned_reverse_ips_grpo/plots/mlp_shifted_linear_reference_1000k/signature_checkpoint_vs_reward_reference_topk.png)

![signature qhat vs loglikelihood](../runs/5taxa_noreplay/learned_reverse/20260825_164705_learned_reverse_5taxa_noreplay_learned_reverse_ips_grpo/plots/mlp_shifted_linear_reference_1000k/signature_qhat_vs_loglikelihood_1000k.png)

![partition calibrated P vs R](../runs/5taxa_noreplay/learned_reverse/20260825_164705_learned_reverse_5taxa_noreplay_learned_reverse_ips_grpo/plots/mlp_shifted_linear_reference_1000k/partition_calibrated_model_probability_vs_reward.png)

</details>

---

### 1.4 PhyloGFN (trajectory balance, paper backend)

**Run:** `final/runs/5taxa_noreplay/phylgfn/20260825_180739_phylgfn_5taxa_noreplay`

#### Training diagnostics

![5taxa PhyloGFN log loss](../runs/5taxa_noreplay/phylgfn/20260825_180739_phylgfn_5taxa_noreplay/log_loss.png)

![5taxa PhyloGFN log partition](../runs/5taxa_noreplay/phylgfn/20260825_180739_phylgfn_5taxa_noreplay/log_partition.png)

![5taxa PhyloGFN grad norm](../runs/5taxa_noreplay/phylgfn/20260825_180739_phylgfn_5taxa_noreplay/grad_norm.png)

#### Sampling — 1M eval

![5taxa PhyloGFN log P vs log R](../runs/5taxa_noreplay/phylgfn/20260825_180739_phylgfn_5taxa_noreplay/plots/reward_probability_eval_1000000/paper_gflownet_log_probability_vs_log_reward.png)

![5taxa PhyloGFN P vs R](../runs/5taxa_noreplay/phylgfn/20260825_180739_phylgfn_5taxa_noreplay/plots/reward_probability_eval_1000000/paper_gflownet_model_probability_vs_reward.png)

---

## 2. Ten-taxa uniform q\_φ ablation

**Run:** `final/runs/10taxa_uniform_reverse_ablation/learned_reverse/20260825_235540_learned_reverse_10taxa_uniform_reverse_ablation_learned_reverse_ips_grpo`

Tests whether IPS normalization alone (frozen uniform P\_B) suffices without fitting q\_φ. Compare to fitted baseline in [P1_RESULTS.md](P1_RESULTS.md). **Interpretation for paper/rebuttal:** see [PAPER_REVISION_PRIORITIES.md](PAPER_REVISION_PRIORITIES.md) (P1 §2).

#### Training curves

![10taxa uniform LR-IPS training curves](../runs/10taxa_uniform_reverse_ablation/learned_reverse/20260825_235540_learned_reverse_10taxa_uniform_reverse_ablation_learned_reverse_ips_grpo/plots/training_curves.png)

#### Sampling — 1M eval

![10taxa uniform P vs R](../runs/10taxa_uniform_reverse_ablation/learned_reverse/20260825_235540_learned_reverse_10taxa_uniform_reverse_ablation_learned_reverse_ips_grpo/plots/mlp_shifted_linear_reference_1000k/model_probability_vs_reward.png)

![10taxa uniform log P vs log R](../runs/10taxa_uniform_reverse_ablation/learned_reverse/20260825_235540_learned_reverse_10taxa_uniform_reverse_ablation_learned_reverse_ips_grpo/plots/mlp_shifted_linear_reference_1000k/log_model_probability_vs_log_reward.png)

<details>
<summary>Additional 10taxa ablation plots</summary>

![topology vs reward ref](../runs/10taxa_uniform_reverse_ablation/learned_reverse/20260825_235540_learned_reverse_10taxa_uniform_reverse_ablation_learned_reverse_ips_grpo/plots/mlp_shifted_linear_reference_1000k/topology_checkpoint_vs_reward_reference.png)

![signature top-k](../runs/10taxa_uniform_reverse_ablation/learned_reverse/20260825_235540_learned_reverse_10taxa_uniform_reverse_ablation_learned_reverse_ips_grpo/plots/mlp_shifted_linear_reference_1000k/signature_checkpoint_vs_reward_reference_topk.png)

![signature qhat vs loglikelihood](../runs/10taxa_uniform_reverse_ablation/learned_reverse/20260825_235540_learned_reverse_10taxa_uniform_reverse_ablation_learned_reverse_ips_grpo/plots/mlp_shifted_linear_reference_1000k/signature_qhat_vs_loglikelihood_1000k.png)

![partition calibrated P vs R](../runs/10taxa_uniform_reverse_ablation/learned_reverse/20260825_235540_learned_reverse_10taxa_uniform_reverse_ablation_learned_reverse_ips_grpo/plots/mlp_shifted_linear_reference_1000k/partition_calibrated_model_probability_vs_reward.png)

</details>

---

## 3. Hyper-Grid 64 (toy benchmark)

All methods trained 25k epochs. Suite-level comparison at final checkpoint.

### 3.1 Suite-level comparison

![Hypergrid64 recovery summary](../runs/hypergrid_64/plots/epoch_9999/recovery_summary.png)

![Hypergrid64 terminal distribution (all methods)](../runs/hypergrid_64/plots/epoch_9999/terminal_distribution_comparison.png)

![Hypergrid64 modes found vs samples](../runs/hypergrid_64/plots/epoch_9999/modes_found_vs_samples.png)

![Hypergrid64 training L1 curve (suite)](../runs/hypergrid_64/plots/epoch_9999/training_l1_curve.png)

![Hypergrid64 training reward curve (suite)](../runs/hypergrid_64/plots/epoch_9999/training_reward_curve.png)

### 3.2 Per-method GT vs sampled distribution

![GT vs GRPO](../runs/hypergrid_64/plots/gt_vs_grpo.png)

![GT vs Count-IPS](../runs/hypergrid_64/plots/gt_vs_count_ips.png)

![GT vs Learned-Reverse IPS](../runs/hypergrid_64/plots/gt_vs_learned_reverse_ips.png)

### 3.3 Per-method training diagnostics

#### GRPO
**Run:** `final/runs/hypergrid_64/grpo/20260824_184125_hypergrid_64_grpo`

![HG64 GRPO training diagnostics](../runs/hypergrid_64/grpo/20260824_184125_hypergrid_64_grpo/plots/training_diagnostics.png)

![HG64 GRPO terminal distribution](../runs/hypergrid_64/grpo/20260824_184125_hypergrid_64_grpo/plots/terminal_distribution_comparison.png)

![HG64 GRPO training L1](../runs/hypergrid_64/grpo/20260824_184125_hypergrid_64_grpo/plots/training_l1_curve.png)

#### Count-IPS
**Run:** `final/runs/hypergrid_64/count_ips/20260824_191534_hypergrid_64_count_ips`

![HG64 Count-IPS training diagnostics](../runs/hypergrid_64/count_ips/20260824_191534_hypergrid_64_count_ips/plots/training_diagnostics.png)

![HG64 Count-IPS terminal distribution](../runs/hypergrid_64/count_ips/20260824_191534_hypergrid_64_count_ips/plots/terminal_distribution_comparison.png)

![HG64 Count-IPS training L1](../runs/hypergrid_64/count_ips/20260824_191534_hypergrid_64_count_ips/plots/training_l1_curve.png)

#### Learned-Reverse IPS
**Run:** `final/runs/hypergrid_64/learned_reverse_ips/20260824_212235_hypergrid_64_learned_reverse_ips`

Final train mean log reward: **−0.67** (epoch 24999).

![HG64 LR-IPS training diagnostics](../runs/hypergrid_64/learned_reverse_ips/20260824_212235_hypergrid_64_learned_reverse_ips/plots/training_diagnostics.png)

![HG64 LR-IPS training L1 live](../runs/hypergrid_64/learned_reverse_ips/20260824_212235_hypergrid_64_learned_reverse_ips/plots/training_l1_live.png)

#### GFlowNet trajectory balance
**Run:** `final/runs/hypergrid_64/trajectory_balance/20260824_224747_hypergrid_64_trajectory_balance`

![HG64 TB training diagnostics](../runs/hypergrid_64/trajectory_balance/20260824_224747_hypergrid_64_trajectory_balance/plots/training_diagnostics.png)

![HG64 TB training L1 live](../runs/hypergrid_64/trajectory_balance/20260824_224747_hypergrid_64_trajectory_balance/plots/training_l1_live.png)

---

## 4. Hyper-Grid 4096 (short validation runs)

100 epochs only — pipeline smoke test at larger grid size.

### Suite comparison

![Hypergrid4096 recovery summary](../runs/hypergrid_4096/plots/recovery_summary.png)

![Hypergrid4096 terminal distribution](../runs/hypergrid_4096/plots/terminal_distribution_comparison.png)

![Hypergrid4096 modes found vs samples](../runs/hypergrid_4096/plots/modes_found_vs_samples.png)

### Per-method

#### GRPO — `final/runs/hypergrid_4096/grpo/20260824_182056_hypergrid_4096_grpo`

![HG4096 GRPO training diagnostics](../runs/hypergrid_4096/grpo/20260824_182056_hypergrid_4096_grpo/plots/training_diagnostics.png)

![HG4096 GRPO terminal distribution](../runs/hypergrid_4096/grpo/20260824_182056_hypergrid_4096_grpo/plots/terminal_distribution_comparison.png)

#### Count-IPS — `final/runs/hypergrid_4096/count_ips/20260824_182056_hypergrid_4096_count_ips`

![HG4096 Count-IPS training diagnostics](../runs/hypergrid_4096/count_ips/20260824_182056_hypergrid_4096_count_ips/plots/training_diagnostics.png)

![HG4096 Count-IPS terminal distribution](../runs/hypergrid_4096/count_ips/20260824_182056_hypergrid_4096_count_ips/plots/terminal_distribution_comparison.png)

---

## 5. Runs started Aug 25 — not yet complete

| Suite | Method | Status (Aug 26) |
|-------|--------|-----------------|
| `27taxa_noreplay_b4096_seed0` | learned_reverse | ~25% (6.2k/25k epochs) |
| `27taxa_noreplay_b4096_seed1` | learned_reverse | ~21% (5.3k/25k) |
| `27taxa_noreplay_b4096_seed2` | learned_reverse | ~19% (4.6k/25k) |
| `27taxa_noreplay_b4096_seed0` | phylgfn | ~44% (~14k/32k est.) |
| `27taxa_noreplay_b4096_seed1` | phylgfn | ~44% |
| `27taxa_noreplay_b4096_seed2` | phylgfn | ~76% (~24k/32k est.) |
| `27taxa_noreplay_b4096_seed0` | grpo | **Failed** (OOM at epoch 0) |

---

## Regenerate

```bash
cd /home/armaan/phylogfn

# Re-run post-train for a completed DNA run (example)
CUDA_VISIBLE_DEVICES=3 .venv/bin/python -u -m final pipeline \
  --suite 5taxa_noreplay --method learned_reverse --skip-train --skip-sample \
  --run-dir final/runs/5taxa_noreplay/learned_reverse/20260825_164705_learned_reverse_5taxa_noreplay_learned_reverse_ips_grpo \
  --cuda-device 0 --device cuda:0

# Hypergrid suite comparison plots
.venv/bin/python -m final.toy.plot_comparison --suite hypergrid_64
.venv/bin/python -m final.toy.plot_comparison --suite hypergrid_4096
```

Comparison metrics JSON paths:
- DNA methods: `{run_dir}/plots/mlp_shifted_linear_reference_1000k/comparison_metrics.json`
- PhyloGFN: `{run_dir}/plots/reward_probability_eval_1000000/comparison_metrics.json`
- Hypergrid: `final/runs/hypergrid_64/plots/epoch_9999/recovery_summary.json`
