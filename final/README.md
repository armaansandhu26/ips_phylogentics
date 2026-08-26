# Final paper experiment harness

Clean, unified setup for the four methods in the paper:

| Method | Implementation | Reward |
|--------|----------------|--------|
| **GRPO** | `grpo_experiments.train --method grpo` | shifted log score |
| **Count IPS** | `grpo_experiments.ips_grpo.train` | shifted log score + count IPS |
| **Learned-reverse IPS** | `learned_reverse_ips.train` | shifted linear target + learned q_φ |
| **PhyloGFN** | `og_code/train.py` (default) | shifted log score (TB) |

Paper-faithful PhyloGFN (upstream ICLR 2024 reward) is available via `"backend": "paper"` in a suite config.

## Layout

```
final/
├── configs/suites/          # one JSON per experiment setting
├── runs/<suite_id>/         # all run outputs
│   ├── suite.json           # links methods -> run dirs
│   ├── grpo/
│   ├── count_ips/
│   ├── learned_reverse/
│   └── phylgfn/
├── paper/
│   ├── PAPER_REVISION_PRIORITIES.md  # reviewer feedback tracker
│   ├── README.md
│   ├── AUG25_COMPLETED_RUNS.md
│   ├── P1_RESULTS.md
│   ├── manifest.json
│   ├── manifest.json
│   └── table2.*, figure6_*.png
├── results/<suite_id>/      # aggregated tables
│   ├── comparison_table.csv
│   └── comparison_table.json
├── pipeline.py              # train + sample + plots (one method)
├── run_suite.py             # all four methods for one suite
└── aggregate.py             # rebuild comparison tables
```

## Quick start

List suites:

```bash
cd /home/armaan/phylogfn
python -m final list
```

Run one method end-to-end (train → 1M sample → plots):

```bash
CUDA_VISIBLE_DEVICES=0 python -m final pipeline \
  --suite 5taxa_noreplay \
  --method grpo \
  --cuda-device 0
```

Run all four methods for a suite:

```bash
CUDA_VISIBLE_DEVICES=0 python -m final run_suite \
  --suite 5taxa_noreplay \
  --cuda-device 0
```

Re-sample / replot an existing run:

```bash
python -m final pipeline \
  --suite 27taxa_noreplay \
  --method learned_reverse \
  --skip-train \
  --run-dir final/runs/27taxa_noreplay/learned_reverse/<timestamp>_...
```

Rebuild comparison table:

```bash
python -m final aggregate --suite 5taxa_noreplay
```

## Suite configs

| Suite | Taxa | Replay | Epochs | Batch | Shift |
|-------|------|--------|--------|-------|-------|
| `5taxa_noreplay` | 5 | no | 10k | 4096 | 3600 |
| `10taxa_noreplay` | 10 | no | 10k | 4096 | 5000 |
| `27taxa_noreplay` | 27 | no | 32k | 1024 | 12000 |
| `27taxa_replay` | 27 | 819+205 | 32k | 819+205 | 12000 |

Each suite JSON defines shared training/sampling settings plus per-method cfg paths.

## Run artifacts (per method)

After a full pipeline, each run directory contains:

| Artifact | GRPO / IPS | Learned-reverse | PhyloGFN |
|----------|------------|-----------------|----------|
| Training metrics | `metrics.jsonl` | `metrics.jsonl` | loss plots + checkpoints |
| Checkpoint | `final_checkpoint.pt` | `final_checkpoint.pt` + `learned_reverse_state.pt` | `checkpoints/checkpoint_*.pt` |
| 1M samples | `sampled_full_diagnostics_1000000.npz` | same | `plots/reward_probability_eval_1000000/*.npz` |
| Main plot | `plots/.../model_probability_vs_reward.png` | same + `training_curves.png` | `og_gflownet_*.png` |
| Metrics | `comparison_metrics.json` | same | same |
| Manifest | `final_manifest.json` | same | same |

Suite-level manifest: `final/runs/<suite_id>/suite.json`

## Dependencies (kept outside `final/`)

This harness orchestrates existing code — it does not duplicate trainers:

```
src/                          # shared env, models, configs
grpo_experiments/core/        # GRPO trainer
grpo_experiments/ips_grpo/    # count IPS
learned_reverse_ips/          # learned-reverse IPS
og_code/                      # shifted-reward PhyloGFN
phylogfn_paper/               # optional paper-faithful PhyloGFN
dataset/                      # DS1 pickles
```

See `MIGRATION.md` for the plan to archive legacy folders after final runs are complete.

## Run GRPO / count IPS

Train on the 4096×4096 Hyper-Grid (TB paper reward, 4 corner modes). Target distribution is `p(x) ∝ R(x)`; evaluation reports **L1 distance** (and TV = L1/2) to that target.

```bash
# GRPO
python -m final hypergrid-pipeline --suite hypergrid_4096 --method grpo

# Count IPS
python -m final hypergrid-pipeline --suite hypergrid_4096 --method count_ips

# Or directly:
python -m grpo_experiments.hypergrid.train_grpo --dataset final/datasets/hypergrid_4096
python -m grpo_experiments.hypergrid.train_count_ips --dataset final/datasets/hypergrid_4096
```

Suite config: `final/configs/suites/hypergrid_4096.json`. Runs go to `final/runs/hypergrid_4096/{grpo,count_ips}/`.

Each run writes `metrics.jsonl`, `eval_metrics.json` (L1 vs target), and `sampled_terminals_*.npz`.

**wandb (hypergrid):** pass `--wandb` to stream live metrics and upload plots:

```bash
python -m final hypergrid-pipeline \
  --suite hypergrid_64 \
  --method grpo \
  --device cuda:0 \
  --wandb \
  --wandb-project phylogfn-final \
  --wandb-entity your-team
```

Both methods share `--wandb-group hypergrid_64`. Live charts: `train/*` scalars every step, `eval/*` every eval epoch. PNGs: `training_l1_live.png` (updates each eval), `training_diagnostics.png` (final). Comparison plots upload to a `{suite}_comparison` run via `plot_comparison --wandb`.

Or set `FINAL_WANDB=1` when using `final/toy/run_hypergrid_64_and_plot.sh`.

Precomputed reward grid (`final/datasets/hypergrid_4096/`):

```
final/datasets/hypergrid_4096/
├── rewards.npy              # 4096×4096 float32 reward grid
├── target_distribution.npz  # probs ∝ rewards
├── meta.json
└── reward_paper_indicator_4_modes.png
```

Build or regenerate:

```bash
python -m final.toy.build_dataset
python -m final.toy.plot_reward
```

Reward (TB paper): `R(x) = R₀ + R₁ ∏ I(|x/(H-1)-0.5| ∈ (0.25,0.5]) + R₂ ∏ I(... ∈ (0.3,0.4))` with R₀=0.1, R₁=0.5, R₂=2.0.

## tmux example (27 taxa, one method)

```bash
tmux new-session -d -s final_27t_grpo \
  "cd /home/armaan/phylogfn && CUDA_VISIBLE_DEVICES=0 \
   /home/armaan/phylogfn/.venv/bin/python -u -m final pipeline \
   --suite 27taxa_noreplay --method grpo --cuda-device 0 \
   2>&1 | tee final/runs/27taxa_noreplay/grpo/pipeline.log"
```

## Notes

- **wandb:** pass `--wandb` to stream training metrics and plots live to [Weights & Biases](https://wandb.ai). Plots are uploaded as each PNG is written during the plot step.

```bash
python -m final pipeline \
  --suite 5taxa_noreplay \
  --method learned_reverse \
  --wandb \
  --wandb-project phylogfn-final \
  --wandb-entity your-team
```

Or enable via env: `FINAL_WANDB=1 WANDB_PROJECT=phylogfn-final`.

- **PhyloGFN disk usage:** upstream trainer saves a checkpoint every epoch. For 27 taxa, patch checkpoint frequency or prune before long runs.
- **Shift alignment:** GRPO-family methods and og_code PhyloGFN use the same `LOG_SCORE_SHIFT` from each suite's yaml. Paper PhyloGFN uses log L directly — not directly comparable without a separate analysis track.
- **Resume:** pass `--resume-from` to `pipeline` for GRPO-family methods; PhyloGFN resume uses `train.py resume`.
