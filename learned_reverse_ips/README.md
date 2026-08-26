# Learned-reverse IPS-GRPO

Learned reverse proposal + forward PPO for phylogenetic tree sampling.

This folder mirrors the clean layout of `og_code/` (original PhyloGFN TB training)
while reusing the shared PhyloGFN stack at repo-root `src/` and GRPO primitives
in `grpo_experiments/core/`.

Training always uses:
- the **MLP reverse policy** (per-step learned q_φ)
- the **full model** (tree topology + categorical branch lengths)

## Algorithm

Each on-policy step uses importance weights

```
weight(τ) = R(x) · q_φ(τ | x) / P_F(τ)
```

Forward advantages are computed with `q_φ` frozen, then the forward policy is
updated with PPO. After that, `q_φ` is fit by maximum likelihood on the same
batch.

## Quick start

From the repository root:

```bash
python -m learned_reverse_ips.train \
  --reward-target shifted_linear \
  --on-policy-batch-size 4096

python -m learned_reverse_ips.train \
  --reward-target shifted_linear \
  --run-name learned_reverse_10taxa \
  --cfg-path src/configs/benchmark_dna_cfgs/discrete_branch_lengths/cfg_0.001binsize_50bins_temperature_anneal_0.4_10taxa_shift5000.yaml \
  --dataset-path dataset/benchmark_datasets/DS1_full.pickle \
  --on-policy-batch-size 4096
```

Runs are written to `learned_reverse_ips/experiments/<timestamp>_learned_reverse_ips_grpo/`.

Example presets live in `experiment_configs/`.

## Compare with

| Method | Entry point |
|--------|-------------|
| PhyloGFN TB (original) | `python og_code/train.py <cfg> <dataset> <output>` |
| IPS-GRPO (count/exact) | `python -m grpo_experiments.ips_grpo.train` |
| Learned-reverse IPS | `python -m learned_reverse_ips.train` |

## Layout

| File | Role |
|------|------|
| `train.py` | CLI entry |
| `config.py` | Experiment dataclass + argparse |
| `runner.py` | Training loop |
| `mlp_policy.py` | Per-step MLP reverse policy |
| `advantages.py` | Log-weight advantages + running normalizer |
| `checkpoint.py` | Save/load paired forward + reverse state |

Shared dependencies (not duplicated here):

- `src/` — PhyloGFN environment, models, rollout
- `grpo_experiments/core/` — GRPO trainer and on-policy buffer
- `grpo_experiments/ips_grpo/config.py` — base IPS experiment config

## Key flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--reward-target` | `likelihood` | `likelihood` or `shifted_linear` (og_code shift ablation) |
| `--reverse-lr` | `1e-3` | Reverse proposal learning rate |
| `--reverse-train-epochs` | `4` | MLE epochs per forward step |
| `--reverse-hidden-size` | `128` | MLP hidden size |
| `--reverse-num-layers` | `2` | MLP trunk depth |
| `--advantage-normalization` | `running` | `batch` or EMA `running` scale |

## Artifacts per run

- `final_checkpoint.pt` — forward PhyloGFN policy
- `learned_reverse_state.pt` — reverse policy + optimizer + normalizer
- `reverse_catalog.json` — run metadata
- `metrics.jsonl`, `epoch_summaries.json` — training logs

### Post-training pipeline (on by default)

After training completes, the runner automatically:

1. Plots training curves → `plots/training_curves.png`
2. Samples 10k terminal trees → `sampled_full_diagnostics_10000.npz`
3. Plots model probability vs reward → `plots/sampling/model_probability_vs_reward.png`

Disable with `--skip-post-train`. Re-run evaluation on an existing run:

```bash
python -m learned_reverse_ips.post_train --run-dir learned_reverse_ips/experiments/<run_dir>
```

Options:

| Flag | Default | Meaning |
|------|---------|---------|
| `--skip-post-train` | off | Skip the post-training pipeline |
| `--post-train-sample-size` | `10000` | Trees to sample after training |
| `--post-train-sample-batch-size` | on-policy B | Sampling batch size |

Note: sampling diagnostics currently require `--reward-target shifted_linear`.
