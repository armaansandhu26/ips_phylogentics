# Train and evaluate the paper-faithful 27-taxa PhyloGFN

This runbook trains the official continuous-branch PhyloGFN configuration on
DS1 and evaluates both terminal sampling density and a dense 100,000-trajectory
pathwise diagnostic from the final checkpoint.

The prepared suites use the published 40% setup:

- 27 taxa (DS1)
- continuous branch lengths
- 500 epochs
- 200 updates per epoch
- 64 on-policy and 64 replay trajectories per update
- 12.8 million total training trajectories
- 100,000 post-training forward trajectories
- three independent seeds: 0, 1, and 2

Run all commands from the parent repository root, not from this directory.

## 1. Verify the environment

```bash
cd /path/to/ips_phylogentics

test -x .venv/bin/python

.venv/bin/python -c \
"import torch, ete3, fvcore, iopath, scipy; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The final experiment harness currently expects Python at
`.venv/bin/python`. Do not start training unless CUDA is reported as available.

## 2. Train and evaluate seed 0

```bash
.venv/bin/python -m final pipeline \
  --suite 27taxa_phylogfn_paper_40pct_seed0 \
  --method phylgfn \
  --cuda-device 0 \
  --device cuda:0 \
  --log-file final/phylogfn_paper_seed0.log
```

The pipeline will:

1. Train the official continuous-branch PhyloGFN for 12.8 million trajectories.
2. Select the final checkpoint automatically.
3. Estimate terminal sampling density for 100 states using 200 backward
   trajectories per state.
4. Calculate marginal log-likelihood with 10 repetitions of 1,024 samples.
5. Sample 100,000 forward trajectories.
6. Save forward and backward log-probabilities, rewards, scores, pathwise
   probabilities, metrics, and plots.

## 3. Expected outputs

The timestamped run will be written under:

```text
final/runs/27taxa_phylogfn_paper_40pct_seed0/phylgfn/<timestamped-run>/
```

Important files are:

```text
checkpoints/checkpoint_000499.pt

plots/reward_probability_eval_100000/
├── paper_gflownet_log_probability_vs_log_reward.png
├── paper_gflownet_model_probability_vs_reward.png
├── pathwise_log_probability_vs_log_reward.png
├── model_probability_vs_reward.png
├── paper_gflownet_evaluation.npz
└── comparison_metrics.json
```

The plots have different purposes:

- `paper_gflownet_log_probability_vs_log_reward.png` uses the paper-style
  terminal-density estimate that accounts for multiple trajectories reaching
  the same terminal tree.
- `paper_gflownet_model_probability_vs_reward.png` is the relative linear-scale
  version of the paper-style terminal-density estimate.
- `pathwise_log_probability_vs_log_reward.png` uses all 100,000 forward
  trajectories and plots `log P_F(tau) - log P_B(tau)` against log reward.
- `model_probability_vs_reward.png` is the dense relative linear-scale plot
  analogous to the MIPS probability-versus-reward figure.

For continuous branch lengths, these quantities are probability densities.
The correct target is the unnormalized posterior density, including the
branch-length prior, rather than the shifted-linear reward used by the MIPS
comparison runs.

## 4. Train the other publication seeds

Run these on separate GPUs or nodes when possible:

```bash
# Seed 1
.venv/bin/python -m final pipeline \
  --suite 27taxa_phylogfn_paper_40pct_seed1 \
  --method phylgfn \
  --cuda-device 1 \
  --device cuda:0 \
  --log-file final/phylogfn_paper_seed1.log

# Seed 2
.venv/bin/python -m final pipeline \
  --suite 27taxa_phylogfn_paper_40pct_seed2 \
  --method phylgfn \
  --cuda-device 2 \
  --device cuda:0 \
  --log-file final/phylogfn_paper_seed2.log
```

`--cuda-device N` controls the physical GPU exposed to the subprocess.
`--device cuda:0` is correct because that GPU becomes device zero inside the
subprocess.

Each run uses ten CPU workers. When launching all three concurrently, use
separate nodes or ensure that at least 30 CPU workers and sufficient RAM are
available. The upstream trainer saves one checkpoint per epoch, so confirm that
there is enough disk space for 500 checkpoints per seed.

## 5. Rerun evaluation without retraining

If training completes but sampling or plotting is interrupted, rerun only the
evaluation from the existing timestamped run:

```bash
.venv/bin/python -m final pipeline \
  --suite 27taxa_phylogfn_paper_40pct_seed0 \
  --method phylgfn \
  --skip-train \
  --run-dir /absolute/path/to/the/timestamped-run \
  --cuda-device 0 \
  --device cuda:0
```

Change the suite, run directory, and GPU for seeds 1 and 2.

## 6. Publication reporting

Report the mean and standard deviation across the three seeds for:

- terminal-density Pearson correlation
- pathwise log-density Pearson correlation
- fitted slope and balance-residual standard deviation
- marginal log-likelihood

Treat the dense 100,000-trajectory plot as a pathwise diagnostic. Use the
paper-style backward-importance-sampling estimate for claims about terminal
sampling density. Keep this official continuous-branch reproduction separate
from the matched shifted-reward/no-replay comparison unless both methods are
retrained with the same reward, branch representation, replay setting, and
trajectory budget.
