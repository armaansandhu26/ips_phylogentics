#!/usr/bin/env bash
# Sample 100k from epoch-1000 checkpoints and build an early 2-panel comparison.
# Safe to run while full 10k-epoch training continues (uses GPUs 2 and 3 by default).

set -euo pipefail
cd "$(dirname "$0")/../../.."
REPO_ROOT="$PWD"
PYTHON="$REPO_ROOT/.venv/bin/python"

LR_RUN="$REPO_ROOT/grpo_experiments/learned_reverse_runs/20260803_124837_learned_reverse_10taxa_mlp_shifted_linear_b4096_learned_reverse_ips_grpo"
GFN_RUN="$REPO_ROOT/og_code/experiments/full_model/20260803_124841_phylgfn_logreward_10taxa_g4096_1m_full_replay_op3277_r819_rb4096"
OUT="$REPO_ROOT/grpo_experiments/comparisons/10taxa/early_epoch1000_100k"
mkdir -p "$OUT"

LR_CKPT="checkpoint_epoch0999.pt"
GFN_CKPT="checkpoints/checkpoint_000999.pt"

echo "==> learned reverse: sample 100k from $LR_CKPT"
CUDA_VISIBLE_DEVICES=2 "$PYTHON" -u \
  grpo_experiments/scripts/sample_learned_reverse_full_diagnostics.py \
  --checkpoint "$LR_RUN" \
  --checkpoint-name "$LR_CKPT" \
  --reverse-state "$LR_RUN/learned_reverse_epoch0999.pt" \
  -n 100000 \
  --batch-size 4096 \
  --seed 0 \
  --device cuda:0 \
  --output "$OUT/learned_reverse_samples_100k.npz"

echo "==> learned reverse: plot"
CUDA_VISIBLE_DEVICES=2 "$PYTHON" \
  grpo_experiments/scripts/plot_full_checkpoint_vs_reward_reference.py \
  --samples "$OUT/learned_reverse_samples_100k.npz" \
  --output-dir "$OUT/learned_reverse" \
  --plot-method learned-reverse \
  --shared-reference

echo "==> gflownet: sample+plot 100k from $GFN_CKPT"
CUDA_VISIBLE_DEVICES=3 "$PYTHON" -u \
  grpo_experiments/scripts/eval_og_gflownet_reward_probability.py \
  --run-dir "$GFN_RUN" \
  --checkpoint "$GFN_CKPT" \
  --dataset dataset/benchmark_datasets/DS1_reduced_10taxa.pickle \
  -n 100000 \
  --batch-size 4096 \
  --seed 0 \
  --device cuda:0 \
  --output-dir "$OUT/gflownet" \
  --shared-reference

echo "==> 2-panel grid"
"$PYTHON" grpo_experiments/scripts/plot_10taxa_early_comparison_grid.py \
  --comparison-dir "$OUT"

echo "==> union-catalog ideal sampling grid"
"$PYTHON" grpo_experiments/scripts/plot_10taxa_early_empirical_ideal_grid.py \
  --comparison-dir "$OUT"

echo "Done. See $OUT/sampling_comparison_grid.png and $OUT/union_ideal_sampling_grid.png"
