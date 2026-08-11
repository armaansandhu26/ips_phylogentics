#!/usr/bin/env bash
# 10-taxa full-model comparison runs (DS1_reduced_10taxa.pickle, log_score_shift=5000)
# Run from repo root.

set -euo pipefail
cd "$(dirname "$0")/../../.."
REPO_ROOT="$PWD"
PYTHON="$REPO_ROOT/.venv/bin/python"

mkdir -p grpo_experiments/comparisons/10taxa/logs
mkdir -p grpo_experiments/comparisons/10taxa/health
mkdir -p grpo_experiments/comparisons/10taxa/status

# Foreground (logs in terminal):
#
# Learned reverse on GPU 1:
# CUDA_VISIBLE_DEVICES=1 "$PYTHON" -u \
#   grpo_experiments/scripts/run_10taxa_full_model_comparison_e2e.py \
#   --method learned_reverse --cuda-device 1
#
# GFlowNet on GPU 0 (run in parallel or after learned reverse):
# CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u \
#   grpo_experiments/scripts/run_10taxa_full_model_comparison_e2e.py \
#   --method gflownet --cuda-device 0

echo "See RUN_COMMANDS.sh comments for foreground commands."
