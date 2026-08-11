#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$RUN_DIR/../../../.." && pwd)"
PLOT_DIR="$RUN_DIR/plots/reward_probability_eval_1000000"
COMP_DIR="$REPO_ROOT/grpo_experiments/comparisons/27taxa/gflownet"
PID="${1:-1731474}"

echo "Waiting for sampling process $PID to finish..."
while kill -0 "$PID" 2>/dev/null; do
  sleep 60
done

echo "Sampling finished. Copying plots to $COMP_DIR"
mkdir -p "$COMP_DIR"
cp "$PLOT_DIR/og_gflownet_model_probability_vs_reward.png" \
  "$COMP_DIR/model_probability_vs_reward.png"
cp "$PLOT_DIR/og_gflownet_partition_calibrated_probability_vs_reward.png" \
  "$COMP_DIR/partition_calibrated_probability_vs_reward.png"
cp "$PLOT_DIR/comparison_metrics.json" \
  "$COMP_DIR/comparison_metrics.json"
echo "Done. Wrote:"
ls -la "$COMP_DIR"
