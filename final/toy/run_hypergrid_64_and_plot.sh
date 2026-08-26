#!/usr/bin/env bash
# Train GRPO + count IPS on hypergrid_64, then plot when both finish.
# Enable wandb: FINAL_WANDB=1 WANDB_PROJECT=phylogfn-final bash final/toy/run_hypergrid_64_and_plot.sh
set -euo pipefail
ROOT="/home/armaan/phylogfn"
cd "$ROOT"
mkdir -p final/runs/hypergrid_64/grpo final/runs/hypergrid_64/count_ips final/runs/hypergrid_64/plots

GRPO_LOG="final/runs/hypergrid_64/grpo/pipeline.log"
IPS_LOG="final/runs/hypergrid_64/count_ips/pipeline.log"

WANDB_ARGS=()
PLOT_WANDB_ARGS=()
if [[ "${FINAL_WANDB:-0}" =~ ^(1|true|yes|on)$ ]]; then
  WANDB_ARGS=(--wandb --wandb-project "${WANDB_PROJECT:-phylogfn-final}" --wandb-group hypergrid_64)
  PLOT_WANDB_ARGS=(--wandb --wandb-project "${WANDB_PROJECT:-phylogfn-final}" --wandb-group hypergrid_64)
  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    WANDB_ARGS+=(--wandb-entity "$WANDB_ENTITY")
    PLOT_WANDB_ARGS+=(--wandb-entity "$WANDB_ENTITY")
  fi
fi

CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m final hypergrid-pipeline \
  --suite hypergrid_64 --method grpo --device cuda:0 \
  "${WANDB_ARGS[@]}" \
  > "$GRPO_LOG" 2>&1 &
GRPO_PID=$!

CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u -m final hypergrid-pipeline \
  --suite hypergrid_64 --method count_ips --device cuda:0 \
  "${WANDB_ARGS[@]}" \
  > "$IPS_LOG" 2>&1 &
IPS_PID=$!

echo "grpo pid=$GRPO_PID log=$GRPO_LOG"
echo "count_ips pid=$IPS_PID log=$IPS_LOG"

wait "$GRPO_PID"
echo "grpo finished"
wait "$IPS_PID"
echo "count_ips finished"

.venv/bin/python -m final.toy.plot_comparison --suite hypergrid_64 "${PLOT_WANDB_ARGS[@]}"
echo "plots written to final/runs/hypergrid_64/plots/"
