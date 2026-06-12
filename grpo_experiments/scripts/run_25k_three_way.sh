#!/usr/bin/env bash
# Launch 25k-step comparison:
#   PhyloGFN — original repo training loop (TrainingDataLoader + TB)
#   GRPO / IPS-GRPO — hybrid replay + policy-IS (grpo_experiments)
#
# PhyloGFN only overrides batch size, step count, and outcome_level vs the yaml
# defaults; everything else stays the PhyloGFN repo setup.
#
# Usage:
#   ./grpo_experiments/scripts/run_25k_three_way.sh
#   DEVICE_PHYLGFN=cuda:2 DEVICE_GRPO=cuda:0 DEVICE_IPS=cuda:3 ./grpo_experiments/scripts/run_25k_three_way.sh
#
#   OUTCOME_LEVEL=topology OUTPUT_ROOT=grpo_experiments/runs/hybrid_25k_track_topo ./grpo_experiments/scripts/run_25k_three_way.sh
#
# After training, three-way plot + final sampling (always 3 methods):
#   OUTPUT_ROOT=grpo_experiments/runs/hybrid_25k_track_topo ./grpo_experiments/scripts/eval_25k_trajectories.sh

set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="$(pwd)"
PY="${ROOT}/.venv/bin/python"
OUTPUT_ROOT="${OUTPUT_ROOT:-grpo_experiments/runs/hybrid_25k_track}"
OUTCOME_LEVEL="${OUTCOME_LEVEL:-signature}"
OUTCOME_TAG="${OUTCOME_TAG:-$([ "$OUTCOME_LEVEL" = topology ] && echo topo || echo sig)}"
ROUNDS="${ROUNDS:-25000}"
SEED="${SEED:-0}"
DS="${DS:-dataset/benchmark_datasets/DS1_reduced.pickle}"
CFG="${CFG:-src/configs/benchmark_dna_cfgs/discrete_branch_lengths/cfg_0.001binsize_50bins_temperature_anneal_0.4.yaml}"
DEVICE_PHYLGFN="${DEVICE_PHYLGFN:-cuda:2}"
DEVICE_GRPO="${DEVICE_GRPO:-cuda:0}"
DEVICE_IPS="${DEVICE_IPS:-cuda:3}"
FRESH="${FRESH:-480}"
REPLAY="${REPLAY:-32}"
WARMSTART="${WARMSTART:-64}"
CHUNK="${CHUNK:-512}"
ENTROPY="${ENTROPY:-0.01}"
PRINT_EVERY="${PRINT_EVERY:-100}"
TRAJ_FLUSH="${TRAJ_FLUSH:-20}"
# Match hybrid total batch 512 = fresh + replay
PHYLGFN_ON_POLICY="${PHYLGFN_ON_POLICY:-480}"
PHYLGFN_REPLAY="${PHYLGFN_REPLAY:-32}"
PHYLGFN_REPLAY_BUFFER="${PHYLGFN_REPLAY_BUFFER:-256}"

mkdir -p "${OUTPUT_ROOT}/logs"

hybrid_common=(
  --dataset "$DS"
  --cfg "$CFG"
  --output "$OUTPUT_ROOT"
  --seed "$SEED"
  --resample-rounds "$ROUNDS"
  --update-cycles 1
  --fresh-buffer-size "$FRESH"
  --replay-sample-size "$REPLAY"
  --replay-warmstart-samples "$WARMSTART"
  --best-tree-buffer-size "$PHYLGFN_REPLAY_BUFFER"
  --rollout-chunk-size "$CHUNK"
  --outcome-level "$OUTCOME_LEVEL"
  --print-every "$PRINT_EVERY"
  --trajectory-flush-every "$TRAJ_FLUSH"
  --grpo-lr 1e-4
  --entropy-coef "$ENTROPY"
)

echo "output_root=${OUTPUT_ROOT}  rounds=${ROUNDS}  outcome_level=${OUTCOME_LEVEL}"
echo "devices: phylgfn=${DEVICE_PHYLGFN} grpo=${DEVICE_GRPO} ips=${DEVICE_IPS}"
echo "phylgfn: standard TrainingDataLoader  batch=${PHYLGFN_ON_POLICY}+${PHYLGFN_REPLAY}=$((PHYLGFN_ON_POLICY + PHYLGFN_REPLAY))  steps=${ROUNDS}"

nohup "$PY" -m grpo_experiments.train \
  --method phylgfn \
  --run-name "ds1_${OUTCOME_TAG}_r${ROUNDS}_phylgfn" \
  --device "$DEVICE_PHYLGFN" \
  --dataset "$DS" \
  --cfg "$CFG" \
  --output "$OUTPUT_ROOT" \
  --seed "$SEED" \
  --epochs "$ROUNDS" \
  --steps-per-epoch 1 \
  --on-policy-batch-size "$PHYLGFN_ON_POLICY" \
  --replay-batch-size "$PHYLGFN_REPLAY" \
  --replay-buffer-size "$PHYLGFN_REPLAY_BUFFER" \
  --outcome-level "$OUTCOME_LEVEL" \
  --print-every "$PRINT_EVERY" \
  > "${OUTPUT_ROOT}/logs/phylgfn_gpu.log" 2>&1 &
echo "started phylgfn (standard TB) pid=$!"

nohup "$PY" -m grpo_experiments.hybrid_grpo.train \
  --run-name "ds1_${OUTCOME_TAG}_r${ROUNDS}_grpo" \
  --device "$DEVICE_GRPO" \
  "${hybrid_common[@]}" \
  > "${OUTPUT_ROOT}/logs/grpo_gpu.log" 2>&1 &
echo "started grpo (hybrid) pid=$!"

nohup "$PY" -m grpo_experiments.hybrid_ips_grpo.train \
  --run-name "ds1_${OUTCOME_TAG}_r${ROUNDS}_ips" \
  --device "$DEVICE_IPS" \
  --ips-prob-floor 0.05 \
  "${hybrid_common[@]}" \
  > "${OUTPUT_ROOT}/logs/ips_gpu.log" 2>&1 &
echo "started ips (hybrid) pid=$!"

echo ""
echo "Monitor: tail -f ${OUTPUT_ROOT}/logs/*.log"
echo "When done, run: OUTPUT_ROOT=${OUTPUT_ROOT} ./grpo_experiments/scripts/eval_25k_trajectories.sh"
