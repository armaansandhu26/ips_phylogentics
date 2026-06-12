#!/usr/bin/env bash
# Phase 0 + 1: Panel A baselines + Panel C/D ablations @ replay64.
# See grpo_experiments/runs/ips_replay_ablation/README.md

set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="$(pwd)"
PY="${ROOT}/.venv/bin/python"
AB_ROOT="${ROOT}/grpo_experiments/runs/ips_replay_ablation"
DS="${DS:-dataset/benchmark_datasets/DS1_reduced.pickle}"
CFG="${CFG:-src/configs/benchmark_dna_cfgs/discrete_branch_lengths/cfg_0.001binsize_50bins_temperature_anneal_0.4.yaml}"
ROUNDS="${ROUNDS:-1000}"
SEED="${SEED:-0}"
GPU_SIG="${GPU_SIG:-cuda:0}"
GPU_TOPO="${GPU_TOPO:-cuda:1}"
FRESH=448
REPLAY=64
BUFFER=512
CHUNK=512
LR=1e-4
ENT_DEFAULT=0.01
PFLOOR_DEFAULT=0.05

mkdir -p "${AB_ROOT}/logs" "${AB_ROOT}/sig" "${AB_ROOT}/topo"
MANIFEST="${AB_ROOT}/manifest_phase01.json"
echo '{"phase":"0+1","runs":[]}' > "${MANIFEST}"

append_manifest() {
  local id="$1" outcome="$2" run_dir="$3"
  "$PY" - <<PY
import json
from pathlib import Path
p = Path("${MANIFEST}")
data = json.loads(p.read_text())
data["runs"].append({"id": "${id}", "outcome": "${outcome}", "run_dir": "${run_dir}"})
p.write_text(json.dumps(data, indent=2) + "\n")
PY
}

run_phylgfn() {
  local outcome="$1" gpu="$2" id="$3"
  local out="${AB_ROOT}/${outcome}"
  local ol="$([ "$outcome" = sig ] && echo signature || echo topology)"
  mkdir -p "${out}"
  echo "[${outcome}] ${id} on ${gpu}"
  "$PY" -m grpo_experiments.train \
    --method phylgfn \
    --run-name "${id}" \
    --device "$gpu" \
    --dataset "$DS" --cfg "$CFG" \
    --output "$out" --seed "$SEED" \
    --epochs "$ROUNDS" --steps-per-epoch 1 \
    --on-policy-batch-size "$FRESH" --replay-batch-size "$REPLAY" \
    --replay-buffer-size "$BUFFER" \
    --outcome-level "$ol" --print-every 100 \
    > "${AB_ROOT}/logs/${outcome}_${id}.log" 2>&1
  local run_dir
  run_dir=$(grep -oP 'saved to: \K.*' "${AB_ROOT}/logs/${outcome}_${id}.log" | tail -1)
  append_manifest "$id" "$outcome" "$run_dir"
}

run_hyb_grpo() {
  local outcome="$1" gpu="$2" id="$3" entropy="${4:-$ENT_DEFAULT}"
  local out="${AB_ROOT}/${outcome}"
  local ol="$([ "$outcome" = sig ] && echo signature || echo topology)"
  echo "[${outcome}] ${id} ent=${entropy} on ${gpu}"
  "$PY" -m grpo_experiments.hybrid_grpo.train \
    --run-name "${id}" --device "$gpu" \
    --dataset "$DS" --cfg "$CFG" --output "$out" --seed "$SEED" \
    --resample-rounds "$ROUNDS" --update-cycles 1 \
    --fresh-buffer-size "$FRESH" --replay-sample-size "$REPLAY" \
    --replay-warmstart-samples 64 --best-tree-buffer-size "$BUFFER" \
    --rollout-chunk-size "$CHUNK" --outcome-level "$ol" \
    --print-every 100 --no-log-trajectories \
    --grpo-lr "$LR" --entropy-coef "$entropy" \
    > "${AB_ROOT}/logs/${outcome}_${id}.log" 2>&1
  local run_dir
  run_dir=$(grep -oP 'saved to: \K.*' "${AB_ROOT}/logs/${outcome}_${id}.log" | tail -1)
  append_manifest "$id" "$outcome" "$run_dir"
}

run_hyb_ips() {
  local outcome="$1" gpu="$2" id="$3" pfloor="${4:-$PFLOOR_DEFAULT}" entropy="${5:-$ENT_DEFAULT}"
  local out="${AB_ROOT}/${outcome}"
  local ol="$([ "$outcome" = sig ] && echo signature || echo topology)"
  echo "[${outcome}] ${id} pfloor=${pfloor} ent=${entropy} on ${gpu}"
  "$PY" -m grpo_experiments.hybrid_ips_grpo.train \
    --run-name "${id}" --device "$gpu" \
    --dataset "$DS" --cfg "$CFG" --output "$out" --seed "$SEED" \
    --resample-rounds "$ROUNDS" --update-cycles 1 \
    --fresh-buffer-size "$FRESH" --replay-sample-size "$REPLAY" \
    --replay-warmstart-samples 64 --best-tree-buffer-size "$BUFFER" \
    --rollout-chunk-size "$CHUNK" --outcome-level "$ol" \
    --print-every 100 --no-log-trajectories \
    --grpo-lr "$LR" --entropy-coef "$entropy" --ips-prob-floor "$pfloor" \
    > "${AB_ROOT}/logs/${outcome}_${id}.log" 2>&1
  local run_dir
  run_dir=$(grep -oP 'saved to: \K.*' "${AB_ROOT}/logs/${outcome}_${id}.log" | tail -1)
  append_manifest "$id" "$outcome" "$run_dir"
}

run_outcome_queue() {
  local outcome="$1" gpu="$2"
  # Panel A
  run_phylgfn "$outcome" "$gpu" "ablation_phylgfn_r64"
  run_hyb_grpo "$outcome" "$gpu" "ablation_hyb_grpo_r64"
  run_hyb_ips "$outcome" "$gpu" "ablation_hyb_ips_r64"
  # Panel C
  run_hyb_ips "$outcome" "$gpu" "ablation_hyb_ips_pfloor_010" 0.01
  run_hyb_ips "$outcome" "$gpu" "ablation_hyb_ips_pfloor_005" 0.005
  run_hyb_ips "$outcome" "$gpu" "ablation_hyb_ips_pfloor_002" 0.002
  # Panel D1
  run_hyb_ips "$outcome" "$gpu" "ablation_hyb_ips_ent_000" "$PFLOOR_DEFAULT" 0
  run_hyb_ips "$outcome" "$gpu" "ablation_hyb_ips_ent_001" "$PFLOOR_DEFAULT" 0.001
  # Panel D2
  run_hyb_grpo "$outcome" "$gpu" "ablation_hyb_grpo_ent_000" 0
  run_hyb_grpo "$outcome" "$gpu" "ablation_hyb_grpo_ent_001" 0.001
}

echo "Phase 0+1 start: sig=${GPU_SIG} topo=${GPU_TOPO} rounds=${ROUNDS}"
run_outcome_queue sig "$GPU_SIG" &
PID_SIG=$!
run_outcome_queue topo "$GPU_TOPO" &
PID_TOPO=$!
wait "$PID_SIG" "$PID_TOPO"
echo "Phase 0+1 training done. Manifest: ${MANIFEST}"
