#!/usr/bin/env bash
# Phase 1b — Panel F replay anchors (r32, r128) + pfloor carry-forward (0.002, 0.005).
# Default: topology only. Set OUTCOMES=topo,sig to run both.
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
GPU_TOPO="${GPU_TOPO:-cuda:1}"
GPU_SIG="${GPU_SIG:-cuda:0}"
OUTCOMES="${OUTCOMES:-topo}"
CHUNK=512
LR=1e-4
ENT_DEFAULT=0.01
PFLOOR_DEFAULT=0.05
WARMSTART=64

FRESH=448
REPLAY=64
BUFFER=512

mkdir -p "${AB_ROOT}/logs" "${AB_ROOT}/sig" "${AB_ROOT}/topo"
MANIFEST="${AB_ROOT}/manifest_phase1b.json"
echo '{"phase":"1b","runs":[]}' > "${MANIFEST}"

append_manifest() {
  local id="$1" outcome="$2" run_dir="$3" replay="$4"
  "$PY" - <<PY
import json
from pathlib import Path
p = Path("${MANIFEST}")
data = json.loads(p.read_text())
data["runs"].append({
    "id": "${id}",
    "outcome": "${outcome}",
    "run_dir": "${run_dir}",
    "replay": int("${replay}"),
})
p.write_text(json.dumps(data, indent=2) + "\n")
PY
}

set_replay_preset() {
  local preset="$1"
  case "$preset" in
    r32)
      FRESH=480
      REPLAY=32
      BUFFER=256
      ;;
    r128)
      FRESH=384
      REPLAY=128
      BUFFER=2048
      ;;
    *)
      echo "Unknown replay preset: ${preset}" >&2
      exit 1
      ;;
  esac
}

run_phylgfn() {
  local outcome="$1" gpu="$2" id="$3" replay_tag="$4"
  local out="${AB_ROOT}/${outcome}"
  local ol="$([ "$outcome" = sig ] && echo signature || echo topology)"
  mkdir -p "${out}"
  echo "[${outcome}] ${id} replay=${REPLAY} on ${gpu}"
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
  append_manifest "$id" "$outcome" "$run_dir" "$REPLAY"
}

run_hyb_grpo() {
  local outcome="$1" gpu="$2" id="$3" replay_tag="$4" entropy="${5:-$ENT_DEFAULT}"
  local out="${AB_ROOT}/${outcome}"
  local ol="$([ "$outcome" = sig ] && echo signature || echo topology)"
  echo "[${outcome}] ${id} replay=${REPLAY} ent=${entropy} on ${gpu}"
  "$PY" -m grpo_experiments.hybrid_grpo.train \
    --run-name "${id}" --device "$gpu" \
    --dataset "$DS" --cfg "$CFG" --output "$out" --seed "$SEED" \
    --resample-rounds "$ROUNDS" --update-cycles 1 \
    --fresh-buffer-size "$FRESH" --replay-sample-size "$REPLAY" \
    --replay-warmstart-samples "$WARMSTART" --best-tree-buffer-size "$BUFFER" \
    --rollout-chunk-size "$CHUNK" --outcome-level "$ol" \
    --print-every 100 --no-log-trajectories \
    --grpo-lr "$LR" --entropy-coef "$entropy" \
    > "${AB_ROOT}/logs/${outcome}_${id}.log" 2>&1
  local run_dir
  run_dir=$(grep -oP 'saved to: \K.*' "${AB_ROOT}/logs/${outcome}_${id}.log" | tail -1)
  append_manifest "$id" "$outcome" "$run_dir" "$REPLAY"
}

run_hyb_ips() {
  local outcome="$1" gpu="$2" id="$3" replay_tag="$4" pfloor="${5:-$PFLOOR_DEFAULT}" entropy="${6:-$ENT_DEFAULT}"
  local out="${AB_ROOT}/${outcome}"
  local ol="$([ "$outcome" = sig ] && echo signature || echo topology)"
  echo "[${outcome}] ${id} replay=${REPLAY} pfloor=${pfloor} ent=${entropy} on ${gpu}"
  "$PY" -m grpo_experiments.hybrid_ips_grpo.train \
    --run-name "${id}" --device "$gpu" \
    --dataset "$DS" --cfg "$CFG" --output "$out" --seed "$SEED" \
    --resample-rounds "$ROUNDS" --update-cycles 1 \
    --fresh-buffer-size "$FRESH" --replay-sample-size "$REPLAY" \
    --replay-warmstart-samples "$WARMSTART" --best-tree-buffer-size "$BUFFER" \
    --rollout-chunk-size "$CHUNK" --outcome-level "$ol" \
    --print-every 100 --no-log-trajectories \
    --grpo-lr "$LR" --entropy-coef "$entropy" --ips-prob-floor "$pfloor" \
    > "${AB_ROOT}/logs/${outcome}_${id}.log" 2>&1
  local run_dir
  run_dir=$(grep -oP 'saved to: \K.*' "${AB_ROOT}/logs/${outcome}_${id}.log" | tail -1)
  append_manifest "$id" "$outcome" "$run_dir" "$REPLAY"
}

run_panel_f() {
  local outcome="$1" gpu="$2"
  local preset id_suffix

  for preset in r32 r128; do
    set_replay_preset "$preset"
    id_suffix="${preset#r}"

    run_phylgfn "$outcome" "$gpu" "ablation_phylgfn_${preset}" "$preset"
    run_hyb_grpo "$outcome" "$gpu" "ablation_hyb_grpo_${preset}" "$preset"
    run_hyb_ips "$outcome" "$gpu" "ablation_hyb_ips_${preset}" "$preset"
    run_hyb_ips "$outcome" "$gpu" "ablation_hyb_ips_pfloor_002_${preset}" "$preset" 0.002
    run_hyb_ips "$outcome" "$gpu" "ablation_hyb_ips_pfloor_005_${preset}" "$preset" 0.005
  done
}

IFS=',' read -ra OUTCOME_LIST <<< "${OUTCOMES}"
PIDS=()
for outcome in "${OUTCOME_LIST[@]}"; do
  outcome="$(echo "$outcome" | xargs)"
  case "$outcome" in
    topo)
      gpu="$GPU_TOPO"
      ;;
    sig)
      gpu="$GPU_SIG"
      ;;
    *)
      echo "Unknown outcome: ${outcome} (use topo and/or sig)" >&2
      exit 1
      ;;
  esac
  echo "Phase 1b queue: outcome=${outcome} gpu=${gpu} rounds=${ROUNDS}"
  run_panel_f "$outcome" "$gpu" &
  PIDS+=($!)
done

for pid in "${PIDS[@]}"; do
  wait "$pid"
done

echo "Phase 1b training done (${#OUTCOME_LIST[@]} outcome(s), 10 runs each)."
echo "Manifest: ${MANIFEST}"
echo "Eval: .venv/bin/python -m grpo_experiments.scripts.eval_ips_replay_ablation_phase1b"
