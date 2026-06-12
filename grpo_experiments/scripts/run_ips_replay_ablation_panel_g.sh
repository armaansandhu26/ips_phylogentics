#!/usr/bin/env bash
# Panel G — replay annealing 128→32 (hyb_grpo + hyb_ips), topo only.
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
GPU="${GPU:-cuda:1}"
OUTCOME="${OUTCOME:-topo}"
CHUNK=512
LR=1e-4
ENT=0.01
PFLOOR=0.05
WARMSTART=64
ANNEAL_START=128
ANNEAL_END=32
TOTAL_BATCH=512
BUFFER=2048

mkdir -p "${AB_ROOT}/logs" "${AB_ROOT}/${OUTCOME}"
MANIFEST="${AB_ROOT}/manifest_panel_g.json"
echo '{"phase":"panel_g","runs":[]}' > "${MANIFEST}"

append_manifest() {
  local id="$1" outcome="$2" run_dir="$3"
  "$PY" - <<PY
import json
from pathlib import Path
p = Path("${MANIFEST}")
data = json.loads(p.read_text())
data["runs"].append({
    "id": "${id}",
    "outcome": "${outcome}",
    "run_dir": "${run_dir}",
    "replay": "anneal_${ANNEAL_START}to${ANNEAL_END}",
})
p.write_text(json.dumps(data, indent=2) + "\n")
PY
}

run_hyb_grpo_anneal() {
  local id="$1"
  local out="${AB_ROOT}/${OUTCOME}"
  local ol="$([ "$OUTCOME" = sig ] && echo signature || echo topology)"
  echo "[${OUTCOME}] ${id} replay_anneal=${ANNEAL_START}->${ANNEAL_END} on ${GPU}"
  "$PY" -m grpo_experiments.hybrid_grpo.train \
    --run-name "${id}" --device "$GPU" \
    --dataset "$DS" --cfg "$CFG" --output "$out" --seed "$SEED" \
    --resample-rounds "$ROUNDS" --update-cycles 1 \
    --fresh-buffer-size 384 --replay-sample-size "$ANNEAL_START" \
    --replay-warmstart-samples "$WARMSTART" --best-tree-buffer-size "$BUFFER" \
    --replay-anneal-start "$ANNEAL_START" --replay-anneal-end "$ANNEAL_END" \
    --replay-anneal-total-batch "$TOTAL_BATCH" \
    --rollout-chunk-size "$CHUNK" --outcome-level "$ol" \
    --print-every 100 --no-log-trajectories \
    --grpo-lr "$LR" --entropy-coef "$ENT" \
    > "${AB_ROOT}/logs/${OUTCOME}_${id}.log" 2>&1
  local run_dir
  run_dir=$(grep -oP 'saved to: \K.*' "${AB_ROOT}/logs/${OUTCOME}_${id}.log" | tail -1)
  append_manifest "$id" "$OUTCOME" "$run_dir"
}

run_hyb_ips_anneal() {
  local id="$1"
  local out="${AB_ROOT}/${OUTCOME}"
  local ol="$([ "$OUTCOME" = sig ] && echo signature || echo topology)"
  echo "[${OUTCOME}] ${id} replay_anneal=${ANNEAL_START}->${ANNEAL_END} pfloor=${PFLOOR} on ${GPU}"
  "$PY" -m grpo_experiments.hybrid_ips_grpo.train \
    --run-name "${id}" --device "$GPU" \
    --dataset "$DS" --cfg "$CFG" --output "$out" --seed "$SEED" \
    --resample-rounds "$ROUNDS" --update-cycles 1 \
    --fresh-buffer-size 384 --replay-sample-size "$ANNEAL_START" \
    --replay-warmstart-samples "$WARMSTART" --best-tree-buffer-size "$BUFFER" \
    --replay-anneal-start "$ANNEAL_START" --replay-anneal-end "$ANNEAL_END" \
    --replay-anneal-total-batch "$TOTAL_BATCH" \
    --rollout-chunk-size "$CHUNK" --outcome-level "$ol" \
    --print-every 100 --no-log-trajectories \
    --grpo-lr "$LR" --entropy-coef "$ENT" --ips-prob-floor "$PFLOOR" \
    > "${AB_ROOT}/logs/${OUTCOME}_${id}.log" 2>&1
  local run_dir
  run_dir=$(grep -oP 'saved to: \K.*' "${AB_ROOT}/logs/${OUTCOME}_${id}.log" | tail -1)
  append_manifest "$id" "$OUTCOME" "$run_dir"
}

echo "Panel G start: outcome=${OUTCOME} gpu=${GPU} rounds=${ROUNDS}"
run_hyb_grpo_anneal "ablation_hyb_grpo_replay_anneal_128to32"
run_hyb_ips_anneal "ablation_hyb_ips_replay_anneal_128to32"
echo "Panel G training done (2 runs)."
echo "Manifest: ${MANIFEST}"
echo "Eval: .venv/bin/python -m grpo_experiments.scripts.eval_ips_replay_ablation_panel_g"
