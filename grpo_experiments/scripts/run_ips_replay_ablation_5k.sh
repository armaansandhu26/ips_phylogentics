#!/usr/bin/env bash
# 5k validation — linear IPS winners @ topo r64 (scheduled after Panel J).
# See grpo_experiments/runs/ips_replay_ablation/README.md

set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="$(pwd)"
PY="${ROOT}/.venv/bin/python"
AB_ROOT="${ROOT}/grpo_experiments/runs/ips_replay_ablation"
DS="${DS:-dataset/benchmark_datasets/DS1_reduced.pickle}"
CFG="${CFG:-src/configs/benchmark_dna_cfgs/discrete_branch_lengths/cfg_0.001binsize_50bins_temperature_anneal_0.4.yaml}"
ROUNDS="${ROUNDS:-5000}"
SEED="${SEED:-0}"
GPU="${GPU:-cuda:1}"
OUTCOME="${OUTCOME:-topo}"
CHUNK=512
LR=1e-4
ENT=0.01
FRESH=448
REPLAY=64
BUFFER=512
WARMSTART=64

mkdir -p "${AB_ROOT}/logs" "${AB_ROOT}/${OUTCOME}"
MANIFEST="${AB_ROOT}/manifest_5k.json"
echo '{"phase":"5k","runs":[]}' > "${MANIFEST}"

append_manifest() {
  local id="$1" outcome="$2" run_dir="$3" pfloor="${4:-}"
  "$PY" - <<PY
import json
from pathlib import Path
p = Path("${MANIFEST}")
data = json.loads(p.read_text())
row = {
    "id": "${id}",
    "outcome": "${outcome}",
    "run_dir": "${run_dir}",
    "replay": 64,
    "rounds": int("${ROUNDS}"),
}
pfloor = "${pfloor}"
if pfloor:
    row["ips_prob_floor"] = float(pfloor)
data["runs"].append(row)
p.write_text(json.dumps(data, indent=2) + "\n")
PY
}

run_phylgfn() {
  local id="$1"
  local out="${AB_ROOT}/${OUTCOME}"
  local ol="$([ "$OUTCOME" = sig ] && echo signature || echo topology)"
  echo "[${OUTCOME}] ${id} replay=${REPLAY} rounds=${ROUNDS} on ${GPU}"
  "$PY" -m grpo_experiments.train \
    --method phylgfn --run-name "${id}" --device "$GPU" \
    --dataset "$DS" --cfg "$CFG" --output "$out" --seed "$SEED" \
    --epochs "$ROUNDS" --steps-per-epoch 1 \
    --on-policy-batch-size "$FRESH" --replay-batch-size "$REPLAY" \
    --replay-buffer-size "$BUFFER" --outcome-level "$ol" \
    --print-every 500 --no-log-trajectories \
    > "${AB_ROOT}/logs/${OUTCOME}_${id}.log" 2>&1
  local run_dir
  run_dir=$(grep -oP 'saved to: \K.*' "${AB_ROOT}/logs/${OUTCOME}_${id}.log" | tail -1)
  append_manifest "$id" "$OUTCOME" "$run_dir"
}

run_hyb_ips() {
  local id="$1" pfloor="$2"
  local out="${AB_ROOT}/${OUTCOME}"
  local ol="$([ "$OUTCOME" = sig ] && echo signature || echo topology)"
  echo "[${OUTCOME}] ${id} pfloor=${pfloor} replay=${REPLAY} rounds=${ROUNDS} on ${GPU}"
  "$PY" -m grpo_experiments.hybrid_ips_grpo.train \
    --run-name "${id}" --device "$GPU" \
    --dataset "$DS" --cfg "$CFG" --output "$out" --seed "$SEED" \
    --resample-rounds "$ROUNDS" --update-cycles 1 \
    --fresh-buffer-size "$FRESH" --replay-sample-size "$REPLAY" \
    --replay-warmstart-samples "$WARMSTART" --best-tree-buffer-size "$BUFFER" \
    --rollout-chunk-size "$CHUNK" --outcome-level "$ol" \
    --print-every 500 --no-log-trajectories \
    --grpo-lr "$LR" --entropy-coef "$ENT" --ips-prob-floor "$pfloor" \
    > "${AB_ROOT}/logs/${OUTCOME}_${id}.log" 2>&1
  local run_dir
  run_dir=$(grep -oP 'saved to: \K.*' "${AB_ROOT}/logs/${OUTCOME}_${id}.log" | tail -1)
  append_manifest "$id" "$OUTCOME" "$run_dir" "$pfloor"
}

echo "5k validation start: outcome=${OUTCOME} gpu=${GPU} rounds=${ROUNDS}"
run_phylgfn "ablation_phylgfn_r64_5k"
run_hyb_ips "ablation_hyb_ips_pfloor_002_5k" 0.002
run_hyb_ips "ablation_hyb_ips_pfloor_005_5k" 0.005
echo "5k training done (3 runs)."
echo "Manifest: ${MANIFEST}"
echo "Eval: .venv/bin/python -m grpo_experiments.scripts.eval_ips_replay_ablation_5k"
