#!/usr/bin/env bash
# Follow-up runs for the Panel F replay=128 comparison:
#   - PhyloGFN baseline
#   - Hybrid IPS-GRPO with ips_prob_floor in {0.002, 1e-6}
#
# Defaults:
#   total batch = 1024 = 896 fresh + 128 replay
#   total epochs / rounds = 100000
#   steps per epoch / update cycles = 1
#   GPUs: phylgfn on cuda:0, IPS runs on cuda:7

set -euo pipefail

cd "$(dirname "$0")/../.."
ROOT="$(pwd)"
PY="${ROOT}/.venv/bin/python"
EXP_ROOT="${ROOT}/grpo_experiments/final_eval_experiment"

DS="${DS:-dataset/benchmark_datasets/DS1_reduced.pickle}"
CFG="${CFG:-src/configs/benchmark_dna_cfgs/discrete_branch_lengths/cfg_0.001binsize_50bins_temperature_anneal_0.4.yaml}"
SEED="${SEED:-0}"
GPU_PHYLGFN="${GPU_PHYLGFN:-cuda:0}"
GPU_IPS="${GPU_IPS:-cuda:7}"
OUTCOME="${OUTCOME:-topo}"

TOTAL_BATCH="${TOTAL_BATCH:-1024}"
REPLAY="${REPLAY:-128}"
FRESH="${FRESH:-896}"
ROUNDS="${ROUNDS:-100000}"

BUFFER="${BUFFER:-2048}"
WARMSTART="${WARMSTART:-64}"
CHUNK="${CHUNK:-512}"
LR="${LR:-1e-4}"
ENTROPY="${ENTROPY:-0.01}"
PRINT_EVERY="${PRINT_EVERY:-1000}"

if [[ "$OUTCOME" == "sig" ]]; then
  OUTCOME_LEVEL="signature"
else
  OUTCOME="topo"
  OUTCOME_LEVEL="topology"
fi

if [[ $((FRESH + REPLAY)) -ne ${TOTAL_BATCH} ]]; then
  echo "FRESH + REPLAY must equal TOTAL_BATCH (${FRESH} + ${REPLAY} != ${TOTAL_BATCH})" >&2
  exit 1
fi

mkdir -p "${EXP_ROOT}/logs" "${EXP_ROOT}/${OUTCOME}"

MANIFEST="${EXP_ROOT}/manifest.json"
cat > "${MANIFEST}" <<EOF
{
  "experiment": "final_eval_followup",
  "outcome": "${OUTCOME}",
  "seed": ${SEED},
  "total_batch": ${TOTAL_BATCH},
  "fresh_batch": ${FRESH},
  "replay_batch": ${REPLAY},
  "rounds": ${ROUNDS},
  "runs": []
}
EOF

append_manifest() {
  local id="$1"
  local method="$2"
  local run_dir="$3"
  local pfloor="${4:-}"
  "$PY" - <<PY
import json
from pathlib import Path

p = Path("${MANIFEST}")
data = json.loads(p.read_text())
row = {
    "id": "${id}",
    "method": "${method}",
    "run_dir": "${run_dir}",
    "outcome": "${OUTCOME}",
    "rounds": int("${ROUNDS}"),
    "fresh_batch": int("${FRESH}"),
    "replay_batch": int("${REPLAY}"),
    "total_batch": int("${TOTAL_BATCH}"),
}
pfloor = "${pfloor}"
if pfloor:
    row["ips_prob_floor"] = float(pfloor)
data["runs"].append(row)
p.write_text(json.dumps(data, indent=2) + "\n")
PY
}

extract_run_dir() {
  local log_path="$1"
  "$PY" - <<PY
from pathlib import Path
import re
text = Path("${log_path}").read_text()
matches = re.findall(r"saved to:\s*(.+)", text)
print(matches[-1] if matches else "")
PY
}

run_phylgfn() {
  local id="finaleval_phylgfn_r128_b1024_100k"
  local log_path="${EXP_ROOT}/logs/${OUTCOME}_${id}.log"
  echo "[${OUTCOME}] ${id} on ${GPU_PHYLGFN}"
  "$PY" -m grpo_experiments.train \
    --method phylgfn \
    --run-name "${id}" \
    --device "${GPU_PHYLGFN}" \
    --dataset "${DS}" \
    --cfg "${CFG}" \
    --output "${EXP_ROOT}/${OUTCOME}" \
    --seed "${SEED}" \
    --epochs "${ROUNDS}" \
    --steps-per-epoch 1 \
    --on-policy-batch-size "${FRESH}" \
    --replay-batch-size "${REPLAY}" \
    --replay-buffer-size "${BUFFER}" \
    --outcome-level "${OUTCOME_LEVEL}" \
    --print-every "${PRINT_EVERY}" \
    > "${log_path}" 2>&1
  local run_dir
  run_dir="$(extract_run_dir "${log_path}")"
  append_manifest "${id}" "phylgfn" "${run_dir}"
}

run_hyb_ips() {
  local pfloor="$1"
  local pfloor_tag="$2"
  local id="finaleval_hyb_ips_pfloor_${pfloor_tag}_r128_b1024_100k"
  local log_path="${EXP_ROOT}/logs/${OUTCOME}_${id}.log"
  echo "[${OUTCOME}] ${id} pfloor=${pfloor} on ${GPU_IPS}"
  "$PY" -m grpo_experiments.hybrid_ips_grpo.train \
    --run-name "${id}" \
    --device "${GPU_IPS}" \
    --dataset "${DS}" \
    --cfg "${CFG}" \
    --output "${EXP_ROOT}/${OUTCOME}" \
    --seed "${SEED}" \
    --resample-rounds "${ROUNDS}" \
    --update-cycles 1 \
    --fresh-buffer-size "${FRESH}" \
    --replay-sample-size "${REPLAY}" \
    --replay-warmstart-samples "${WARMSTART}" \
    --best-tree-buffer-size "${BUFFER}" \
    --rollout-chunk-size "${CHUNK}" \
    --outcome-level "${OUTCOME_LEVEL}" \
    --print-every "${PRINT_EVERY}" \
    --no-log-trajectories \
    --grpo-lr "${LR}" \
    --entropy-coef "${ENTROPY}" \
    --ips-prob-floor "${pfloor}" \
    > "${log_path}" 2>&1
  local run_dir
  run_dir="$(extract_run_dir "${log_path}")"
  append_manifest "${id}" "hybrid_ips_grpo" "${run_dir}" "${pfloor}"
}

echo "Final eval follow-up start: outcome=${OUTCOME} rounds=${ROUNDS} total_batch=${TOTAL_BATCH} fresh=${FRESH} replay=${REPLAY}"
echo "Device split: phylgfn=${GPU_PHYLGFN}, ips=${GPU_IPS}"

run_phylgfn &
PHYLGFN_PID=$!
run_hyb_ips 0.002 002
run_hyb_ips 1e-6 1em6
wait "${PHYLGFN_PID}"

echo "Final eval follow-up complete (3 runs)."
echo "Manifest: ${MANIFEST}"
echo "Logs: ${EXP_ROOT}/logs"
