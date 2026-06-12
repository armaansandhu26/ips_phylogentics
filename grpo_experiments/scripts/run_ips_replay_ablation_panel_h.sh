#!/usr/bin/env bash
# Panel H — delayed IPS (start rounds 500 and 750), topo @ r64, pfloor=0.005.
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
PFLOOR=0.005
FRESH=448
REPLAY=64
BUFFER=512
WARMSTART=64

mkdir -p "${AB_ROOT}/logs" "${AB_ROOT}/${OUTCOME}"
MANIFEST="${AB_ROOT}/manifest_panel_h.json"
echo '{"phase":"panel_h","runs":[]}' > "${MANIFEST}"

append_manifest() {
  local id="$1" outcome="$2" run_dir="$3" ips_start="$4"
  "$PY" - <<PY
import json
from pathlib import Path
p = Path("${MANIFEST}")
data = json.loads(p.read_text())
data["runs"].append({
    "id": "${id}",
    "outcome": "${outcome}",
    "run_dir": "${run_dir}",
    "replay": 64,
    "ips_start_round": int("${ips_start}"),
})
p.write_text(json.dumps(data, indent=2) + "\n")
PY
}

run_hyb_ips_delayed() {
  local id="$1" ips_start="$2"
  local out="${AB_ROOT}/${OUTCOME}"
  local ol="$([ "$OUTCOME" = sig ] && echo signature || echo topology)"
  echo "[${OUTCOME}] ${id} ips_start_round=${ips_start} pfloor=${PFLOOR} replay=${REPLAY} on ${GPU}"
  "$PY" -m grpo_experiments.hybrid_ips_grpo.train \
    --run-name "${id}" --device "$GPU" \
    --dataset "$DS" --cfg "$CFG" --output "$out" --seed "$SEED" \
    --resample-rounds "$ROUNDS" --update-cycles 1 \
    --fresh-buffer-size "$FRESH" --replay-sample-size "$REPLAY" \
    --replay-warmstart-samples "$WARMSTART" --best-tree-buffer-size "$BUFFER" \
    --rollout-chunk-size "$CHUNK" --outcome-level "$ol" \
    --print-every 100 --no-log-trajectories \
    --grpo-lr "$LR" --entropy-coef "$ENT" \
    --ips-prob-floor "$PFLOOR" --ips-start-round "$ips_start" \
    > "${AB_ROOT}/logs/${OUTCOME}_${id}.log" 2>&1
  local run_dir
  run_dir=$(grep -oP 'saved to: \K.*' "${AB_ROOT}/logs/${OUTCOME}_${id}.log" | tail -1)
  append_manifest "$id" "$OUTCOME" "$run_dir" "$ips_start"
}

echo "Panel H start: outcome=${OUTCOME} gpu=${GPU} rounds=${ROUNDS}"
run_hyb_ips_delayed "ablation_hyb_ips_delayed500" 500
run_hyb_ips_delayed "ablation_hyb_ips_delayed750" 750
echo "Panel H training done (2 runs)."
echo "Manifest: ${MANIFEST}"
echo "Eval: .venv/bin/python -m grpo_experiments.scripts.eval_ips_replay_ablation_panel_h"
