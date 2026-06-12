#!/usr/bin/env bash
# Sample 10k trees per r128 method on cuda:1/2/3, save raw data, then plot.
#
# Usage:
#   bash grpo_experiments/final_eval_experiment/run_r128_10k_sampling_gpus.sh

set -euo pipefail

cd "$(dirname "$0")/../.."
ROOT="$(pwd)"
PY="${ROOT}/.venv/bin/python"
EXP_ROOT="${ROOT}/grpo_experiments/final_eval_experiment"

PLOT_MANIFEST="${EXP_ROOT}/manifest_r128_50k_plot.json"
EVAL_OUT="${EXP_ROOT}/eval/topo/r128_50k_checkpoint_sampling_10k_40bins"
LOG_DIR="${EXP_ROOT}/logs/r128_10k_sampling"

mkdir -p "${LOG_DIR}"

if [[ ! -f "${PLOT_MANIFEST}" ]]; then
  "$PY" - <<PY
import json
from pathlib import Path

src = Path("${EXP_ROOT}/manifest_r128_50k.json")
dst = Path("${PLOT_MANIFEST}")
data = json.loads(src.read_text())
for row in data.get("runs", []):
    row["replay_batch"] = 128
    row["total_batch"] = 512
    row["fresh_batch"] = 384
    row["rounds"] = 50000
dst.write_text(json.dumps(data, indent=2) + "\n")
print(f"Wrote plot manifest: {dst}")
PY
fi

COMMON_ARGS=(
  --manifest "${PLOT_MANIFEST}"
  --output-dir "${EVAL_OUT}"
  --samples 10000
  --batch-size 128
  --n-bins 40
  --sample-only
)

echo "=== Sampling PhyloGFN on cuda:1 ==="
"$PY" "${EXP_ROOT}/eval_sampling_10k.py" \
  "${COMMON_ARGS[@]}" \
  --run-index 0 \
  --device cuda:1 \
  --seed 0 \
  > "${LOG_DIR}/phylgfn_cuda1.log" 2>&1 &
PID0=$!

echo "=== Sampling hybrid GRPO on cuda:2 ==="
"$PY" "${EXP_ROOT}/eval_sampling_10k.py" \
  "${COMMON_ARGS[@]}" \
  --run-index 1 \
  --device cuda:2 \
  --seed 1 \
  > "${LOG_DIR}/hyb_grpo_cuda2.log" 2>&1 &
PID1=$!

echo "=== Sampling hybrid IPS-GRPO on cuda:3 ==="
"$PY" "${EXP_ROOT}/eval_sampling_10k.py" \
  "${COMMON_ARGS[@]}" \
  --run-index 2 \
  --device cuda:3 \
  --seed 2 \
  > "${LOG_DIR}/hyb_ips_cuda3.log" 2>&1 &
PID2=$!

wait "${PID0}" "${PID1}" "${PID2}"

echo "=== Generating sampling plots from raw bundles ==="
"$PY" "${EXP_ROOT}/eval_sampling_10k.py" \
  --manifest "${PLOT_MANIFEST}" \
  --output-dir "${EVAL_OUT}" \
  --samples 10000 \
  --n-bins 40 \
  --plot-only

echo "=== Signature mass scatter (from raw) ==="
"$PY" "${EXP_ROOT}/eval_signature_mass_scatter.py" \
  --manifest "${PLOT_MANIFEST}" \
  --output-dir "${EVAL_OUT}" \
  --samples 10000 \
  --group-by signature \
  --from-raw

echo "=== Topology mass scatter (from raw) ==="
"$PY" "${EXP_ROOT}/eval_signature_mass_scatter.py" \
  --manifest "${PLOT_MANIFEST}" \
  --output-dir "${EVAL_OUT}" \
  --samples 10000 \
  --group-by topology \
  --from-raw

echo "Done."
echo "  raw samples -> ${EVAL_OUT}/raw_samples"
echo "  plots       -> ${EVAL_OUT}"
echo "  logs        -> ${LOG_DIR}"
