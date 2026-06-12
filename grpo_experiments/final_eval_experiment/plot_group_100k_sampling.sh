#!/usr/bin/env bash
# 100k checkpoint sampling plots for one completed 50k group.
#
# Usage:
#   bash grpo_experiments/final_eval_experiment/plot_group_100k_sampling.sh r32
#   bash grpo_experiments/final_eval_experiment/plot_group_100k_sampling.sh r128 cuda:7

set -euo pipefail

cd "$(dirname "$0")/../.."
ROOT="$(pwd)"
PY="${ROOT}/.venv/bin/python"
EXP_ROOT="${ROOT}/grpo_experiments/final_eval_experiment"

GROUP="${1:-}"
DEVICE="${2:-cuda:7}"
SAMPLES="${SAMPLES:-100000}"
N_BINS="${N_BINS:-40}"

if [[ -z "${GROUP}" ]]; then
  echo "Usage: $0 <r32|r64|r128> [device]" >&2
  exit 1
fi

PLOT_MANIFEST="${EXP_ROOT}/manifest_${GROUP}_50k_plot.json"
EVAL_OUT="${EXP_ROOT}/eval/topo/${GROUP}_50k_checkpoint_sampling_${SAMPLES//000}k_${N_BINS}bins"

if [[ ! -f "${PLOT_MANIFEST}" ]]; then
  echo "Plot manifest not found: ${PLOT_MANIFEST}" >&2
  echo "Run plot_group_50k.sh ${GROUP} first to create it." >&2
  exit 1
fi

echo "=== 100k checkpoint sampling: ${GROUP} on ${DEVICE} ==="
"$PY" "${EXP_ROOT}/eval_sampling_10k.py" \
  --manifest "${PLOT_MANIFEST}" \
  --output-dir "${EVAL_OUT}" \
  --device "${DEVICE}" \
  --samples "${SAMPLES}" \
  --n-bins "${N_BINS}"

echo "=== Signature mass scatter (${SAMPLES}): ${GROUP} ==="
"$PY" "${EXP_ROOT}/eval_signature_mass_scatter.py" \
  --manifest "${PLOT_MANIFEST}" \
  --output-dir "${EVAL_OUT}" \
  --device "${DEVICE}" \
  --samples "${SAMPLES}" \
  --group-by signature

echo "=== Topology mass scatter (${SAMPLES}): ${GROUP} ==="
"$PY" "${EXP_ROOT}/eval_signature_mass_scatter.py" \
  --manifest "${PLOT_MANIFEST}" \
  --output-dir "${EVAL_OUT}" \
  --device "${DEVICE}" \
  --samples "${SAMPLES}" \
  --group-by topology

echo "Done: ${GROUP}"
echo "  sampling eval -> ${EVAL_OUT}"
