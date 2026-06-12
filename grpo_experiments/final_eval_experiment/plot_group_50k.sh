#!/usr/bin/env bash
# Plot training curves and checkpoint sampling for one completed 50k group.
#
# Usage:
#   bash grpo_experiments/final_eval_experiment/plot_group_50k.sh r32
#   bash grpo_experiments/final_eval_experiment/plot_group_50k.sh r128 cuda:7

set -euo pipefail

cd "$(dirname "$0")/../.."
ROOT="$(pwd)"
PY="${ROOT}/.venv/bin/python"
EXP_ROOT="${ROOT}/grpo_experiments/final_eval_experiment"

GROUP="${1:-}"
DEVICE="${2:-cuda:7}"

if [[ -z "${GROUP}" ]]; then
  echo "Usage: $0 <r32|r64|r128> [device]" >&2
  exit 1
fi

case "${GROUP}" in
  r32) REPLAY=32 ;;
  r64) REPLAY=64 ;;
  r128) REPLAY=128 ;;
  *)
    echo "Unknown group: ${GROUP}" >&2
    exit 1
    ;;
esac

SRC_MANIFEST="${EXP_ROOT}/manifest_${GROUP}_50k.json"
PLOT_MANIFEST="${EXP_ROOT}/manifest_${GROUP}_50k_plot.json"
TRAIN_OUT="${EXP_ROOT}/training_curves/${GROUP}_50k"
EVAL_OUT="${EXP_ROOT}/eval/topo/${GROUP}_50k_checkpoint_sampling_10k_40bins"

if [[ ! -f "${SRC_MANIFEST}" ]]; then
  echo "Manifest not found: ${SRC_MANIFEST}" >&2
  exit 1
fi

"$PY" - <<PY
import json
from pathlib import Path

src = Path("${SRC_MANIFEST}")
dst = Path("${PLOT_MANIFEST}")
data = json.loads(src.read_text())
for row in data.get("runs", []):
    row["replay_batch"] = int("${REPLAY}")
    row["total_batch"] = 512
    row["fresh_batch"] = 512 - int("${REPLAY}")
    row["rounds"] = 50000
dst.write_text(json.dumps(data, indent=2) + "\n")
print(f"Wrote plot manifest: {dst} ({len(data.get('runs', []))} runs)")
PY

echo "=== Training curves: ${GROUP} ==="
"$PY" "${EXP_ROOT}/plot_training_curves.py" \
  --manifest "${PLOT_MANIFEST}" \
  --output-dir "${TRAIN_OUT}"

echo "=== Checkpoint sampling (10k): ${GROUP} on ${DEVICE} ==="
"$PY" "${EXP_ROOT}/eval_sampling_10k.py" \
  --manifest "${PLOT_MANIFEST}" \
  --output-dir "${EVAL_OUT}" \
  --device "${DEVICE}" \
  --samples 10000 \
  --n-bins 40

echo "=== Signature mass scatter: ${GROUP} ==="
"$PY" "${EXP_ROOT}/eval_signature_mass_scatter.py" \
  --manifest "${PLOT_MANIFEST}" \
  --output-dir "${EVAL_OUT}" \
  --device "${DEVICE}" \
  --samples 10000 \
  --group-by signature

echo "=== Topology mass scatter: ${GROUP} ==="
"$PY" "${EXP_ROOT}/eval_signature_mass_scatter.py" \
  --manifest "${PLOT_MANIFEST}" \
  --output-dir "${EVAL_OUT}" \
  --device "${DEVICE}" \
  --samples 10000 \
  --group-by topology

echo "Done: ${GROUP}"
echo "  training curves -> ${TRAIN_OUT}"
echo "  sampling eval   -> ${EVAL_OUT}"
