#!/usr/bin/env bash
# Evaluate a completed sanity matrix (training curves + final sampling).
#
# Usage:
#   bash grpo_experiments/scripts/run_matrix_eval.sh grpo_experiments/runs/sanity_matrix/20260530_120000
#   SAMPLES=1000 ESTIMATE_MLL=1 bash grpo_experiments/scripts/run_matrix_eval.sh <matrix_root>

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi

MATRIX_ROOT="${1:?usage: run_matrix_eval.sh <matrix_root>}"
MANIFEST="$MATRIX_ROOT/manifest.json"
EVAL_DIR="$MATRIX_ROOT/eval"
SAMPLING_DIR="$MATRIX_ROOT/sampling"
SAMPLES="${SAMPLES:-1000}"
N_BINS="${N_BINS:-10}"
ESTIMATE_MLL="${ESTIMATE_MLL:-0}"

if [[ ! -f "$MANIFEST" ]]; then
  echo "ERROR: missing manifest: $MANIFEST" >&2
  exit 1
fi

echo "=== evaluate training metrics ==="
"$PYTHON" -m grpo_experiments.scripts.evaluate_runs \
  --manifest "$MANIFEST" \
  --output-dir "$EVAL_DIR"

echo
echo "=== compare final sampling ==="
EXTRA=()
if [[ "$ESTIMATE_MLL" == "1" ]]; then
  EXTRA+=(--estimate-mll)
fi
"$PYTHON" -m grpo_experiments.scripts.compare_sampling \
  --manifest "$MANIFEST" \
  --output-dir "$SAMPLING_DIR" \
  --samples "$SAMPLES" \
  --n-bins "$N_BINS" \
  "${EXTRA[@]}"

echo
echo "=== outputs ==="
echo "  training curves: $EVAL_DIR/training_curves.png"
echo "  training summary: $EVAL_DIR/evaluation_summary.json"
echo "  sampling plot:   $SAMPLING_DIR/sampling_comparison.png"
echo "  sampling summary: $SAMPLING_DIR/sampling_summary.json"
