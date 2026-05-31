#!/usr/bin/env bash
# Launch the 100-epoch matrix across GPUs 0, 1, 2 (2 experiments per GPU).
#
# Usage (repo root):
#   bash grpo_experiments/scripts/run_matrix_100epoch_parallel.sh
#   MATRIX_ID=20260601_120000 bash grpo_experiments/scripts/run_matrix_100epoch_parallel.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MATRIX_ID="${MATRIX_ID:-$(date +%Y%m%d_%H%M%S)}"
MATRIX_ROOT="${MATRIX_ROOT:-grpo_experiments/runs/matrix_100epoch/${MATRIX_ID}}"
SCRIPT="$ROOT/grpo_experiments/scripts/run_matrix_100epoch.sh"

mkdir -p "$MATRIX_ROOT/train"

echo "=== launching 100-epoch matrix on GPUs 0,1,2 ==="
echo "matrix_root: $MATRIX_ROOT"
echo "  GPU 0 → exp1, exp2"
echo "  GPU 1 → exp3, exp4"
echo "  GPU 2 → exp5, exp6"
echo

PIDS=()
for pair in "1 0" "2 1" "3 2"; do
  set -- $pair
  shard="$1"
  gpu="$2"
  log_file="$MATRIX_ROOT/shard${shard}.log"
  echo "starting shard $shard on GPU $gpu → $log_file"
  (
    export MATRIX_ROOT MATRIX_ID SHARD="$shard" DEVICE=cuda:0
    export CUDA_VISIBLE_DEVICES="$gpu"
    exec bash "$SCRIPT"
  ) > "$log_file" 2>&1 &
  PIDS+=("$!")
done

fail=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done

echo
echo "=== merging manifest ==="
MATRIX_ROOT="$MATRIX_ROOT" MERGE_MANIFEST=1 bash "$SCRIPT"

echo
if [[ "$fail" -ne 0 ]]; then
  echo "ERROR: one or more shards failed — check $MATRIX_ROOT/shard*.log" >&2
  exit 1
fi

echo "=== all shards complete ==="
echo "manifest: $MATRIX_ROOT/manifest.json"
echo "logs:     $MATRIX_ROOT/shard1.log (GPU 0)"
echo "          $MATRIX_ROOT/shard2.log (GPU 1)"
echo "          $MATRIX_ROOT/shard3.log (GPU 2)"
echo
echo "Eval when done:"
echo "  bash grpo_experiments/scripts/run_matrix_eval.sh $MATRIX_ROOT"
