#!/usr/bin/env bash
# Three-way eval (PhyloGFN, GRPO, IPS) for hybrid_25k_track* runs.

set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PWD}/.venv/bin/python"
ROOT="${OUTPUT_ROOT:-grpo_experiments/runs/hybrid_25k_track}"
ROUNDS="${ROUNDS:-25000}"
DEVICE="${DEVICE:-cuda:0}"
SAMPLES="${SAMPLES:-1000}"

_pick_best_run() {
  local suffix="$1"
  shift
  local best="" best_n=0
  local d n token skip
  for d in "${ROOT}"/*"${suffix}"; do
    [[ -d "$d" ]] || continue
    skip=0
    for token in "$@"; do
      if [[ "$d" == *"${token}"* ]]; then
        skip=1
        break
      fi
    done
    (( skip )) && continue
    [[ -f "${d}/metrics.jsonl" ]] || continue
    n=$(wc -l < "${d}/metrics.jsonl")
    if (( n > best_n )); then
      best_n=$n
      best=$d
    fi
  done
  if [[ -z "$best" ]]; then
    echo "missing run matching *${suffix} under ${ROOT}" >&2
    exit 1
  fi
  echo "$best"
}

PHYLGFN_DIR="${PHYLGFN_DIR:-$(_pick_best_run _phylgfn_phylgfn)}"
if [[ -z "${GRPO_DIR:-}" ]]; then
  GRPO_DIR="$(_pick_best_run _grpo_hybrid_grpo softmax)"
fi
if [[ -z "${IPS_DIR:-}" ]]; then
  IPS_DIR="$(_pick_best_run _ips_hybrid_ips_grpo softmax)"
fi

echo "phylgfn: $PHYLGFN_DIR"
echo "grpo:    $GRPO_DIR"
echo "ips:     $IPS_DIR"

# Remove legacy two-method-only outputs.
rm -rf "${ROOT}/combined_eval"
rm -f "${ROOT}/late_training_eval"/*.png "${ROOT}/late_training_eval"/*.json 2>/dev/null || true

"$PY" -m grpo_experiments.scripts.compare_three_way_eval \
  --output-root "$ROOT" \
  --phylgfn-dir "$PHYLGFN_DIR" \
  --grpo-dir "$GRPO_DIR" \
  --ips-dir "$IPS_DIR" \
  --rounds "$ROUNDS" \
  --samples "$SAMPLES" \
  --device "$DEVICE"

if [[ "${RUN_LATE_COMPARE:-0}" == "1" ]]; then
  LATE_STEP_MIN="${LATE_STEP_MIN:-20000}"
  LATE_STEP_MAX="${LATE_STEP_MAX:-24999}"
  LATE_VS_FINAL_DIR="${ROOT}/late_vs_final_eval"
  mkdir -p "$LATE_VS_FINAL_DIR"
  rm -f "${LATE_VS_FINAL_DIR}"/*
  "$PY" -m grpo_experiments.scripts.compare_three_way_eval \
    --output-root "$LATE_VS_FINAL_DIR" \
    --phylgfn-dir "$PHYLGFN_DIR" \
    --grpo-dir "$GRPO_DIR" \
    --ips-dir "$IPS_DIR" \
    --rounds "$ROUNDS" \
    --samples "$SAMPLES" \
    --device "$DEVICE" \
    --global-step-min "$LATE_STEP_MIN" \
    --global-step-max "$LATE_STEP_MAX"
fi

echo "done: ${ROOT}/training_eval  ${ROOT}/sampling_eval  ${ROOT}/training_vs_final_eval  ${ROOT}/trajectory_plots"
