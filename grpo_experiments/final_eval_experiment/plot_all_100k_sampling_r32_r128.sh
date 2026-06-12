#!/usr/bin/env bash
# Run 100k sampling plots for completed r32 and r128 groups in parallel.
#
# Default split:
#   r32  -> cuda:1
#   r128 -> cuda:2

set -euo pipefail

cd "$(dirname "$0")/../.."
RUNNER="${PWD}/grpo_experiments/final_eval_experiment/plot_group_100k_sampling.sh"
DEVICE_R32="${DEVICE_R32:-cuda:1}"
DEVICE_R128="${DEVICE_R128:-cuda:2}"

echo "Scheduling 100k sampling:"
echo "  r32  -> ${DEVICE_R32}"
echo "  r128 -> ${DEVICE_R128}"

bash "${RUNNER}" r32 "${DEVICE_R32}" &
PID_R32=$!

bash "${RUNNER}" r128 "${DEVICE_R128}" &
PID_R128=$!

wait "${PID_R32}" "${PID_R128}"

echo "All 100k sampling plots complete for r32 and r128."
