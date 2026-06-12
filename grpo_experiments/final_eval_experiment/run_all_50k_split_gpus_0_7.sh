#!/usr/bin/env bash
# Schedule the three 50k group configs across GPUs:
#   - r32 on cuda:0
#   - r64 on cuda:0
#   - r128 on cuda:7
#
# This creates two queues:
#   queue A on cuda:0 runs r32 then r64
#   queue B on cuda:7 runs r128

set -euo pipefail

cd "$(dirname "$0")/../.."
ROOT="$(pwd)"
RUNNER="${ROOT}/grpo_experiments/final_eval_experiment/run_group_config.sh"

GPU0="${GPU0:-cuda:0}"
GPU7="${GPU7:-cuda:7}"

CONFIG_R32="${ROOT}/grpo_experiments/final_eval_experiment/config_r32_50k.json"
CONFIG_R64="${ROOT}/grpo_experiments/final_eval_experiment/config_r64_50k.json"
CONFIG_R128="${ROOT}/grpo_experiments/final_eval_experiment/config_r128_50k.json"

queue_gpu0() {
  echo "Queue on ${GPU0}: r32 -> r64"
  bash "${RUNNER}" "${CONFIG_R32}" "${GPU0}"
  bash "${RUNNER}" "${CONFIG_R64}" "${GPU0}"
}

queue_gpu7() {
  echo "Queue on ${GPU7}: r128"
  bash "${RUNNER}" "${CONFIG_R128}" "${GPU7}"
}

echo "Scheduling 50k final-eval groups across GPUs"
echo "  ${GPU0}: r32, r64"
echo "  ${GPU7}: r128"

queue_gpu0 &
PID_GPU0=$!

queue_gpu7 &
PID_GPU7=$!

wait "${PID_GPU0}" "${PID_GPU7}"

echo "All scheduled 50k group runs finished."
