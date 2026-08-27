#!/usr/bin/env bash
# Submit four parallel batch jobs: one per method, each training to iteration 500.
# MIPS resumes from the existing partial checkpoint; the other three start fresh.

set -euo pipefail

cd "$(dirname "$0")"

J_RGFN=$(qsub -N rgfn_p500 -v METHOD=rgfn run.pbs)
J_GRPO=$(qsub -N grpo_p500 -v METHOD=grpo run.pbs)
J_COUNT=$(qsub -N count_p500 -v METHOD=count_ips_grpo run.pbs)
J_MIPS=$(qsub -N mips_p500 -v METHOD=mips_grpo run.pbs)

echo "Submitted phase 1 (parallel, target iter 500):"
echo "  RGFN:          ${J_RGFN}"
echo "  GRPO:          ${J_GRPO}"
echo "  Count IPS-GRPO:${J_COUNT}"
echo "  MIPS-GRPO:     ${J_MIPS}"
echo
echo "Monitor: qstat -u \$USER"
echo "When all finish, check checkpoints:"
echo "  bash check_phase1.sh"
echo
echo "Then start phase 2 (500 -> 2500):"
echo "  bash submit_phase2.sh"
