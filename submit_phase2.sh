#!/usr/bin/env bash
# Submit four parallel batch jobs: resume from iter-500 checkpoints to 2500, then sample.
# Run only after check_phase1.sh looks good.

set -euo pipefail

cd "$(dirname "$0")"

TARGET_ITER=2500

J_RGFN=$(qsub -N rgfn_p2500 -v "METHOD=rgfn,TARGET_ITER=${TARGET_ITER},DO_SAMPLE=1" run.pbs)
J_GRPO=$(qsub -N grpo_p2500 -v "METHOD=grpo,TARGET_ITER=${TARGET_ITER},DO_SAMPLE=1" run.pbs)
J_COUNT=$(qsub -N count_p2500 -v "METHOD=count_ips_grpo,TARGET_ITER=${TARGET_ITER},DO_SAMPLE=1" run.pbs)
J_MIPS=$(qsub -N mips_p2500 -v "METHOD=mips_grpo,TARGET_ITER=${TARGET_ITER},DO_SAMPLE=1" run.pbs)

echo "Submitted phase 2 (parallel, target iter ${TARGET_ITER} + 50k sample each):"
echo "  RGFN:          ${J_RGFN}"
echo "  GRPO:          ${J_GRPO}"
echo "  Count IPS-GRPO:${J_COUNT}"
echo "  MIPS-GRPO:     ${J_MIPS}"
echo
echo "Monitor: qstat -u \$USER"
