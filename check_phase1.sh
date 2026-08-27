#!/usr/bin/env bash
# Print phase-1 status for each method without loading PyTorch on the login node.

set -euo pipefail

cd "$(dirname "$0")"

declare -A RUNS=(
  [rgfn]="molecule_synthesis/runs/seh_paper_medium/rgfn/seed_0/batch"
  [grpo]="molecule_synthesis/runs/seh_paper_medium/grpo/seed_0/batch"
  [count_ips_grpo]="molecule_synthesis/runs/seh_paper_medium/count_ips_grpo/seed_0/batch"
  [mips_grpo]="molecule_synthesis/runs/seh_paper_medium/mips_grpo/seed_0/20260827_112154"
)

TARGET_EPOCH=499
all_ok=1

echo "=== PBS jobs (if still queued/running, checkpoints are not ready yet) ==="
qstat -u "${USER}" 2>/dev/null | rg 'p500|p2500|seh_medium' || echo "(no matching jobs in queue)"

echo
echo "=== Checkpoint status ==="
for method in rgfn grpo count_ips_grpo mips_grpo; do
  run_dir="${RUNS[$method]}"
  ckpt="${run_dir}/train/checkpoints/last_gfn.pt"
  epoch_file="${run_dir}/train/checkpoints/last_epoch.txt"
  manifest="${run_dir}/manifest.json"

  if [[ ! -f "${ckpt}" ]]; then
    echo "${method}: MISSING ${ckpt}"
    all_ok=0
    continue
  fi

  ls -lh "${ckpt}"
  epoch=""
  if [[ -f "${epoch_file}" ]]; then
    epoch="$(tr -d '[:space:]' < "${epoch_file}")"
  elif [[ -f "${manifest}" ]]; then
    epoch="$(python3 - <<PY
import json
data = json.load(open("${manifest}"))
metrics = data.get("best_metrics") or {}
print(metrics.get("epoch", ""))
PY
)"
  fi

  if [[ -n "${epoch}" ]]; then
    echo "${method}: epoch=${epoch}  (${run_dir})"
    if [[ "${epoch}" -lt "${TARGET_EPOCH}" ]]; then
      echo "  warning: expected epoch >= ${TARGET_EPOCH} before phase 2"
      all_ok=0
    elif [[ "${epoch}" -gt "${TARGET_EPOCH}" ]]; then
      echo "  note: past phase-1 target (${TARGET_EPOCH})"
    fi
  else
    echo "${method}: checkpoint exists, training manifest not written yet"
    echo "  (${run_dir})"
    echo "  note: manifest.json appears only after a training job finishes"
    all_ok=0
  fi
  echo
done

if [[ "${all_ok}" == "1" ]]; then
  echo "All four methods reached epoch ${TARGET_EPOCH}. Safe to run: bash submit_phase2.sh"
else
  echo "Not ready for phase 2 yet."
  echo "Wait until all four p500 jobs finish, then rerun: bash check_phase1.sh"
fi
