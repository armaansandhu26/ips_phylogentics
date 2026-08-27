#!/usr/bin/env bash
# Report live training iteration progress for seh_paper_medium PBS jobs.
# Safe on login nodes (no PyTorch import).
#
# Usage:
#   bash check_training_progress.sh                    # all running *_p* jobs
#   bash check_training_progress.sh rgfn_p2000 mips_p2000
#   TARGET_ITER=2000 bash check_training_progress.sh # override default target

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

declare -A RUN_DIRS=(
  [rgfn]="${ROOT}/molecule_synthesis/runs/seh_paper_medium/rgfn/seed_0/batch"
  [grpo]="${ROOT}/molecule_synthesis/runs/seh_paper_medium/grpo/seed_0/batch"
  [count_ips_grpo]="${ROOT}/molecule_synthesis/runs/seh_paper_medium/count_ips_grpo/seed_0/batch"
  [mips_grpo]="${ROOT}/molecule_synthesis/runs/seh_paper_medium/mips_grpo/seed_0/20260827_112154"
)

job_method() {
  local job_name="$1"
  case "${job_name}" in
    rgfn_*) echo rgfn ;;
    grpo_*) echo grpo ;;
    count_*) echo count_ips_grpo ;;
    mips_*) echo mips_grpo ;;
    *) echo "" ;;
  esac
}

job_target_iter() {
  local job_name="$1"
  if [[ "${job_name}" =~ _p([0-9]+)$ ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    echo "${TARGET_ITER:-500}"
  fi
}

read_start_iter() {
  local file="$1"
  local run_dir="$2"
  local start=""
  if [[ -f "$file" ]]; then
    start="$(python3 - "$file" <<'PY'
import re, sys
text = open(sys.argv[1], errors="replace").read()
m = re.findall(r"Loaded checkpoint from (\d+) iteration", text)
print(m[-1] if m else "")
PY
)"
  fi
  if [[ -z "$start" && -f "${run_dir}/train/checkpoints/last_epoch.txt" ]]; then
    start="$(python3 - <<PY
epoch = int(open("${run_dir}/train/checkpoints/last_epoch.txt").read().strip())
print(epoch + 1)
PY
)"
  fi
  echo "${start:-0}"
}

latest_tqdm_progress() {
  local file="$1"
  [[ -f "$file" ]] || return 1
  python3 - "$file" <<'PY'
import re, sys
text = open(sys.argv[1], errors="replace").read()
matches = re.findall(r"\|\s*(\d+)/(\d+)\s*\[", text)
if matches:
    local, job_total = matches[-1]
    print(f"{local} {job_total}")
PY
}

latest_wandb_progress() {
  local run_dir="$1"
  [[ -d "${run_dir}/logs/wandb" ]] || return 1
  python3 - "${run_dir}/logs/wandb" <<'PY'
import re, sys
from pathlib import Path

wandb_root = Path(sys.argv[1])
best = None
for wb in sorted(wandb_root.glob("offline-run-*/run-*.wandb"), key=lambda p: p.stat().st_mtime):
    data = wb.read_bytes().decode("latin1", errors="ignore")
    matches = re.findall(r"\|\s*(\d+)/(\d+)\s*\[", data)
    if matches:
        local, job_total = matches[-1]
        best = (local, job_total)
if best:
    print(f"{best[0]} {best[1]}")
PY
}

progress_for_job() {
  local job_name="$1"
  local job_id="$2"
  local elapsed="$3"
  local method target_iter run_dir stdout_log start_iter="" local_iter="" job_total="" source=""

  method="$(job_method "${job_name}")"
  target_iter="$(job_target_iter "${job_name}")"
  run_dir="${RUN_DIRS[$method]:-}"
  stdout_log="${ROOT}/${job_name}.o${job_id}"

  if [[ -n "$run_dir" ]]; then
    start_iter="$(read_start_iter "$stdout_log" "$run_dir")"
    if read -r local_iter job_total < <(latest_tqdm_progress "$stdout_log" 2>/dev/null); then
      source="stdout"
    elif read -r local_iter job_total < <(latest_wandb_progress "$run_dir" 2>/dev/null); then
      source="wandb"
    fi
  fi

  printf "%-12s id=%-8s elapsed=%-6s" "$job_name" "$job_id" "$elapsed"
  if [[ -n "$local_iter" && -n "$job_total" ]]; then
    abs_iter="$(python3 - <<PY
start = int("${start_iter:-0}")
local = int("${local_iter}")
print(start + local)
PY
)"
    abs_pct="$(python3 - <<PY
print(f"{100 * int('${abs_iter}') / int('${target_iter}'):.1f}")
PY
)"
    job_pct="$(python3 - <<PY
print(f"{100 * int('${local_iter}') / int('${job_total}'):.1f}")
PY
)"
    printf "  total=%s/%s (%s%%)" "$abs_iter" "$target_iter" "$abs_pct"
    printf "  job=%s/%s (%s%%)" "$local_iter" "$job_total" "$job_pct"
    [[ -n "$source" ]] && printf "  [%s]" "$source"
  elif [[ -n "$start_iter" && "$start_iter" != "0" ]]; then
    printf "  total=%s/%s (starting)" "$start_iter" "$target_iter"
  else
    printf "  progress=unknown (target total=%s)" "$target_iter"
  fi
  printf "\n"
}

echo "=== Training progress ==="
echo "time: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo

if ! qstat -u "${USER}" >/tmp/check_training_progress_qstat.$$ 2>/dev/null; then
  echo "qstat unavailable"
  rm -f /tmp/check_training_progress_qstat.$$
  exit 1
fi

declare -a FILTER=()
if (($# > 0)); then
  FILTER=("$@")
fi

found=0
while IFS= read -r line; do
  [[ "$line" =~ ^[0-9]+ ]] || continue
  job_id="${line%%.*}"
  job_name="$(awk '{print $4}' <<<"$line")"
  state="$(awk '{print $10}' <<<"$line")"
  elapsed="$(awk '{print $11}' <<<"$line")"

  if ((${#FILTER[@]} > 0)); then
    match=0
    for want in "${FILTER[@]}"; do
      [[ "$job_name" == "$want" ]] && match=1 && break
    done
    (( match )) || continue
  else
    [[ "$job_name" =~ _(p[0-9]+|p2500)$ ]] || continue
  fi

  found=1
  progress_for_job "$job_name" "$job_id" "$elapsed"
  [[ "$state" != "R" ]] && echo "  (state=${state}, not running)"
done < <(tail -n +3 /tmp/check_training_progress_qstat.$$)

rm -f /tmp/check_training_progress_qstat.$$

if (( found == 0 )); then
  if ((${#FILTER[@]} > 0)); then
    echo "No matching jobs in queue for: ${FILTER[*]}"
  else
    echo "No matching training jobs currently in queue."
  fi
fi
