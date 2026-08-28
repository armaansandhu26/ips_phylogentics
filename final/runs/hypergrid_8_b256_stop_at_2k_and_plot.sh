#!/usr/bin/env bash
# Stop hypergrid 8×8 b256 jobs at epoch 2000, resume paused 27t jobs as GPUs empty, then plot.
set -euo pipefail
cd /home/armaan/phylogfn

TARGET_EPOCH=2000
POLL_SEC=30
LOG=final/runs/hypergrid_8_b256_watcher.log

declare -A RESUME_PIDS=(
  [1]="1514523"
  [2]="1583704 1583705"
  [3]="1583706"
)
declare -A RESUMED=([1]=0 [2]=0 [3]=0)

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

get_epoch() {
  local suite=$1 method=$2
  local mp
  mp=$(find "final/runs/${suite}/${method}" -name metrics.jsonl 2>/dev/null | head -1)
  [[ -z "$mp" ]] && echo 0 && return
  python3 -c "import json; print(int(json.loads(open('${mp}').readlines()[-1])['epoch']))"
}

get_pid() {
  local suite=$1 method=$2
  ps aux | grep "[h]ypergrid-pipeline --suite ${suite} --method ${method}" | awk '{print $2}' | head -1
}

hypergrid_count_on_gpu() {
  local gpu=$1
  ps aux | grep '[h]ypergrid-pipeline --suite hypergrid_8_b256' | grep -c "cuda:${gpu}" || true
}

all_hypergrid_count() {
  ps aux | grep -c '[h]ypergrid-pipeline --suite hypergrid_8_b256' || true
}

stop_jobs_at_target() {
  local seed method suite epoch pid
  for seed in 0 1 2; do
    for method in grpo count_ips learned_reverse_ips; do
      suite="hypergrid_8_b256_seed${seed}"
      pid=$(get_pid "$suite" "$method")
      [[ -z "$pid" ]] && continue
      epoch=$(get_epoch "$suite" "$method")
      if [[ "$epoch" -ge "$TARGET_EPOCH" ]]; then
        log "STOP ${suite} ${method} epoch=${epoch} PID=${pid}"
        kill "$pid" 2>/dev/null || true
      fi
    done
  done
}

resume_freed_gpus() {
  local gpu count
  for gpu in 1 2 3; do
    [[ "${RESUMED[$gpu]}" -eq 1 ]] && continue
    count=$(hypergrid_count_on_gpu "$gpu")
    if [[ "$count" -eq 0 ]]; then
      log "GPU ${gpu} empty — resuming: ${RESUME_PIDS[$gpu]}"
      for pid in ${RESUME_PIDS[$gpu]}; do
        if kill -CONT "$pid" 2>/dev/null; then
          log "  CONT ${pid} OK"
        else
          log "  CONT ${pid} failed"
        fi
      done
      RESUMED[$gpu]=1
    fi
  done
}

run_plots() {
  log "All hypergrid jobs stopped — running plot_comparison per seed"
  for seed in 0 1 2; do
    base="final/runs/hypergrid_8_b256_seed${seed}"
    grpo=$(find "$base/grpo" -maxdepth 1 -type d -name '2026*' | sort | tail -1)
    ips=$(find "$base/count_ips" -maxdepth 1 -type d -name '2026*' | sort | tail -1)
    mips=$(find "$base/learned_reverse_ips" -maxdepth 1 -type d -name '2026*' | sort | tail -1)
    out="final/runs/hypergrid_8_b256/plots/seed${seed}"
    mkdir -p "$out"
    log "plot seed${seed} -> ${out}"
    .venv/bin/python -m final.toy.plot_comparison \
      --suite hypergrid_8 --all-methods --last-common-checkpoint \
      --grpo-run "$grpo" --ips-run "$ips" --mips-grpo-run "$mips" \
      --out-dir "$out" --num-samples 10000 \
      >> "$LOG" 2>&1 || log "plot seed${seed} FAILED"
  done
  log "Plots complete."
}

log "Watcher started (target epoch=${TARGET_EPOCH}, poll=${POLL_SEC}s)"

while true; do
  remaining=$(all_hypergrid_count)
  if [[ "$remaining" -eq 0 ]]; then
    resume_freed_gpus
    run_plots
    exit 0
  fi
  stop_jobs_at_target
  resume_freed_gpus
  sleep "$POLL_SEC"
done
