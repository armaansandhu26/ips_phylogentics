#!/usr/bin/env bash
# Pause 27t jobs on GPU 1 & 3 at next checkpoint (SIGSTOP — resume with kill -CONT PID).
set -euo pipefail

LOG="/home/armaan/phylogfn/final/runs/pause_27t_at_checkpoint.log"
REPO="/home/armaan/phylogfn"

LRIPS_PID=1514523
LRIPS_CKPT="${REPO}/final/runs/27taxa_noreplay_b4096_seed1/learned_reverse/20260825_210645_learned_reverse_27taxa_noreplay_b4096_seed1_learned_reverse_ips_grpo/checkpoint_epoch5999.pt"

PHYLGFN_PID=1583706
PHYLGFN_CKPT="${REPO}/final/runs/27taxa_noreplay_b4096_seed2/phylgfn/20260825_223123_phylgfn_27taxa_noreplay_b4096_seed2/checkpoints/checkpoint_025999.pt"

log() { echo "$(date -Iseconds) $*" | tee -a "$LOG"; }

pause_if_running() {
  local pid=$1
  local label=$2
  if kill -0 "$pid" 2>/dev/null; then
    kill -STOP "$pid"
    log "PAUSED $label (PID $pid) via SIGSTOP"
  else
    log "SKIP $label — PID $pid not running"
  fi
}

wait_for_ckpt() {
  local ckpt=$1
  local label=$2
  local pid=$3
  log "Watching $label — waiting for $(basename "$ckpt") (PID $pid)"
  while [[ ! -f "$ckpt" ]]; do
    if ! kill -0 "$pid" 2>/dev/null; then
      log "ERROR $label PID $pid exited before checkpoint"
      return 1
    fi
    sleep 30
  done
  sleep 5
  log "Checkpoint found: $ckpt"
  pause_if_running "$pid" "$label"
}

log "=== pause watcher started ==="
log "LR-IPS seed1 target: checkpoint_epoch5999.pt (~epoch 6000)"
log "PhyloGFN seed2 target: checkpoint_025999.pt (~epoch 26000)"
log "Resume later: kill -CONT $LRIPS_PID $PHYLGFN_PID"

wait_for_ckpt "$LRIPS_CKPT" "27t LR-IPS seed1 (GPU 1)" "$LRIPS_PID" &
wait_for_ckpt "$PHYLGFN_CKPT" "27t PhyloGFN seed2 (GPU 3)" "$PHYLGFN_PID" &
wait
log "=== both watchers done ==="
