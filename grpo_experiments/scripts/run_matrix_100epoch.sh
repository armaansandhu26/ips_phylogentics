#!/usr/bin/env bash
# 100-epoch × 100-step matrix (6 experiments). Run one shard on a single GPU, or all
# sequentially. Locked configs from the 100-epoch plan.
#
# Usage (repo root):
#   bash grpo_experiments/scripts/run_matrix_100epoch.sh
#   SHARD=2 MATRIX_ROOT=... bash grpo_experiments/scripts/run_matrix_100epoch.sh
#
# Parallel (3 GPUs):
#   bash grpo_experiments/scripts/run_matrix_100epoch_parallel.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi

# --- locked training config ---
EPOCHS="${EPOCHS:-100}"
STEPS="${STEPS:-100}"
SEED="${SEED:-0}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5}"
ON_POLICY="${ON_POLICY:-256}"
REPLAY="${REPLAY:-256}"
REPLAY_HEAP="${REPLAY_HEAP:-1024}"
BUFFER="${BUFFER:-512}"
GRPO_LR="${GRPO_LR:-1e-4}"
DEVICE="${DEVICE:-cuda:0}"

# Policy IS (exp4, exp6): resample often, same total updates as replay runs.
RESAMPLE_ROUNDS="${RESAMPLE_ROUNDS:-5000}"
UPDATE_CYCLES="${UPDATE_CYCLES:-2}"
POLICY_IS_LR="${POLICY_IS_LR:-5e-5}"
IS_RATIO_CLIP="${IS_RATIO_CLIP:-0.2}"
IS_RATIO_MAX="${IS_RATIO_MAX:-5.0}"
IS_LOG_RATIO_MAX="${IS_LOG_RATIO_MAX:-2.0}"
IPS_PROB_FLOOR="${IPS_PROB_FLOOR:-0.05}"
ROLLOUT_CHUNK="${ROLLOUT_CHUNK:-128}"

DS="dataset/benchmark_datasets/DS1_reduced.pickle"
CFG="src/configs/benchmark_dna_cfgs/discrete_branch_lengths/cfg_0.001binsize_50bins_temperature_anneal_0.4.yaml"

MATRIX_ID="${MATRIX_ID:-$(date +%Y%m%d_%H%M%S)}"
MATRIX_ROOT="${MATRIX_ROOT:-grpo_experiments/runs/matrix_100epoch/${MATRIX_ID}}"
OUTPUT_ROOT="$MATRIX_ROOT/train"
MANIFEST="$MATRIX_ROOT/manifest.json"

# SHARD: 1=exp1-2, 2=exp3-4, 3=exp5-6, unset=all
SHARD="${SHARD:-}"
if [[ -n "$SHARD" ]]; then
  MANIFEST_ROWS_FILE="$MATRIX_ROOT/manifest_rows_shard${SHARD}.tsv"
else
  MANIFEST_ROWS_FILE="$MATRIX_ROOT/manifest_rows.tsv"
fi

mkdir -p "$OUTPUT_ROOT" "$MATRIX_ROOT"

declare -A RUN_LABELS=(
  [exp1_phylgfn_replay]="PhyloGFN + replay"
  [exp2_phylgfn_noreplay]="PhyloGFN no replay"
  [exp3_grpo_replay]="GRPO + replay"
  [exp4_grpo_is]="GRPO + policy IS"
  [exp5_ips_replay]="IPS-GRPO + replay"
  [exp6_ips_is]="IPS-GRPO + policy IS"
)

declare -A RUN_METHODS=(
  [exp1_phylgfn_replay]="phylgfn"
  [exp2_phylgfn_noreplay]="phylgfn"
  [exp3_grpo_replay]="grpo"
  [exp4_grpo_is]="grpo"
  [exp5_ips_replay]="ips_grpo"
  [exp6_ips_is]="ips_grpo"
)

should_run() {
  local run_id="$1"
  case "$SHARD" in
    "") return 0 ;;
    1) [[ "$run_id" == exp1_* || "$run_id" == exp2_* ]] ;;
    2) [[ "$run_id" == exp3_* || "$run_id" == exp4_* ]] ;;
    3) [[ "$run_id" == exp5_* || "$run_id" == exp6_* ]] ;;
    *)
      echo "ERROR: SHARD must be 1, 2, or 3 (got: $SHARD)" >&2
      exit 1
      ;;
  esac
}

find_run_dir() {
  local run_name="$1"
  local method_suffix="$2"
  local matches
  matches="$(find "$OUTPUT_ROOT" -maxdepth 1 -type d -name "*_${run_name}_${method_suffix}" | sort)"
  if [[ -z "$matches" ]]; then
    echo "ERROR: could not find run dir for ${run_name} under ${OUTPUT_ROOT}" >&2
    exit 1
  fi
  echo "$matches" | tail -1
}

append_manifest_row() {
  local run_id="$1" label="$2" method="$3" run_dir="$4"
  echo -e "${run_id}\t${label}\t${method}\t${run_dir}" >> "$MANIFEST_ROWS_FILE"
}

write_manifest() {
  local -a rows_files=("$@")
  local rows_json=""
  rows_json="$(printf '%s\n' "${rows_files[@]}" | "$PYTHON" -c 'import json,sys; print(json.dumps(sys.stdin.read().splitlines()))')"
  "$PYTHON" - <<PY
import json
from pathlib import Path

manifest_path = Path("$MANIFEST")
output_root = Path("$OUTPUT_ROOT")
rows_files = json.loads('''$rows_json''')

runs = []
seen = set()
for rows_file in rows_files:
    path = Path(rows_file)
    if not path.exists():
        continue
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        run_id, label, method, run_dir = line.split("\t")
        if run_id in seen:
            continue
        seen.add(run_id)
        runs.append({
            "id": run_id,
            "label": label,
            "method": method,
            "run_dir": run_dir,
        })

runs.sort(key=lambda row: row["id"])

payload = {
    "matrix_id": manifest_path.parent.name,
    "output_root": str(output_root.resolve()),
    "shared": {
        "epochs": int("$EPOCHS"),
        "steps_per_epoch": int("$STEPS"),
        "total_replay_updates": int("$EPOCHS") * int("$STEPS"),
        "seed": int("$SEED"),
        "checkpoint_every": int("$CHECKPOINT_EVERY"),
        "on_policy_batch_size": int("$ON_POLICY"),
        "replay_batch_size": int("$REPLAY"),
        "replay_buffer_size": int("$REPLAY_HEAP"),
        "grpo_group_size_replay": int("$ON_POLICY") + int("$REPLAY"),
        "policy_is": {
            "buffer_size": int("$BUFFER"),
            "resample_rounds": int("$RESAMPLE_ROUNDS"),
            "update_cycles": int("$UPDATE_CYCLES"),
            "total_policy_is_updates": int("$RESAMPLE_ROUNDS") * int("$UPDATE_CYCLES"),
            "is_ratio_clip": float("$IS_RATIO_CLIP"),
            "is_ratio_max": float("$IS_RATIO_MAX"),
            "is_log_ratio_max": float("$IS_LOG_RATIO_MAX"),
            "ips_prob_floor": float("$IPS_PROB_FLOOR"),
            "policy_is_lr": float("$POLICY_IS_LR"),
            "rollout_chunk_size": int("$ROLLOUT_CHUNK"),
        },
        "dataset": "$DS",
        "grpo_lr": float("$GRPO_LR"),
        "gpus": {"shard1": "cuda:0", "shard2": "cuda:1", "shard3": "cuda:2"},
    },
    "runs": runs,
}
manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
print(f"wrote manifest: {manifest_path} ({len(runs)} runs)")
PY
}

if [[ "${MERGE_MANIFEST:-}" == "1" ]]; then
  write_manifest \
    "$MATRIX_ROOT/manifest_rows_shard1.tsv" \
    "$MATRIX_ROOT/manifest_rows_shard2.tsv" \
    "$MATRIX_ROOT/manifest_rows_shard3.tsv"
  exit 0
fi

POLICY_IS_ARGS=(
  --resample-rounds "$RESAMPLE_ROUNDS"
  --update-cycles "$UPDATE_CYCLES"
  --is-ratio-clip "$IS_RATIO_CLIP"
  --is-ratio-max "$IS_RATIO_MAX"
  --is-log-ratio-max "$IS_LOG_RATIO_MAX"
  --rollout-chunk-size "$ROLLOUT_CHUNK"
)

run_train() {
  local run_id="$1"
  shift
  if ! should_run "$run_id"; then
    return 0
  fi
  echo "--- training: ${run_id} on ${DEVICE} ---"
  "$PYTHON" "$@"
  local method_suffix="${RUN_METHODS[$run_id]}"
  local run_dir
  run_dir="$(find_run_dir "$run_id" "$method_suffix")"
  append_manifest_row "$run_id" "${RUN_LABELS[$run_id]}" "$method_suffix" "$run_dir"
  echo "run_dir: ${run_dir}"
  echo
}

echo "=== matrix 100epoch ==="
echo "matrix_root:  $MATRIX_ROOT"
echo "shard:        ${SHARD:-all}"
echo "device:       $DEVICE"
echo "replay G:     $((ON_POLICY + REPLAY)) = ${ON_POLICY} + ${REPLAY}, heap=${REPLAY_HEAP}"
echo "policy IS:    buffer=${BUFFER}, rounds=${RESAMPLE_ROUNDS}, cycles=${UPDATE_CYCLES}, clip=${IS_RATIO_CLIP}, max_w=${IS_RATIO_MAX}, log_clip=${IS_LOG_RATIO_MAX}, lr=${POLICY_IS_LR}"
echo "schedule:     ${EPOCHS} epochs × ${STEPS} steps (replay) | ${RESAMPLE_ROUNDS}×${UPDATE_CYCLES} (policy IS)"
echo "checkpoint:   every ${CHECKPOINT_EVERY} resample rounds / epochs"
echo

: > "$MANIFEST_ROWS_FILE"

# 1. PhyloGFN + best-tree replay
run_train exp1_phylgfn_replay \
  -m grpo_experiments.train --method phylgfn \
  --run-name exp1_phylgfn_replay \
  --output "$OUTPUT_ROOT" \
  --dataset "$DS" --cfg "$CFG" \
  --epochs "$EPOCHS" --steps-per-epoch "$STEPS" --seed "$SEED" --device "$DEVICE" \
  --on-policy-batch-size "$ON_POLICY" --replay-batch-size "$REPLAY" \
  --replay-buffer-size "$REPLAY_HEAP" \
  --grpo-lr "$GRPO_LR" \
  --checkpoint-every "$CHECKPOINT_EVERY"

# 2. PhyloGFN, no replay
run_train exp2_phylgfn_noreplay \
  -m grpo_experiments.train --method phylgfn \
  --run-name exp2_phylgfn_noreplay \
  --output "$OUTPUT_ROOT" \
  --dataset "$DS" --cfg "$CFG" \
  --epochs "$EPOCHS" --steps-per-epoch "$STEPS" --seed "$SEED" --device "$DEVICE" \
  --on-policy-batch-size "$BUFFER" --disable-replay \
  --grpo-lr "$GRPO_LR" \
  --checkpoint-every "$CHECKPOINT_EVERY"

# 3. GRPO + best-tree replay
run_train exp3_grpo_replay \
  -m grpo_experiments.train --method grpo \
  --run-name exp3_grpo_replay \
  --output "$OUTPUT_ROOT" \
  --dataset "$DS" --cfg "$CFG" \
  --epochs "$EPOCHS" --steps-per-epoch "$STEPS" --seed "$SEED" --device "$DEVICE" \
  --on-policy-batch-size "$ON_POLICY" --replay-batch-size "$REPLAY" \
  --replay-buffer-size "$REPLAY_HEAP" \
  --grpo-lr "$GRPO_LR" \
  --checkpoint-every "$CHECKPOINT_EVERY"

# 4. GRPO + policy IS
run_train exp4_grpo_is \
  -m grpo_experiments.train --method grpo --enable-policy-is \
  --run-name exp4_grpo_is \
  --output "$OUTPUT_ROOT" \
  --dataset "$DS" --cfg "$CFG" \
  --epochs "$EPOCHS" --steps-per-epoch "$STEPS" --seed "$SEED" --device "$DEVICE" \
  --on-policy-batch-size "$BUFFER" --buffer-size "$BUFFER" --disable-replay \
  --grpo-lr "$POLICY_IS_LR" \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  "${POLICY_IS_ARGS[@]}"

# 5. IPS-GRPO + best-tree replay
run_train exp5_ips_replay \
  -m grpo_experiments.ips_grpo.train \
  --run-name exp5_ips_replay \
  --output "$OUTPUT_ROOT" \
  --dataset "$DS" --cfg "$CFG" \
  --epochs "$EPOCHS" --steps-per-epoch "$STEPS" --seed "$SEED" --device "$DEVICE" \
  --on-policy-batch-size "$ON_POLICY" --replay-batch-size "$REPLAY" \
  --replay-buffer-size "$REPLAY_HEAP" \
  --outcome-level topology \
  --grpo-lr "$GRPO_LR" \
  --checkpoint-every "$CHECKPOINT_EVERY"

# 6. IPS-GRPO + policy IS
run_train exp6_ips_is \
  -m grpo_experiments.ips_grpo.train --enable-policy-is \
  --run-name exp6_ips_is \
  --output "$OUTPUT_ROOT" \
  --dataset "$DS" --cfg "$CFG" \
  --epochs "$EPOCHS" --steps-per-epoch "$STEPS" --seed "$SEED" --device "$DEVICE" \
  --on-policy-batch-size "$BUFFER" --buffer-size "$BUFFER" --disable-replay \
  --outcome-level topology \
  --grpo-lr "$POLICY_IS_LR" \
  --ips-prob-floor "$IPS_PROB_FLOOR" \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  "${POLICY_IS_ARGS[@]}"

if [[ -z "$SHARD" ]]; then
  write_manifest "$MANIFEST_ROWS_FILE"
else
  echo "shard ${SHARD} complete (manifest merged by parallel launcher)"
fi

echo "=== shard done ==="
echo "matrix_root: $MATRIX_ROOT"
if [[ -f "$MANIFEST" ]]; then
  echo "manifest:    $MANIFEST"
fi
