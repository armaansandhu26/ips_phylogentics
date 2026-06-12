#!/usr/bin/env bash
# Continue a partially failed sanity matrix (exp3 resume + exp4-6).
# Usage: MATRIX_ROOT=grpo_experiments/runs/sanity_matrix/20260531_083428 bash ...

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
MATRIX_ROOT="${MATRIX_ROOT:?set MATRIX_ROOT to the sanity matrix directory}"
OUTPUT_ROOT="$MATRIX_ROOT/train"
MANIFEST_ROWS_FILE="$MATRIX_ROOT/manifest_rows.tsv"
DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-5}"
STEPS="${STEPS:-10}"
SEED="${SEED:-0}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1}"
ON_POLICY="${ON_POLICY:-256}"
REPLAY="${REPLAY:-256}"
BUFFER="${BUFFER:-512}"
REPLAY_HEAP="${REPLAY_HEAP:-512}"
GRPO_LR="${GRPO_LR:-1e-4}"
DS="dataset/benchmark_datasets/DS1_reduced.pickle"
CFG="src/configs/benchmark_dna_cfgs/discrete_branch_lengths/cfg_0.001binsize_50bins_temperature_anneal_0.4.yaml"

EXP3_DIR="$OUTPUT_ROOT/20260531_083459_exp3_grpo_replay_grpo"

append_manifest() {
  local run_id="$1" label="$2" method="$3" run_dir="$4"
  echo -e "${run_id}\t${label}\t${method}\t${run_dir}" >> "$MANIFEST_ROWS_FILE"
}

write_manifest() {
  "$PYTHON" - <<PY
import json
from pathlib import Path

manifest_path = Path("$MATRIX_ROOT/manifest.json")
rows_file = Path("$MANIFEST_ROWS_FILE")
runs = []
for line in rows_file.read_text().splitlines():
    if not line.strip():
        continue
    run_id, label, method, run_dir = line.split("\t")
    runs.append({"id": run_id, "label": label, "method": method, "run_dir": run_dir})

payload = {
    "matrix_id": manifest_path.parent.name,
    "output_root": str(Path("$OUTPUT_ROOT").resolve()),
    "shared": {
        "epochs": int("$EPOCHS"),
        "steps_per_epoch": int("$STEPS"),
        "seed": int("$SEED"),
        "group_size": int("$ON_POLICY") + int("$REPLAY"),
        "on_policy_batch_size": int("$ON_POLICY"),
        "replay_batch_size": int("$REPLAY"),
        "policy_is_buffer_size": int("$BUFFER"),
        "device": "$DEVICE",
    },
    "runs": runs,
}
manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
print(f"wrote manifest: {manifest_path}")
PY
}

echo "=== continuing matrix at $MATRIX_ROOT on $DEVICE ==="

echo "--- resume exp3_grpo_replay ---"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON" -m grpo_experiments.train --method grpo \
  --resume-from "$EXP3_DIR" \
  --output "$OUTPUT_ROOT" \
  --dataset "$DS" --cfg "$CFG" \
  --epochs "$EPOCHS" --steps-per-epoch "$STEPS" --seed "$SEED" --device "$DEVICE" \
  --on-policy-batch-size "$ON_POLICY" --replay-batch-size "$REPLAY" \
  --replay-buffer-size "$REPLAY_HEAP" \
  --grpo-lr "$GRPO_LR" \
  --checkpoint-every "$CHECKPOINT_EVERY"
append_manifest exp3_grpo_replay "GRPO + replay" grpo "$EXP3_DIR"

run_new() {
  local run_id="$1" label="$2" method_suffix="$3"
  shift 3
  echo "--- training: ${run_id} ---"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON" "$@"
  local run_dir
  run_dir="$(find "$OUTPUT_ROOT" -maxdepth 1 -type d -name "*_${run_id}_${method_suffix}" | sort | tail -1)"
  append_manifest "$run_id" "$label" "$method_suffix" "$run_dir"
  echo "run_dir: $run_dir"
}

run_new exp4_grpo_is "GRPO + policy IS" grpo \
  -m grpo_experiments.train --method grpo --enable-policy-is \
  --run-name exp4_grpo_is \
  --output "$OUTPUT_ROOT" \
  --dataset "$DS" --cfg "$CFG" \
  --epochs "$EPOCHS" --steps-per-epoch "$STEPS" --seed "$SEED" --device "$DEVICE" \
  --on-policy-batch-size "$BUFFER" --buffer-size "$BUFFER" --disable-replay \
  --grpo-lr "${POLICY_IS_LR:-5e-5}" \
  --resample-rounds "${RESAMPLE_ROUNDS:-25}" \
  --update-cycles "${UPDATE_CYCLES:-2}" \
  --is-ratio-clip "${IS_RATIO_CLIP:-0.2}" \
  --is-ratio-max "${IS_RATIO_MAX:-5.0}" \
  --is-log-ratio-max "${IS_LOG_RATIO_MAX:-2.0}" \
  --checkpoint-every "$CHECKPOINT_EVERY"

run_new exp5_ips_replay "IPS-GRPO + replay" ips_grpo \
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

run_new exp6_ips_is "IPS-GRPO + policy IS" ips_grpo \
  -m grpo_experiments.ips_grpo.train --enable-policy-is \
  --run-name exp6_ips_is \
  --output "$OUTPUT_ROOT" \
  --dataset "$DS" --cfg "$CFG" \
  --epochs "$EPOCHS" --steps-per-epoch "$STEPS" --seed "$SEED" --device "$DEVICE" \
  --on-policy-batch-size "$BUFFER" --buffer-size "$BUFFER" --disable-replay \
  --outcome-level topology \
  --grpo-lr "${POLICY_IS_LR:-5e-5}" \
  --ips-prob-floor "${IPS_PROB_FLOOR:-0.05}" \
  --resample-rounds "${RESAMPLE_ROUNDS:-25}" \
  --update-cycles "${UPDATE_CYCLES:-2}" \
  --is-ratio-clip "${IS_RATIO_CLIP:-0.2}" \
  --is-ratio-max "${IS_RATIO_MAX:-5.0}" \
  --is-log-ratio-max "${IS_LOG_RATIO_MAX:-2.0}" \
  --checkpoint-every "$CHECKPOINT_EVERY"

write_manifest
echo "=== continue complete ==="
