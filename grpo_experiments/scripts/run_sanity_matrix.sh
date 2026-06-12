#!/usr/bin/env bash
# Run the 6-experiment sanity matrix (PhyloGFN / GRPO / IPS-GRPO).
#
# Usage (repo root):
#   bash grpo_experiments/scripts/run_sanity_matrix.sh
#   EPOCHS=10 bash grpo_experiments/scripts/run_sanity_matrix.sh
#
# After training:
#   .venv/bin/python grpo_experiments/scripts/evaluate_runs.py \
#     --manifest "$MANIFEST" --output-dir "$MATRIX_ROOT/eval"
#   .venv/bin/python grpo_experiments/scripts/compare_sampling.py \
#     --manifest "$MANIFEST" --output-dir "$MATRIX_ROOT/sampling" --samples 512

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi

EPOCHS="${EPOCHS:-5}"
STEPS="${STEPS:-10}"
SEED="${SEED:-0}"
# Checkpoints: set CHECKPOINT_EVERY>0 to save periodic .pt files plus training_state.json
# for resume. Continue a run with:
#   python -m grpo_experiments.train --resume-from <run_dir> --epochs 200 ...
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1}"
ON_POLICY="${ON_POLICY:-256}"
REPLAY="${REPLAY:-256}"
BUFFER="${BUFFER:-512}"
REPLAY_HEAP="${REPLAY_HEAP:-512}"
GRPO_LR="${GRPO_LR:-1e-4}"
POLICY_IS_LR="${POLICY_IS_LR:-5e-5}"
UPDATE_CYCLES="${UPDATE_CYCLES:-2}"
RESAMPLE_ROUNDS="${RESAMPLE_ROUNDS:-25}"
IS_RATIO_CLIP="${IS_RATIO_CLIP:-0.2}"
IS_RATIO_MAX="${IS_RATIO_MAX:-5.0}"
IS_LOG_RATIO_MAX="${IS_LOG_RATIO_MAX:-2.0}"
IPS_PROB_FLOOR="${IPS_PROB_FLOOR:-0.05}"
DEVICE="${DEVICE:-cuda:0}"

DS="dataset/benchmark_datasets/DS1_reduced.pickle"
CFG="src/configs/benchmark_dna_cfgs/discrete_branch_lengths/cfg_0.001binsize_50bins_temperature_anneal_0.4.yaml"

MATRIX_ID="${MATRIX_ID:-$(date +%Y%m%d_%H%M%S)}"
MATRIX_ROOT="${MATRIX_ROOT:-grpo_experiments/runs/sanity_matrix/${MATRIX_ID}}"
OUTPUT_ROOT="$MATRIX_ROOT/train"
MANIFEST="$MATRIX_ROOT/manifest.json"

mkdir -p "$OUTPUT_ROOT"

declare -a RUN_IDS=(
  "exp1_phylgfn_replay"
  "exp2_phylgfn_noreplay"
  "exp3_grpo_replay"
  "exp4_grpo_is"
  "exp5_ips_replay"
  "exp6_ips_is"
)

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

write_manifest() {
  local rows_file="$1"
  "$PYTHON" - <<PY
import json
from pathlib import Path

manifest_path = "$MANIFEST"
output_root = "$OUTPUT_ROOT"
rows_file = "$rows_file"
epochs = int("$EPOCHS")
steps = int("$STEPS")
seed = int("$SEED")
on_policy = int("$ON_POLICY")
replay = int("$REPLAY")
buffer = int("$BUFFER")

runs = []
with open(rows_file) as handle:
    for line in handle:
        line = line.strip()
        if not line:
            continue
        run_id, label, method, run_dir = line.split("\t")
        runs.append({
            "id": run_id,
            "label": label,
            "method": method,
            "run_dir": run_dir,
        })

payload = {
    "matrix_id": Path(manifest_path).parent.name,
    "output_root": str(Path(output_root).resolve()),
    "shared": {
        "epochs": epochs,
        "steps_per_epoch": steps,
        "seed": seed,
        "group_size": on_policy + replay,
        "on_policy_batch_size": on_policy,
        "replay_batch_size": replay,
        "policy_is_buffer_size": buffer,
        "dataset": "dataset/benchmark_datasets/DS1_reduced.pickle",
    },
    "runs": runs,
}
Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
Path(manifest_path).write_text(json.dumps(payload, indent=2) + "\n")
print(f"wrote manifest: {manifest_path}")
PY
}

echo "=== sanity matrix ==="
echo "matrix_root:  $MATRIX_ROOT"
echo "output_root:  $OUTPUT_ROOT"
echo "G (replay):   $((ON_POLICY + REPLAY)) = ${ON_POLICY} on-policy + ${REPLAY} replay"
echo "G (policy IS): ${BUFFER}"
echo "epochs/steps: ${EPOCHS}/${STEPS}  seed=${SEED}  checkpoint_every=${CHECKPOINT_EVERY}"
echo "device:       ${DEVICE}"
echo

MANIFEST_ROWS_FILE="$MATRIX_ROOT/manifest_rows.tsv"
: > "$MANIFEST_ROWS_FILE"

POLICY_IS_ARGS=(
  --resample-rounds "$RESAMPLE_ROUNDS"
  --update-cycles "$UPDATE_CYCLES"
  --is-ratio-clip "$IS_RATIO_CLIP"
  --is-ratio-max "$IS_RATIO_MAX"
  --is-log-ratio-max "$IS_LOG_RATIO_MAX"
)

run_train() {
  local run_id="$1"
  shift
  echo "--- training: ${run_id} ---"
  "$PYTHON" "$@"
  local method_suffix="${RUN_METHODS[$run_id]}"
  local run_dir
  run_dir="$(find_run_dir "$run_id" "$method_suffix")"
  echo -e "${run_id}\t${RUN_LABELS[$run_id]}\t${method_suffix}\t${run_dir}" >> "$MANIFEST_ROWS_FILE"
  echo "run_dir: ${run_dir}"
  echo
}

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

write_manifest "$MANIFEST_ROWS_FILE"

echo "=== done ==="
echo "manifest: $MANIFEST"
echo
echo "Next steps:"
echo "  MANIFEST=$MANIFEST"
echo "  $PYTHON -m grpo_experiments.scripts.evaluate_runs \\"
echo "    --manifest \$MANIFEST --output-dir $MATRIX_ROOT/eval"
echo "  $PYTHON -m grpo_experiments.scripts.compare_sampling \\"
echo "    --manifest \$MANIFEST --output-dir $MATRIX_ROOT/sampling --samples 1000 --n-bins 10"
