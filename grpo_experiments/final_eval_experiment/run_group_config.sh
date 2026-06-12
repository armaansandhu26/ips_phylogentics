#!/usr/bin/env bash
# Run one final-eval group config on a single GPU.
# Usage:
#   bash grpo_experiments/final_eval_experiment/run_group_config.sh \
#     grpo_experiments/final_eval_experiment/config_r32_50k.json cuda:0

set -euo pipefail

cd "$(dirname "$0")/../.."
ROOT="$(pwd)"
PY="${ROOT}/.venv/bin/python"

CONFIG_PATH="${1:-}"
GPU="${2:-cuda:0}"

if [[ -z "${CONFIG_PATH}" ]]; then
  echo "Usage: $0 <config-json> [gpu]" >&2
  exit 1
fi

if [[ "${CONFIG_PATH}" != /* ]]; then
  CONFIG_PATH="${ROOT}/${CONFIG_PATH}"
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config file not found: ${CONFIG_PATH}" >&2
  exit 1
fi

read_config_field() {
  local field="$1"
  "$PY" - <<PY
import json
from pathlib import Path
cfg = json.loads(Path("${CONFIG_PATH}").read_text())
value = cfg["${field}"]
print(value)
PY
}

GROUP="$(read_config_field group)"
OUTCOME="$(read_config_field outcome)"
OUTPUT_ROOT="$("$PY" - <<PY
import json
from pathlib import Path
cfg = json.loads(Path("${CONFIG_PATH}").read_text())
out = cfg["output_root"]
if not out.startswith("/"):
    out = str(Path("${ROOT}") / out)
print(out)
PY
)"
LOG_DIR="${ROOT}/grpo_experiments/final_eval_experiment/logs"
MANIFEST="${ROOT}/grpo_experiments/final_eval_experiment/manifest_${GROUP}_50k.json"

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"

cat > "${MANIFEST}" <<EOF
{
  "experiment": "final_eval_group_run",
  "group": "${GROUP}",
  "config_path": "${CONFIG_PATH}",
  "gpu": "${GPU}",
  "runs": []
}
EOF

append_manifest() {
  local id="$1"
  local method="$2"
  local run_dir="$3"
  local pfloor="${4:-}"
  "$PY" - <<PY
import json
from pathlib import Path

p = Path("${MANIFEST}")
data = json.loads(p.read_text())
row = {
    "id": "${id}",
    "method": "${method}",
    "run_dir": "${run_dir}",
    "group": "${GROUP}",
    "outcome": "${OUTCOME}",
    "gpu": "${GPU}",
}
pfloor = "${pfloor}"
if pfloor:
    row["ips_prob_floor"] = float(pfloor)
data["runs"].append(row)
p.write_text(json.dumps(data, indent=2) + "\n")
PY
}

extract_run_dir() {
  local log_path="$1"
  "$PY" - <<PY
from pathlib import Path
import re
text = Path("${log_path}").read_text()
matches = re.findall(r"saved to:\s*(.+)", text)
print(matches[-1] if matches else "")
PY
}

run_from_row() {
  local row_idx="$1"
  "$PY" - <<PY
import json
import shlex
from pathlib import Path

cfg = json.loads(Path("${CONFIG_PATH}").read_text())
row = cfg["runs"][${row_idx}]

def get(key, default=None):
    if key in row:
        return row[key]
    return cfg.get(key, default)

method = row["method"]
run_name = row["run_name"]
dataset = cfg["dataset_path"]
cfg_path = cfg["cfg_path"]
output_root = cfg["output_root"]
if not output_root.startswith("/"):
    output_root = str(Path("${ROOT}") / output_root)
outcome_level = cfg["outcome_level"]
seed = get("seed", cfg.get("seed", 0))
print_every = get("print_every", cfg.get("print_every", 1000))
checkpoint_every = get("checkpoint_every", cfg.get("checkpoint_every", 0))
rollout_chunk_size = get("rollout_chunk_size", cfg.get("rollout_chunk_size", 512))
grpo_lr = get("grpo_lr", cfg.get("grpo_lr", 1e-4))
entropy_coef = get("entropy_coef", cfg.get("entropy_coef", 0.01))
warmstart = get("replay_warmstart_samples", cfg.get("replay_warmstart_samples", 64))

cmd = ["${PY}"]
pfloor = ""

if method == "phylgfn":
    cmd += [
        "-m", "grpo_experiments.train",
        "--method", "phylgfn",
        "--run-name", run_name,
        "--device", "${GPU}",
        "--dataset", dataset,
        "--cfg", cfg_path,
        "--output", output_root,
        "--seed", str(seed),
        "--epochs", str(row["epochs"]),
        "--steps-per-epoch", str(row["steps_per_epoch"]),
        "--on-policy-batch-size", str(row["on_policy_batch_size"]),
        "--replay-batch-size", str(row["replay_batch_size"]),
        "--replay-buffer-size", str(row["replay_buffer_size"]),
        "--outcome-level", outcome_level,
        "--checkpoint-every", str(checkpoint_every),
        "--print-every", str(print_every),
    ]
elif method == "hybrid_grpo":
    cmd += [
        "-m", "grpo_experiments.hybrid_grpo.train",
        "--run-name", run_name,
        "--device", "${GPU}",
        "--dataset", dataset,
        "--cfg", cfg_path,
        "--output", output_root,
        "--seed", str(seed),
        "--resample-rounds", str(row["resample_rounds"]),
        "--update-cycles", str(row["update_cycles"]),
        "--fresh-buffer-size", str(row["fresh_buffer_size"]),
        "--replay-sample-size", str(row["replay_sample_size"]),
        "--best-tree-buffer-size", str(row["best_tree_buffer_size"]),
        "--replay-warmstart-samples", str(warmstart),
        "--rollout-chunk-size", str(rollout_chunk_size),
        "--outcome-level", outcome_level,
        "--checkpoint-every", str(checkpoint_every),
        "--print-every", str(print_every),
        "--grpo-lr", str(grpo_lr),
        "--entropy-coef", str(entropy_coef),
        "--no-log-trajectories",
    ]
elif method == "hybrid_ips_grpo":
    pfloor = str(row["ips_prob_floor"])
    cmd += [
        "-m", "grpo_experiments.hybrid_ips_grpo.train",
        "--run-name", run_name,
        "--device", "${GPU}",
        "--dataset", dataset,
        "--cfg", cfg_path,
        "--output", output_root,
        "--seed", str(seed),
        "--resample-rounds", str(row["resample_rounds"]),
        "--update-cycles", str(row["update_cycles"]),
        "--fresh-buffer-size", str(row["fresh_buffer_size"]),
        "--replay-sample-size", str(row["replay_sample_size"]),
        "--best-tree-buffer-size", str(row["best_tree_buffer_size"]),
        "--replay-warmstart-samples", str(warmstart),
        "--rollout-chunk-size", str(rollout_chunk_size),
        "--outcome-level", outcome_level,
        "--checkpoint-every", str(checkpoint_every),
        "--print-every", str(print_every),
        "--grpo-lr", str(grpo_lr),
        "--entropy-coef", str(entropy_coef),
        "--ips-prob-floor", pfloor,
        "--no-log-trajectories",
    ]
else:
    raise SystemExit(f"Unsupported method: {method}")

log_path = Path("${LOG_DIR}") / f"{cfg['group']}_{run_name}.log"
print(method)
print(run_name)
print(log_path)
print(pfloor)
print(shlex.join(cmd))
PY
}

NUM_RUNS="$("$PY" - <<PY
import json
from pathlib import Path
cfg = json.loads(Path("${CONFIG_PATH}").read_text())
print(len(cfg["runs"]))
PY
)"

echo "Starting group=${GROUP} on ${GPU} using ${CONFIG_PATH}"

for ((i = 0; i < NUM_RUNS; i++)); do
  mapfile -t ROW_INFO < <(run_from_row "${i}")
  METHOD="${ROW_INFO[0]}"
  RUN_NAME="${ROW_INFO[1]}"
  LOG_PATH="${ROW_INFO[2]}"
  PFLOOR="${ROW_INFO[3]}"
  CMD="${ROW_INFO[4]}"

  echo "[${GROUP}] ${RUN_NAME} (${METHOD}) on ${GPU}"
  bash -lc "${CMD}" > "${LOG_PATH}" 2>&1

  RUN_DIR="$(extract_run_dir "${LOG_PATH}")"
  append_manifest "${RUN_NAME}" "${METHOD}" "${RUN_DIR}" "${PFLOOR}"
done

echo "Completed group=${GROUP} on ${GPU}"
echo "Manifest: ${MANIFEST}"
