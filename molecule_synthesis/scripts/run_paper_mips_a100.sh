#!/usr/bin/env bash
set -euo pipefail

# One-command setup, verification, and launch for the first publication-scale
# MIPS-GRPO run. Re-running reuses the virtual environment and pip cache.

PYTHON_BIN="python3.11"
VENV_PATH=""
CACHE_PATH=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOLECULE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${MOLECULE_DIR}/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: run_paper_mips_a100.sh [options]

Options:
  --python PATH       Python 3.11 executable (default: python3.11)
  --venv PATH         Persistent virtual environment path
                      (default: <repo>/.venv-rgfn-cu118)
  --cache-dir PATH    Persistent pip cache path
                      (default: <repo>/.cache/pip)
  -h, --help          Show this help

Run this inside an interactive scheduler allocation or tmux session with an
A100. The script installs/reuses the environment, verifies MIPS, and launches
paper-scale sEH seed 0 including 100,000 final-checkpoint samples.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --venv)
      VENV_PATH="$2"
      shift 2
      ;;
    --cache-dir)
      CACHE_PATH="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv-rgfn-cu118}"
CACHE_PATH="${CACHE_PATH:-${REPO_ROOT}/.cache/pip}"

bash "${SCRIPT_DIR}/bootstrap_a100.sh" \
  --python "${PYTHON_BIN}" \
  --venv "${VENV_PATH}" \
  --cache-dir "${CACHE_PATH}"

VENV_PYTHON="${VENV_PATH}/bin/python"
cd "${REPO_ROOT}"

"${VENV_PYTHON}" -c \
  'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(f"cuda=OK device={torch.cuda.get_device_name(0)}")'
"${VENV_PYTHON}" -m pytest -q molecule_synthesis/tests

RUN_METADATA_DIR="${REPO_ROOT}/molecule_synthesis/runs/seh_paper_main"
mkdir -p "${RUN_METADATA_DIR}"
"${VENV_PYTHON}" -m pip freeze > "${RUN_METADATA_DIR}/environment.freeze.txt"
git rev-parse HEAD > "${RUN_METADATA_DIR}/code_commit.txt"

echo "Launching corrected on-policy MIPS-GRPO: paper-scale sEH, seed 0."
echo "Expected configuration: 4000x100 trajectories, max depth 4, no replay, no exploration."
echo "MIPS: w(tau)=R(x)q_phi(tau|x)/P_F(tau), forward LR 1e-4, reverse LR 1e-3, four reverse MLE updates."

/usr/bin/time -p "${VENV_PYTHON}" -m molecule_synthesis.pipeline \
  --suite seh_paper_main \
  --seed 0 \
  --method mips_grpo \
  --device cuda \
  --wandb-mode offline

"${VENV_PYTHON}" -m molecule_synthesis.verify_mips_run \
  --suite-dir "${RUN_METADATA_DIR}" \
  --seed 0
