#!/usr/bin/env bash
set -euo pipefail

# Fast, repeatable Linux/A100 setup for the public sEH and QED suites.
# A persistent venv and pip cache can be shared across multiple repository
# clones, while the small editable RGFN install is refreshed for each clone.

PYTHON_BIN="python3.11"
VENV_PATH=""
CACHE_PATH=""
RGFN_URL="https://github.com/koziarskilab/RGFN.git"
RGFN_COMMIT="6ce59169f855ed18f34ba4e8279de93bee306e4f"
ENV_VERSION="seh-cu118-v1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOLECULE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${MOLECULE_DIR}/.." && pwd)"
RGFN_DIR="${MOLECULE_DIR}/external/RGFN"
PATCH_PATH="${MOLECULE_DIR}/patches/rgfn_minimal_proxies.patch"
REQUIREMENTS_PATH="${MOLECULE_DIR}/environments/requirements-seh-cu118.txt"

usage() {
  cat <<'EOF'
Usage: bootstrap_a100.sh [options]

Options:
  --python PATH       Python 3.11 executable (default: python3.11)
  --venv PATH         Persistent virtual environment path
                      (default: <repo>/.venv-rgfn-cu118)
  --cache-dir PATH    Persistent pip cache path
                      (default: <repo>/.cache/pip)
  -h, --help          Show this help

For reuse across repository clones, put --venv and --cache-dir on persistent
scratch or shared storage. Do not reuse this Linux/CUDA environment on macOS.
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

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "bootstrap_a100.sh requires Linux; use setup_env.sh for macOS/CPU." >&2
  exit 1
fi
if ! command -v git >/dev/null 2>&1; then
  echo "git is required." >&2
  exit 1
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python 3.11 was not found at ${PYTHON_BIN}." >&2
  echo "Load your cluster's Python 3.11 module or pass --python /path/to/python3.11." >&2
  exit 1
fi

PYTHON_VERSION="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${PYTHON_VERSION}" != "3.11" ]]; then
  echo "Python 3.11 is required; ${PYTHON_BIN} reports ${PYTHON_VERSION}." >&2
  exit 1
fi

VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv-rgfn-cu118}"
CACHE_PATH="${CACHE_PATH:-${REPO_ROOT}/.cache/pip}"
mkdir -p "${CACHE_PATH}" "${MOLECULE_DIR}/external"
export PIP_CACHE_DIR="${CACHE_PATH}"

if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
  echo "Creating virtual environment at ${VENV_PATH}"
  "${PYTHON_BIN}" -m venv "${VENV_PATH}"
fi
VENV_PYTHON="${VENV_PATH}/bin/python"
MARKER_PATH="${VENV_PATH}/.${ENV_VERSION}"

if [[ ! -d "${RGFN_DIR}/.git" ]]; then
  echo "Cloning pinned RGFN checkout"
  git clone --filter=blob:none "${RGFN_URL}" "${RGFN_DIR}"
elif [[ "$(git -C "${RGFN_DIR}" remote get-url origin)" != "${RGFN_URL}" ]]; then
  echo "Refusing to modify ${RGFN_DIR}: unexpected Git remote." >&2
  exit 1
fi

if ! git -C "${RGFN_DIR}" cat-file -e "${RGFN_COMMIT}^{commit}" 2>/dev/null; then
  git -C "${RGFN_DIR}" fetch --depth 1 origin "${RGFN_COMMIT}"
fi
CURRENT_COMMIT="$(git -C "${RGFN_DIR}" rev-parse HEAD)"
if [[ "${CURRENT_COMMIT}" != "${RGFN_COMMIT}" ]]; then
  if git -C "${RGFN_DIR}" diff --quiet && git -C "${RGFN_DIR}" diff --cached --quiet; then
    git -C "${RGFN_DIR}" checkout --detach "${RGFN_COMMIT}"
  else
    echo "RGFN has local changes and is not at the pinned commit." >&2
    echo "Expected ${RGFN_COMMIT}; found ${CURRENT_COMMIT}." >&2
    exit 1
  fi
fi

if git -C "${RGFN_DIR}" apply --check "${PATCH_PATH}" 2>/dev/null; then
  git -C "${RGFN_DIR}" apply "${PATCH_PATH}"
elif ! git -C "${RGFN_DIR}" apply --reverse --check "${PATCH_PATH}" 2>/dev/null; then
  echo "Unable to apply or recognize ${PATCH_PATH}." >&2
  exit 1
fi

if [[ ! -f "${MARKER_PATH}" ]]; then
  echo "Installing pinned CUDA 11.8 environment (first run only)"
  "${VENV_PYTHON}" -m pip install --upgrade 'pip<25.3'
  "${VENV_PYTHON}" -m pip install \
    torch==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu118
  "${VENV_PYTHON}" -m pip install \
    dgl==2.2.1+cu118 \
    -f https://data.dgl.ai/wheels/torch-2.3/cu118/repo.html
  "${VENV_PYTHON}" -m pip install \
    --prefer-binary \
    -r "${REQUIREMENTS_PATH}"
  touch "${MARKER_PATH}"
else
  echo "Reusing installed environment at ${VENV_PATH}"
fi

# This is quick and intentionally repeated: an editable install must point at
# the RGFN checkout in the current clone, not an older clone's absolute path.
"${VENV_PYTHON}" -m pip install --no-deps -e "${RGFN_DIR}"

cd "${REPO_ROOT}"
"${VENV_PYTHON}" -m molecule_synthesis.prefetch_assets --seh
"${VENV_PYTHON}" -m molecule_synthesis.preflight --strict-commit
"${VENV_PYTHON}" -m molecule_synthesis.pipeline \
  --suite seh_reduced_a100 \
  --seed 0 \
  --method all \
  --dry-run

cat <<EOF

server_setup=OK
environment=${VENV_PATH}
pip_cache=${CACHE_PATH}

Activate with:
  source ${VENV_PATH}/bin/activate

Then launch the timed seed-0 job from:
  cd ${REPO_ROOT}
  /usr/bin/time -p python -m molecule_synthesis.pipeline --suite seh_reduced_a100 --seed 0 --method mips_grpo
EOF
