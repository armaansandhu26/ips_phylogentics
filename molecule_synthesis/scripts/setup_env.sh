#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="rgfn-molecules"
ACCELERATOR="cpu"
RGFN_URL="https://github.com/koziarskilab/RGFN.git"
RGFN_COMMIT="6ce59169f855ed18f34ba4e8279de93bee306e4f"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOLECULE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RGFN_DIR="${MOLECULE_DIR}/external/RGFN"
MINIMAL_PROXY_PATCH="${MOLECULE_DIR}/patches/rgfn_minimal_proxies.patch"

usage() {
  echo "Usage: $0 [--accelerator cpu|cu118] [--env-name NAME]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --accelerator)
      ACCELERATOR="$2"
      shift 2
      ;;
    --env-name)
      ENV_NAME="$2"
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

if [[ "${ACCELERATOR}" != "cpu" && "${ACCELERATOR}" != "cu118" ]]; then
  echo "--accelerator must be cpu or cu118" >&2
  exit 2
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required; install Miniconda/Anaconda first." >&2
  exit 1
fi

CREATED_CHECKOUT=0
if [[ ! -d "${RGFN_DIR}/.git" ]]; then
  mkdir -p "${MOLECULE_DIR}/external"
  git clone "${RGFN_URL}" "${RGFN_DIR}"
  CREATED_CHECKOUT=1
elif [[ "$(git -C "${RGFN_DIR}" remote get-url origin)" != "${RGFN_URL}" ]]; then
  echo "Refusing to modify ${RGFN_DIR}: it is not the expected RGFN checkout." >&2
  exit 1
fi

if ! git -C "${RGFN_DIR}" cat-file -e "${RGFN_COMMIT}^{commit}" 2>/dev/null; then
  git -C "${RGFN_DIR}" fetch origin "${RGFN_COMMIT}"
fi
if [[ "${CREATED_CHECKOUT}" == "1" ]]; then
  git -C "${RGFN_DIR}" checkout --detach "${RGFN_COMMIT}"
fi
CURRENT_COMMIT="$(git -C "${RGFN_DIR}" rev-parse HEAD)"
if [[ "${CURRENT_COMMIT}" != "${RGFN_COMMIT}" ]]; then
  echo "RGFN exists at ${CURRENT_COMMIT}; expected ${RGFN_COMMIT}." >&2
  echo "Move that checkout aside or explicitly check out the pinned commit, then rerun." >&2
  exit 1
fi

if git -C "${RGFN_DIR}" apply --check "${MINIMAL_PROXY_PATCH}" 2>/dev/null; then
  git -C "${RGFN_DIR}" apply "${MINIMAL_PROXY_PATCH}"
elif ! git -C "${RGFN_DIR}" apply --reverse --check "${MINIMAL_PROXY_PATCH}" 2>/dev/null; then
  echo "Unable to apply or recognize ${MINIMAL_PROXY_PATCH}." >&2
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create --name "${ENV_NAME}" python=3.11.8 -y
fi

if [[ "${ACCELERATOR}" == "cu118" ]]; then
  conda run -n "${ENV_NAME}" python -m pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu118
  conda run -n "${ENV_NAME}" python -m pip install dgl==2.2.1+cu118 -f https://data.dgl.ai/wheels/torch-2.3/cu118/repo.html
else
  conda run -n "${ENV_NAME}" python -m pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
  conda run -n "${ENV_NAME}" python -m pip install dgl==1.1.2 -f https://data.dgl.ai/wheels/torch-2.3/cpu/repo.html
fi

if [[ "$(uname -s)" == "Darwin" && "${ACCELERATOR}" == "cpu" ]]; then
  # RGFN pins PyTDC 1.0.6. On Apple Silicon its transitive dependencies are
  # currently unsatisfiable (datasets requires pyarrow>=15 while tiledbsoma
  # requires pyarrow<13). The QED experiment does not use TDC, docking, or the
  # private GNEprop models, so install the released package with the minimal
  # runtime dependencies needed by the reaction environment and trainer.
  conda run -n "${ENV_NAME}" python -m pip install \
    'setuptools<81' numpy==1.26.4 gin-config==0.5.0 more-itertools==10.1.0 \
    pandas==2.1.4 openpyxl==3.1.4 rdkit==2023.9.5 \
    pydantic==2.6.3 tqdm==4.66.3 wandb==0.15.12 \
    torch-geometric==2.5.3 torchmetrics==1.2.0 dgllife==0.3.2 \
    wurlitzer==3.1.0 openbabel-wheel==3.1.1.19 meeko==0.5.1 \
    pytest==7.4.2 xlsxwriter==3.2.0
  conda run -n "${ENV_NAME}" python -m pip install PyTDC==1.0.6 --no-deps
  conda run -n "${ENV_NAME}" python -m pip install -e "${RGFN_DIR}" --no-deps
else
  conda run -n "${ENV_NAME}" python -m pip install -e "${RGFN_DIR}"
fi

conda run -n "${ENV_NAME}" python -m pip install matplotlib==3.9.2

echo "Environment ready. Activate it with: conda activate ${ENV_NAME}"
echo "Then run: python -m molecule_synthesis.preflight --strict-commit"
