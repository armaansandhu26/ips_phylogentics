#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "No Python executable found; set PYTHON_BIN=/path/to/python." >&2
    exit 127
  fi
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RGFN_ROOT="${RGFN_ROOT:-${REPO_ROOT}/molecule_synthesis/external/RGFN}"
export DGLBACKEND="${DGLBACKEND:-pytorch}"
export RGFN_MINIMAL_PROXIES="${RGFN_MINIMAL_PROXIES:-1}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -m unittest discover -s molecule_synthesis/tests -v
"${PYTHON_BIN}" -m molecule_synthesis.preflight --rgfn-root "${RGFN_ROOT}" --strict-commit
"${PYTHON_BIN}" -m molecule_synthesis.pipeline \
  --suite qed_smoke \
  --method all \
  --rgfn-root "${RGFN_ROOT}" \
  --dry-run

if [[ "${FULL_SMOKE:-0}" == "1" ]]; then
  "${PYTHON_BIN}" -m molecule_synthesis.pipeline \
    --suite qed_smoke \
    --method all \
    --rgfn-root "${RGFN_ROOT}"
else
  echo "Set FULL_SMOKE=1 to execute one CPU training step for every method."
fi
