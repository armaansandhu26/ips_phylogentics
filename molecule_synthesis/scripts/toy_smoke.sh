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

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -m molecule_synthesis.toy.pipeline \
  --method all \
  --device cpu \
  --steps 400 \
  --batch-size 512 \
  --assert-expected
