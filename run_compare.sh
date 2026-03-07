#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -z "${PYTHON_BIN:-}" && -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
"$PYTHON_BIN" "$SCRIPT_DIR/run_cantera_eval.py" --config "$SCRIPT_DIR/configs/gri30_small_compare_template.yaml" "$@"
