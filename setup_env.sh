#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/.venv}"

"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV_DIR/bin/python" -m pip install -r "$SCRIPT_DIR/requirements.txt"

"$VENV_DIR/bin/python" - <<'PY'
import cantera as ct
import yaml
print("python env ready")
print("cantera", ct.__version__)
print("pyyaml", yaml.__version__)
PY

echo "[OK] Environment is ready: $VENV_DIR"
echo "Run with: PYTHON_BIN=$VENV_DIR/bin/python ./run_compare.sh"
