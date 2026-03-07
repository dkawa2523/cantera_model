#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="python3.11"
  else
    PYTHON_BIN="python3"
  fi
fi
VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/.venv}"
ALLOW_NON_311="${ALLOW_NON_311:-0}"

"$PYTHON_BIN" - <<'PY' "$ALLOW_NON_311"
import sys
allow = str(sys.argv[1]).strip() == "1"
major_minor = f"{sys.version_info.major}.{sys.version_info.minor}"
if major_minor != "3.11" and not allow:
    raise SystemExit(
        f"[ERROR] setup_env.sh requires Python 3.11 (detected {major_minor}). "
        "Set PYTHON_BIN=python3.11 or override temporarily with ALLOW_NON_311=1."
    )
print(f"[setup_env] interpreter: {sys.executable} ({major_minor})")
PY

"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" - <<'PY' "$ALLOW_NON_311" "$VENV_DIR"
import sys
allow = str(sys.argv[1]).strip() == "1"
venv_dir = str(sys.argv[2])
major_minor = f"{sys.version_info.major}.{sys.version_info.minor}"
if major_minor != "3.11" and not allow:
    raise SystemExit(
        f"[ERROR] Existing venv uses Python {major_minor}: {venv_dir}. "
        "Remove the venv and rerun with PYTHON_BIN=python3.11 "
        "or override temporarily with ALLOW_NON_311=1."
    )
print(f"[setup_env] venv interpreter: {sys.executable} ({major_minor})")
PY
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV_DIR/bin/python" -m pip install -r "$SCRIPT_DIR/requirements.txt"

"$VENV_DIR/bin/python" - <<'PY'
import cantera as ct
import optuna
import torch
import yaml
print("python env ready")
print("cantera", ct.__version__)
print("pyyaml", yaml.__version__)
print("torch", torch.__version__)
print("optuna", optuna.__version__)
try:
    import torch_geometric
    print("torch_geometric", torch_geometric.__version__)
except Exception as exc:
    raise RuntimeError(f"torch-geometric import failed: {exc}")
try:
    import tgp
    print("tgp", getattr(tgp, "__version__", "unknown"))
except Exception as exc:
    raise RuntimeError(f"tgp import failed: {exc}")
PY

echo "[OK] Environment is ready: $VENV_DIR"
echo "Run with: PYTHON_BIN=$VENV_DIR/bin/python ./run_compare.sh"
