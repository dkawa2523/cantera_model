#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -z "${PYTHON_BIN:-}" && -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

usage() {
  cat <<'EOF'
Usage:
  ./run_dev_checks.sh [RUN_TAG] [options]
  ./run_dev_checks.sh --run-tag RUN_TAG [options]

Options:
  --run-tag <id>           Run tag prefix (default: dev)
  --pooling-trials <n>     Trials for pooling tune smoke (default: 1)
  --pooling-backend <name> Override pooling.model.backend and tuning.backend_choices
                           in temporary pooling config (e.g. pyg, numpy)
  --skip-pooling-tune      Skip tune_pooling smoke step
  -h, --help               Show this help
EOF
}

RUN_TAG="dev"
POOLING_TRIALS="1"
POOLING_BACKEND=""
SKIP_POOLING_TUNE="0"

if [[ $# -gt 0 && "${1}" != -* ]]; then
  RUN_TAG="$1"
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-tag)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] --run-tag requires a value" >&2
        exit 2
      fi
      RUN_TAG="$2"
      shift 2
      ;;
    --pooling-trials)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] --pooling-trials requires a value" >&2
        exit 2
      fi
      POOLING_TRIALS="$2"
      shift 2
      ;;
    --pooling-backend)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] --pooling-backend requires a value" >&2
        exit 2
      fi
      POOLING_BACKEND="$2"
      shift 2
      ;;
    --skip-pooling-tune)
      SKIP_POOLING_TUNE="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! [[ "$POOLING_TRIALS" =~ ^[0-9]+$ ]] || [[ "$POOLING_TRIALS" -lt 1 ]]; then
  echo "[ERROR] --pooling-trials must be a positive integer: $POOLING_TRIALS" >&2
  exit 2
fi

TRACE_RUN_ID="${RUN_TAG}_tiny_trace"
TRACE_PATH="$SCRIPT_DIR/artifacts/traces/${TRACE_RUN_ID}.h5"
TMP_CFG="/tmp/reduce_trace_cfg_${RUN_TAG}.yaml"
NET_CFG="/tmp/reduce_network_cfg_${RUN_TAG}.yaml"
POOL_CFG="/tmp/reduce_pooling_cfg_${RUN_TAG}.yaml"

"$PYTHON_BIN" - <<'PY' "$SCRIPT_DIR/configs/reduce_pooling_mvp.yaml" "$POOL_CFG" "$POOLING_BACKEND"
import sys
from pathlib import Path
import yaml

base = Path(sys.argv[1])
out = Path(sys.argv[2])
backend = str(sys.argv[3]).strip()
cfg = yaml.safe_load(base.read_text())
if backend:
    cfg.setdefault("pooling", {}).setdefault("model", {})["backend"] = backend
    cfg.setdefault("pooling", {}).setdefault("tuning", {})["backend_choices"] = [backend]
out.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"pooling config: {out}")
PY

TOTAL_STEPS=10
if [[ "$SKIP_POOLING_TUNE" == "1" ]]; then
  TOTAL_STEPS=9
fi
STEP_IDX=0
step() {
  STEP_IDX=$((STEP_IDX + 1))
  printf "\n[%d/%d] %s\n" "$STEP_IDX" "$TOTAL_STEPS" "$1"
}

step "Unit tests"
"$PYTHON_BIN" -m pytest -q tests

step "Tiny evaluator smoke"
"$PYTHON_BIN" "$SCRIPT_DIR/run_cantera_eval.py" \
  --config "$SCRIPT_DIR/configs/gri30_tiny_quick.yaml" \
  --run-id "${RUN_TAG}_tiny_eval"

step "Generate tiny trace"
"$PYTHON_BIN" "$SCRIPT_DIR/run_cantera_trace.py" \
  --config "$SCRIPT_DIR/configs/gri30_tiny_trace.yaml" \
  --run-id "$TRACE_RUN_ID"

step "Build network artifacts from trace"
"$PYTHON_BIN" "$SCRIPT_DIR/run_build_network.py" \
  --trace-h5 "$TRACE_PATH" \
  --run-id "${RUN_TAG}_tiny_network" \
  --output-root "$SCRIPT_DIR/artifacts/network"

step "Reduction validate (synthetic fallback path)"
"$PYTHON_BIN" -m cantera_model.cli.reduce_validate \
  --config "$SCRIPT_DIR/configs/reduce_surrogate_aggressive.yaml" \
  --run-id "${RUN_TAG}_reduce_synth"

step "Build trace/network-aware temporary configs"
"$PYTHON_BIN" - <<'PY' "$SCRIPT_DIR/configs/reduce_surrogate_aggressive.yaml" "$TMP_CFG" "$TRACE_PATH"
import sys
from pathlib import Path
import yaml
base = Path(sys.argv[1])
out = Path(sys.argv[2])
trace = Path(sys.argv[3])
cfg = yaml.safe_load(base.read_text())
cfg["trace_h5"] = str(trace)
out.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"temp config: {out}")
PY
"$PYTHON_BIN" - <<'PY' "$SCRIPT_DIR/configs/reduce_surrogate_aggressive.yaml" "$NET_CFG" "$SCRIPT_DIR/artifacts/network/${RUN_TAG}_tiny_network"
import sys
from pathlib import Path
import yaml
base = Path(sys.argv[1])
out = Path(sys.argv[2])
net = Path(sys.argv[3])
cfg = yaml.safe_load(base.read_text())
cfg["network_dir"] = str(net)
cfg.pop("trace_h5", None)
out.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"network config: {out}")
PY

step "Reduction validate (trace_h5 path)"
"$PYTHON_BIN" -m cantera_model.cli.reduce_validate \
  --config "$TMP_CFG" \
  --run-id "${RUN_TAG}_reduce_trace"

step "Reduction validate (network_dir path)"
"$PYTHON_BIN" -m cantera_model.cli.reduce_validate \
  --config "$NET_CFG" \
  --run-id "${RUN_TAG}_reduce_network"

step "Reduction validate (pooling mode)"
"$PYTHON_BIN" -m cantera_model.cli.reduce_validate \
  --config "$POOL_CFG" \
  --run-id "${RUN_TAG}_reduce_pooling"

if [[ "$SKIP_POOLING_TUNE" == "1" ]]; then
  printf "\n[SKIP] Pooling tuning smoke (requested by --skip-pooling-tune)\n"
else
  step "Pooling tuning smoke (3-source)"
  "$PYTHON_BIN" -m cantera_model.cli.tune_pooling \
    --config "$POOL_CFG" \
    --run-id "${RUN_TAG}_pooling_tune" \
    --trace-h5 "$TRACE_PATH" \
    --network-dir "$SCRIPT_DIR/artifacts/network/${RUN_TAG}_tiny_network" \
    --max-trials "$POOLING_TRIALS" \
    --output-root "$SCRIPT_DIR/reports/tuning_pooling" \
    --apply-best
fi

printf "\n[OK] Dev checks finished\n"
printf "%s\n" "- run_tag: ${RUN_TAG}"
printf "%s\n" "- python_bin: ${PYTHON_BIN}"
printf "%s\n" "- pooling_backend_override: ${POOLING_BACKEND:-<none>}"
printf "%s\n" "- pooling_trials: ${POOLING_TRIALS}"
printf "%s\n" "- tiny eval: runs/${RUN_TAG}_tiny_eval/summary.json"
printf "%s\n" "- tiny trace: artifacts/traces/${TRACE_RUN_ID}.h5"
printf "%s\n" "- tiny network: artifacts/network/${RUN_TAG}_tiny_network/summary.json"
printf "%s\n" "- synthetic reduce: reports/${RUN_TAG}_reduce_synth/summary.json"
printf "%s\n" "- trace reduce: reports/${RUN_TAG}_reduce_trace/summary.json"
printf "%s\n" "- network reduce: reports/${RUN_TAG}_reduce_network/summary.json"
printf "%s\n" "- pooling reduce: reports/${RUN_TAG}_reduce_pooling/summary.json"
if [[ "$SKIP_POOLING_TUNE" != "1" ]]; then
  printf "%s\n" "- pooling tune: reports/tuning_pooling/${RUN_TAG}_pooling_tune/summary.json"
fi
