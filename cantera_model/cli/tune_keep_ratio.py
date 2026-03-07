from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("config must be a mapping")
    return data


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def _deep_copy(obj: Any) -> Any:
    return json.loads(json.dumps(obj))


def _collect_changed_paths(before: Any, after: Any, *, prefix: str = "") -> list[dict[str, Any]]:
    if isinstance(before, dict) and isinstance(after, dict):
        out: list[dict[str, Any]] = []
        keys = sorted(set(before.keys()) | set(after.keys()))
        for key in keys:
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in before:
                out.append({"path": path, "before": None, "after": after[key]})
                continue
            if key not in after:
                out.append({"path": path, "before": before[key], "after": None})
                continue
            out.extend(_collect_changed_paths(before[key], after[key], prefix=path))
        return out

    if isinstance(before, list) and isinstance(after, list):
        if before == after:
            return []
        return [{"path": prefix or "<root>", "before": before, "after": after}]

    if before != after:
        return [{"path": prefix or "<root>", "before": before, "after": after}]
    return []


def _set_auto_tune_params(cfg: dict[str, Any], params: dict[str, float]) -> None:
    learn = cfg.setdefault("learnckpp", {})
    adaptive = learn.setdefault("adaptive_keep_ratio", {})
    auto = adaptive.setdefault("auto_tune", {})

    auto["safety_weight"] = float(params["safety_weight"])
    auto["risk_keep_boost"] = float(params["risk_keep_boost"])

    src = dict(auto.get("source_risk") or {})
    src["trace_h5"] = float(params["source_risk_trace"])
    src["network_dir"] = float(params["source_risk_network"])
    auto["source_risk"] = src


def _grid(calib_cfg: dict[str, Any], max_trials: int | None) -> list[dict[str, float]]:
    safety = [float(x) for x in (calib_cfg.get("safety_weight_values") or [2.0])]
    keep_boost = [float(x) for x in (calib_cfg.get("risk_keep_boost_values") or [0.35])]
    trace_risk = [float(x) for x in (calib_cfg.get("source_risk_trace_values") or [0.40])]
    network_risk = [float(x) for x in (calib_cfg.get("source_risk_network_values") or [0.40])]

    rows: list[dict[str, float]] = []
    for sw, kb, tr, nr in itertools.product(safety, keep_boost, trace_risk, network_risk):
        rows.append(
            {
                "safety_weight": sw,
                "risk_keep_boost": kb,
                "source_risk_trace": tr,
                "source_risk_network": nr,
            }
        )
    if max_trials is not None and max_trials > 0:
        rows = rows[:max_trials]
    return rows


def _run_reduce_validate(
    *,
    cfg_path: Path,
    run_id: str,
    python_bin: str,
) -> tuple[bool, dict[str, Any], str]:
    proc = subprocess.run(
        [python_bin, "-m", "cantera_model.cli.reduce_validate", "--config", str(cfg_path), "--run-id", run_id],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return False, {}, (proc.stderr or proc.stdout)
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return False, {}, f"invalid json output: {exc}"
    return True, payload, ""


def _score_result(payload: dict[str, Any]) -> tuple[float, dict[str, float]]:
    selected = dict(payload.get("selected_metrics") or {})
    gate_passed = bool(payload.get("gate_passed", False))

    overall_select_ratio = float(selected.get("overall_select_ratio", 1.0))
    mean_rel_diff = float(selected.get("mean_rel_diff", 1.0))

    compression_term = max(0.0, 1.0 - overall_select_ratio)
    error_term = max(0.0, 1.0 - min(mean_rel_diff / 0.40, 1.0))
    gate_term = 1.0 if gate_passed else -2.0

    score = (2.0 * compression_term) + (1.0 * error_term) + gate_term
    return score, {
        "compression_term": compression_term,
        "error_term": error_term,
        "gate_term": gate_term,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    seen = set(keys)
    for row in rows[1:]:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid-tune learnckpp adaptive keep-ratio auto_tune parameters")
    parser.add_argument("--config", required=True, help="Path to base reduce config")
    parser.add_argument("--run-id", required=True, help="Run id prefix")
    parser.add_argument("--trace-h5", default=None, help="Optional trace_h5 path for trace mode")
    parser.add_argument("--network-dir", default=None, help="Optional network_dir path for network mode")
    parser.add_argument("--output-root", default="reports/tuning", help="Output root directory")
    parser.add_argument("--max-trials", type=int, default=0, help="Max parameter trials (0 means all)")
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable to call reduce_validate")
    parser.add_argument(
        "--apply-best",
        action="store_true",
        help="Write a config with best tuned parameters applied",
    )
    parser.add_argument(
        "--applied-config-out",
        default=None,
        help="Output path for applied config (default: <output_dir>/applied_config.yaml)",
    )
    parser.add_argument(
        "--apply-inplace",
        action="store_true",
        help="If set with --apply-best, overwrite the input --config after writing backup",
    )
    args = parser.parse_args()

    base_cfg_path = Path(args.config).resolve()
    base_cfg = _load_yaml(base_cfg_path)

    if str((base_cfg.get("reduction") or {}).get("mode", "baseline")) != "learnckpp":
        raise ValueError("tune_keep_ratio requires reduction.mode=learnckpp")
    if args.apply_inplace and not args.apply_best:
        raise ValueError("--apply-inplace requires --apply-best")

    learn = dict(base_cfg.get("learnckpp") or {})
    adaptive = dict(learn.get("adaptive_keep_ratio") or {})
    calib_cfg = dict(adaptive.get("calibration") or {})
    trials = _grid(calib_cfg, (None if args.max_trials <= 0 else int(args.max_trials)))
    if not trials:
        raise ValueError("no calibration trials were generated")

    modes: list[tuple[str, str | None]] = [("synthetic", None)]
    trace_path = args.trace_h5 or base_cfg.get("trace_h5")
    network_dir = args.network_dir or base_cfg.get("network_dir")
    if trace_path:
        modes.append(("trace_h5", str(trace_path)))
    if network_dir:
        modes.append(("network_dir", str(network_dir)))

    out_dir = Path(args.output_root).resolve() / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    trial_rows: list[dict[str, Any]] = []

    best_key = None
    best_score = -float("inf")

    for i, params in enumerate(trials):
        trial_score = 0.0
        trial_gate_all = True
        mode_rows: list[dict[str, Any]] = []

        for mode, path_val in modes:
            cfg = _deep_copy(base_cfg)
            cfg["report_dir"] = str(out_dir / "reports")
            _set_auto_tune_params(cfg, params)

            if mode == "synthetic":
                cfg.pop("trace_h5", None)
                cfg.pop("network_dir", None)
            elif mode == "trace_h5":
                cfg["trace_h5"] = str(path_val)
                cfg.pop("network_dir", None)
            elif mode == "network_dir":
                cfg["network_dir"] = str(path_val)
                cfg.pop("trace_h5", None)

            cfg_path = out_dir / f"trial_{i:03d}_{mode}.yaml"
            _write_yaml(cfg_path, cfg)
            run_id = f"{args.run_id}_trial{i:03d}_{mode}"

            ok, payload, err = _run_reduce_validate(cfg_path=cfg_path, run_id=run_id, python_bin=args.python_bin)
            if not ok:
                trial_gate_all = False
                mode_rows.append(
                    {
                        "trial": i,
                        "mode": mode,
                        "ok": False,
                        "error": err,
                        **params,
                    }
                )
                continue

            score, terms = _score_result(payload)
            trial_score += score
            gate = bool(payload.get("gate_passed", False))
            trial_gate_all = trial_gate_all and gate
            selected = dict(payload.get("selected_metrics") or {})
            mode_rows.append(
                {
                    "trial": i,
                    "mode": mode,
                    "ok": True,
                    "gate_passed": gate,
                    "score": score,
                    "overall_select_ratio": float(selected.get("overall_select_ratio", 1.0)),
                    "mean_rel_diff": float(selected.get("mean_rel_diff", 1.0)),
                    "compression_term": terms["compression_term"],
                    "error_term": terms["error_term"],
                    "gate_term": terms["gate_term"],
                    **params,
                }
            )

        trial_rows.extend(mode_rows)
        mode_count = max(1, len(modes))
        mean_trial_score = trial_score / mode_count
        if not trial_gate_all:
            mean_trial_score -= 5.0

        key = (
            round(mean_trial_score, 12),
            int(trial_gate_all),
            -params["safety_weight"],
            -params["risk_keep_boost"],
            -params["source_risk_trace"],
            -params["source_risk_network"],
        )
        if key > (best_key if best_key is not None else (-float("inf"), -1, 0, 0, 0, 0)):
            best_key = key
            best_score = mean_trial_score

    # Pick best parameter row by aggregated score key.
    candidate_to_score: dict[tuple[float, float, float, float], float] = {}
    candidate_to_gate: dict[tuple[float, float, float, float], bool] = {}
    for row in trial_rows:
        if not bool(row.get("ok", False)):
            continue
        k = (
            float(row["safety_weight"]),
            float(row["risk_keep_boost"]),
            float(row["source_risk_trace"]),
            float(row["source_risk_network"]),
        )
        candidate_to_score[k] = candidate_to_score.get(k, 0.0) + float(row.get("score", 0.0))
        candidate_to_gate[k] = candidate_to_gate.get(k, True) and bool(row.get("gate_passed", False))

    best_params = None
    best_val = -float("inf")
    for k, val in candidate_to_score.items():
        gate_ok = candidate_to_gate.get(k, False)
        cmp_val = val + (1000.0 if gate_ok else 0.0)
        if cmp_val > best_val:
            best_val = cmp_val
            best_params = {
                "safety_weight": k[0],
                "risk_keep_boost": k[1],
                "source_risk_trace": k[2],
                "source_risk_network": k[3],
                "gate_all": gate_ok,
                "aggregate_score": val,
            }

    if best_params is None:
        raise RuntimeError("failed to evaluate any valid tuning trial")

    summary: dict[str, Any] = {
        "run_id": args.run_id,
        "modes": [m for m, _ in modes],
        "trials": len(trials),
        "best_score": best_score,
        "best_params": best_params,
        "output_dir": str(out_dir),
    }

    # Emit recommended config patch values.
    recommended = {
        "learnckpp": {
            "adaptive_keep_ratio": {
                "auto_tune": {
                    "safety_weight": float(best_params["safety_weight"]),
                    "risk_keep_boost": float(best_params["risk_keep_boost"]),
                    "source_risk": {
                        "trace_h5": float(best_params["source_risk_trace"]),
                        "network_dir": float(best_params["source_risk_network"]),
                    },
                }
            }
        }
    }

    _write_csv(out_dir / "trials.csv", trial_rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (out_dir / "best_params.yaml").write_text(yaml.safe_dump(recommended, sort_keys=False))

    if args.apply_best:
        applied_cfg = _deep_copy(base_cfg)
        _set_auto_tune_params(applied_cfg, best_params)
        changed_entries = _collect_changed_paths(base_cfg, applied_cfg)

        if args.applied_config_out:
            applied_path = Path(args.applied_config_out).resolve()
        else:
            applied_path = out_dir / "applied_config.yaml"
        applied_path.parent.mkdir(parents=True, exist_ok=True)
        _write_yaml(applied_path, applied_cfg)

        summary["apply_best"] = {
            "enabled": True,
            "applied_config_path": str(applied_path),
            "inplace": bool(args.apply_inplace),
            "changed_count": int(len(changed_entries)),
            "changed_paths": changed_entries,
        }
        if args.apply_inplace:
            backup_path = out_dir / f"{base_cfg_path.name}.backup.yaml"
            _write_yaml(backup_path, _deep_copy(base_cfg))
            _write_yaml(base_cfg_path, applied_cfg)
            summary["apply_best"]["backup_path"] = str(backup_path)
            summary["apply_best"]["inplace_target"] = str(base_cfg_path)

        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
