from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import optuna
except Exception:  # pragma: no cover
    optuna = None


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


def _run_reduce_validate(*, cfg_path: Path, run_id: str, python_bin: str) -> tuple[bool, dict[str, Any], str]:
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


def _score_result(payload: dict[str, Any]) -> tuple[float, dict[str, float]]:
    selected = dict(payload.get("selected_metrics") or {})
    gate_passed = bool(payload.get("gate_passed", False))

    species_before = float(selected.get("species_before", 1.0))
    species_after = float(selected.get("species_after", species_before))
    reactions_before = float(selected.get("reactions_before", 1.0))
    reactions_after = float(selected.get("reactions_after", reactions_before))
    mean_rel_diff = float(selected.get("mean_rel_diff", 1.0))
    hard_ban_viol = float(selected.get("hard_ban_violations", 0.0))

    species_ratio = species_after / max(species_before, 1.0)
    reaction_ratio = reactions_after / max(reactions_before, 1.0)
    compression_term = max(0.0, (1.0 - species_ratio) * 0.6 + (1.0 - reaction_ratio) * 0.4)
    error_term = max(0.0, 1.0 - min(mean_rel_diff / 0.40, 1.0))
    hard_ban_term = -5.0 * hard_ban_viol
    gate_term = 1.0 if gate_passed else -3.0

    score = (2.0 * compression_term) + (1.0 * error_term) + hard_ban_term + gate_term
    return score, {
        "compression_term": compression_term,
        "error_term": error_term,
        "hard_ban_term": hard_ban_term,
        "gate_term": gate_term,
    }


def _resolve_modes(base_cfg: dict[str, Any], trace_h5: str | None, network_dir: str | None) -> list[tuple[str, str | None]]:
    modes: list[tuple[str, str | None]] = [("synthetic", None)]
    trace_path = trace_h5 or base_cfg.get("trace_h5")
    network_path = network_dir or base_cfg.get("network_dir")
    if trace_path:
        modes.append(("trace_h5", str(trace_path)))
    if network_path:
        modes.append(("network_dir", str(network_path)))
    return modes


def _pooling_tuning_cfg(base_cfg: dict[str, Any]) -> dict[str, Any]:
    pooling_cfg = dict(base_cfg.get("pooling") or {})
    return dict(pooling_cfg.get("tuning") or {})


def _range_pair(raw: Any, default_low: float, default_high: float) -> tuple[float, float]:
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        lo = float(raw[0])
        hi = float(raw[1])
        if lo > hi:
            lo, hi = hi, lo
        return lo, hi
    return float(default_low), float(default_high)


def _sample_params_optuna(trial: Any, tuning: dict[str, Any]) -> dict[str, Any]:
    graph_choices = list(tuning.get("graph_choices") or ["species", "bipartite"])
    backend_choices = list(tuning.get("backend_choices") or ["numpy"])

    t_lo, t_hi = _range_pair(tuning.get("temperature_range"), 0.5, 1.0)
    r_lo, r_hi = _range_pair(tuning.get("ratio_scale_range"), 0.85, 1.15)
    ph_lo, ph_hi = _range_pair(tuning.get("penalty_phase_range"), 0.4, 1.4)
    ch_lo, ch_hi = _range_pair(tuning.get("penalty_charge_range"), 0.4, 1.4)
    ra_lo, ra_hi = _range_pair(tuning.get("penalty_radical_range"), 0.1, 0.9)
    ro_lo, ro_hi = _range_pair(tuning.get("penalty_role_range"), 0.1, 0.9)

    return {
        "graph": str(trial.suggest_categorical("graph", graph_choices)),
        "backend": str(trial.suggest_categorical("backend", backend_choices)),
        "temperature": float(trial.suggest_float("temperature", t_lo, t_hi)),
        "ratio_scale": float(trial.suggest_float("ratio_scale", r_lo, r_hi)),
        "penalty_phase": float(trial.suggest_float("penalty_phase", ph_lo, ph_hi)),
        "penalty_charge": float(trial.suggest_float("penalty_charge", ch_lo, ch_hi)),
        "penalty_radical": float(trial.suggest_float("penalty_radical", ra_lo, ra_hi)),
        "penalty_role": float(trial.suggest_float("penalty_role", ro_lo, ro_hi)),
    }


def _sample_params_random(rng: np.random.Generator, tuning: dict[str, Any]) -> dict[str, Any]:
    graph_choices = list(tuning.get("graph_choices") or ["species", "bipartite"])
    backend_choices = list(tuning.get("backend_choices") or ["numpy"])

    t_lo, t_hi = _range_pair(tuning.get("temperature_range"), 0.5, 1.0)
    r_lo, r_hi = _range_pair(tuning.get("ratio_scale_range"), 0.85, 1.15)
    ph_lo, ph_hi = _range_pair(tuning.get("penalty_phase_range"), 0.4, 1.4)
    ch_lo, ch_hi = _range_pair(tuning.get("penalty_charge_range"), 0.4, 1.4)
    ra_lo, ra_hi = _range_pair(tuning.get("penalty_radical_range"), 0.1, 0.9)
    ro_lo, ro_hi = _range_pair(tuning.get("penalty_role_range"), 0.1, 0.9)

    return {
        "graph": str(graph_choices[int(rng.integers(0, len(graph_choices)))]),
        "backend": str(backend_choices[int(rng.integers(0, len(backend_choices)))]),
        "temperature": float(rng.uniform(t_lo, t_hi)),
        "ratio_scale": float(rng.uniform(r_lo, r_hi)),
        "penalty_phase": float(rng.uniform(ph_lo, ph_hi)),
        "penalty_charge": float(rng.uniform(ch_lo, ch_hi)),
        "penalty_radical": float(rng.uniform(ra_lo, ra_hi)),
        "penalty_role": float(rng.uniform(ro_lo, ro_hi)),
    }


def _apply_pooling_params(cfg: dict[str, Any], params: dict[str, Any]) -> None:
    cfg.setdefault("reduction", {})["mode"] = "pooling"

    pooling = cfg.setdefault("pooling", {})
    pooling["graph"] = str(params["graph"])

    model = pooling.setdefault("model", {})
    model["backend"] = str(params["backend"])

    train = pooling.setdefault("train", {})
    train["temperature"] = float(params["temperature"])

    constraint_cfg = pooling.setdefault("constraint_cfg", {})
    soft = constraint_cfg.setdefault("soft", {})
    penalty = soft.setdefault("penalty", {})
    penalty["phase"] = float(params["penalty_phase"])
    penalty["charge"] = float(params["penalty_charge"])
    penalty["radical"] = float(params["penalty_radical"])
    penalty["role"] = float(params["penalty_role"])

    ratio_scale = float(params["ratio_scale"])
    search = cfg.setdefault("search", {})
    stages = list(search.get("stages") or [])
    for stage in stages:
        try:
            base_ratio = float(stage.get("target_ratio", 1.0))
        except (TypeError, ValueError):
            continue
        stage["target_ratio"] = float(np.clip(base_ratio * ratio_scale, 0.05, 1.0))


def _evaluate_trial(
    *,
    trial_id: int,
    params: dict[str, Any],
    base_cfg: dict[str, Any],
    modes: list[tuple[str, str | None]],
    out_dir: Path,
    python_bin: str,
) -> tuple[float, bool, list[dict[str, Any]]]:
    trial_score = 0.0
    trial_gate_all = True
    mode_rows: list[dict[str, Any]] = []

    for mode, path_val in modes:
        cfg = _deep_copy(base_cfg)
        cfg["report_dir"] = str(out_dir / "reports")
        _apply_pooling_params(cfg, params)

        if mode == "synthetic":
            cfg.pop("trace_h5", None)
            cfg.pop("network_dir", None)
        elif mode == "trace_h5":
            cfg["trace_h5"] = str(path_val)
            cfg.pop("network_dir", None)
        elif mode == "network_dir":
            cfg["network_dir"] = str(path_val)
            cfg.pop("trace_h5", None)

        cfg_path = out_dir / f"trial_{trial_id:03d}_{mode}.yaml"
        _write_yaml(cfg_path, cfg)
        run_id = f"{out_dir.name}_trial{trial_id:03d}_{mode}"

        ok, payload, err = _run_reduce_validate(cfg_path=cfg_path, run_id=run_id, python_bin=python_bin)
        if not ok:
            trial_gate_all = False
            mode_rows.append({"trial": trial_id, "mode": mode, "ok": False, "error": err, **params})
            trial_score -= 5.0
            continue

        score, terms = _score_result(payload)
        selected = dict(payload.get("selected_metrics") or {})
        gate = bool(payload.get("gate_passed", False))

        trial_gate_all = trial_gate_all and gate
        trial_score += score
        mode_rows.append(
            {
                "trial": trial_id,
                "mode": mode,
                "ok": True,
                "gate_passed": gate,
                "score": float(score),
                "species_before": int(selected.get("species_before", 0)),
                "species_after": int(selected.get("species_after", 0)),
                "reactions_before": int(selected.get("reactions_before", 0)),
                "reactions_after": int(selected.get("reactions_after", 0)),
                "mean_rel_diff": float(selected.get("mean_rel_diff", 1.0)),
                "hard_ban_violations": int(selected.get("hard_ban_violations", 0)),
                "compression_term": terms["compression_term"],
                "error_term": terms["error_term"],
                "hard_ban_term": terms["hard_ban_term"],
                "gate_term": terms["gate_term"],
                **params,
            }
        )

    mode_count = max(1, len(modes))
    mean_score = trial_score / mode_count
    if not trial_gate_all:
        mean_score -= 2.5
    return mean_score, trial_gate_all, mode_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune pooling hyper-parameters with Optuna and apply best config")
    parser.add_argument("--config", required=True, help="Path to base reduce config")
    parser.add_argument("--run-id", required=True, help="Run id prefix")
    parser.add_argument("--trace-h5", default=None, help="Optional trace_h5 path for trace mode")
    parser.add_argument("--network-dir", default=None, help="Optional network_dir path for network mode")
    parser.add_argument("--output-root", default="reports/tuning_pooling", help="Output root directory")
    parser.add_argument("--max-trials", type=int, default=8, help="Max optimization trials")
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable to call reduce_validate")
    parser.add_argument("--apply-best", action="store_true", help="Write a config with best tuned parameters applied")
    parser.add_argument("--applied-config-out", default=None, help="Output path for applied config")
    parser.add_argument("--apply-inplace", action="store_true", help="If set with --apply-best, overwrite input config")
    args = parser.parse_args()

    base_cfg_path = Path(args.config).resolve()
    base_cfg = _load_yaml(base_cfg_path)
    if args.apply_inplace and not args.apply_best:
        raise ValueError("--apply-inplace requires --apply-best")

    out_dir = Path(args.output_root).resolve() / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    modes = _resolve_modes(base_cfg, args.trace_h5, args.network_dir)
    tuning = _pooling_tuning_cfg(base_cfg)
    seed = int(tuning.get("seed", 7))
    max_trials = max(1, int(args.max_trials))

    trial_rows: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_params: dict[str, Any] | None = None
    best_gate_all = False
    backend_name = "random"

    if optuna is not None:
        backend_name = "optuna"
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def objective(trial: Any) -> float:
            params = _sample_params_optuna(trial, tuning)
            score, gate_all, rows = _evaluate_trial(
                trial_id=int(trial.number),
                params=params,
                base_cfg=base_cfg,
                modes=modes,
                out_dir=out_dir,
                python_bin=args.python_bin,
            )
            for row in rows:
                row["search_backend"] = backend_name
            trial_rows.extend(rows)
            trial.set_user_attr("params", params)
            trial.set_user_attr("gate_all", gate_all)
            return score

        study.optimize(objective, n_trials=max_trials)
        trial = study.best_trial
        best_score = float(trial.value)
        best_params = dict(trial.user_attrs.get("params") or {})
        best_gate_all = bool(trial.user_attrs.get("gate_all", False))
    else:
        rng = np.random.default_rng(seed)
        for i in range(max_trials):
            params = _sample_params_random(rng, tuning)
            score, gate_all, rows = _evaluate_trial(
                trial_id=i,
                params=params,
                base_cfg=base_cfg,
                modes=modes,
                out_dir=out_dir,
                python_bin=args.python_bin,
            )
            for row in rows:
                row["search_backend"] = backend_name
            trial_rows.extend(rows)

            if score > best_score:
                best_score = score
                best_params = dict(params)
                best_gate_all = bool(gate_all)

    if best_params is None:
        raise RuntimeError("failed to evaluate pooling tuning trials")

    # Re-aggregate score by parameter set for stable reporting.
    agg: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in trial_rows:
        if not bool(row.get("ok", False)):
            continue
        key = (
            row.get("graph"),
            row.get("backend"),
            float(row.get("temperature", 0.0)),
            float(row.get("ratio_scale", 0.0)),
            float(row.get("penalty_phase", 0.0)),
            float(row.get("penalty_charge", 0.0)),
            float(row.get("penalty_radical", 0.0)),
            float(row.get("penalty_role", 0.0)),
        )
        if key not in agg:
            agg[key] = {"score": 0.0, "gate_all": True}
        agg[key]["score"] += float(row.get("score", 0.0))
        agg[key]["gate_all"] = bool(agg[key]["gate_all"] and bool(row.get("gate_passed", False)))

    best_key = None
    best_cmp = -float("inf")
    for key, meta in agg.items():
        cmp_val = float(meta["score"]) + (1000.0 if bool(meta["gate_all"]) else 0.0)
        if cmp_val > best_cmp:
            best_cmp = cmp_val
            best_key = key

    if best_key is not None:
        best_params = {
            "graph": str(best_key[0]),
            "backend": str(best_key[1]),
            "temperature": float(best_key[2]),
            "ratio_scale": float(best_key[3]),
            "penalty_phase": float(best_key[4]),
            "penalty_charge": float(best_key[5]),
            "penalty_radical": float(best_key[6]),
            "penalty_role": float(best_key[7]),
        }
        best_gate_all = bool(agg[best_key]["gate_all"])

    summary: dict[str, Any] = {
        "run_id": args.run_id,
        "modes": [m for m, _ in modes],
        "trials": int(max_trials),
        "search_backend": backend_name,
        "best_score": float(best_score),
        "best_params": {**best_params, "gate_all": bool(best_gate_all)},
        "output_dir": str(out_dir),
    }

    recommended = {
        "reduction": {"mode": "pooling"},
        "pooling": {
            "graph": str(best_params["graph"]),
            "model": {"backend": str(best_params["backend"])},
            "train": {"temperature": float(best_params["temperature"])},
            "constraint_cfg": {
                "soft": {
                    "penalty": {
                        "phase": float(best_params["penalty_phase"]),
                        "charge": float(best_params["penalty_charge"]),
                        "radical": float(best_params["penalty_radical"]),
                        "role": float(best_params["penalty_role"]),
                    }
                }
            },
        },
        "search": {
            "stages": [
                {"target_ratio": "base * ratio_scale", "ratio_scale": float(best_params["ratio_scale"])},
            ]
        },
    }

    _write_csv(out_dir / "trials.csv", trial_rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (out_dir / "best_params.yaml").write_text(yaml.safe_dump(recommended, sort_keys=False))

    if args.apply_best:
        applied_cfg = _deep_copy(base_cfg)
        _apply_pooling_params(applied_cfg, best_params)
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
