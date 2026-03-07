from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

LARGE9: list[tuple[str, str, str]] = [
    ("diamond", "baseline", "configs/reduce_diamond_benchmarks_large_baseline.yaml"),
    ("diamond", "learnckpp", "configs/reduce_diamond_benchmarks_large_learnckpp.yaml"),
    ("diamond", "pooling", "configs/reduce_diamond_benchmarks_large_pooling.yaml"),
    ("sif4", "baseline", "configs/reduce_sif4_benchmark_sin3n4_large_baseline.yaml"),
    ("sif4", "learnckpp", "configs/reduce_sif4_benchmark_sin3n4_large_learnckpp.yaml"),
    ("sif4", "pooling", "configs/reduce_sif4_benchmark_sin3n4_large_pooling.yaml"),
    ("ac", "baseline", "configs/reduce_ac_benchmark_large_baseline.yaml"),
    ("ac", "learnckpp", "configs/reduce_ac_benchmark_large_learnckpp.yaml"),
    ("ac", "pooling", "configs/reduce_ac_benchmark_large_pooling.yaml"),
]


def _run_id(prefix: str, bench: str, mode: str) -> str:
    return f"{prefix}_{bench}_large_{mode}"


def _summary_path(report_dir: Path, run_id: str) -> Path:
    return report_dir / run_id / "summary.json"


def _load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"invalid summary payload: {path}")
    return payload


def _reaction_reduction(summary: dict[str, Any]) -> float:
    selected = dict(summary.get("selected_metrics") or {})
    before = float(selected.get("reactions_before") or 0.0)
    after = float(selected.get("reactions_after") or 0.0)
    if before <= 0.0:
        return 0.0
    return float(1.0 - (after / before))


def _compare(report_dir: Path, base_prefix: str, new_prefix: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for bench, mode, _ in LARGE9:
        base_id = _run_id(base_prefix, bench, mode)
        new_id = _run_id(new_prefix, bench, mode)
        base = _load_summary(_summary_path(report_dir, base_id))
        new = _load_summary(_summary_path(report_dir, new_id))

        base_selected = dict(base.get("selected_metrics") or {})
        new_selected = dict(new.get("selected_metrics") or {})

        row = {
            "bench": bench,
            "mode": mode,
            "base_run_id": base_id,
            "new_run_id": new_id,
            "base_gate_passed": bool(base.get("gate_passed", False)),
            "new_gate_passed": bool(new.get("gate_passed", False)),
            "base_primary_blocker_layer": str(base.get("primary_blocker_layer", "none")),
            "new_primary_blocker_layer": str(new.get("primary_blocker_layer", "none")),
            "base_reaction_reduction": _reaction_reduction(base),
            "new_reaction_reduction": _reaction_reduction(new),
            "reaction_reduction_delta": _reaction_reduction(new) - _reaction_reduction(base),
            "mandatory_mean_delta": float(new_selected.get("mean_rel_diff_mandatory") or 0.0)
            - float(base_selected.get("mean_rel_diff_mandatory") or 0.0),
            "optional_mean_delta": float(new_selected.get("mean_rel_diff_optional") or 0.0)
            - float(base_selected.get("mean_rel_diff_optional") or 0.0),
        }
        rows.append(row)

    return {
        "base_prefix": base_prefix,
        "new_prefix": new_prefix,
        "rows": rows,
        "gate_passed_9of9": bool(all(bool(r["new_gate_passed"]) for r in rows)),
        "reaction_reduction_non_degradation": bool(all(float(r["reaction_reduction_delta"]) >= 0.0 for r in rows)),
        "mandatory_mean_non_degradation": bool(all(float(r["mandatory_mean_delta"]) <= 1.0e-6 for r in rows)),
        "optional_mean_non_degradation": bool(all(float(r["optional_mean_delta"]) <= 1.0e-6 for r in rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run large9 suite, summarize each bench, and compare against a baseline prefix")
    parser.add_argument("--run-prefix", default="eval38", help="Run-id prefix for the new execution set")
    parser.add_argument("--compare-prefix", default="eval37", help="Run-id prefix used as baseline for non-regression comparison")
    parser.add_argument("--report-dir", default="reports", help="Report root directory")
    parser.add_argument("--python-bin", default=str(Path(".venv/bin/python")), help="Python executable used to run CLI commands")
    parser.add_argument("--skip-run", action="store_true", help="Skip reduce_validate execution and only regenerate summaries/comparison")
    args = parser.parse_args()

    report_dir = Path(args.report_dir).resolve()
    pybin = str(Path(args.python_bin))

    if not args.skip_run:
        for bench, mode, cfg in LARGE9:
            run_id = _run_id(args.run_prefix, bench, mode)
            cmd = [
                pybin,
                "-m",
                "cantera_model.cli.reduce_validate",
                "--config",
                cfg,
                "--run-id",
                run_id,
            ]
            subprocess.run(cmd, check=True)

    for bench in ("diamond", "sif4", "ac"):
        entries: list[str] = []
        for mode in ("baseline", "learnckpp", "pooling"):
            entries.extend(["--entry", f"{mode}:{_run_id(args.run_prefix, bench, mode)}"])
        out = report_dir / f"{bench}_large_eval_summary_{args.run_prefix}.json"
        cmd = [
            pybin,
            "-m",
            "cantera_model.cli.summarize_reduction_eval",
            "--report-dir",
            str(report_dir),
            "--output",
            str(out),
            *entries,
        ]
        subprocess.run(cmd, check=True)

    comp = _compare(report_dir, args.compare_prefix, args.run_prefix)
    comp_path = report_dir / f"large9_compare_{args.compare_prefix}_vs_{args.run_prefix}.json"
    comp_path.write_text(json.dumps(comp, ensure_ascii=False, indent=2))
    print(json.dumps(comp, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
