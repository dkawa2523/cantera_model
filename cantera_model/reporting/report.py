from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    extra: list[str] = []
    seen = set(keys)
    for row in rows[1:]:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            extra.append(key)
    keys.extend(extra)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _objective_tuple(row: dict[str, Any]) -> tuple[float, float, float]:
    try:
        species = float(row.get("species_after", float("inf")))
    except (TypeError, ValueError):
        species = float("inf")
    try:
        reactions = float(row.get("reactions_after", float("inf")))
    except (TypeError, ValueError):
        reactions = float("inf")
    try:
        err = float(row.get("mean_rel_diff", float("inf")))
    except (TypeError, ValueError):
        err = float("inf")
    return species, reactions, err


def _dominates(a: tuple[float, float, float], b: tuple[float, float, float]) -> bool:
    return (a[0] <= b[0] and a[1] <= b[1] and a[2] <= b[2]) and (a != b)


def _is_gate_floor_passed(row: dict[str, Any]) -> bool:
    gate_passed = bool(row.get("gate_passed", True))
    floor_passed = bool(row.get("floor_passed", True))
    balance_passed = bool(row.get("balance_gate_passed", True))
    cluster_guard_passed = bool(row.get("cluster_guard_passed", True))
    return gate_passed and floor_passed and balance_passed and cluster_guard_passed


def _pareto_front(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    filtered = [r for r in rows if _is_gate_floor_passed(r)]
    if not filtered:
        return []
    objs = [_objective_tuple(r) for r in filtered]
    out: list[dict[str, Any]] = []
    for i, row in enumerate(filtered):
        dominated = False
        for j, other in enumerate(filtered):
            if i == j:
                continue
            if _dominates(objs[j], objs[i]):
                dominated = True
                break
        if not dominated:
            out.append(row)
    return out


def write_report(
    report_dir: str | Path,
    *,
    run_id: str,
    stage_rows: list[dict[str, Any]],
    selected_stage: str | None,
    summary_payload: dict[str, Any],
) -> Path:
    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = dict(summary_payload)
    summary["run_id"] = run_id
    summary["selected_stage"] = selected_stage

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    lines = [
        f"# Reduction Report: {run_id}",
        "",
        f"- selected_stage: {selected_stage}",
        f"- hard_ban_violations: {summary.get('hard_ban_violations', 'n/a')}",
        f"- gate_passed: {summary.get('gate_passed', False)}",
    ]

    reduction_trace = dict(summary.get("reduction_trace") or {})
    trend = list(reduction_trace.get("candidate_trend") or [])
    if trend:
        lines.extend(
            [
                "",
                "## Candidate Trend",
                "",
                "| stage | species_after | reactions_after | overall_candidates | overall_selected | select_ratio | mean_rel_diff | floor_passed | balance_gate_passed | cluster_guard | weighted_cov | top_weighted_cov | essential_cov | max_cluster_ratio | balance_mode | selection_score | rs_upper_eff | active_cov_floor_eff | balance_margin | dynamic |",
                "|---|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|---:|---:|---:|---:|---|---:|---:|---:|---:|:---:|",
            ]
        )
        for row in trend:
            lines.append(
                "| {stage} | {species_after} | {reactions_after} | {overall_candidates} | {overall_selected} | {overall_select_ratio:.4f} | {mean_rel_diff:.4f} | {floor_passed} | {balance_gate_passed} | {cluster_guard_passed} | {weighted_cov:.4f} | {top_weighted_cov:.4f} | {essential_cov:.4f} | {max_cluster_ratio:.4f} | {balance_mode} | {selection_score:.4f} | {rs_upper_effective:.4f} | {active_cov_effective_floor:.4f} | {balance_margin:.4f} | {dynamic_applied} |".format(
                    stage=str(row.get("stage", "")),
                    species_after=int(row.get("species_after", 0)),
                    reactions_after=int(row.get("reactions_after", 0)),
                    overall_candidates=int(row.get("overall_candidates", 0)),
                    overall_selected=int(row.get("overall_selected", 0)),
                    overall_select_ratio=float(row.get("overall_select_ratio", 0.0)),
                    mean_rel_diff=float(row.get("mean_rel_diff", 0.0)),
                    floor_passed=("yes" if bool(row.get("floor_passed", True)) else "no"),
                    balance_gate_passed=("yes" if bool(row.get("balance_gate_passed", True)) else "no"),
                    cluster_guard_passed=("yes" if bool(row.get("cluster_guard_passed", True)) else "no"),
                    weighted_cov=float(row.get("weighted_active_species_coverage", 0.0)),
                    top_weighted_cov=float(
                        row.get("active_species_coverage_top_weighted", row.get("weighted_active_species_coverage", 0.0))
                    ),
                    essential_cov=float(row.get("essential_species_coverage", 1.0)),
                    max_cluster_ratio=float(row.get("max_cluster_size_ratio", 0.0)),
                    balance_mode=str(row.get("balance_mode", "binary")),
                    selection_score=float(row.get("selection_score", 0.0)),
                    rs_upper_effective=float(row.get("rs_upper_effective", 0.0)),
                    active_cov_effective_floor=float(row.get("active_cov_effective_floor", 0.0)),
                    balance_margin=float(row.get("balance_margin", 0.0)),
                    dynamic_applied=("yes" if bool(row.get("balance_dynamic_applied", False)) else "no"),
                )
            )

    cluster_preview = dict(reduction_trace.get("cluster_preview") or {})
    selected_clusters = list(cluster_preview.get("selected_stage_clusters") or [])
    if selected_clusters:
        lines.extend(
            [
                "",
                "## Selected Stage Clusters",
                "",
                "| cluster_id | size | members_sample | elements |",
                "|---:|---:|---|---|",
            ]
        )
        for row in selected_clusters:
            members = ", ".join(str(x) for x in list(row.get("members_sample") or []))
            elems = ", ".join(str(x) for x in list(row.get("elements") or []))
            lines.append(
                f"| {int(row.get('cluster_id', 0))} | {int(row.get('size', 0))} | {members} | {elems} |"
            )

    (out_dir / "report.md").write_text("\n".join(lines))

    _write_csv(out_dir / "metrics.csv", stage_rows)
    _write_csv(out_dir / "pareto.csv", _pareto_front(stage_rows))
    return out_dir
