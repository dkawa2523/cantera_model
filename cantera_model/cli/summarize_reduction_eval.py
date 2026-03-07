from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cantera_model.eval.diagnostic_schema import project_entry, validate_summary_schema


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"summary must be a JSON object: {path}")
    return data


def _compression_ratio(entry: dict[str, Any], before_key: str, after_key: str) -> float:
    try:
        before = float(entry.get(before_key) or 0.0)
        after = float(entry.get(after_key) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if before <= 0.0:
        return 0.0
    return float(1.0 - (after / before))


def _add_mode_collapse_warning(rows: list[dict[str, Any]]) -> None:
    if len(rows) <= 1:
        for row in rows:
            row["mode_collapse_warning"] = False
        return
    mandatory_means = [float(row.get("mean_rel_diff_mandatory") or 0.0) for row in rows]
    optional_means = [float(row.get("mean_rel_diff_optional") or 0.0) for row in rows]
    species_reductions = [_compression_ratio(row, "species_before", "species_after") for row in rows]
    reaction_reductions = [_compression_ratio(row, "reactions_before", "reactions_after") for row in rows]
    mandatory_spread = float(max(mandatory_means) - min(mandatory_means)) if mandatory_means else 0.0
    optional_spread = float(max(optional_means) - min(optional_means)) if optional_means else 0.0
    species_spread = float(max(species_reductions) - min(species_reductions)) if species_reductions else 0.0
    reaction_spread = float(max(reaction_reductions) - min(reaction_reductions)) if reaction_reductions else 0.0
    warning = bool(
        mandatory_spread <= 0.03
        and optional_spread <= 0.03
        and (species_spread >= 0.20 or reaction_spread >= 0.30)
    )
    for row in rows:
        row["mode_collapse_warning"] = bool(warning)


def _parse_bool(value: str | bool | None, *, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _normalize_split_mode(value: Any) -> str:
    if value is None:
        return "__none__"
    return str(value).strip().lower() or "__none__"


def _normalize_kfolds(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize reduction run summaries into one JSON")
    parser.add_argument(
        "--entry",
        action="append",
        required=True,
        help="Mode and run id in the form '<mode>:<run_id>', e.g. baseline:reduce_x",
    )
    parser.add_argument("--report-dir", default="reports", help="Base reports directory")
    parser.add_argument(
        "--output",
        default="reports/diamond_benchmarks_diamond_eval_summary.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--enforce-same-split",
        default="true",
        help="Fail fast when split_mode or effective_kfolds differ across entries (true/false)",
    )
    args = parser.parse_args()

    report_dir = Path(args.report_dir).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for raw in args.entry:
        token = str(raw)
        if ":" not in token:
            raise ValueError(f"invalid --entry format: {token!r}")
        mode, run_id = token.split(":", 1)
        mode = mode.strip()
        run_id = run_id.strip()
        if not mode or not run_id:
            raise ValueError(f"invalid --entry value: {token!r}")

        summary_path = report_dir / run_id / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"summary not found: {summary_path}")
        summary = _load_json(summary_path)
        contract_cfg = dict(summary.get("evaluation_contract") or {})
        strict_schema = bool(contract_cfg.get("diagnostic_schema_strict", False))
        validate_summary_schema(summary, strict=strict_schema)
        rows.append(project_entry(mode, run_id, summary))

    if _parse_bool(args.enforce_same_split, default=True) and len(rows) > 1:
        split_modes = {_normalize_split_mode(row.get("split_mode")) for row in rows}
        if len(split_modes) != 1:
            detail = ", ".join(
                f"{row.get('mode')}:{row.get('run_id')}={row.get('split_mode')}" for row in rows
            )
            raise ValueError(f"split_mode mismatch across entries: {detail}")
        kfolds = {_normalize_kfolds(row.get("effective_kfolds")) for row in rows}
        if len(kfolds) != 1:
            detail = ", ".join(
                f"{row.get('mode')}:{row.get('run_id')}={row.get('effective_kfolds')}" for row in rows
            )
            raise ValueError(f"effective_kfolds mismatch across entries: {detail}")
    _add_mode_collapse_warning(rows)

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "report_dir": str(report_dir),
        "entries": rows,
        "mode_collapse_warning": bool(any(bool(row.get("mode_collapse_warning")) for row in rows)),
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
