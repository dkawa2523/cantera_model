from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"summary must be a JSON object: {path}")
    return data


def _entry_from_summary(mode: str, run_id: str, summary: dict[str, Any]) -> dict[str, Any]:
    selected = dict(summary.get("selected_metrics") or {})
    split = dict(summary.get("surrogate_split") or {})
    fallback_reason = selected.get("learnckpp_fallback_reason") or summary.get("failure_reason")
    return {
        "mode": mode,
        "run_id": run_id,
        "gate_passed": bool(summary.get("gate_passed")),
        "selected_stage": summary.get("selected_stage"),
        "species_before": selected.get("species_before"),
        "species_after": selected.get("species_after"),
        "reactions_before": selected.get("reactions_before"),
        "reactions_after": selected.get("reactions_after"),
        "pass_rate": selected.get("pass_rate"),
        "mean_rel_diff": selected.get("mean_rel_diff"),
        "conservation_violation": selected.get("conservation_violation"),
        "hard_ban_violations": summary.get("hard_ban_violations"),
        "split_mode": split.get("mode"),
        "effective_kfolds": split.get("effective_kfolds"),
        "qoi_metrics_count": selected.get("qoi_metrics_count"),
        "integral_qoi_count": selected.get("integral_qoi_count", summary.get("qoi_integral_count")),
        "fallback_reason": fallback_reason,
    }


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
        rows.append(_entry_from_summary(mode, run_id, summary))

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "report_dir": str(report_dir),
        "entries": rows,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
