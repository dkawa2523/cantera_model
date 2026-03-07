import json
import subprocess
import sys
from pathlib import Path


def test_summarize_reduction_eval_fails_fast_when_strict_schema_missing(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    run_dir = reports / "r0"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "evaluation_contract": {"diagnostic_schema_strict": True},
        "gate_passed": True,
        "primary_blocker_layer": "none",
        "selected_stage": "A",
        "selected_metrics": {},
        "gate_evidence": {"selected_stage_evidence": {"mandatory_validity_passed": True}},
        "surrogate_split": {"mode": "adaptive_kfold", "effective_kfolds": 3},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary))

    output_path = tmp_path / "summary_eval.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "cantera_model.cli.summarize_reduction_eval",
            "--report-dir",
            str(reports),
            "--output",
            str(output_path),
            "--entry",
            "baseline:r0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "diagnostic schema missing keys" in proc.stderr


def test_summarize_reduction_eval_allows_missing_schema_when_not_strict(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    run_dir = reports / "r0"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "evaluation_contract": {"diagnostic_schema_strict": False},
        "gate_passed": True,
        "primary_blocker_layer": "none",
        "selected_stage": "A",
        "selected_metrics": {},
        "gate_evidence": {"selected_stage_evidence": {"mandatory_validity_passed": True}},
        "surrogate_split": {"mode": "adaptive_kfold", "effective_kfolds": 3},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary))

    output_path = tmp_path / "summary_eval.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "cantera_model.cli.summarize_reduction_eval",
            "--report-dir",
            str(reports),
            "--output",
            str(output_path),
            "--entry",
            "baseline:r0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
