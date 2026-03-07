import json
import subprocess
import sys
from pathlib import Path

from cantera_model.eval.diagnostic_schema import project_entry


def test_summarize_uses_project_entry_consistently(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports"
    run_dir = report_dir / "r0"
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "gate_passed": True,
        "primary_blocker_layer": "none",
        "selected_stage": "A",
        "selected_metrics": {
            "species_before": 10,
            "species_after": 5,
            "reactions_before": 20,
            "reactions_after": 8,
            "pass_rate": 0.8,
            "mean_rel_diff": 0.2,
        },
        "surrogate_split": {"mode": "adaptive_kfold", "effective_kfolds": 3},
        "evaluation_contract": {"diagnostic_schema_strict": False},
        "gate_evidence": {
            "selected_stage_evidence": {
                "mandatory_validity_passed": True,
                "coverage_gate_passed": True,
                "error_gate_passed": True,
                "mandatory_error_passed": True,
                "optional_error_passed": True,
                "error_fail_reason_primary": "none",
                "validity_fail_reason_primary": "none",
                "primary_blocker_layer": "none",
            }
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary))

    out = tmp_path / "summary_eval.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "cantera_model.cli.summarize_reduction_eval",
            "--report-dir",
            str(report_dir),
            "--output",
            str(out),
            "--entry",
            "baseline:r0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    payload = json.loads(out.read_text())
    entry = payload["entries"][0]
    projected = project_entry("baseline", "r0", summary)
    for key, value in projected.items():
        assert entry[key] == value
