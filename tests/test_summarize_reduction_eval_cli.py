import json
import subprocess
import sys
from pathlib import Path


def test_summarize_reduction_eval_cli(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    for mode, run_id in [("baseline", "r0"), ("learnckpp", "r1")]:
        out_dir = reports / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "gate_passed": True,
            "selected_stage": "C",
            "hard_ban_violations": 0,
            "failure_reason": None,
            "selected_metrics": {
                "species_before": 10,
                "species_after": 5,
                "reactions_before": 20,
                "reactions_after": 8,
                "pass_rate": 0.8,
                "mean_rel_diff": 0.2,
                "conservation_violation": 0.0,
                "learnckpp_fallback_reason": None,
            },
        }
        (out_dir / "summary.json").write_text(json.dumps(summary))

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
            "--entry",
            "learnckpp:r1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(output_path.read_text())
    assert len(payload["entries"]) == 2
    assert payload["entries"][0]["mode"] == "baseline"
