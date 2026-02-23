import json
import subprocess
import sys
from pathlib import Path


def test_summarize_reduction_eval_emits_split_and_qoi_fields(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    out_dir = reports / "r0"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "gate_passed": True,
        "selected_stage": "B",
        "hard_ban_violations": 0,
        "failure_reason": None,
        "qoi_integral_count": 2,
        "surrogate_split": {
            "mode": "kfold",
            "effective_kfolds": 3,
        },
        "selected_metrics": {
            "species_before": 20,
            "species_after": 8,
            "reactions_before": 40,
            "reactions_after": 15,
            "pass_rate": 0.8,
            "mean_rel_diff": 0.2,
            "conservation_violation": 0.0,
            "qoi_metrics_count": 9,
            "integral_qoi_count": 2,
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
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    payload = json.loads(output_path.read_text())
    entry = payload["entries"][0]
    assert entry["split_mode"] == "kfold"
    assert entry["effective_kfolds"] == 3
    assert entry["qoi_metrics_count"] == 9
    assert entry["integral_qoi_count"] == 2
