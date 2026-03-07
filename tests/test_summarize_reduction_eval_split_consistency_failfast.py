import json
import subprocess
import sys
from pathlib import Path


def _write_summary(path: Path, *, split_mode: str, kfolds: int) -> None:
    summary = {
        "gate_passed": True,
        "selected_stage": "A",
        "hard_ban_violations": 0,
        "surrogate_split": {"mode": split_mode, "effective_kfolds": kfolds},
        "selected_metrics": {
            "species_before": 10,
            "species_after": 5,
            "reactions_before": 20,
            "reactions_after": 8,
            "pass_rate": 0.8,
            "mean_rel_diff": 0.2,
            "conservation_violation": 0.0,
        },
    }
    path.write_text(json.dumps(summary))


def test_summarize_reduction_eval_fails_fast_on_split_mismatch(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    (reports / "r0").mkdir(parents=True, exist_ok=True)
    (reports / "r1").mkdir(parents=True, exist_ok=True)
    _write_summary(reports / "r0" / "summary.json", split_mode="adaptive_kfold", kfolds=3)
    _write_summary(reports / "r1" / "summary.json", split_mode="in_sample", kfolds=1)

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
            "pooling:r1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "split_mode mismatch across entries" in proc.stderr


def test_summarize_reduction_eval_can_disable_split_failfast(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    (reports / "r0").mkdir(parents=True, exist_ok=True)
    (reports / "r1").mkdir(parents=True, exist_ok=True)
    _write_summary(reports / "r0" / "summary.json", split_mode="adaptive_kfold", kfolds=3)
    _write_summary(reports / "r1" / "summary.json", split_mode="in_sample", kfolds=1)

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
            "--enforce-same-split",
            "false",
            "--entry",
            "baseline:r0",
            "--entry",
            "pooling:r1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
