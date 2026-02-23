import json

from cantera_model.reporting.report import write_report


def test_report_outputs(tmp_path) -> None:
    out = write_report(
        tmp_path / "r1",
        run_id="r1",
        stage_rows=[
            {
                "stage": "A",
                "species_after": 5,
                "reactions_after": 10,
                "hard_ban_violations": 0,
                "gate_passed": True,
            }
        ],
        selected_stage="A",
        summary_payload={"hard_ban_violations": 0, "gate_passed": True},
    )
    summary = json.loads((out / "summary.json").read_text())
    assert summary["selected_stage"] == "A"
    assert summary["hard_ban_violations"] == 0
    assert (out / "pareto.csv").exists()
