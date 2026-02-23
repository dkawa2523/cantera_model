import csv

from cantera_model.reporting.report import write_report


def test_report_pareto_frontier_subset(tmp_path) -> None:
    stage_rows = [
        {"stage": "A", "species_after": 8, "reactions_after": 15, "mean_rel_diff": 0.05, "hard_ban_violations": 0, "gate_passed": True},
        {"stage": "B", "species_after": 7, "reactions_after": 15, "mean_rel_diff": 0.06, "hard_ban_violations": 0, "gate_passed": True},
        {"stage": "C", "species_after": 9, "reactions_after": 20, "mean_rel_diff": 0.20, "hard_ban_violations": 0, "gate_passed": False},
    ]
    out = write_report(
        tmp_path / "r",
        run_id="r",
        stage_rows=stage_rows,
        selected_stage="A",
        summary_payload={"hard_ban_violations": 0, "gate_passed": True},
    )

    with (out / "metrics.csv").open("r", newline="") as handle:
        metrics = list(csv.DictReader(handle))
    with (out / "pareto.csv").open("r", newline="") as handle:
        pareto = list(csv.DictReader(handle))

    assert len(metrics) == 3
    assert len(pareto) == 2
    assert {row["stage"] for row in pareto} == {"A", "B"}
