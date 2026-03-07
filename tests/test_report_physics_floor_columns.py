import csv

from cantera_model.reporting.report import write_report


def test_report_contains_floor_columns_and_filtered_pareto(tmp_path) -> None:
    stage_rows = [
        {
            "stage": "A",
            "species_after": 5,
            "reactions_after": 1,
            "mean_rel_diff": 0.05,
            "gate_passed": False,
            "floor_passed": False,
            "floor_min_species": 3,
            "floor_min_reactions": 2,
            "selection_score": 0.1,
        },
        {
            "stage": "B",
            "species_after": 6,
            "reactions_after": 4,
            "mean_rel_diff": 0.10,
            "gate_passed": True,
            "floor_passed": True,
            "floor_min_species": 3,
            "floor_min_reactions": 2,
            "selection_score": 0.8,
        },
    ]
    out = write_report(
        tmp_path / "phys",
        run_id="phys",
        stage_rows=stage_rows,
        selected_stage="B",
        summary_payload={"gate_passed": True, "hard_ban_violations": 0},
    )
    with (out / "metrics.csv").open("r", newline="") as handle:
        metrics = list(csv.DictReader(handle))
    with (out / "pareto.csv").open("r", newline="") as handle:
        pareto = list(csv.DictReader(handle))

    assert metrics
    assert "selection_score" in metrics[0]
    assert "floor_passed" in metrics[0]
    assert "floor_min_species" in metrics[0]
    assert "floor_min_reactions" in metrics[0]
    assert len(pareto) == 1
    assert pareto[0]["stage"] == "B"
