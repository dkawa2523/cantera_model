import csv

from cantera_model.reporting.report import write_report


def test_report_balance_columns_and_pareto_filter(tmp_path) -> None:
    rows = [
        {
            "stage": "A",
            "species_after": 10,
            "reactions_after": 2,
            "mean_rel_diff": 0.1,
            "gate_passed": False,
            "floor_passed": True,
            "balance_gate_passed": False,
            "reaction_species_ratio": 0.2,
            "active_species_coverage": 0.9,
            "nu_rank_ratio": 0.4,
            "balance_violations": "min_reaction_species_ratio",
        },
        {
            "stage": "B",
            "species_after": 8,
            "reactions_after": 8,
            "mean_rel_diff": 0.2,
            "gate_passed": True,
            "floor_passed": True,
            "balance_gate_passed": True,
            "reaction_species_ratio": 1.0,
            "active_species_coverage": 0.9,
            "nu_rank_ratio": 0.7,
            "balance_violations": "",
        },
    ]
    out = write_report(
        tmp_path / "r_bal",
        run_id="r_bal",
        stage_rows=rows,
        selected_stage="B",
        summary_payload={"gate_passed": True, "hard_ban_violations": 0},
    )
    with (out / "metrics.csv").open("r", newline="") as handle:
        metrics = list(csv.DictReader(handle))
    with (out / "pareto.csv").open("r", newline="") as handle:
        pareto = list(csv.DictReader(handle))

    assert "reaction_species_ratio" in metrics[0]
    assert "active_species_coverage" in metrics[0]
    assert "nu_rank_ratio" in metrics[0]
    assert "balance_gate_passed" in metrics[0]
    assert "balance_violations" in metrics[0]
    assert len(pareto) == 1
    assert pareto[0]["stage"] == "B"
