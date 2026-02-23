import csv

from cantera_model.reporting.report import write_report


def test_report_includes_cluster_guard_columns_and_pareto_filter(tmp_path) -> None:
    rows = [
        {
            "stage": "A",
            "species_after": 8,
            "reactions_after": 13,
            "mean_rel_diff": 0.10,
            "gate_passed": True,
            "floor_passed": True,
            "balance_gate_passed": True,
            "cluster_guard_passed": False,
            "active_species_coverage_top_weighted": 0.95,
            "max_cluster_size_ratio": 0.60,
        },
        {
            "stage": "B",
            "species_after": 10,
            "reactions_after": 14,
            "mean_rel_diff": 0.12,
            "gate_passed": True,
            "floor_passed": True,
            "balance_gate_passed": True,
            "cluster_guard_passed": True,
            "active_species_coverage_top_weighted": 0.92,
            "max_cluster_size_ratio": 0.40,
        },
    ]
    out = write_report(
        tmp_path / "r_cluster_guard",
        run_id="r_cluster_guard",
        stage_rows=rows,
        selected_stage="B",
        summary_payload={"gate_passed": True, "hard_ban_violations": 0, "reduction_trace": {"candidate_trend": rows}},
    )
    with (out / "metrics.csv").open("r", newline="") as handle:
        metrics = list(csv.DictReader(handle))
    with (out / "pareto.csv").open("r", newline="") as handle:
        pareto = list(csv.DictReader(handle))

    assert "active_species_coverage_top_weighted" in metrics[0]
    assert "max_cluster_size_ratio" in metrics[0]
    assert "cluster_guard_passed" in metrics[0]
    assert len(pareto) == 1
    assert pareto[0]["stage"] == "B"
