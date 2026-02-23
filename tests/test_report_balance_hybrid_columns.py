import csv

from cantera_model.reporting.report import write_report


def test_report_hybrid_balance_columns(tmp_path) -> None:
    rows = [
        {
            "stage": "A",
            "species_after": 12,
            "reactions_after": 3,
            "mean_rel_diff": 0.1,
            "gate_passed": True,
            "floor_passed": True,
            "balance_gate_passed": False,
            "reaction_species_ratio": 0.25,
            "active_species_coverage": 0.50,
            "weighted_active_species_coverage": 0.60,
            "active_species_coverage_top_weighted": 0.62,
            "essential_species_coverage": 0.70,
            "nu_rank_ratio": 0.4,
            "max_cluster_size_ratio": 0.50,
            "cluster_guard_passed": False,
            "balance_mode": "hybrid",
            "balance_violations": "min_weighted_active_species_coverage",
        },
        {
            "stage": "B",
            "species_after": 10,
            "reactions_after": 8,
            "mean_rel_diff": 0.2,
            "gate_passed": True,
            "floor_passed": True,
            "balance_gate_passed": True,
            "reaction_species_ratio": 0.8,
            "active_species_coverage": 0.6,
            "weighted_active_species_coverage": 0.84,
            "active_species_coverage_top_weighted": 0.86,
            "essential_species_coverage": 0.86,
            "nu_rank_ratio": 0.6,
            "max_cluster_size_ratio": 0.30,
            "cluster_guard_passed": True,
            "balance_mode": "hybrid",
            "balance_violations": "",
        },
    ]
    out = write_report(
        tmp_path / "r_bal_hybrid",
        run_id="r_bal_hybrid",
        stage_rows=rows,
        selected_stage="B",
        summary_payload={
            "gate_passed": True,
            "hard_ban_violations": 0,
            "reduction_trace": {"candidate_trend": rows},
        },
    )
    with (out / "metrics.csv").open("r", newline="") as handle:
        metrics = list(csv.DictReader(handle))
    with (out / "pareto.csv").open("r", newline="") as handle:
        pareto = list(csv.DictReader(handle))
    report_md = (out / "report.md").read_text()

    assert "weighted_active_species_coverage" in metrics[0]
    assert "active_species_coverage_top_weighted" in metrics[0]
    assert "essential_species_coverage" in metrics[0]
    assert "max_cluster_size_ratio" in metrics[0]
    assert "cluster_guard_passed" in metrics[0]
    assert "balance_mode" in metrics[0]
    assert "| weighted_cov | top_weighted_cov | essential_cov | max_cluster_ratio | balance_mode |" in report_md
    assert len(pareto) == 1
    assert pareto[0]["stage"] == "B"
