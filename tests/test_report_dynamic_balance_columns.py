import csv

from cantera_model.reporting.report import write_report


def test_report_dynamic_balance_columns_are_emitted(tmp_path) -> None:
    rows = [
        {
            "stage": "A",
            "species_after": 12,
            "reactions_after": 8,
            "mean_rel_diff": 0.12,
            "gate_passed": True,
            "floor_passed": True,
            "balance_gate_passed": True,
            "cluster_guard_passed": True,
            "reaction_species_ratio": 0.66,
            "active_species_coverage": 0.62,
            "weighted_active_species_coverage": 0.84,
            "active_species_coverage_top_weighted": 0.86,
            "essential_species_coverage": 0.88,
            "nu_rank_ratio": 0.55,
            "max_cluster_size_ratio": 0.32,
            "balance_mode": "hybrid",
            "balance_dynamic_applied": True,
            "balance_dynamic_complexity": 0.75,
            "rs_upper_effective": 5.5,
            "active_cov_effective_floor": 0.40,
            "balance_margin": 0.03,
        }
    ]
    out = write_report(
        tmp_path / "dynamic_report",
        run_id="dynamic_report",
        stage_rows=rows,
        selected_stage="A",
        summary_payload={
            "gate_passed": True,
            "hard_ban_violations": 0,
            "reduction_trace": {"candidate_trend": rows},
        },
    )
    with (out / "metrics.csv").open("r", newline="") as handle:
        metrics = list(csv.DictReader(handle))

    report_md = (out / "report.md").read_text()
    assert "balance_dynamic_applied" in metrics[0]
    assert "balance_dynamic_complexity" in metrics[0]
    assert "rs_upper_effective" in metrics[0]
    assert "active_cov_effective_floor" in metrics[0]
    assert "| rs_upper_eff | active_cov_floor_eff | balance_margin | dynamic |" in report_md
