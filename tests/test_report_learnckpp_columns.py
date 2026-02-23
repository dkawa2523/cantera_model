import csv

from cantera_model.reporting.report import write_report


def test_report_metrics_include_learnckpp_columns_from_union_keys(tmp_path) -> None:
    out = write_report(
        tmp_path / "r_learn",
        run_id="r_learn",
        stage_rows=[
            {
                "stage": "A",
                "species_after": 5,
                "reactions_after": 9,
                "mean_rel_diff": 0.1,
            },
            {
                "stage": "B",
                "species_after": 4,
                "reactions_after": 6,
                "mean_rel_diff": 0.2,
                "overall_candidates": 10,
                "overall_selected": 6,
                "overall_select_ratio": 0.6,
            },
        ],
        selected_stage="A",
        summary_payload={"gate_passed": True, "hard_ban_violations": 0},
    )

    with (out / "metrics.csv").open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = list(reader.fieldnames or [])

    assert "overall_candidates" in headers
    assert "overall_selected" in headers
    assert "overall_select_ratio" in headers
