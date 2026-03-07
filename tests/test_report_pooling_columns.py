import csv

from cantera_model.reporting.report import write_report


def test_report_metrics_include_pooling_columns_from_union_keys(tmp_path) -> None:
    out = write_report(
        tmp_path / "r_pooling",
        run_id="r_pooling",
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
                "pooling_clusters": 4,
                "pooling_constraint_loss": 0.13,
                "pooling_hard_ban_violations": 0,
                "pooling_artifact_path": "artifacts/pooling/x.npz",
            },
        ],
        selected_stage="A",
        summary_payload={"gate_passed": True, "hard_ban_violations": 0},
    )

    with (out / "metrics.csv").open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = list(reader.fieldnames or [])

    assert "pooling_clusters" in headers
    assert "pooling_constraint_loss" in headers
    assert "pooling_hard_ban_violations" in headers
    assert "pooling_artifact_path" in headers
