from cantera_model.reporting.report import write_report


def test_report_markdown_contains_reduction_trace_sections(tmp_path) -> None:
    out = write_report(
        tmp_path / "r_trace",
        run_id="r_trace",
        stage_rows=[
            {
                "stage": "A",
                "species_after": 5,
                "reactions_after": 9,
                "mean_rel_diff": 0.11,
                "overall_candidates": 20,
                "overall_selected": 10,
                "overall_select_ratio": 0.5,
            }
        ],
        selected_stage="A",
        summary_payload={
            "gate_passed": True,
            "hard_ban_violations": 0,
            "reduction_trace": {
                "candidate_trend": [
                    {
                        "stage": "A",
                        "species_after": 5,
                        "reactions_after": 9,
                        "overall_candidates": 20,
                        "overall_selected": 10,
                        "overall_select_ratio": 0.5,
                        "mean_rel_diff": 0.11,
                    }
                ],
                "cluster_preview": {
                    "selected_stage": "A",
                    "selected_stage_clusters": [
                        {
                            "cluster_id": 0,
                            "size": 2,
                            "members_sample": ["CH4", "CH"],
                            "elements": ["C", "H"],
                        }
                    ],
                },
            },
        },
    )

    report_text = (out / "report.md").read_text()
    assert "## Candidate Trend" in report_text
    assert "## Selected Stage Clusters" in report_text
    assert "CH4, CH" in report_text
