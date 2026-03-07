from cantera_model.cli.summarize_reduction_eval import _add_mode_collapse_warning


def test_mode_collapse_warning_turns_true_on_large_compression_gap_with_similar_error() -> None:
    rows = [
        {
            "mode": "baseline",
            "mean_rel_diff_mandatory": 0.22,
            "mean_rel_diff_optional": 0.31,
            "species_before": 100,
            "species_after": 80,
            "reactions_before": 300,
            "reactions_after": 240,
        },
        {
            "mode": "pooling",
            "mean_rel_diff_mandatory": 0.23,
            "mean_rel_diff_optional": 0.30,
            "species_before": 100,
            "species_after": 40,
            "reactions_before": 300,
            "reactions_after": 120,
        },
    ]
    _add_mode_collapse_warning(rows)
    assert rows[0]["mode_collapse_warning"] is True
    assert rows[1]["mode_collapse_warning"] is True


def test_mode_collapse_warning_false_when_error_gap_is_not_small() -> None:
    rows = [
        {
            "mode": "baseline",
            "mean_rel_diff_mandatory": 0.22,
            "mean_rel_diff_optional": 0.31,
            "species_before": 100,
            "species_after": 80,
            "reactions_before": 300,
            "reactions_after": 240,
        },
        {
            "mode": "learnckpp",
            "mean_rel_diff_mandatory": 0.40,
            "mean_rel_diff_optional": 0.55,
            "species_before": 100,
            "species_after": 40,
            "reactions_before": 300,
            "reactions_after": 120,
        },
    ]
    _add_mode_collapse_warning(rows)
    assert rows[0]["mode_collapse_warning"] is False
    assert rows[1]["mode_collapse_warning"] is False
