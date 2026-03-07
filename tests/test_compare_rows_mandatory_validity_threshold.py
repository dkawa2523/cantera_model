from cantera_model.eval.cantera_runner import compare_rows


def _rows(values: list[tuple[str, float, float, float]]) -> list[dict[str, float | str]]:
    out: list[dict[str, float | str]] = []
    for case_id, m1, m2, m3 in values:
        out.append({"case_id": case_id, "m1": m1, "m2": m2, "m3": m3, "aux": 1.0})
    return out


def test_compare_rows_mandatory_min_valid_count_fail_and_split_counts() -> None:
    baseline = _rows([("c1", 1.0, 1.0, 1.0), ("c2", 1.0, 1.0, 1.0)])
    candidate = _rows(
        [("c1", 1.0, float("nan"), float("nan")), ("c2", 1.0, float("nan"), float("nan"))]
    )

    _, summary = compare_rows(
        baseline,
        candidate,
        rel_eps=1.0e-12,
        rel_tolerance=0.20,
        mandatory_validity={
            "mandatory_metrics": ["m1", "m2", "m3", "m_missing"],
            "mandatory_hard_mode": "min_valid_count",
            "min_valid_mandatory_count_abs": 2,
            "min_valid_mandatory_ratio": 0.67,
            "min_valid_mandatory_cap_by_total": True,
        },
    )

    assert summary["mandatory_total_metric_count"] == 3
    assert summary["valid_mandatory_metric_count"] == 1
    assert summary["active_invalid_mandatory_metric_count"] == 2
    assert summary["inactive_mandatory_metric_count"] == 1
    assert summary["invalid_mandatory_metric_count"] == 3
    assert summary["min_valid_mandatory_count_effective"] == 2
    assert summary["mandatory_validity_passed"] is False


def test_compare_rows_mandatory_min_valid_count_pass() -> None:
    baseline = _rows([("c1", 1.0, 1.0, 1.0), ("c2", 1.0, 1.0, 1.0)])
    candidate = _rows([("c1", 1.0, 1.5, 1.0), ("c2", 1.0, 1.5, 1.0)])

    _, summary = compare_rows(
        baseline,
        candidate,
        rel_eps=1.0e-12,
        rel_tolerance=0.20,
        mandatory_validity={
            "mandatory_metrics": ["m1", "m2", "m3"],
            "mandatory_hard_mode": "min_valid_count",
            "min_valid_mandatory_count_abs": 2,
            "min_valid_mandatory_ratio": 0.67,
            "min_valid_mandatory_cap_by_total": True,
        },
    )

    assert summary["mandatory_total_metric_count"] == 3
    assert summary["valid_mandatory_metric_count"] == 2
    assert summary["mandatory_validity_passed"] is True
