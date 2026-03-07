from cantera_model.cli.reduce_validate import _resolve_surrogate_split


def _conditions(n: int) -> list[dict[str, object]]:
    return [{"case_id": f"c{i}"} for i in range(n)]


def test_adaptive_kfold_policy_switches_by_case_count() -> None:
    eval_cfg = {
        "surrogate_split": {
            "mode": "adaptive_kfold",
            "kfold_policy": {
                "min_cases_for_kfold": 4,
                "default_k": 2,
                "k_by_case_count": {6: 3, 8: 3},
            },
        }
    }

    split3 = _resolve_surrogate_split(_conditions(3), eval_cfg)
    assert split3["requested_mode"] == "adaptive_kfold"
    assert split3["mode"] == "in_sample"
    assert split3["fallback_reason"] == "insufficient_cases_for_kfold"
    assert split3["effective_kfolds"] == 0

    split4 = _resolve_surrogate_split(_conditions(4), eval_cfg)
    assert split4["mode"] == "kfold"
    assert split4["effective_kfolds"] == 2
    assert split4["case_count"] == 4
    assert split4["fold_sizes"] == [2, 2]

    split6 = _resolve_surrogate_split(_conditions(6), eval_cfg)
    assert split6["mode"] == "kfold"
    assert split6["effective_kfolds"] == 3
    assert split6["fold_sizes"] == [2, 2, 2]

    split8 = _resolve_surrogate_split(_conditions(8), eval_cfg)
    assert split8["mode"] == "kfold"
    assert split8["effective_kfolds"] == 3
    assert sorted(split8["fold_sizes"]) == [2, 3, 3]
