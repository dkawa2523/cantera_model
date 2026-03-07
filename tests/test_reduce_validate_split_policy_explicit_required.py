import pytest

from cantera_model.cli.reduce_validate import _resolve_surrogate_split


def _conditions(n: int) -> list[dict[str, object]]:
    return [{"case_id": f"c{i}"} for i in range(n)]


def test_adaptive_kfold_requires_explicit_policy_when_enforced() -> None:
    eval_cfg = {"surrogate_split": {"mode": "adaptive_kfold", "enforce_explicit_policy": True}}
    with pytest.raises(ValueError, match="kfold_policy is required"):
        _resolve_surrogate_split(_conditions(6), eval_cfg)


def test_adaptive_kfold_accepts_explicit_policy_when_enforced() -> None:
    eval_cfg = {
        "surrogate_split": {
            "mode": "adaptive_kfold",
            "enforce_explicit_policy": True,
            "kfold_policy": {
                "min_cases_for_kfold": 4,
                "default_k": 2,
                "k_by_case_count": {6: 3, 8: 3},
            },
        }
    }
    split = _resolve_surrogate_split(_conditions(6), eval_cfg)
    assert split["mode"] == "kfold"
    assert split["effective_kfolds"] == 3
