import pytest

from cantera_model.cli.reduce_validate import (
    _resolve_evaluation_contract,
    _validate_evaluation_contract,
)


def test_resolve_evaluation_contract_sets_defaults() -> None:
    cfg: dict = {}
    contract = _resolve_evaluation_contract(cfg)
    assert contract["version"] == "v1"
    assert contract["enforce"] is False
    assert contract["invariants_profile"] == "strict_physical_v1"
    assert contract["evaluation_profile"] == "tiered_v35"
    assert contract["run_policy_profile"] == "adaptive_kfold_strict_v1"
    assert contract["diagnostic_schema_strict"] is False


def test_validate_evaluation_contract_requires_tiered_and_explicit_thresholds() -> None:
    eval_cfg = {
        "error_aggregation": {"mode": "legacy_all_metric"},
        "surrogate_split": {"mode": "adaptive_kfold", "enforce_explicit_policy": True},
    }
    contract = {
        "version": "v1",
        "enforce": True,
        "invariants_profile": "strict_physical_v1",
        "evaluation_profile": "tiered_v35",
        "run_policy_profile": "adaptive_kfold_strict_v1",
        "diagnostic_schema_strict": True,
    }
    with pytest.raises(ValueError, match="error_aggregation.mode=tiered"):
        _validate_evaluation_contract(eval_cfg, contract)


def test_validate_evaluation_contract_requires_adaptive_kfold_explicit_policy() -> None:
    eval_cfg = {
        "error_aggregation": {"mode": "tiered", "require_explicit_thresholds": True},
        "surrogate_split": {"mode": "adaptive_kfold", "enforce_explicit_policy": False},
    }
    contract = {
        "version": "v1",
        "enforce": True,
        "invariants_profile": "strict_physical_v1",
        "evaluation_profile": "tiered_v35",
        "run_policy_profile": "adaptive_kfold_strict_v1",
        "diagnostic_schema_strict": True,
    }
    with pytest.raises(ValueError, match="enforce_explicit_policy=true"):
        _validate_evaluation_contract(eval_cfg, contract)
