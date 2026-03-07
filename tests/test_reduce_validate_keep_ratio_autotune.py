from cantera_model.cli.reduce_validate import _auto_tune_learnckpp_keep_ratio


def test_auto_tune_keep_ratio_prefers_more_compression_when_risk_low() -> None:
    ratio, meta = _auto_tune_learnckpp_keep_ratio(
        base_keep_ratio=0.30,
        overall_candidates=100,
        min_keep_ratio=0.01,
        max_keep_ratio=1.0,
        data_source="synthetic",
        split_mode="in_sample",
        prev_stage_physical={"passed": True, "raw_conservation_violation": 0.0, "raw_negative_steps": 0},
        cfg={
            "enabled": True,
            "multipliers": [0.8, 1.0, 1.2],
            "compression_weight": 1.0,
            "safety_weight": 1.0,
            "min_keep_floor": 0.05,
            "risk_keep_boost": 0.1,
            "risk_cap": 3.0,
            "source_risk": {"synthetic": 0.0},
            "split_risk": {"in_sample": 0.0},
        },
    )
    assert ratio <= 0.30
    assert meta["enabled"] is True


def test_auto_tune_keep_ratio_prefers_safety_when_risk_high() -> None:
    ratio, meta = _auto_tune_learnckpp_keep_ratio(
        base_keep_ratio=0.30,
        overall_candidates=100,
        min_keep_ratio=0.01,
        max_keep_ratio=1.0,
        data_source="trace_h5",
        split_mode="kfold",
        prev_stage_physical={"passed": False, "raw_conservation_violation": 3.0, "raw_negative_steps": 60},
        cfg={
            "enabled": True,
            "multipliers": [0.8, 1.0, 1.2],
            "compression_weight": 1.0,
            "safety_weight": 6.0,
            "min_keep_floor": 0.10,
            "risk_keep_boost": 0.6,
            "risk_cap": 3.0,
            "source_risk": {"trace_h5": 0.5},
            "split_risk": {"kfold": 0.3},
            "physical_risk": {
                "cons_norm": 1.0,
                "neg_norm": 20.0,
                "cons_weight": 1.0,
                "neg_weight": 1.0,
                "fail_penalty": 1.0,
            },
        },
    )
    assert ratio >= 0.30
    assert meta["risk"] > 1.0
