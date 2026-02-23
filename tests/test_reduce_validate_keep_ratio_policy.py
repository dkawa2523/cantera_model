from cantera_model.cli.reduce_validate import _resolve_learnckpp_target_keep_ratio


def test_keep_ratio_policy_adapts_by_source_and_feedback() -> None:
    ratio, meta = _resolve_learnckpp_target_keep_ratio(
        base_keep_ratio=0.30,
        prune_keep_ratio=0.50,
        stage_idx=0,
        data_source="trace_h5",
        split_mode="in_sample",
        max_mean_rel=0.40,
        prev_mean_rel_diff=0.35,
        prev_stage_physical=None,
        adaptive_cfg={
            "enabled": True,
            "min_keep_ratio": 0.10,
            "max_keep_ratio": 0.95,
            "source_multiplier": {"trace_h5": 1.10, "default": 1.00},
            "split_multiplier": {"in_sample": 1.00, "default": 1.00},
            "stage_multiplier": [1.20, 1.0, 0.9],
            "feedback_trigger_ratio": 0.80,
            "feedback_multiplier": 1.15,
        },
    )
    # raw=0.15 -> *1.10*1.20*1.15=0.2277
    assert abs(ratio - 0.2277) < 1.0e-6
    assert meta["feedback_multiplier"] == 1.15


def test_keep_ratio_policy_disable_returns_raw() -> None:
    ratio, meta = _resolve_learnckpp_target_keep_ratio(
        base_keep_ratio=0.30,
        prune_keep_ratio=0.50,
        stage_idx=2,
        data_source="synthetic",
        split_mode="in_sample",
        max_mean_rel=0.40,
        prev_mean_rel_diff=None,
        prev_stage_physical=None,
        adaptive_cfg={"enabled": False},
    )
    assert abs(ratio - 0.15) < 1.0e-12
    assert meta["enabled"] is False


def test_keep_ratio_policy_physical_feedback_boost() -> None:
    ratio, meta = _resolve_learnckpp_target_keep_ratio(
        base_keep_ratio=0.30,
        prune_keep_ratio=0.50,
        stage_idx=1,
        data_source="synthetic",
        split_mode="in_sample",
        max_mean_rel=0.40,
        prev_mean_rel_diff=0.10,
        prev_stage_physical={
            "passed": False,
            "raw_conservation_violation": 2.0,
            "raw_negative_steps": 20,
        },
        adaptive_cfg={
            "enabled": True,
            "min_keep_ratio": 0.10,
            "max_keep_ratio": 0.95,
            "source_multiplier": {"synthetic": 1.0, "default": 1.0},
            "split_multiplier": {"in_sample": 1.0, "default": 1.0},
            "stage_multiplier": [1.0, 1.0, 1.0],
            "feedback_trigger_ratio": 0.80,
            "feedback_multiplier": 1.15,
            "physical_feedback": {
                "enabled": True,
                "fail_multiplier": 1.30,
                "cons_multiplier": 1.10,
                "negative_multiplier": 1.10,
                "raw_conservation_threshold": 1.0,
                "raw_negative_steps_threshold": 8,
            },
        },
    )
    # raw=0.15, physical path picks max multiplier=1.30
    assert abs(ratio - 0.195) < 1.0e-12
    assert meta["physical_feedback_multiplier"] == 1.30
    assert "prev_stage_failed" in meta["physical_feedback_reason"]
