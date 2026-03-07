from cantera_model.cli.reduce_validate import _resolve_learnckpp_target_keep_ratio


def test_keep_ratio_structure_feedback_boosts_target_ratio() -> None:
    ratio, meta = _resolve_learnckpp_target_keep_ratio(
        base_keep_ratio=0.30,
        prune_keep_ratio=0.50,
        stage_idx=1,
        data_source="synthetic",
        split_mode="in_sample",
        max_mean_rel=0.40,
        prev_mean_rel_diff=0.10,
        prev_stage_physical=None,
        prev_stage_structure={"domain_deficit": 0.20, "balance_margin": -0.10},
        adaptive_cfg={
            "enabled": True,
            "min_keep_ratio": 0.10,
            "max_keep_ratio": 0.95,
            "source_multiplier": {"synthetic": 1.0, "default": 1.0},
            "split_multiplier": {"in_sample": 1.0, "default": 1.0},
            "stage_multiplier": [1.0, 1.0, 1.0],
            "feedback_trigger_ratio": 0.80,
            "feedback_multiplier": 1.15,
            "structure_feedback": {
                "enabled": True,
                "alpha_domain": 0.8,
                "beta_balance": 0.5,
                "max_multiplier": 1.35,
            },
        },
    )
    assert abs(ratio - 0.1815) < 1.0e-12
    assert abs(float(meta["structure_feedback_multiplier"]) - 1.21) < 1.0e-12


def test_keep_ratio_structure_feedback_defaults_to_one_when_disabled() -> None:
    ratio, meta = _resolve_learnckpp_target_keep_ratio(
        base_keep_ratio=0.30,
        prune_keep_ratio=0.50,
        stage_idx=1,
        data_source="synthetic",
        split_mode="in_sample",
        max_mean_rel=0.40,
        prev_mean_rel_diff=0.10,
        prev_stage_physical=None,
        prev_stage_structure={"domain_deficit": 0.20, "balance_margin": -0.10},
        adaptive_cfg={
            "enabled": True,
            "min_keep_ratio": 0.10,
            "max_keep_ratio": 0.95,
            "source_multiplier": {"synthetic": 1.0, "default": 1.0},
            "split_multiplier": {"in_sample": 1.0, "default": 1.0},
            "stage_multiplier": [1.0, 1.0, 1.0],
        },
    )
    assert abs(ratio - 0.15) < 1.0e-12
    assert abs(float(meta["structure_feedback_multiplier"]) - 1.0) < 1.0e-12
