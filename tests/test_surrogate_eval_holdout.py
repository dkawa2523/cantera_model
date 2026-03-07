from cantera_model.eval.surrogate_eval import fit_lightweight_surrogate, run_surrogate_cases


def _data():
    conditions = [
        {"case_id": "c0", "T0": 900.0, "P0_atm": 1.0, "phi": 0.9, "t_end": 0.10},
        {"case_id": "c1", "T0": 950.0, "P0_atm": 1.2, "phi": 1.0, "t_end": 0.11},
        {"case_id": "c2", "T0": 1000.0, "P0_atm": 1.4, "phi": 1.1, "t_end": 0.12},
        {"case_id": "c3", "T0": 1050.0, "P0_atm": 1.6, "phi": 1.2, "t_end": 0.13},
    ]
    qoi = {"species_last": ["CH4", "O2"], "species_max": ["OH"]}
    baseline = run_surrogate_cases({"global_scale": 1.0}, conditions, qoi)
    return conditions, qoi, baseline


def test_fit_lightweight_surrogate_holdout_split() -> None:
    conditions, qoi, baseline = _data()
    model = fit_lightweight_surrogate(
        conditions,
        baseline,
        qoi,
        l2=1.0e-6,
        split_cfg={"mode": "holdout", "holdout_ratio": 0.25},
    )
    split_meta = dict(model.get("split_meta") or {})
    assert split_meta["requested_mode"] == "holdout"
    assert split_meta["mode"] in {"holdout", "in_sample"}
    if split_meta["mode"] == "holdout":
        assert len(split_meta["test_case_ids"]) >= 1
        assert len(split_meta["train_case_ids"]) >= 2


def test_fit_lightweight_surrogate_holdout_fallback_small_cases() -> None:
    conditions, qoi, baseline = _data()
    model = fit_lightweight_surrogate(
        conditions[:2],
        baseline[:2],
        qoi,
        l2=1.0e-6,
        split_cfg={"mode": "holdout", "holdout_ratio": 0.5, "min_train_cases": 2},
    )
    split_meta = dict(model.get("split_meta") or {})
    assert split_meta["requested_mode"] == "holdout"
    assert split_meta["mode"] == "in_sample"
    assert split_meta["fallback_reason"] == "insufficient_cases_for_holdout"
