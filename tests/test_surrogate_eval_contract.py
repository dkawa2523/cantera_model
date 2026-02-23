import numpy as np

from cantera_model.eval.surrogate_eval import (
    compare_with_baseline,
    fit_lightweight_surrogate,
    run_surrogate_cases,
    run_surrogate_traces,
)


def test_surrogate_eval_contract() -> None:
    conditions = [
        {"case_id": "c0", "T0": 1000.0, "phi": 1.0, "t_end": 0.1},
        {"case_id": "c1", "T0": 1100.0, "phi": 0.8, "t_end": 0.2},
    ]
    qoi = {"species_last": ["CH4", "O2"], "species_max": ["OH"]}

    baseline = run_surrogate_cases({"global_scale": 1.0}, conditions, qoi)
    candidate = run_surrogate_cases({"reference_rows": baseline, "global_scale": 1.05}, conditions, qoi)

    _, summary = compare_with_baseline(
        baseline,
        candidate,
        {"rel_tolerance": 0.4, "rel_eps": 1.0e-12},
    )
    assert summary.cases == 2
    assert summary.qoi_metrics_count >= 3


def test_linear_surrogate_fit_and_predict() -> None:
    conditions = [
        {"case_id": "c0", "T0": 1000.0, "P0_atm": 1.0, "phi": 1.0, "t_end": 0.1},
        {"case_id": "c1", "T0": 1150.0, "P0_atm": 2.0, "phi": 0.9, "t_end": 0.2},
        {"case_id": "c2", "T0": 1300.0, "P0_atm": 5.0, "phi": 1.1, "t_end": 0.15},
    ]
    qoi = {"species_last": ["CH4", "O2"], "species_max": ["OH"]}
    baseline = run_surrogate_cases({"global_scale": 1.0}, conditions, qoi)

    model = fit_lightweight_surrogate(conditions, baseline, qoi, l2=1.0e-6)
    candidate = run_surrogate_cases(
        {"linear_surrogate": model, "perturb_scale": 0.05, "perturb_seed": 7},
        conditions,
        qoi,
    )
    _, summary = compare_with_baseline(
        baseline,
        candidate,
        {"rel_tolerance": 0.5, "rel_eps": 1.0e-12},
    )
    assert summary.cases == 3
    assert summary.mean_rel_diff is not None


def test_run_surrogate_traces_contract() -> None:
    traces = [
        {
            "case_id": "c0",
            "time": np.asarray([0.0, 0.1, 0.2], dtype=float),
            "X": np.asarray([[0.5, 0.5], [0.4, 0.6], [0.3, 0.7]], dtype=float),
            "wdot": np.asarray([[-0.1, 0.1], [-0.05, 0.05], [0.0, 0.0]], dtype=float),
        }
    ]
    out = run_surrogate_traces({"wdot_scale": 1.0}, traces)
    assert len(out) == 1
    assert out[0]["X"].shape == (3, 2)
    assert out[0]["wdot"].shape == (3, 2)
