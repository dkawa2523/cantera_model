import numpy as np

from cantera_model.reduction.prune_gate import train_prune_gate


def test_prune_lambda_controls_sparsity() -> None:
    rng = np.random.default_rng(0)
    nu = rng.normal(size=(5, 12))
    rop = np.abs(rng.normal(size=(80, 12)))
    wdot = rop @ nu.T + 0.01 * rng.normal(size=(80, 5))

    keep_lo = train_prune_gate(nu, rop, wdot, lambda_l0=1.0e-4, seed=0)
    keep_hi = train_prune_gate(nu, rop, wdot, lambda_l0=5.0e-2, seed=0)

    assert int(keep_hi.sum()) <= int(keep_lo.sum())
    assert keep_lo.dtype == bool


def test_prune_min_keep_count_is_enforced() -> None:
    rng = np.random.default_rng(1)
    nu = rng.normal(size=(4, 10))
    rop = np.abs(rng.normal(size=(40, 10)))
    wdot = rop @ nu.T

    keep, details = train_prune_gate(
        nu,
        rop,
        wdot,
        lambda_l0=1.0,
        threshold=0.99,
        min_keep_count=3,
        seed=1,
        return_details=True,
    )
    assert int(keep.sum()) >= 3
    assert details["min_keep_count"] == 3
    assert details["status"] in {"ok", "forced_min_keep", "fallback_non_finite"}


def test_prune_target_keep_ratio_exact() -> None:
    rng = np.random.default_rng(2)
    nu = rng.normal(size=(5, 20))
    rop = np.abs(rng.normal(size=(60, 20)))
    wdot = rop @ nu.T

    keep, details = train_prune_gate(
        nu,
        rop,
        wdot,
        lambda_l0=0.01,
        target_keep_ratio=0.35,
        enforce_target_exact=True,
        seed=2,
        return_details=True,
    )
    assert int(keep.sum()) == int(round(20 * 0.35))
    assert details["status"] in {"ok_targeted", "fallback_non_finite_targeted", "fallback_all_pruned_targeted"}
