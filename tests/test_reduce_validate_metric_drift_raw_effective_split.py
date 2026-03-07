from cantera_model.cli.reduce_validate import _effective_metric_drift


def test_effective_metric_drift_keeps_raw_and_caps_effective() -> None:
    raw = 1.92
    effective = _effective_metric_drift(raw, cap=1.30)
    assert abs(raw - 1.92) < 1.0e-12
    assert abs(effective - 1.30) < 1.0e-12


def test_effective_metric_drift_defaults_when_cap_invalid() -> None:
    effective = _effective_metric_drift(1.75, cap=0.5)
    assert abs(effective - 1.30) < 1.0e-12
