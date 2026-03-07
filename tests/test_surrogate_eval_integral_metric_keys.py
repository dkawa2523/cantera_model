from cantera_model.eval.surrogate_eval import _default_metric_keys


def test_default_metric_keys_include_integral_metrics() -> None:
    keys = _default_metric_keys(
        {
            "species_last": ["H2"],
            "species_max": ["OH"],
            "species_integral": ["H2", "CH4"],
            "deposition_integral": ["C(d)"],
        }
    )

    assert "X_last:H2" in keys
    assert "X_max:OH" in keys
    assert "X_int:H2" in keys
    assert "X_int:CH4" in keys
    assert "dep_int:C(d)" in keys
    assert len(keys) == len(set(keys))
