import pytest

from cantera_model.cli.reduce_validate import _validate_tiered_error_aggregation_config


def test_tiered_error_aggregation_requires_explicit_threshold_keys() -> None:
    with pytest.raises(ValueError, match="require_explicit_thresholds"):
        _validate_tiered_error_aggregation_config({"mode": "tiered"}, require_explicit=True)


def test_legacy_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="mode must be 'tiered'"):
        _validate_tiered_error_aggregation_config({"mode": "legacy_all_metric"})
