import pytest

from cantera_model.eval.diagnostic_schema import validate_summary_schema


def _valid_summary() -> dict:
    return {
        "gate_passed": True,
        "primary_blocker_layer": "none",
        "selected_metrics": {},
        "gate_evidence": {
            "selected_stage_evidence": {
                "mandatory_validity_passed": True,
                "coverage_gate_passed": True,
                "error_gate_passed": True,
                "mandatory_error_passed": True,
                "optional_error_passed": True,
                "error_fail_reason_primary": "none",
                "validity_fail_reason_primary": "none",
                "primary_blocker_layer": "none",
            }
        },
    }


def test_validate_required_diagnostic_schema_passes_when_keys_present() -> None:
    assert validate_summary_schema(_valid_summary(), strict=True) is True


def test_validate_required_diagnostic_schema_raises_when_strict() -> None:
    summary = _valid_summary()
    del summary["gate_evidence"]["selected_stage_evidence"]["optional_error_passed"]
    with pytest.raises(ValueError, match="diagnostic schema missing keys"):
        validate_summary_schema(summary, strict=True)


def test_validate_required_diagnostic_schema_returns_false_when_not_strict() -> None:
    summary = _valid_summary()
    del summary["gate_passed"]
    assert validate_summary_schema(summary, strict=False) is False
