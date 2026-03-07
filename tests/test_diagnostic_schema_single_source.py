from cantera_model.cli import reduce_validate
from cantera_model.cli import summarize_reduction_eval
from cantera_model.eval import diagnostic_schema


def test_diagnostic_schema_is_single_source() -> None:
    assert hasattr(diagnostic_schema, "REQUIRED_TOP_KEYS")
    assert hasattr(diagnostic_schema, "REQUIRED_STAGE_KEYS")
    assert hasattr(diagnostic_schema, "project_entry")

    # Legacy duplicated schema definitions are removed from CLI modules.
    assert not hasattr(summarize_reduction_eval, "_REQUIRED_DIAGNOSTIC_TOP_KEYS")
    assert not hasattr(summarize_reduction_eval, "_REQUIRED_DIAGNOSTIC_STAGE_KEYS")
    assert not hasattr(reduce_validate, "_validate_required_diagnostic_schema")
