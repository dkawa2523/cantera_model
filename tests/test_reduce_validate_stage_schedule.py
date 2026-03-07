from cantera_model.cli.reduce_validate import _resolve_stage_scheduled_value


def test_resolve_stage_scheduled_value_list_and_dict() -> None:
    assert _resolve_stage_scheduled_value([1, 2, 3], stage_idx=1, stage_name="B", default=0) == 2
    assert _resolve_stage_scheduled_value([1, 2], stage_idx=5, stage_name="C", default=0) == 2
    assert _resolve_stage_scheduled_value({"A": 10, "default": 20}, stage_idx=0, stage_name="A", default=0) == 10
    assert _resolve_stage_scheduled_value({"default": 20}, stage_idx=2, stage_name="C", default=0) == 20
    assert _resolve_stage_scheduled_value(None, stage_idx=0, stage_name="A", default=7) == 7
