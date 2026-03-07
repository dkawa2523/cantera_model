from cantera_model.cli.reduce_validate import _derive_blockers


def test_derive_blockers_keeps_error_fail_reason() -> None:
    selected = {
        "mandatory_validity_passed": True,
        "error_gate_passed": False,
        "error_fail_reason_primary": "mandatory_tail",
        "hard_ban_violations": 0,
        "physical_gate_passed": True,
        "floor_passed": True,
        "balance_gate_passed": True,
        "cluster_guard_passed": True,
    }
    blockers = _derive_blockers(selected)
    assert blockers["primary_blocker_layer"] == "error"
    assert blockers["error_fail_reason_primary"] == "mandatory_tail"
    assert blockers["validity_fail_reason_primary"] == "none"


def test_derive_blockers_sets_none_error_reason_when_error_passed() -> None:
    selected = {
        "mandatory_validity_passed": True,
        "error_gate_passed": True,
        "error_fail_reason_primary": "mandatory_tail",
        "hard_ban_violations": 0,
        "physical_gate_passed": True,
        "floor_passed": True,
        "balance_gate_passed": True,
        "cluster_guard_passed": True,
    }
    blockers = _derive_blockers(selected)
    assert blockers["primary_blocker_layer"] == "none"
    assert blockers["error_fail_reason_primary"] == "none"
