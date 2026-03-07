from cantera_model.cli.reduce_validate import _derive_blockers


def test_derive_blockers_priority_and_reason() -> None:
    selected = {
        "mandatory_validity_passed": False,
        "error_gate_passed": False,
        "error_fail_reason_primary": "mandatory_tail",
        "hard_ban_violations": 1,
        "physical_gate_passed": False,
        "floor_passed": False,
        "balance_gate_passed": False,
        "cluster_guard_passed": False,
    }
    blockers = _derive_blockers(selected)
    assert blockers["primary_blocker_layer"] == "validity"
    assert blockers["secondary_blockers"] == ["error", "structure"]
    assert blockers["validity_fail_reason_primary"] == "mandatory_threshold_not_met"
    assert blockers["error_fail_reason_primary"] == "mandatory_tail"
    assert blockers["structure_fail_reasons"] == [
        "hard_ban",
        "physical_gate",
        "physics_floor",
        "balance_gate",
        "cluster_guard",
    ]


def test_derive_blockers_none_when_all_passed() -> None:
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
    assert blockers["secondary_blockers"] == []
    assert blockers["validity_fail_reason_primary"] == "none"
    assert blockers["error_fail_reason_primary"] == "none"
