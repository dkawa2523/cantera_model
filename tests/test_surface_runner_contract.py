from pathlib import Path

import pytest

from cantera_model.eval.cantera_runner import load_conditions, run_mechanism


try:
    import cantera  # noqa: F401
except Exception:  # pragma: no cover - environment dependent
    cantera = None


def test_surface_runner_contract() -> None:
    if cantera is None:
        pytest.skip("cantera is required")

    conditions = load_conditions(
        Path("cantera_model/benchmarks/benchmarks/diamond_cvd/conditions.csv"),
        mode="surface_batch",
    )

    rows = run_mechanism(
        "diamond.yaml",
        "unused",
        conditions[:1],
        fuel="CH4:1.0",
        oxidizer="O2:1.0, N2:3.76",
        n_steps=30,
        species_last=[],
        species_max=[],
        mode="surface_batch",
        surface_cfg={"interface_phase": "diamond_100", "gas_phase": "gas"},
        qoi_cfg={
            "selectors": [
                "gas_X:H2:final",
                "surface_theta:c6HH:final",
                "deposition_rate:C(d):mean",
            ]
        },
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["case_id"] == "base"
    assert "gas_X:H2:final" in row
    assert "surface_theta:c6HH:final" in row
    assert "deposition_rate:C(d):mean" in row
