from pathlib import Path

from cantera_model.eval.conditions import load_conditions


def test_surface_conditions_schema_diamond() -> None:
    rows = load_conditions(
        Path("cantera_model/benchmarks/benchmarks/diamond_cvd/conditions.csv"),
        mode="surface_batch",
    )
    assert rows
    first = rows[0]
    assert first["case_id"] == "base"
    assert first["T_K"] > 0.0
    assert first["P_Pa"] > 0.0
    assert first["composition"]
    assert first["n_steps"] >= 2
    assert first["area"] > 0.0


def test_surface_conditions_schema_sif4() -> None:
    rows = load_conditions(
        Path("cantera_model/benchmarks/benchmarks/sif4_sin3n4_cvd/conditions.csv"),
        mode="surface_batch",
    )
    assert rows
    first = rows[0]
    assert first["case_id"] == "base"
    assert first["P_Pa"] > 0.0
    assert first["composition"]
