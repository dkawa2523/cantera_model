from pathlib import Path

import pytest

from cantera_model.eval.conditions import load_conditions


def test_pressure_atm_to_pa(tmp_path: Path) -> None:
    csv_path = tmp_path / "atm.csv"
    csv_path.write_text(
        "case_id,T_K,P_atm,composition,t_end_s,n_steps,area\n"
        "c0,1200,0.5,\"H2:1.0\",0.1,10,1.0\n"
    )
    rows = load_conditions(csv_path, mode="surface_batch")
    assert rows[0]["P_Pa"] == pytest.approx(0.5 * 101325.0)


def test_pressure_torr_to_pa(tmp_path: Path) -> None:
    csv_path = tmp_path / "torr.csv"
    csv_path.write_text(
        "case_id,T_K,P_Torr,composition,t_end_s,n_steps,area\n"
        "c0,1200,2.0,\"H2:1.0\",0.1,10,1.0\n"
    )
    rows = load_conditions(csv_path, mode="surface_batch")
    assert rows[0]["P_Pa"] == pytest.approx(2.0 * 133.32236842105263)
