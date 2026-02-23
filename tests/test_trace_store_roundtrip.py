import numpy as np

from cantera_model.io.trace_store import load_case_bundle, save_case_bundle
from cantera_model.types import CaseBundle, CaseTrace


def test_trace_roundtrip(tmp_path) -> None:
    case = CaseTrace(
        case_id="c0",
        time=np.array([0.0, 0.1]),
        temperature=np.array([1000.0, 1100.0]),
        pressure=np.array([101325.0, 101325.0]),
        X=np.array([[0.9, 0.1], [0.8, 0.2]]),
        wdot=np.array([[0.0, 0.0], [0.1, -0.1]]),
        rop=np.array([[1.0], [0.5]]),
        species_names=["A", "B"],
        reaction_eqs=["A=>B"],
        meta={"note": "ok"},
    )
    bundle = CaseBundle(
        mechanism_path="m.yaml",
        phase="gas",
        species_names=["A", "B"],
        reaction_eqs=["A=>B"],
        cases=[case],
        meta={"v": 1},
    )
    out = save_case_bundle(tmp_path / "trace.h5", bundle)
    loaded = load_case_bundle(out)

    assert loaded.phase == "gas"
    assert loaded.cases[0].case_id == "c0"
    assert loaded.cases[0].meta["note"] == "ok"
    assert np.allclose(loaded.cases[0].X, case.X)
