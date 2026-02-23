import numpy as np

from cantera_model.types import CaseTrace


def test_case_trace_shapes_ok() -> None:
    trace = CaseTrace(
        case_id="c0",
        time=np.array([0.0, 0.1]),
        temperature=np.array([1000.0, 1100.0]),
        pressure=np.array([101325.0, 101325.0]),
        X=np.array([[0.9, 0.1], [0.8, 0.2]]),
        wdot=np.array([[0.0, 0.0], [0.1, -0.1]]),
        rop=np.array([[1.0], [0.5]]),
        species_names=["A", "B"],
        reaction_eqs=["A=>B"],
    )
    assert trace.X.shape == (2, 2)


def test_case_trace_shape_error() -> None:
    try:
        CaseTrace(
            case_id="c0",
            time=np.array([0.0, 0.1]),
            temperature=np.array([1000.0]),
            pressure=np.array([101325.0, 101325.0]),
            X=np.array([[0.9, 0.1], [0.8, 0.2]]),
            wdot=np.array([[0.0, 0.0], [0.1, -0.1]]),
            rop=np.array([[1.0], [0.5]]),
            species_names=["A", "B"],
            reaction_eqs=["A=>B"],
        )
    except ValueError as exc:
        assert "temperature" in str(exc)
    else:
        raise AssertionError("CaseTrace should validate shapes")
