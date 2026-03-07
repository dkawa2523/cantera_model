from types import SimpleNamespace

import numpy as np

from cantera_model.cli.reduce_validate import _qoi_from_trace_case


def test_qoi_from_trace_case_integral_metrics() -> None:
    case = SimpleNamespace(
        species_names=["H2", "C(d)", "CH4"],
        time=np.asarray([0.0, 1.0, 2.0], dtype=float),
        temperature=np.asarray([900.0, 950.0, 1000.0], dtype=float),
        X=np.asarray(
            [
                [0.1, 0.0, 0.8],
                [0.2, 0.2, 0.6],
                [0.3, 0.4, 0.4],
            ],
            dtype=float,
        ),
    )
    qoi = _qoi_from_trace_case(
        case,
        {
            "species_last": ["H2"],
            "species_max": ["CH4"],
            "species_integral": ["H2"],
            "deposition_integral": ["C(d)"],
        },
    )
    assert "X_last:H2" in qoi
    assert "X_max:CH4" in qoi
    assert "X_int:H2" in qoi
    assert "dep_int:C(d)" in qoi
    assert float(qoi["X_int:H2"]) > 0.0
    assert float(qoi["dep_int:C(d)"]) > 0.0
