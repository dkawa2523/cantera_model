import pytest

from cantera_model.eval.qoi import extract_qoi


def test_extract_qoi_integral_selectors() -> None:
    row_ctx = {
        "time": [0.0, 1.0, 2.0],
        "gas_species_names": ["A", "B"],
        "surface_species_names": ["S1", "S2"],
        "gas_X": [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]],
        "surface_theta": [[0.2, 0.8], [0.3, 0.7], [0.1, 0.9]],
        "deposition_rate": {"D": [0.0, 2.0, 4.0]},
    }
    out = extract_qoi(
        row_ctx,
        {
            "selectors": [
                "gas_X:A:integral",
                "surface_theta:S1:integral",
                "deposition_rate:D:integral",
            ]
        },
    )
    assert out["gas_X:A:integral"] == pytest.approx(1.0)
    assert out["surface_theta:S1:integral"] == pytest.approx(0.45)
    assert out["deposition_rate:D:integral"] == pytest.approx(4.0)
