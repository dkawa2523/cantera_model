import pytest

from cantera_model.eval.qoi import extract_qoi


def test_extract_surface_qoi_selectors() -> None:
    row_ctx = {
        "gas_species_names": ["A", "B"],
        "surface_species_names": ["S1", "S2"],
        "gas_X": [[0.2, 0.8], [0.3, 0.7]],
        "surface_theta": [[0.6, 0.4], [0.5, 0.5]],
        "deposition_rate": {"D": [1.0, 3.0]},
    }
    out = extract_qoi(
        row_ctx,
        {
            "selectors": [
                "gas_X:A:final",
                "surface_theta:S1:final",
                "deposition_rate:D:mean",
            ]
        },
    )
    assert out["gas_X:A:final"] == pytest.approx(0.3)
    assert out["surface_theta:S1:final"] == pytest.approx(0.5)
    assert out["deposition_rate:D:mean"] == pytest.approx(2.0)
