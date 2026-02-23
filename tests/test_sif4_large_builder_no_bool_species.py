from __future__ import annotations

from pathlib import Path

import pytest

from cantera_model.benchmark_sif4_sin3n4_cvd.tools import build_sif4_large_mech as builder


def test_loader_preserves_no_token_as_species_name(tmp_path: Path) -> None:
    p = tmp_path / "no_token.yaml"
    p.write_text(
        "\n".join(
            [
                "phases:",
                "  - name: gas",
                "    species: [NO]",
                "species:",
                "  - name: NO",
                "    composition: {N: 1, O: 1}",
            ]
        )
        + "\n"
    )
    loaded = builder._load_yaml(p)
    assert loaded["species"][0]["name"] == "NO"
    assert isinstance(loaded["species"][0]["name"], str)


def test_loader_rejects_bool_species_tokens(tmp_path: Path) -> None:
    p = tmp_path / "bad_bool.yaml"
    p.write_text(
        "\n".join(
            [
                "phases:",
                "  - name: gas",
                "    species: [true]",
                "species:",
                "  - name: H2",
                "    composition: {H: 2}",
            ]
        )
        + "\n"
    )
    with pytest.raises(ValueError):
        _ = builder._load_yaml(p)
