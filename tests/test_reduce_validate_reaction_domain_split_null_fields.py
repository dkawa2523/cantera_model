from pathlib import Path


def test_reduce_validate_no_negative_one_reaction_split_sentinel() -> None:
    src = Path("cantera_model/cli/reduce_validate.py").read_text()
    assert "gas_reactions_after = -1" not in src
    assert "surface_reactions_after = -1" not in src
