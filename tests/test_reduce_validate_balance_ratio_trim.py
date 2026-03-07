import numpy as np

from cantera_model.cli.reduce_validate import _trim_keep_by_reaction_species_ratio


def test_trim_keep_by_reaction_species_ratio_drops_low_importance_first() -> None:
    keep = np.array([True, True, True, True, True, True, True, True], dtype=bool)
    # species_after=2, max_ratio=3.0 -> allowed=6
    # low-importance kept reactions (idx=2,4) should be dropped first.
    importance = np.array([9.0, 8.0, 0.1, 7.0, 0.2, 6.0, 5.0, 4.0], dtype=float)
    trimmed, meta = _trim_keep_by_reaction_species_ratio(
        keep,
        species_after=2,
        max_reaction_species_ratio=3.0,
        min_keep_count=1,
        importance=importance,
    )
    assert bool(meta["applied"]) is True
    assert int(meta["selected_before"]) == 8
    assert int(meta["selected_after"]) == 6
    assert int(meta["dropped_reactions"]) == 2
    assert trimmed.tolist() == [True, True, False, True, False, True, True, True]


def test_trim_keep_by_reaction_species_ratio_respects_min_keep_count() -> None:
    keep = np.array([True, True, True, True, True, True], dtype=bool)
    trimmed, meta = _trim_keep_by_reaction_species_ratio(
        keep,
        species_after=1,
        max_reaction_species_ratio=2.0,
        min_keep_count=5,
        importance=np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float),
    )
    assert bool(meta["applied"]) is True
    assert int(np.sum(trimmed)) == 5
    assert float(meta["ratio_after"]) == 5.0

