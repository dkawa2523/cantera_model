from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

try:
    import cantera as ct
except ImportError:  # pragma: no cover
    ct = None


def _require_cantera() -> None:
    if ct is None:
        raise ImportError("cantera is required for stoichiometry builders")


def build_nu(
    mech_path: str | Path,
    phase: str = "gri30",
    *,
    split_reversible: bool = False,
) -> tuple[csr_matrix, list[str], list[str]]:
    _require_cantera()
    gas = ct.Solution(str(mech_path), phase)
    species_names = list(gas.species_names)
    reactions = gas.reactions()

    react = np.asarray(gas.reactant_stoich_coeffs, dtype=float)
    prod = np.asarray(gas.product_stoich_coeffs, dtype=float)
    nu = prod - react

    if not split_reversible:
        reaction_eqs = [r.equation for r in reactions]
        return csr_matrix(nu), species_names, reaction_eqs

    cols: list[np.ndarray] = []
    reaction_eqs: list[str] = []
    for idx, reaction in enumerate(reactions):
        cols.append(nu[:, idx])
        reaction_eqs.append(f"{reaction.equation} [fwd]")
        if bool(getattr(reaction, "reversible", False)):
            cols.append(-nu[:, idx])
            reaction_eqs.append(f"{reaction.equation} [rev]")

    stacked = np.stack(cols, axis=1) if cols else np.empty((gas.n_species, 0), dtype=float)
    return csr_matrix(stacked), species_names, reaction_eqs


def build_element_matrix(
    mech_path: str | Path,
    species_names: list[str] | None = None,
    *,
    phase: str = "gri30",
) -> tuple[np.ndarray, list[str], list[str]]:
    _require_cantera()
    gas = ct.Solution(str(mech_path), phase)
    all_species = {sp.name: sp for sp in gas.species()}
    ordered_species = species_names or list(gas.species_names)
    elements = list(gas.element_names)

    A = np.zeros((len(elements), len(ordered_species)), dtype=float)
    for j, name in enumerate(ordered_species):
        sp = all_species.get(name)
        if sp is None:
            continue
        comp = sp.composition
        for i, elem in enumerate(elements):
            A[i, j] = float(comp.get(elem, 0.0))
    return A, elements, ordered_species


def element_conservation_residual(A: np.ndarray, nu: np.ndarray | csr_matrix) -> np.ndarray:
    dense_nu = nu.toarray() if hasattr(nu, "toarray") else np.asarray(nu, dtype=float)
    return np.asarray(A, dtype=float) @ dense_nu


def extract_species_meta(
    mech_path: str | Path,
    *,
    phase: str = "gri30",
) -> list[dict[str, object]]:
    _require_cantera()
    gas = ct.Solution(str(mech_path), phase)
    out: list[dict[str, object]] = []
    for sp in gas.species():
        charge = float(getattr(sp, "charge", 0.0) or 0.0)
        out.append(
            {
                "name": sp.name,
                "composition": {k: float(v) for k, v in dict(sp.composition).items()},
                "phase": str(phase),
                "charge": charge,
                "radical": False,
                "role": "",
            }
        )
    return out
