"""tools.build_sif4_large_mech

Build a *large* SiF4/NH3 -> Si3N4 deposition benchmark mechanism for Cantera.

Goals
-----
- Keep **physically sourced** kinetics (Arrhenius / sticking) by reusing published mechanisms.
- Increase mechanism size so reduction / network tooling has a harder benchmark.

How
---
1) Start from Cantera's example mechanism: ``SiF4_NH3_mec.yaml`` (gas + interface ``SI3N4``)
2) Merge a large gas mechanism (default: ``gri30.yaml``) into the gas phase
   - On **species name collisions**, keep the *base* species definition
3) Multi-site surface expansion:
   - Clone surface species and surface reactions across multiple "site families" (prefixes)
   - Example: ``X(s)`` -> ``t_X(s)``, ``s_X(s)``, ``k_X(s)``

The resulting file can be used directly with ct.Interface(..., "SI3N4")

Notes
-----
- This utility does *not* require Cantera at build time (pure YAML transform).
- Verification (load + counts) should be done with ``tools/check_mechanism_size.py``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import copy
import hashlib
import json
import re

import yaml


@dataclass(frozen=True)
class BuildResult:
    out_path: Path
    n_species_base: int
    n_species_added: int
    n_reactions_base: int
    n_reactions_added: int
    surface_species_base: int
    surface_reactions_base: int
    site_families: Tuple[str, ...]


def _sha1_of_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=_CanteraYamlLoader)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    _validate_species_tokens(data, source=str(path))
    return data


def _dump_yaml(data: Dict, path: Path) -> None:
    # Keep ordering as-is (PyYAML 6 uses dict insertion order)
    text = yaml.safe_dump(
        data,
        sort_keys=False,
        width=120,
        allow_unicode=True,
    )
    path.write_text(text, encoding="utf-8")


class _CanteraYamlLoader(yaml.SafeLoader):
    """YAML loader that avoids YAML 1.1 NO/ON/OFF implicit-bool pitfalls."""


_CanteraYamlLoader.yaml_implicit_resolvers = copy.deepcopy(yaml.SafeLoader.yaml_implicit_resolvers)
for key, resolvers in list(_CanteraYamlLoader.yaml_implicit_resolvers.items()):
    _CanteraYamlLoader.yaml_implicit_resolvers[key] = [
        (tag, regex)
        for tag, regex in resolvers
        if tag != "tag:yaml.org,2002:bool"
    ]
_CanteraYamlLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
    list("tTfF"),
)


def _validate_species_tokens(data: Dict, *, source: str) -> None:
    species = data.get("species", [])
    if not isinstance(species, list):
        raise ValueError(f"{source}: top-level species must be a list")

    for idx, sp in enumerate(species):
        if not isinstance(sp, dict):
            continue
        name = sp.get("name")
        if isinstance(name, bool):
            raise ValueError(f"{source}: species[{idx}].name must be string, got bool")
        if name is None or not isinstance(name, str):
            raise ValueError(f"{source}: species[{idx}].name must be string")
        if not name.strip():
            raise ValueError(f"{source}: species[{idx}].name cannot be empty")

    phases = data.get("phases", [])
    if not isinstance(phases, list):
        raise ValueError(f"{source}: top-level phases must be a list")
    for p_idx, phase in enumerate(phases):
        if not isinstance(phase, dict):
            continue
        phase_species = phase.get("species")
        if phase_species is None:
            continue
        if not isinstance(phase_species, list):
            raise ValueError(f"{source}: phases[{p_idx}].species must be list")
        for s_idx, item in enumerate(phase_species):
            if isinstance(item, bool):
                raise ValueError(f"{source}: phases[{p_idx}].species[{s_idx}] must not be bool")
            if isinstance(item, str):
                continue
            if isinstance(item, dict):
                for k, v in item.items():
                    if isinstance(k, bool) or isinstance(v, bool):
                        raise ValueError(
                            f"{source}: phases[{p_idx}].species[{s_idx}] mapping must not contain bool"
                        )
                continue
            raise ValueError(
                f"{source}: phases[{p_idx}].species[{s_idx}] must be string or mapping, got {type(item).__name__}"
            )


def _species_list(data: Dict) -> List[Dict]:
    sp = data.get("species", [])
    if sp is None:
        return []
    if not isinstance(sp, list):
        raise ValueError("Top-level 'species' must be a list")
    return sp


def _reactions_list(data: Dict) -> List[Dict]:
    rx = data.get("reactions", [])
    if rx is None:
        return []
    if not isinstance(rx, list):
        raise ValueError("Top-level 'reactions' must be a list")
    return rx


def _reaction_refs_for_phase(phase: Dict) -> List[str]:
    refs = phase.get("reactions")
    if refs is None:
        return []
    if isinstance(refs, str):
        refs = [refs]
    if not isinstance(refs, list) or not all(isinstance(x, str) for x in refs):
        raise ValueError(f"Phase '{phase.get('name', '<unknown>')}' has invalid 'reactions' field")
    return [x.strip() for x in refs if x and x.strip()]


def _resolve_phase_reaction_sections(data: Dict, phase: Dict) -> List[Tuple[str, List[Dict]]]:
    refs = _reaction_refs_for_phase(phase)
    sections: List[Tuple[str, List[Dict]]] = []
    seen: set[str] = set()

    def _add_named_section(section_name: str) -> None:
        if section_name in seen:
            return
        raw = data.get(section_name)
        if raw is None:
            data[section_name] = []
            raw = data[section_name]
        if not isinstance(raw, list):
            raise ValueError(f"Reaction section '{section_name}' must be a list")
        sections.append((section_name, raw))
        seen.add(section_name)

    if not refs:
        if "reactions" in data:
            _add_named_section("reactions")
        return sections

    for ref in refs:
        low = ref.lower()
        if low == "none":
            continue
        if low in {"all", "declared-species"}:
            if "reactions" in data:
                _add_named_section("reactions")
            continue
        _add_named_section(ref)

    return sections


def _collect_phase_reactions(data: Dict, phase: Dict) -> List[Dict]:
    reactions: List[Dict] = []
    for _, section in _resolve_phase_reaction_sections(data, phase):
        reactions.extend([r for r in section if isinstance(r, dict)])
    return reactions


def _index_by_name(items: List[Dict]) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for d in items:
        if not isinstance(d, dict) or "name" not in d:
            continue
        out[str(d["name"])] = d
    return out


def _find_phase(data: Dict, name: str) -> Dict:
    phases = data.get("phases", [])
    if not isinstance(phases, list):
        raise ValueError("Top-level 'phases' must be a list")
    for ph in phases:
        if isinstance(ph, dict) and ph.get("name") == name:
            return ph
    raise KeyError(f"Phase '{name}' not found")


def _find_gas_phase(data: Dict) -> Dict:
    phases = data.get("phases", [])
    if not isinstance(phases, list):
        raise ValueError("Top-level 'phases' must be a list")
    # Prefer a phase explicitly named 'gas'
    for ph in phases:
        if isinstance(ph, dict) and ph.get("name") == "gas":
            return ph
    # Else, any ideal-gas phase
    for ph in phases:
        if isinstance(ph, dict) and ph.get("thermo") == "ideal-gas":
            return ph
    raise KeyError("Gas phase not found")


_TERM_RE = re.compile(r"^\s*(?:(?P<nu>[0-9]+(?:\.[0-9]+)?)\s+)?(?P<sp>.+?)\s*$")


def _parse_side(side: str) -> List[Tuple[Optional[str], str]]:
    """Parse one side of a reaction equation into [(coeff, species), ...]."""
    terms: List[Tuple[Optional[str], str]] = []
    for raw in side.split("+"):
        raw = raw.strip()
        if not raw:
            continue
        m = _TERM_RE.match(raw)
        if not m:
            terms.append((None, raw))
            continue
        nu = m.group("nu")
        sp = m.group("sp").strip()
        terms.append((nu, sp))
    return terms


def _format_side(terms: List[Tuple[Optional[str], str]]) -> str:
    out_terms = []
    for nu, sp in terms:
        if nu is None:
            out_terms.append(sp)
        else:
            out_terms.append(f"{nu} {sp}")
    return " + ".join(out_terms)


def _rewrite_equation(eqn: str, sp_map: Dict[str, str]) -> str:
    # Find arrow
    if "<=>" in eqn:
        arrow = "<=>"
    elif "=>" in eqn:
        arrow = "=>"
    elif "=" in eqn:
        arrow = "="
    else:
        return eqn

    lhs, rhs = eqn.split(arrow)
    lhs_terms = _parse_side(lhs)
    rhs_terms = _parse_side(rhs)

    def repl(terms: List[Tuple[Optional[str], str]]) -> List[Tuple[Optional[str], str]]:
        new_terms: List[Tuple[Optional[str], str]] = []
        for nu, sp in terms:
            new_terms.append((nu, sp_map.get(sp, sp)))
        return new_terms

    lhs2 = _format_side(repl(lhs_terms))
    rhs2 = _format_side(repl(rhs_terms))
    return f"{lhs2} {arrow} {rhs2}"


def _reaction_mentions_any(eqn: str, species: Iterable[str]) -> bool:
    # Use parse-based tokenization to avoid substring false positives
    if "<=>" in eqn:
        arrow = "<=>"
    elif "=>" in eqn:
        arrow = "=>"
    elif "=" in eqn:
        arrow = "="
    else:
        return False

    lhs, rhs = eqn.split(arrow)
    sp_set = set(species)
    for _, sp in _parse_side(lhs) + _parse_side(rhs):
        if sp in sp_set:
            return True
    return False


def _equation_signature(
    eqn: str,
    *,
    cancel_common_terms: bool = True,
) -> Tuple[bool, Tuple[Tuple[str, float], ...], Tuple[Tuple[str, float], ...]] | None:
    if not isinstance(eqn, str) or not eqn.strip():
        return None
    if "<=>" in eqn:
        arrow = "<=>"
        reversible = True
    elif "=>" in eqn:
        arrow = "=>"
        reversible = False
    elif "=" in eqn:
        arrow = "="
        reversible = True
    else:
        return None

    lhs_raw, rhs_raw = eqn.split(arrow, 1)

    def _to_map(side: str) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for nu_txt, sp in _parse_side(side):
            coeff = float(nu_txt) if nu_txt is not None else 1.0
            out[sp] = out.get(sp, 0.0) + coeff
        return out

    lhs = _to_map(lhs_raw)
    rhs = _to_map(rhs_raw)

    if cancel_common_terms:
        # Normalize third-body / spectator terms (same species appears on both sides).
        for sp in sorted(set(lhs) & set(rhs)):
            c = min(lhs[sp], rhs[sp])
            if c <= 0.0:
                continue
            lhs[sp] -= c
            rhs[sp] -= c
            if lhs[sp] <= 1.0e-12:
                lhs.pop(sp, None)
            if rhs[sp] <= 1.0e-12:
                rhs.pop(sp, None)

    left = tuple(sorted((sp, round(coeff, 12)) for sp, coeff in lhs.items() if coeff > 1.0e-12))
    right = tuple(sorted((sp, round(coeff, 12)) for sp, coeff in rhs.items() if coeff > 1.0e-12))
    if reversible and right < left:
        left, right = right, left
    return (reversible, left, right)


def _map_keys(d: Dict, sp_map: Dict[str, str]) -> Dict:
    """Return a shallow copy of dict d with keys mapped if they match a species."""
    out: Dict = {}
    for k, v in d.items():
        k2 = sp_map.get(k, k)
        out[k2] = v
    return out


def _clone_surface(
    data: Dict,
    interface_name: str,
    site_families: Sequence[str],
) -> Tuple[int, int]:
    """Clone surface species + surface reactions across site families.

    Returns
    -------
    (n_surface_species_base, n_surface_reactions_base)
    """

    iface = _find_phase(data, interface_name)

    base_species_field = iface.get("species")
    if not isinstance(base_species_field, list) or not all(isinstance(x, str) for x in base_species_field):
        raise ValueError(
            f"Interface phase '{interface_name}' must list its species explicitly as a list of strings."
        )

    surf_species: List[str] = list(base_species_field)
    if not surf_species:
        raise ValueError(f"Interface phase '{interface_name}' has empty species list")

    all_species = _species_list(data)
    sp_by_name = _index_by_name(all_species)

    surface_reaction_sections = _resolve_phase_reaction_sections(data, iface)
    if not surface_reaction_sections and "reactions" in data:
        # Backward-compatible fallback for mechanisms that rely on top-level `reactions`.
        surface_reaction_sections = [("reactions", _reactions_list(data))]
    if not surface_reaction_sections:
        raise ValueError(f"No reaction sections found for interface phase '{interface_name}'")

    # Build per-family species map
    fam_maps: Dict[str, Dict[str, str]] = {}
    for fam in site_families:
        fam_maps[fam] = {sp: f"{fam}_{sp}" for sp in surf_species}

    # Clone species definitions
    new_species_defs: List[Dict] = []
    for fam in site_families:
        sp_map = fam_maps[fam]
        for sp in surf_species:
            if sp not in sp_by_name:
                raise KeyError(
                    f"Surface species '{sp}' not found in top-level species definitions."
                )
            new_def = copy.deepcopy(sp_by_name[sp])
            new_def["name"] = sp_map[sp]
            new_species_defs.append(new_def)

    # Remove original surface species defs from global list (optional but safer)
    all_species2 = [s for s in all_species if not (isinstance(s, dict) and s.get("name") in set(surf_species))]
    all_species2.extend(new_species_defs)
    data["species"] = all_species2

    n_surface_reactions_base = 0
    for section_name, section_rxns in surface_reaction_sections:
        surf_rxns: List[Dict] = []
        non_surf_rxns: List[Dict] = []
        for r in section_rxns:
            eqn = r.get("equation") if isinstance(r, dict) else None
            if isinstance(eqn, str) and _reaction_mentions_any(eqn, surf_species):
                surf_rxns.append(r)
            else:
                non_surf_rxns.append(r)

        # In named interface blocks (e.g. SI3N4-reactions), treat all entries as surface
        # even if token parsing missed one.
        if not surf_rxns and section_name != "reactions":
            surf_rxns = [r for r in section_rxns if isinstance(r, dict)]
            non_surf_rxns = []

        n_surface_reactions_base += len(surf_rxns)
        new_surf_rxns: List[Dict] = []
        for fam in site_families:
            sp_map = fam_maps[fam]
            for r in surf_rxns:
                nr = copy.deepcopy(r)
                if "equation" in nr and isinstance(nr["equation"], str):
                    nr["equation"] = _rewrite_equation(nr["equation"], sp_map)

                # Map any species-keyed dicts commonly used by Cantera.
                if "orders" in nr and isinstance(nr["orders"], dict):
                    nr["orders"] = _map_keys(nr["orders"], sp_map)
                if "efficiencies" in nr and isinstance(nr["efficiencies"], dict):
                    nr["efficiencies"] = _map_keys(nr["efficiencies"], sp_map)

                # Coverage dependencies can appear under several keys; do a best-effort mapping.
                if "coverage-dependencies" in nr and isinstance(nr["coverage-dependencies"], dict):
                    nr["coverage-dependencies"] = _map_keys(nr["coverage-dependencies"], sp_map)
                if "rate-constant" in nr and isinstance(nr["rate-constant"], dict):
                    rc = nr["rate-constant"]
                    if "coverage-dependencies" in rc and isinstance(rc["coverage-dependencies"], dict):
                        rc["coverage-dependencies"] = _map_keys(rc["coverage-dependencies"], sp_map)

                new_surf_rxns.append(nr)

        data[section_name] = non_surf_rxns + new_surf_rxns

    # Update interface species list to all cloned species
    new_surf_species_list: List[str] = []
    for fam in site_families:
        new_surf_species_list.extend([fam_maps[fam][sp] for sp in surf_species])
    iface["species"] = new_surf_species_list

    # Update initial coverages (if present)
    state = iface.get("state")
    if not isinstance(state, dict):
        state = {}
        iface["state"] = state

    base_cov = state.get("coverages")
    new_cov: Dict[str, float] = {}

    if isinstance(base_cov, dict) and base_cov:
        # Spread existing coverage across families
        n_f = float(len(site_families))
        for sp, val in base_cov.items():
            if sp in surf_species:
                for fam in site_families:
                    new_cov[fam_maps[fam][sp]] = float(val) / n_f
            else:
                new_cov[str(sp)] = float(val)
    else:
        # Fallback: assign all coverage to an "empty site" if we can detect one.
        empty_candidates: List[str] = []
        for sp in surf_species:
            d = sp_by_name.get(sp)
            if not isinstance(d, dict):
                continue
            comp = d.get("composition")
            if isinstance(comp, dict) and len(comp) == 0:
                empty_candidates.append(sp)

        empty = empty_candidates[0] if empty_candidates else surf_species[0]
        n_f = float(len(site_families))
        for fam in site_families:
            new_cov[fam_maps[fam][empty]] = 1.0 / n_f

    state["coverages"] = new_cov

    return (len(surf_species), n_surface_reactions_base)


def _merge_gas_mechanism(
    base: Dict,
    extra: Dict,
) -> Tuple[int, int, int, int]:
    """Merge species + reactions from extra into base (in-place).

    Returns:
      (n_species_base, n_species_added, n_reactions_base, n_reactions_added)
    """

    base_species = _species_list(base)
    extra_species = _species_list(extra)

    sp_base_by = _index_by_name(base_species)

    n_sp_base = len([s for s in base_species if isinstance(s, dict) and "name" in s])
    base_gas = _find_gas_phase(base)
    base_gas_reaction_sections = _resolve_phase_reaction_sections(base, base_gas)
    n_rx_base = sum(len(section) for _, section in base_gas_reaction_sections)

    extra_gas = _find_gas_phase(extra)
    extra_rxns = _collect_phase_reactions(extra, extra_gas)
    if not extra_rxns and "reactions" in extra:
        extra_rxns = [r for r in _reactions_list(extra) if isinstance(r, dict)]

    # Add only non-colliding species
    add_sp: List[Dict] = []
    collisions: List[str] = []
    for s in extra_species:
        if not isinstance(s, dict) or "name" not in s:
            continue
        name = str(s["name"])
        if name in sp_base_by:
            collisions.append(name)
            continue
        add_sp.append(s)

    base_species.extend(copy.deepcopy(add_sp))
    base["species"] = base_species

    base_gas_reactions = _collect_phase_reactions(base, base_gas)
    existing_signatures: set[Tuple[bool, Tuple[Tuple[str, float], ...], Tuple[Tuple[str, float], ...]]] = set()
    for r in base_gas_reactions:
        sig = _equation_signature(str(r.get("equation", "")))
        if sig is not None:
            existing_signatures.add(sig)

    existing_strict_signatures: set[Tuple[bool, Tuple[Tuple[str, float], ...], Tuple[Tuple[str, float], ...]]] = set()
    for r in base_gas_reactions:
        strict_sig = _equation_signature(str(r.get("equation", "")), cancel_common_terms=False)
        if strict_sig is not None:
            existing_strict_signatures.add(strict_sig)

    extra_sig_counts: Dict[Tuple[bool, Tuple[Tuple[str, float], ...], Tuple[Tuple[str, float], ...]], int] = {}
    for r in extra_rxns:
        if not isinstance(r, dict):
            continue
        sig = _equation_signature(str(r.get("equation", "")))
        if sig is None:
            continue
        extra_sig_counts[sig] = extra_sig_counts.get(sig, 0) + 1

    add_rxns: List[Dict] = []
    n_rx_duplicates_skipped = 0
    kept_extra_strict_signatures: set[Tuple[bool, Tuple[Tuple[str, float], ...], Tuple[Tuple[str, float], ...]]] = set()
    for r in extra_rxns:
        if not isinstance(r, dict):
            continue
        sig = _equation_signature(str(r.get("equation", "")))
        strict_sig = _equation_signature(str(r.get("equation", "")), cancel_common_terms=False)
        unique_in_extra = sig is not None and extra_sig_counts.get(sig, 0) <= 1
        if unique_in_extra and sig in existing_signatures:
            n_rx_duplicates_skipped += 1
            continue
        if strict_sig is not None and strict_sig in existing_strict_signatures:
            n_rx_duplicates_skipped += 1
            continue
        if strict_sig is not None and strict_sig in kept_extra_strict_signatures:
            n_rx_duplicates_skipped += 1
            continue
        copied = copy.deepcopy(r)
        copied.pop("duplicate", None)
        add_rxns.append(copied)
        if unique_in_extra and sig is not None:
            existing_signatures.add(sig)
        if strict_sig is not None:
            kept_extra_strict_signatures.add(strict_sig)

    # Add extra gas reactions to the gas phase reaction section.
    if base_gas_reaction_sections:
        target_section_name, target_rxns = base_gas_reaction_sections[0]
        target_rxns.extend(add_rxns)
        base[target_section_name] = target_rxns
    else:
        base_rxns = _reactions_list(base)
        base_rxns.extend(add_rxns)
        base["reactions"] = base_rxns

    # Update gas phase species list and elements
    gas = base_gas

    # Extend explicit species list if present
    gas_species_field = gas.get("species")
    if isinstance(gas_species_field, list) and all(isinstance(x, str) for x in gas_species_field):
        existing = set(gas_species_field)
        for s in add_sp:
            gas_species_field.append(str(s["name"]))
        # Keep as list
        gas["species"] = gas_species_field
    elif isinstance(gas_species_field, str):
        # 'all' or similar -> nothing to do
        pass
    else:
        # Unknown structure; don't try to edit.
        pass

    # Elements: union existing with all elements appearing in added species
    el_field = gas.get("elements")
    if isinstance(el_field, list):
        el_set = set(str(e) for e in el_field)
        for s in add_sp:
            comp = s.get("composition")
            if isinstance(comp, dict):
                el_set.update(str(k) for k in comp.keys())
        gas["elements"] = sorted(el_set)
    gas.setdefault("explicit-third-body-duplicates", "modify-efficiency")

    # Store collision info in description for transparency
    if collisions:
        desc = base.get("description", "")
        if not isinstance(desc, str):
            desc = str(desc)
        desc += "\n\n[merge note] Species collisions while merging extra gas mechanism were skipped (kept base definition):\n"
        desc += ", ".join(sorted(set(collisions)))[:2000]
        base["description"] = desc
    if n_rx_duplicates_skipped:
        desc = base.get("description", "")
        if not isinstance(desc, str):
            desc = str(desc)
        desc += (
            "\n\n[merge note] Reactions skipped as duplicates while merging extra gas mechanism: "
            f"{n_rx_duplicates_skipped}\n"
        )
        base["description"] = desc

    n_sp_added = len(add_sp)
    n_rx_added = len(add_rxns)

    return (n_sp_base, n_sp_added, n_rx_base, n_rx_added)


def build_large_sif4_mech(
    *,
    base_mech_path: Path,
    extra_gas_mech_path: Path,
    out_dir: Path,
    interface_name: str = "SI3N4",
    site_families: Sequence[str] = ("t", "s", "k"),
    out_name: str | None = None,
    force_rebuild: bool = False,
) -> Path:
    """Build and cache the augmented mechanism.

    Parameters
    ----------
    base_mech_path:
      Path to SiF4/NH3 base mechanism YAML.
    extra_gas_mech_path:
      Path to the large gas mechanism YAML to merge in.
    out_dir:
      Output directory (cache). File name is derived from inputs.
    interface_name:
      Name of the surface/interface phase to clone.
    site_families:
      Sequence of prefixes; each creates a cloned copy of surface species/reactions.

    Returns
    -------
    Path to the generated YAML.
    """

    base_mech_path = Path(base_mech_path)
    extra_gas_mech_path = Path(extra_gas_mech_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_hash = _sha1_of_file(base_mech_path)[:8]
    extra_hash = _sha1_of_file(extra_gas_mech_path)[:8]
    fam_tag = f"multisite{len(site_families)}"
    default_name = f"SiF4_NH3_mec_large__{extra_gas_mech_path.stem}__{fam_tag}__{base_hash}_{extra_hash}.yaml"
    out_path = out_dir / str(out_name or default_name)

    # Simple cache: if exists, trust it.
    if out_path.exists() and not force_rebuild:
        return out_path

    base = _load_yaml(base_mech_path)
    extra = _load_yaml(extra_gas_mech_path)

    # Merge extra gas mechanism
    _merge_gas_mechanism(base, extra)

    # Clone surface
    _clone_surface(base, interface_name=interface_name, site_families=tuple(site_families))

    # Annotate
    gen = base.get("generator", "")
    if not isinstance(gen, str):
        gen = str(gen)
    base["generator"] = (gen + "\n" if gen else "") + "cantera_netwark.build_sif4_large_mech"

    desc = base.get("description", "")
    if not isinstance(desc, str):
        desc = str(desc)
    add = (
        "\n\n[benchmark augmentation]\n"
        f"- merged extra gas mechanism: {extra_gas_mech_path.name} (species collisions skipped)\n"
        f"- surface multi-site cloning: {interface_name} across families {list(site_families)}\n"
        f"- source hashes: base={base_hash}, extra={extra_hash}\n"
    )
    base["description"] = desc + add

    _validate_species_tokens(base, source="generated_mechanism")
    _dump_yaml(base, out_path)
    return out_path


# Backward-compatible alias used by run scripts
build_large_sif4_mech = build_large_sif4_mech


def _parse_site_families(raw: str) -> tuple[str, ...]:
    vals = tuple(x.strip() for x in raw.split(",") if x.strip())
    if not vals:
        raise ValueError("--site-families must contain at least one family name")
    if len(set(vals)) != len(vals):
        raise ValueError("--site-families must be unique")
    return vals


def main() -> int:
    parser = argparse.ArgumentParser(description="Build large SiF4/NH3 multisite mechanism")
    parser.add_argument(
        "--base-mech",
        default="cantera_model/benchmarks/mechanisms/SiF4_NH3_mec.yaml",
        help="Path to base SiF4/NH3 mechanism YAML.",
    )
    parser.add_argument(
        "--extra-gas-mech",
        default="assets/mechanisms/gri30.yaml",
        help="Path to extra gas mechanism YAML.",
    )
    parser.add_argument(
        "--out-dir",
        default="cantera_model/benchmark_sif4_sin3n4_cvd/mechanisms",
        help="Output directory for generated mechanism.",
    )
    parser.add_argument(
        "--out-name",
        default="SiF4_NH3_mec_large__gri30__multisite3.yaml",
        help="Output file name.",
    )
    parser.add_argument(
        "--interface-name",
        default="SI3N4",
        help="Surface/interface phase name to clone.",
    )
    parser.add_argument(
        "--site-families",
        default="t,s,k",
        help="Comma-separated site-family prefixes.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Regenerate even if output file exists.",
    )
    args = parser.parse_args()

    site_families = _parse_site_families(str(args.site_families))
    out_path = build_large_sif4_mech(
        base_mech_path=Path(str(args.base_mech)).resolve(),
        extra_gas_mech_path=Path(str(args.extra_gas_mech)).resolve(),
        out_dir=Path(str(args.out_dir)).resolve(),
        interface_name=str(args.interface_name),
        site_families=site_families,
        out_name=str(args.out_name),
        force_rebuild=bool(args.force_rebuild),
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "out_path": str(out_path),
                "site_families": list(site_families),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
