from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cantera_model.io.trace_store import load_case_bundle
from cantera_model.network.flux import build_flux, reaction_importance
from cantera_model.network.stoich import build_nu, extract_species_meta


def _formula_guess(name: str) -> dict[str, float]:
    out: dict[str, float] = {}
    token = ""
    num = ""

    def flush() -> None:
        nonlocal token, num
        if not token:
            return
        n = float(num) if num else 1.0
        out[token] = out.get(token, 0.0) + n
        token = ""
        num = ""

    for ch in name:
        if ch.isupper():
            flush()
            token = ch
        elif ch.islower() and token:
            token += ch
        elif ch.isdigit() and token:
            num += ch
        else:
            flush()
    flush()
    return out


def _guess_species_meta(species_names: list[str], phase: str) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "composition": _formula_guess(name),
            "phase": phase,
            "charge": 0,
            "radical": False,
            "role": "",
        }
        for name in species_names
    ]


def _build_element_matrix_from_meta(species_meta: list[dict[str, Any]], species_names: list[str]) -> tuple[np.ndarray, list[str]]:
    by_name = {str(m.get("name", "")): dict(m) for m in species_meta}
    ordered_meta = [by_name.get(name, {"name": name, "composition": {}}) for name in species_names]

    elems = sorted({e for m in ordered_meta for e in (m.get("composition") or {}).keys()})
    A = np.zeros((len(elems), len(species_names)), dtype=float)
    for j, meta in enumerate(ordered_meta):
        comp = meta.get("composition") or {}
        for i, elem in enumerate(elems):
            A[i, j] = float(comp.get(elem, 0.0))
    return A, elems


def _reorder_rows(source_names: list[str], target_names: list[str], matrix: np.ndarray) -> np.ndarray:
    source_idx = {name: i for i, name in enumerate(source_names)}
    out = np.zeros((len(target_names), matrix.shape[1]), dtype=float)
    for t_idx, name in enumerate(target_names):
        s_idx = source_idx.get(name)
        if s_idx is not None:
            out[t_idx, :] = matrix[s_idx, :]
    return out


def _resolve_nu(bundle: Any, rop: np.ndarray, wdot: np.ndarray) -> np.ndarray:
    nu_meta = (bundle.meta or {}).get("nu")
    if nu_meta is not None:
        nu = np.asarray(nu_meta, dtype=float)
        return nu

    try:
        nu_sparse, mech_species, _ = build_nu(bundle.mechanism_path, bundle.phase)
        nu = nu_sparse.toarray()
        if list(mech_species) != list(bundle.species_names):
            nu = _reorder_rows(list(mech_species), list(bundle.species_names), nu)
        return nu
    except Exception:
        # Canteraなし環境向けフォールバック
        return np.linalg.lstsq(rop, wdot, rcond=None)[0].T


def _resolve_species_meta(bundle: Any) -> list[dict[str, Any]]:
    species_meta = list((bundle.meta or {}).get("species_meta") or [])
    if species_meta:
        return species_meta

    try:
        species_meta = extract_species_meta(bundle.mechanism_path, phase=bundle.phase)
    except Exception:
        species_meta = _guess_species_meta(list(bundle.species_names), bundle.phase)
    return species_meta


def _case_dt(time: np.ndarray) -> np.ndarray:
    t = np.asarray(time, dtype=float)
    if t.size < 2:
        return np.ones_like(t)
    dt = np.empty_like(t)
    dt[0] = max(t[1] - t[0], 1.0e-12)
    dt[1:] = np.maximum(np.diff(t), 1.0e-12)
    return dt


def _concat_dt(bundle: Any) -> np.ndarray:
    dts: list[np.ndarray] = []
    for case in bundle.cases:
        dts.append(_case_dt(np.asarray(case.time, dtype=float)))
    return np.concatenate(dts, axis=0)


def _parse_phase_fractions(raw: str | None) -> tuple[list[str], np.ndarray]:
    if raw is None or str(raw).strip() == "":
        return ["all"], np.asarray([1.0], dtype=float)

    names: list[str] = []
    vals: list[float] = []
    for token in str(raw).split(","):
        part = token.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"invalid phase fraction token: {part!r}")
        name, frac = part.split(":", 1)
        name = name.strip()
        val = float(frac.strip())
        if not name:
            raise ValueError("phase name cannot be empty")
        if val <= 0.0:
            raise ValueError("phase fraction must be > 0")
        names.append(name)
        vals.append(val)

    if not names:
        return ["all"], np.asarray([1.0], dtype=float)

    arr = np.asarray(vals, dtype=float)
    arr = arr / float(np.sum(arr))
    return names, arr


def _phase_indices_for_case(time: np.ndarray, phase_fracs: np.ndarray) -> np.ndarray:
    if phase_fracs.size == 1:
        return np.zeros(time.shape[0], dtype=int)

    t = np.asarray(time, dtype=float)
    if t.size == 0:
        return np.zeros(0, dtype=int)
    span = max(float(t[-1] - t[0]), 1.0e-12)
    tau = np.clip((t - float(t[0])) / span, 0.0, 1.0)

    bounds = np.cumsum(phase_fracs)
    bounds[-1] = 1.0
    idx = np.searchsorted(bounds, tau, side="right")
    return np.clip(idx, 0, phase_fracs.size - 1)


def _normalize_flux_matrix(F: np.ndarray) -> np.ndarray:
    arr = np.asarray(F, dtype=float)
    max_val = float(np.max(arr)) if arr.size else 0.0
    if max_val > 0.0:
        arr = arr / max_val
    return arr


def _build_phase_aggregates(
    bundle: Any,
    nu: np.ndarray,
    phase_names: list[str],
    phase_fracs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_phase = len(phase_names)
    n_species, n_reactions = nu.shape

    F_by_phase = np.zeros((n_phase, n_species, n_species), dtype=float)
    I_by_phase = np.zeros((n_phase, n_reactions), dtype=float)

    for case in bundle.cases:
        rop_case = np.asarray(case.rop, dtype=float)
        time_case = np.asarray(case.time, dtype=float)
        dt_case = _case_dt(time_case)
        phase_idx = _phase_indices_for_case(time_case, phase_fracs)

        for p in range(n_phase):
            mask = phase_idx == p
            if not np.any(mask):
                continue
            rop_local = rop_case[mask]
            dt_local = dt_case[mask]
            I_by_phase[p] += reaction_importance(rop_local, dt_local)
            F_by_phase[p] += build_flux(nu, rop_local, dt_local, normalize=False)

    for p in range(n_phase):
        F_by_phase[p] = _normalize_flux_matrix(F_by_phase[p])

    return F_by_phase, I_by_phase


def _parse_bool(raw: str | bool | None, *, default: bool) -> bool:
    if raw is None:
        return bool(default)
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {raw!r}")


def _save_state_artifacts(bundle: Any, out_dir: Path) -> dict[str, Any]:
    times: list[np.ndarray] = []
    xs: list[np.ndarray] = []
    case_slices: list[dict[str, Any]] = []
    cursor = 0

    for case in bundle.cases:
        t = np.asarray(case.time, dtype=float)
        x = np.asarray(case.X, dtype=float)
        if x.shape[0] != t.shape[0]:
            raise ValueError(f"X/time length mismatch for case_id={case.case_id}")
        start = cursor
        end = cursor + int(t.shape[0])
        case_slices.append(
            {
                "case_id": str(case.case_id),
                "start": start,
                "end": end,  # exclusive
                "n_steps": int(t.shape[0]),
                "t_start": (None if t.size == 0 else float(t[0])),
                "t_end": (None if t.size == 0 else float(t[-1])),
            }
        )
        cursor = end
        times.append(t)
        xs.append(x)

    time_concat = np.concatenate(times, axis=0) if times else np.zeros((0,), dtype=float)
    x_concat = np.concatenate(xs, axis=0) if xs else np.zeros((0, len(bundle.species_names)), dtype=float)
    np.save(out_dir / "time.npy", np.asarray(time_concat, dtype=float))
    np.save(out_dir / "X.npy", np.asarray(x_concat, dtype=float))
    (out_dir / "case_slices.json").write_text(json.dumps(case_slices, ensure_ascii=False, indent=2))

    return {
        "state_saved": True,
        "state_rows": int(time_concat.shape[0]),
        "state_cases": int(len(case_slices)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build nu/A/F_bar/I_reaction from trace_h5")
    parser.add_argument("--trace-h5", required=True, help="Path to CaseBundle .h5")
    parser.add_argument("--run-id", required=False, default=None, help="Output run id")
    parser.add_argument("--output-root", required=False, default="artifacts/network", help="Output root directory")
    parser.add_argument(
        "--phase-fractions",
        required=False,
        default=None,
        help="Optional comma-separated normalized phase bins, e.g. 'pulse:0.25,purge:0.25,pulse2:0.25,purge2:0.25'",
    )
    parser.add_argument(
        "--save-state",
        required=False,
        default="true",
        help="Whether to save time/X/case_slices artifacts (true|false). default=true",
    )
    args = parser.parse_args()

    trace_path = Path(args.trace_h5).resolve()
    bundle = load_case_bundle(trace_path)
    if not bundle.cases:
        raise ValueError("trace_h5 has no cases")

    run_id = args.run_id or f"network_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir = Path(args.output_root).resolve() / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    rop = np.concatenate([np.asarray(c.rop, dtype=float) for c in bundle.cases], axis=0)
    wdot = np.concatenate([np.asarray(c.wdot, dtype=float) for c in bundle.cases], axis=0)
    dt = _concat_dt(bundle)

    nu = _resolve_nu(bundle, rop, wdot)
    if nu.shape != (len(bundle.species_names), len(bundle.reaction_eqs)):
        raise ValueError("resolved nu has invalid shape")

    species_meta = _resolve_species_meta(bundle)
    A, elements = _build_element_matrix_from_meta(species_meta, list(bundle.species_names))

    F_bar = build_flux(nu, rop, dt)
    I_reaction = reaction_importance(rop, dt)

    phase_names, phase_fracs = _parse_phase_fractions(args.phase_fractions)
    F_by_phase, I_by_phase = _build_phase_aggregates(bundle, nu, phase_names, phase_fracs)
    save_state = _parse_bool(args.save_state, default=True)

    np.save(out_dir / "nu.npy", np.asarray(nu, dtype=float))
    np.save(out_dir / "A.npy", np.asarray(A, dtype=float))
    np.save(out_dir / "F_bar.npy", np.asarray(F_bar, dtype=float))
    np.save(out_dir / "I_reaction.npy", np.asarray(I_reaction, dtype=float))
    np.save(out_dir / "rop.npy", np.asarray(rop, dtype=float))
    np.save(out_dir / "wdot.npy", np.asarray(wdot, dtype=float))
    np.save(out_dir / "dt.npy", np.asarray(dt, dtype=float))
    np.save(out_dir / "F_bar_by_phase.npy", np.asarray(F_by_phase, dtype=float))
    np.save(out_dir / "I_reaction_by_phase.npy", np.asarray(I_by_phase, dtype=float))

    (out_dir / "species_names.json").write_text(json.dumps(list(bundle.species_names), ensure_ascii=False, indent=2))
    (out_dir / "reaction_eqs.json").write_text(json.dumps(list(bundle.reaction_eqs), ensure_ascii=False, indent=2))
    (out_dir / "elements.json").write_text(json.dumps(elements, ensure_ascii=False, indent=2))
    (out_dir / "species_meta.json").write_text(json.dumps(species_meta, ensure_ascii=False, indent=2))
    (out_dir / "phase_names.json").write_text(json.dumps(phase_names, ensure_ascii=False, indent=2))
    (out_dir / "phase_fractions.json").write_text(json.dumps(phase_fracs.tolist(), ensure_ascii=False, indent=2))
    conditions = [dict((c.meta or {}).get("conditions") or {"case_id": c.case_id}) for c in bundle.cases]
    (out_dir / "conditions.json").write_text(json.dumps(conditions, ensure_ascii=False, indent=2))
    state_summary = {"state_saved": False, "state_rows": 0, "state_cases": 0}
    if save_state:
        state_summary = _save_state_artifacts(bundle, out_dir)

    summary = {
        "status": "ok",
        "run_id": run_id,
        "trace_h5": str(trace_path),
        "output_dir": str(out_dir),
        "cases": len(bundle.cases),
        "species": len(bundle.species_names),
        "reactions": len(bundle.reaction_eqs),
        "elements": len(elements),
        "phases": len(phase_names),
        "phase_names": phase_names,
        "i_reaction_nonzero": int(np.count_nonzero(I_reaction > 0.0)),
        **state_summary,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
