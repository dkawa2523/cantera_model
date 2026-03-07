from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from cantera_model.types import CaseBundle, CaseTrace


def save_case_bundle(path: str | Path, bundle: CaseBundle) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out, "w") as handle:
        handle.attrs["mechanism_path"] = bundle.mechanism_path
        handle.attrs["phase"] = bundle.phase
        handle.attrs["species_names"] = json.dumps(bundle.species_names)
        handle.attrs["reaction_eqs"] = json.dumps(bundle.reaction_eqs)
        handle.attrs["meta"] = json.dumps(bundle.meta)

        cases_grp = handle.create_group("cases")
        for case in bundle.cases:
            grp = cases_grp.create_group(case.case_id)
            grp.create_dataset("time", data=np.asarray(case.time, dtype=float))
            grp.create_dataset("temperature", data=np.asarray(case.temperature, dtype=float))
            grp.create_dataset("pressure", data=np.asarray(case.pressure, dtype=float))
            grp.create_dataset("X", data=np.asarray(case.X, dtype=float))
            grp.create_dataset("wdot", data=np.asarray(case.wdot, dtype=float))
            grp.create_dataset("rop", data=np.asarray(case.rop, dtype=float))
            grp.attrs["meta"] = json.dumps(case.meta)

    return out


def load_case_bundle(path: str | Path) -> CaseBundle:
    src = Path(path)
    with h5py.File(src, "r") as handle:
        mechanism_path = str(handle.attrs["mechanism_path"])
        phase = str(handle.attrs["phase"])
        species_names = json.loads(str(handle.attrs["species_names"]))
        reaction_eqs = json.loads(str(handle.attrs["reaction_eqs"]))
        meta = json.loads(str(handle.attrs.get("meta", "{}")))

        cases: list[CaseTrace] = []
        for case_id in handle["cases"].keys():
            grp = handle["cases"][case_id]
            case = CaseTrace(
                case_id=case_id,
                time=np.asarray(grp["time"]),
                temperature=np.asarray(grp["temperature"]),
                pressure=np.asarray(grp["pressure"]),
                X=np.asarray(grp["X"]),
                wdot=np.asarray(grp["wdot"]),
                rop=np.asarray(grp["rop"]),
                species_names=list(species_names),
                reaction_eqs=list(reaction_eqs),
                meta=json.loads(str(grp.attrs.get("meta", "{}"))),
            )
            cases.append(case)

    return CaseBundle(
        mechanism_path=mechanism_path,
        phase=phase,
        species_names=list(species_names),
        reaction_eqs=list(reaction_eqs),
        cases=sorted(cases, key=lambda c: c.case_id),
        meta=meta,
    )
