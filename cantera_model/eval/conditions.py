from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any


def _as_float(value: Any, field: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid float for {field}: {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"non-finite value for {field}: {value!r}")
    return out


def _convert_pressure_to_pa(value: float, unit: str) -> float:
    lower = unit.strip().lower()
    if lower == "pa":
        return value
    if lower == "atm":
        return value * 101325.0
    if lower == "torr":
        return value * 133.32236842105263
    if lower == "bar":
        return value * 100000.0
    raise ValueError(f"unsupported pressure unit: {unit}")


def _find_pressure_column(fieldnames: set[str], schema_cfg: dict[str, Any]) -> tuple[str, str]:
    explicit = schema_cfg.get("pressure_column")
    if explicit:
        col = str(explicit)
        if col not in fieldnames:
            raise ValueError(f"conditions missing pressure column: {col}")
        unit = str(schema_cfg.get("pressure_unit") or "").strip().lower()
        if not unit:
            raise ValueError("schema.pressure_unit is required with schema.pressure_column")
        return col, unit

    candidates = [
        ("P_Pa", "pa"),
        ("P_pa", "pa"),
        ("P0_Pa", "pa"),
        ("P_atm", "atm"),
        ("P0_atm", "atm"),
        ("P_Torr", "torr"),
        ("P_torr", "torr"),
        ("P_bar", "bar"),
        ("P0_bar", "bar"),
    ]
    for col, unit in candidates:
        if col in fieldnames:
            return col, unit
    raise ValueError("conditions missing pressure column (supported: P_Pa/P_atm/P_Torr/P_bar)")


def load_conditions(
    path: Path,
    mode: str = "gas_homogeneous",
    schema_cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    schema = dict(schema_cfg or {})
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])

        if mode == "gas_homogeneous":
            required = {"T0", "P0_atm", "phi", "t_end"}
            missing = required - fieldnames
            if missing:
                raise ValueError(f"conditions missing required columns: {sorted(missing)}")
            for idx, row in enumerate(reader):
                case_id = (row.get("case_id") or "").strip() or f"row_{idx:04d}"
                rows.append(
                    {
                        "case_id": case_id,
                        "T0": _as_float(row.get("T0"), "T0"),
                        "P0_atm": _as_float(row.get("P0_atm"), "P0_atm"),
                        "phi": _as_float(row.get("phi"), "phi"),
                        "t_end": _as_float(row.get("t_end"), "t_end"),
                    }
                )
        elif mode == "surface_batch":
            t_col = str(schema.get("temperature_column", "T_K"))
            x_col = str(schema.get("composition_column", "composition"))
            end_col = str(schema.get("time_column", "t_end_s"))
            area_col = str(schema.get("area_column", "area"))
            n_steps_col = str(schema.get("n_steps_column", "n_steps"))
            pressure_col, pressure_unit = _find_pressure_column(fieldnames, schema)

            required = {t_col, x_col, end_col, pressure_col}
            missing = required - fieldnames
            if missing:
                raise ValueError(f"conditions missing required columns: {sorted(missing)}")

            default_area = float(schema.get("default_area", 1.0))
            default_n_steps = int(schema.get("default_n_steps", 200))
            for idx, row in enumerate(reader):
                case_id = (row.get("case_id") or "").strip() or f"row_{idx:04d}"
                area_raw = row.get(area_col)
                n_steps_raw = row.get(n_steps_col)
                rows.append(
                    {
                        "case_id": case_id,
                        "T_K": _as_float(row.get(t_col), t_col),
                        "P_Pa": _convert_pressure_to_pa(_as_float(row.get(pressure_col), pressure_col), pressure_unit),
                        "composition": str(row.get(x_col) or "").strip(),
                        "t_end_s": _as_float(row.get(end_col), end_col),
                        "n_steps": int(_as_float(n_steps_raw, n_steps_col)) if n_steps_raw not in (None, "") else default_n_steps,
                        "area": _as_float(area_raw, area_col) if area_raw not in (None, "") else default_area,
                    }
                )
        else:
            raise ValueError(f"unsupported simulation mode: {mode}")

    if not rows:
        raise ValueError("conditions CSV has no rows")
    return rows
