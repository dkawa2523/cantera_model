from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml


def cantera_available() -> bool:
    try:
        import cantera  # noqa: F401
    except Exception:
        return False
    return True


def build_sif4_large_trace(tmp_path: Path, *, run_id: str = "sif4_large_trace_smoke") -> Path:
    trace_root = tmp_path / "traces"
    cfg = yaml.safe_load(Path("configs/sif4_benchmark_sin3n4_large_trace.yaml").read_text())
    cfg["conditions_csv"] = str(
        Path(
            "cantera_model/benchmark_sif4_sin3n4_cvd/benchmarks/sif4_sin3n4_cvd_large/conditions_quick.csv"
        ).resolve()
    )
    cfg["trace_output"] = {"root": str(trace_root)}
    cfg["integration"] = {"n_steps": 16}
    cfg["simulation"]["surface"]["mechanism"] = str(
        Path(
            "cantera_model/benchmark_sif4_sin3n4_cvd/mechanisms/SiF4_NH3_mec_large__gri30__multisite3.yaml"
        ).resolve()
    )
    cfg_path = tmp_path / "trace_cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "cantera_model.cli.run_surface_trace",
            "--config",
            str(cfg_path),
            "--run-id",
            run_id,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)
    trace_path = trace_root / f"{run_id}.h5"
    if not trace_path.exists():
        raise RuntimeError(f"trace not found: {trace_path}")
    return trace_path


def run_reduce_mode(
    *,
    tmp_path: Path,
    config_path: str,
    trace_path: Path,
    run_id: str,
) -> Path:
    cfg = yaml.safe_load(Path(config_path).read_text())
    cfg["trace_h5"] = str(trace_path)
    cfg["report_dir"] = str(tmp_path / "reports")
    cfg_path = tmp_path / f"{run_id}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "cantera_model.cli.reduce_validate",
            "--config",
            str(cfg_path),
            "--run-id",
            run_id,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)

    return tmp_path / "reports" / run_id / "summary.json"
