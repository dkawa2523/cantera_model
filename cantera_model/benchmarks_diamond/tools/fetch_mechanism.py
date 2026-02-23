"""
Utility for obtaining mechanism YAML files needed for the benchmark cases.

Strategy:
1) If the file exists locally under <pack>/mechanisms/, use it.
2) If Cantera is installed, try to locate the file in Cantera's data directories.
3) Otherwise, download from one of the provided URLs and cache it under <pack>/mechanisms/.

This allows the benchmark pack to be lightweight while still being "downloadable".
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional
import urllib.request


def _print(msg: str) -> None:
    print(f"[fetch_mechanism] {msg}", file=sys.stderr)


def _try_cantera_data_dirs(filename: str) -> Optional[Path]:
    try:
        import cantera as ct  # type: ignore
    except Exception:
        return None

    # Cantera keeps a list of data search directories.
    try:
        dirs = ct.get_data_directories()
    except Exception:
        # Older Cantera versions
        try:
            dirs = ct.get_data_directories  # type: ignore
        except Exception:
            return None

    # Some distributions store "example_data" under the data dir
    candidates = []
    for d in dirs:
        d = Path(d)
        candidates.append(d / filename)
        candidates.append(d / "example_data" / filename)

        # Also try common underscore/hyphen variants
        if "-" in filename:
            candidates.append(d / filename.replace("-", "_"))
            candidates.append(d / "example_data" / filename.replace("-", "_"))
        if "_" in filename:
            candidates.append(d / filename.replace("_", "-"))
            candidates.append(d / "example_data" / filename.replace("_", "-"))

    for p in candidates:
        if p.exists():
            return p.resolve()

    return None


def _download(url: str, dest: Path, timeout: float = 60.0) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        _print(f"Downloading: {url}")
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "cantera-benchmark-pack/1.0 (+https://cantera.org)"
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
        dest.write_bytes(data)
        _print(f"Saved: {dest}")
        return True
    except Exception as e:
        _print(f"Download failed: {e}")
        return False


def resolve_mechanism(
    filename: str,
    urls: Iterable[str],
    cache_dir: Path,
    allow_cantera_lookup: bool = True,
) -> Path:
    """
    Return a local filesystem path to the mechanism YAML.

    Args:
        filename: Target YAML file name (e.g., 'SiF4_NH3_mec.yaml')
        urls: Candidate download URLs (raw GitHub, etc.)
        cache_dir: Directory to store downloaded/cached files (e.g., <pack>/mechanisms)
        allow_cantera_lookup: If True, attempt to locate via installed Cantera.

    Returns:
        Path to a local copy of the mechanism YAML.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    local = (cache_dir / filename).resolve()
    if local.exists():
        return local

    # Allow a user override:
    # If they set CANTERA_MECH_PATH=/path/to/file.yaml we use that.
    override = os.environ.get("CANTERA_MECH_PATH", "").strip()
    if override:
        p = Path(override)
        if p.exists():
            _print(f"Using CANTERA_MECH_PATH override: {p}")
            return p.resolve()

    # Try Cantera installed data directory
    if allow_cantera_lookup:
        p = _try_cantera_data_dirs(filename)
        if p is not None:
            _print(f"Found in Cantera data dirs: {p}")
            # Copy into cache for reproducibility
            try:
                local.write_bytes(p.read_bytes())
                _print(f"Copied into cache: {local}")
                return local
            except Exception:
                return p

    # Try downloads
    for url in urls:
        if _download(url, local):
            return local

    raise FileNotFoundError(
        f"Could not obtain mechanism '{filename}'.\n"
        f"Tried cache_dir={cache_dir}, Cantera data dirs, and URLs:\n"
        + "\n".join(urls)
        + "\n\n"
        "Tip: install Cantera (pip/conda) so the mechanism is available locally, "
        "or set CANTERA_MECH_PATH to a local YAML file."
    )
