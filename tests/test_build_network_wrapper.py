import subprocess
import sys


def test_build_network_wrapper_help_runs() -> None:
    proc = subprocess.run(
        [sys.executable, "run_build_network.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "--trace-h5" in proc.stdout
