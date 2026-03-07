import subprocess
import sys


def test_wrapper_help_runs() -> None:
    proc = subprocess.run(
        [sys.executable, "run_cantera_eval.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "--config" in proc.stdout
