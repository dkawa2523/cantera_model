import subprocess


def test_run_dev_checks_help() -> None:
    proc = subprocess.run(
        ["bash", "run_dev_checks.sh", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "--pooling-trials" in proc.stdout
    assert "--pooling-backend" in proc.stdout


def test_run_dev_checks_invalid_trials_rejected() -> None:
    proc = subprocess.run(
        ["bash", "run_dev_checks.sh", "--pooling-trials", "0", "--skip-pooling-tune"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "--pooling-trials" in (proc.stderr + proc.stdout)
