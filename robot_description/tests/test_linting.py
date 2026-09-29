"""ruff is this repo's only linter/formatter (no black/isort/flake8 -- see
CLAUDE.md). Runs `ruff check` and `ruff format --check` as subprocesses over
the whole repo, so it stays correct as files are added without needing a
matching file list here.

Static: no Isaac Lab, no GPU. Requires `ruff` to be installed (it is a `dev`
extra in pyproject.toml, not a runtime dependency of the package itself).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent


def _ruff_available() -> bool:
    if shutil.which("ruff") is not None:
        return True
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "--version"],
        capture_output=True,
        cwd=REPO_ROOT,
    )
    return result.returncode == 0


def _run_ruff(*args: str) -> subprocess.CompletedProcess:
    if shutil.which("ruff") is not None:
        cmd = ["ruff", *args]
    else:
        cmd = [sys.executable, "-m", "ruff", *args]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)


@pytest.mark.static
@pytest.mark.skipif(not _ruff_available(), reason="ruff is not installed")
def test_ruff_check() -> None:
    result = _run_ruff("check", ".")
    assert result.returncode == 0, f"ruff check failed:\n{result.stdout}\n{result.stderr}"


@pytest.mark.static
@pytest.mark.skipif(not _ruff_available(), reason="ruff is not installed")
def test_ruff_format_check() -> None:
    result = _run_ruff("format", "--check", ".")
    assert result.returncode == 0, f"ruff format --check failed:\n{result.stdout}\n{result.stderr}"
