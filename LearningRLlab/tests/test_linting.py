"""Static, GPU-free lint gate.

Runs ``ruff format --check`` and ``ruff check`` over the whole repository via
``subprocess``. Marked ``static`` so ``pytest -m static`` is a GPU-free gate that runs
on Windows -- no Isaac Lab, no container, no simulation -- exactly what a reviewer
should run before pushing.
"""

# Required: without this, the PEP 604 union below raises TypeError at pytest
# COLLECTION on Python < 3.10, silently running 0 tests instead of failing loudly.
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _ruff_cmd() -> list[str] | None:
    """Return the base command used to invoke ruff, or None if ruff is unavailable.

    Prefers a ``ruff`` binary on PATH; falls back to ``python -m ruff`` (how this repo
    runs it on a bare Windows dev box with no PATH entry for console scripts).
    """
    if shutil.which("ruff") is not None:
        return ["ruff"]
    try:
        subprocess.run(
            [sys.executable, "-m", "ruff", "--version"],
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return [sys.executable, "-m", "ruff"]


@pytest.mark.static
def test_ruff_format_check():
    base = _ruff_cmd()
    if base is None:
        pytest.skip("ruff is not installed (neither on PATH nor as a Python module)")

    result = subprocess.run(
        [*base, "format", "--check", "--diff", str(REPO_ROOT)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"ruff format check failed -- run `ruff format .` from the repo root to fix:\n{result.stdout}\n{result.stderr}"
    )


@pytest.mark.static
def test_ruff_check():
    base = _ruff_cmd()
    if base is None:
        pytest.skip("ruff is not installed (neither on PATH nor as a Python module)")

    result = subprocess.run(
        [*base, "check", "--no-fix", str(REPO_ROOT)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "ruff check failed -- run `ruff check --fix .` from the repo root to apply "
        f"auto-fixes, then address any remaining issues by hand:\n{result.stdout}\n{result.stderr}"
    )
