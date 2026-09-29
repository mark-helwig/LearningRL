"""Regression tests for `mesh_ops.check_mesh_refs`.

Load-bearing: the only guard against the `.STL` vs `.stl` class of bug,
guaranteed eventually since authoring happens on case-insensitive Windows and
execution on case-sensitive Linux. Two real Windows footguns motivate these
cases specifically: `Path.is_file()` fast-paths past case differences, and
`Path.resolve()` silently rewrites a path's casing to match the on-disk
file -- both invisible to a happy-path-only test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from robot_description.tools.mesh_ops import check_mesh_refs, iter_mesh_refs

pytestmark = pytest.mark.static

_URDF = """<?xml version="1.0"?>
<robot name="t">
  <link name="a">
    <visual><geometry><mesh filename="meshes/{good}"/></geometry></visual>
  </link>
  <link name="b">
    <visual><geometry><mesh filename="meshes/{bad}"/></geometry></visual>
  </link>
</robot>
"""


def _build(tmp_path: Path, on_disk: str, good: str, bad: str) -> Path:
    (tmp_path / "meshes").mkdir()
    (tmp_path / "meshes" / on_disk).write_bytes(b"solid\nendsolid\n")
    (tmp_path / "meshes" / good).write_bytes(b"solid\nendsolid\n")
    urdf = tmp_path / "t.urdf"
    urdf.write_text(_URDF.format(good=good, bad=bad), encoding="utf-8")
    return urdf


def test_case_mismatch_is_detected(tmp_path: Path) -> None:
    """A reference to `part.stl` when `PART.STL` is on disk must be flagged.

    This is THE test. It passes trivially on Linux and is the one that regressed
    twice on Windows.
    """
    urdf = _build(tmp_path, on_disk="PART.STL", good="ok.STL", bad="part.stl")
    audit = check_mesh_refs(iter_mesh_refs(urdf))
    flagged = " ".join(str(x) for x in audit.case_mismatches).lower()
    assert audit.case_mismatches, "mis-cased 'part.stl' vs on-disk 'PART.STL' was NOT flagged"
    assert "part.stl" in flagged


def test_correctly_cased_reference_is_not_flagged(tmp_path: Path) -> None:
    """No false positive when the casing already matches exactly."""
    urdf = _build(tmp_path, on_disk="PART.STL", good="ok.STL", bad="PART.STL")
    audit = check_mesh_refs(iter_mesh_refs(urdf))
    assert not audit.case_mismatches
    assert not audit.missing


def test_missing_mesh_is_detected(tmp_path: Path) -> None:
    """A reference with no file of any casing is 'missing', not 'case mismatch'."""
    urdf = _build(tmp_path, on_disk="PART.STL", good="ok.STL", bad="absent.STL")
    audit = check_mesh_refs(iter_mesh_refs(urdf))
    assert audit.missing, "absent.STL was not reported missing"
    assert not any("absent" in str(x).lower() for x in audit.case_mismatches)
