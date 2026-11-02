"""The converted USD's inertia tensors equal the vendor URDF's.

Reads the USD a real spawn left in `build/<robot>/`, so it needs one prior spawn (any train or
play run) and `pxr`; it skips without either. If upstream fixes the importer, the marker is still
set and this fails with a rotated tensor -- turn off `RobotUrdfFileCfg.fix_principal_axes` then.

Only links the importer leaves unmerged are compared: not a fixed-joint child, no fixed-joint
children.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from robot_description import paths

pytest.importorskip("pxr", reason="reads the converted USD")

from robot_description.tools.usd_inertia_fix import MARKER  # noqa: E402

ROBOTS = ["unitree_g1", "roboparty_rpo"]


def _urdf_tensors(urdf: Path) -> dict[str, np.ndarray]:
    root = ET.parse(urdf).getroot()
    fixed_child, fixed_parent = set(), set()
    for joint in root.findall("joint"):
        if joint.get("type") == "fixed":
            fixed_child.add(joint.find("child").get("link"))
            fixed_parent.add(joint.find("parent").get("link"))
    out = {}
    for link in root.findall("link"):
        name, inertial = link.get("name"), link.find("inertial")
        if inertial is None or name in fixed_child | fixed_parent:
            continue
        a = {k: float(v) for k, v in inertial.find("inertia").attrib.items()}
        out[name] = np.array(
            [
                [a["ixx"], a["ixy"], a["ixz"]],
                [a["ixy"], a["iyy"], a["iyz"]],
                [a["ixz"], a["iyz"], a["izz"]],
            ]
        )
    return out


def _rotation(w: float, x: float, y: float, z: float) -> np.ndarray:
    """Standard (column-vector) rotation matrix of a unit quaternion."""
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


@pytest.mark.sim
@pytest.mark.parametrize("robot", ROBOTS)
def test_usd_inertia_matches_urdf(robot: str) -> None:
    from pxr import Usd, UsdPhysics

    urdf = Path(paths.urdf_path(robot))
    usd = paths.robot_build_dir(robot) / urdf.stem / f"{urdf.stem}.usda"
    if not (urdf.is_file() and usd.is_file()):
        pytest.skip(f"no converted USD yet: {usd}")

    stage = Usd.Stage.Open(str(usd))
    marked = stage.GetRootLayer().customLayerData.get(MARKER)
    assert marked, f"{usd} was not through the inertia fix"
    expected = _urdf_tensors(urdf)
    checked = 0
    for prim in stage.Traverse():
        if prim.GetName() not in expected or not prim.HasAPI(UsdPhysics.MassAPI):
            continue
        mass = UsdPhysics.MassAPI(prim)
        q = mass.GetPrincipalAxesAttr().Get()
        rot = _rotation(q.GetReal(), *q.GetImaginary())
        got = rot @ np.diag(mass.GetDiagonalInertiaAttr().Get()) @ rot.T
        want = expected[prim.GetName()]
        err = np.linalg.norm(got - want) / np.linalg.norm(want)
        assert err < 0.01, f"{prim.GetName()}: inertia tensor off by {100 * err:.1f}%"
        checked += 1
    assert checked, f"no link of {robot} compared -- prim names changed?"
