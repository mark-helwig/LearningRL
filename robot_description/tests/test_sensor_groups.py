"""Prove each robot's contact-sensor groups name bodies that actually exist.

A body pattern matching nothing is a `ValueError` out of Isaac Lab's
`resolve_matching_names` the first time the scene is built -- loud, but only
once a simulator is running. This catches it earlier.

The body set is DERIVED from the vendor URDF rather than read from a fixture:
`merge_fixed_joints=True` (urdf_common.py) leaves exactly the root link plus
every revolute joint's child link, so a vendor bump surfaces here instead of
drifting from a stale hand-maintained copy of the link list.

Needs Isaac Lab (sensor.py imports it) and a `vendor/` checkout; skips cleanly
without either, so it is a no-op on a CPU-only machine.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from robot_description import paths

pytest.importorskip("isaaclab", reason="sensor.py builds real ContactSensorCfg objects")

from robot_description.robots.roboparty_rpo import sensor as rpo_sensor  # noqa: E402
from robot_description.robots.unitree_g1 import sensor as g1_sensor  # noqa: E402

GROUPS = {
    "unitree_g1": [g1_sensor.TERMINATION_CONTACT, g1_sensor.FEET_CONTACT],
    "roboparty_rpo": [rpo_sensor.TERMINATION_CONTACT, rpo_sensor.FEET_CONTACT],
}
robots = pytest.mark.parametrize("robot", list(GROUPS))


def _body_names(robot: str) -> set[str]:
    """Root link + every revolute joint's child link, i.e. what survives
    `merge_fixed_joints=True`."""
    urdf = Path(paths.urdf_path(robot))
    if not urdf.is_file():
        pytest.skip(f"vendor URDF not checked out: {urdf}")

    root = ET.parse(urdf).getroot()
    children, movable = set(), set()
    for joint in root.findall("joint"):
        child = joint.find("child").get("link")
        children.add(child)
        if joint.get("type") != "fixed":
            movable.add(child)

    roots = {link.get("name") for link in root.findall("link")} - children - {"world"}
    return roots | movable


@pytest.mark.sim
@robots
def test_every_pattern_matches_a_real_body(robot: str) -> None:
    bodies = _body_names(robot)
    for group in GROUPS[robot]:
        for pattern in group.bodies:
            matched = [name for name in bodies if re.fullmatch(pattern, name)]
            assert matched, (
                f"group {group.name!r} pattern {pattern!r} matches no {robot} body. "
                f"Remember fixed-joint links are merged away and are not addressable."
            )


@pytest.mark.sim
@robots
def test_groups_partition_all_bodies(robot: str) -> None:
    """Termination and feet must be disjoint and together cover every body --
    a body in neither is one nothing observes."""
    bodies = _body_names(robot)

    claims: dict[str, list[str]] = {name: [] for name in bodies}
    for group in GROUPS[robot]:
        for name in bodies:
            if any(re.fullmatch(pattern, name) for pattern in group.bodies):
                claims[name].append(group.name)

    unclaimed = sorted(n for n, g in claims.items() if not g)
    overlapping = sorted(n for n, g in claims.items() if len(g) > 1)
    assert not unclaimed, f"bodies in no sensor group: {unclaimed}"
    assert not overlapping, f"bodies claimed by more than one group: {overlapping}"
