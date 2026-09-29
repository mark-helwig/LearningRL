"""Back-compat alias for `urdf_common.RobotUrdfFileCfg`.

The shared URDF-importer spawn config moved to `urdf_common.py` and lost its
`Unitree` prefix when `roboparty_rpo` (not a Unitree robot) started using it.
This module stays so callers that import the old name keep working. New code
should import `RobotUrdfFileCfg` from `robot_description.robots.urdf_common`.
"""

from __future__ import annotations

from robot_description.robots.urdf_common import RobotUrdfFileCfg

UnitreeUrdfFileCfg = RobotUrdfFileCfg

__all__ = ["UnitreeUrdfFileCfg"]
