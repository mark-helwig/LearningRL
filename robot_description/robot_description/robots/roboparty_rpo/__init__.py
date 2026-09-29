"""RoboParty ROBOTO_ORIGIN / RPO (23-DoF) Isaac Lab config.

    from robot_description.robots.roboparty_rpo import RPO_CFG

`RPO_CFG` is the `DelayedPDActuatorCfg` variant, matching RoboParty's own
config; `RPO_IMPLICIT_CFG` is the solver-side-PD fallback. See `robot.py`.

Re-exports `robot.py`/`sensor.py` exactly as `unitree_g1/__init__.py` does.
"""

from __future__ import annotations

from .robot import RPO_CFG, RPO_IMPLICIT_CFG, RPO_JOINT_NAMES
from .sensor import FEET_CONTACT as RPO_FEET_CONTACT
from .sensor import TERMINATION_CONTACT as RPO_TERMINATION_CONTACT

__all__ = [
    "RPO_CFG",
    "RPO_FEET_CONTACT",
    "RPO_IMPLICIT_CFG",
    "RPO_JOINT_NAMES",
    "RPO_TERMINATION_CONTACT",
]
