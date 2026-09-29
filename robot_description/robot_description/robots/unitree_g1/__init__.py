"""Unitree G1 (29-DoF) Isaac Lab config.

Re-exports the public surface of `robot.py` and `sensor.py` so downstream code
(LearningRLlab) can do:

    from robot_description.robots.unitree_g1 import G1_CFG

which is this package's published interface. Both modules import Isaac Lab at
module scope, so importing this package requires Isaac Lab.
"""

from __future__ import annotations

from .robot import G1_CFG, G1_JOINT_NAMES
from .sensor import FEET_CONTACT as G1_FEET_CONTACT
from .sensor import TERMINATION_CONTACT as G1_TERMINATION_CONTACT

__all__ = [
    "G1_CFG",
    "G1_FEET_CONTACT",
    "G1_JOINT_NAMES",
    "G1_TERMINATION_CONTACT",
]
