"""Task-local MDP terms for LearningRL-RPO-LocoManipulation-v0.

The G1 locomanipulation task's ``mdp`` namespace (which re-exports the G1 walk's, F6) for the stock terms and the
hand-tracking pieces, then this task's: the Walk-v1 terms ported from HCRLlab (``walk_rewards``,
``walk_observations``) and the local ``HandPosCommandCfg`` (the RPO wrist variant), which wins over the G1 one.

``RPOHandPosCommand`` is not exported, for the same reason ``HandPosCommand`` is not: it
imports ``CommandTerm``, which reaches ``pxr`` (F9). ``HandPosCommandCfg.class_type`` names it.
"""

from LearningRLlab.tasks.locomanipulation.g1.mdp import *  # noqa: F403

from .commands_cfg import HandPosCommandCfg
from .walk_observations import feet_air_contact_time, illegal_external_contact, stance_type
from .walk_rewards import (
    body_orientation_exp,
    foot_impact_force,
    foot_slip_exp,
    gait_clearance,
    gait_contact_clock,
    joint_pos_limits_normalized,
    self_collision_fraction,
)
from .wrist import WRIST_OFFSET
