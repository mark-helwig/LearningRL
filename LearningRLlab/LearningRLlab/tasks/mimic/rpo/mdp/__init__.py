"""Task-local MDP terms for LearningRL-RPO-BeyondMimic-v0.

Same re-export rule as ``locomotion/g1/mdp``: upstream's *leaf* modules, never
``isaaclab.envs.mdp`` itself (F9 -- that package drags in ``pxr`` before Kit exists). That rule also
decides how this package is split: ``commands_cfg.py`` holds the command cfg and is imported here,
while ``commands.py`` (``MotionCommand``, which needs ``Articulation`` and the debug-vis markers) is
reached only through the cfg's lazy ``class_type`` string.
"""

from isaaclab.envs.mdp.actions.actions_cfg import *  # noqa: F403
from isaaclab.envs.mdp.events import *  # noqa: F403
from isaaclab.envs.mdp.observations import *  # noqa: F403
from isaaclab.envs.mdp.rewards import *  # noqa: F403
from isaaclab.envs.mdp.terminations import *  # noqa: F403

from .commands_cfg import MotionCommandCfg
from .events import randomize_joint_default_pos
from .observations import motion_anchor_ori_b, motion_anchor_pos_b, robot_body_ori_b, robot_body_pos_b
from .rewards import (
    motion_global_anchor_orientation_error_exp,
    motion_global_anchor_position_error_exp,
    motion_global_body_angular_velocity_error_exp,
    motion_global_body_linear_velocity_error_exp,
    motion_relative_body_orientation_error_exp,
    motion_relative_body_position_error_exp,
)
from .terminations import bad_anchor_ori, bad_anchor_pos_z_only
