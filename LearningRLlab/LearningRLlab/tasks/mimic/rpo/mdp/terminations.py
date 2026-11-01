"""Termination terms for LearningRL-RPO-BeyondMimic-v0. Adapted from RoboParty's robolab, see ``NOTICE.md``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg

    from .commands import MotionCommand


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """The robot's anchor is more than ``threshold`` metres above or below the reference's."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


def bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    """The anchor's tilt (gravity's z in its own frame) is more than ``threshold`` off the reference's."""
    asset = env.scene[asset_cfg.name]

    command: MotionCommand = env.command_manager.get_term(command_name)
    gravity_direction = math_utils.normalize(asset.data.GRAVITY_VEC_W.torch)
    motion_projected_gravity_b = math_utils.quat_apply_inverse(command.anchor_quat_w, gravity_direction)

    robot_projected_gravity_b = math_utils.quat_apply_inverse(command.robot_anchor_quat_w, gravity_direction)

    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold
