"""Arm-posture reward for the RPO's two-forearm squeeze.

Imported by the env cfg before Kit starts, so it may not import ``pxr`` (F9): the command term class
is referenced only for typing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands import RPOHandPosCommand


def forearm_level(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """``exp(-|z_wrist - z_elbow| / std)`` per active hand, averaged over the active hands; 0 for an idle env.

    The RPO holds a box between its forearms, so the grip is best with each forearm lying level along the
    box's side, elbow at the wrist's height. ``hand_track`` only says where the WRIST goes; this term asks
    for the forearm root (the ``<side>_elbow_yaw_link`` origin, i.e. the elbow end) to sit at the same height.
    Hands without a target are left free (no reward, no penalty), so idle arms can hang naturally.
    """
    cmd: RPOHandPosCommand = env.command_manager.get_term(command_name)
    wrist = cmd.hand_pos_w()
    elbow = cmd.robot.data.body_pos_w.torch[:, cmd.body_ids]
    dz = (wrist[..., 2] - elbow[..., 2]).abs()
    active = cmd.active_hand_mask.float()
    n_active = active.sum(dim=1)
    reward = (torch.exp(-dz / std) * active).sum(dim=1) / n_active.clamp(min=1.0)
    return torch.where(n_active > 0.0, reward, torch.zeros_like(reward))
