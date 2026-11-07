from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands import HandPosCommand


def hand_pos_tracking_reward(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """``exp(-|error| / std)`` per active hand, averaged over the active hands; 1 for an idle env.

    The error is measured in the yaw-aligned root frame against the velocity-advanced target
    (:meth:`HandPosCommand.target_pos_b`), so it does not grow just because the body walks.
    """
    hand_cmd: HandPosCommand = env.command_manager.get_term(command_name)
    error = torch.linalg.norm(hand_cmd.hand_pos_b() - hand_cmd.target_pos_b(), dim=-1)
    active = hand_cmd.active_hand_mask.float()
    n_active = active.sum(dim=1)
    reward = (torch.exp(-error / std) * active).sum(dim=1) / n_active.clamp(min=1.0)
    return torch.where(n_active > 0.0, reward, torch.ones_like(reward))
