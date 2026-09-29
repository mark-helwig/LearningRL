from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def view_middle_env(env: ManagerBasedEnv, env_ids: torch.Tensor | None) -> None:
    """Point the viewer (and ``--video``) at env ``num_envs // 2``, a middle terrain column for any ``--num_envs``."""
    if env.viewport_camera_controller is not None:
        env.viewport_camera_controller.set_view_env_index(env.num_envs // 2)
