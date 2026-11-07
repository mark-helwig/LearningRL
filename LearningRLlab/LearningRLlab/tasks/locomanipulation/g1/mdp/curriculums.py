from __future__ import annotations

from typing import TYPE_CHECKING, Any

from isaaclab.envs.mdp.curriculums import modify_term_cfg

if TYPE_CHECKING:
    from collections.abc import Sequence

    from isaaclab.envs import ManagerBasedRLEnv


def linear_ramp(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    old_value: Any,
    start_step: int,
    end_step: int,
    start_value: float,
    end_value: float,
) -> float:
    """``modify_fn`` for upstream's ``modify_term_cfg``: ``start_value`` until ``start_step``, then a straight
    line to ``end_value`` at ``end_step``, held after. Steps are ``env.common_step_counter`` (env steps, not
    PPO iterations: one iteration is ``num_steps_per_env`` of them)."""
    t = env.common_step_counter
    alpha = min(max((t - start_step) / (end_step - start_step), 0.0), 1.0)
    value = start_value + alpha * (end_value - start_value)
    return modify_term_cfg.NO_CHANGE if value == old_value else value
