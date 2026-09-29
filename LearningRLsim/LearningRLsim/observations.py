"""The `@observation` library: one entry per `func` name `config.yaml` can emit.

LearningRLlab defines no observation functions of its own -- every term comes from
upstream Isaac Lab's `envs/mdp/observations.py` via a wildcard re-export -- so
this is a reimplementation of that module's locomotion subset against MuJoCo
state, and each entry names the Isaac Lab quantity it reproduces.

Dispatch is on `func`, never on `name`: a task inheriting upstream's naming
calls the same functions `joint_pos` and `actions` that the shipped tasks call
`joint_pos_rel` and `last_action`.

Post-processing (clip, then scale) is applied by `assemble()`, not by the
functions, so it happens once and in one order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from LearningRLsim.registry import OBSERVATIONS, observation
from LearningRLsim.runconfig import ObsTermCfg, RunConfigError

if TYPE_CHECKING:
    from LearningRLsim.context import EvalContext


def _accept(term: ObsTermCfg, *allowed: str) -> dict[str, Any]:
    """Reject any exported param this implementation does not honour.

    Silently ignoring one is the failure mode this whole design is built to
    avoid: `asset_cfg` with a `joint_ids` subset, for instance, would make the
    real term return a slice of the joints while ours returns all of them --
    same shape only by coincidence, and no error anywhere.
    """
    unknown = sorted(set(term.params) - set(allowed))
    if unknown:
        raise RunConfigError(
            f"observation term '{term.name}' ({term.func}) was exported with param(s) "
            f"{unknown}, which LearningRLsim's implementation does not honour. Accepted: "
            f"{sorted(allowed)}. Implement them rather than ignoring them."
        )
    return term.params


# ---------------------------------------------------------------------------
# Root state
# ---------------------------------------------------------------------------
@observation("base_ang_vel")
def _base_ang_vel(ctx: EvalContext, term: ObsTermCfg) -> np.ndarray:
    """Isaac Lab `root_ang_vel_b`. MuJoCo's `qvel[3:6]` is already body-frame."""
    _accept(term)
    return ctx.sim.base_ang_vel_b


@observation("base_lin_vel")
def _base_lin_vel(ctx: EvalContext, term: ObsTermCfg) -> np.ndarray:
    """Isaac Lab `root_lin_vel_b`. MuJoCo's `qvel[0:3]` is WORLD-frame, so it
    is rotated in -- unlike the angular half above."""
    _accept(term)
    return ctx.sim.base_lin_vel_b


@observation("projected_gravity")
def _projected_gravity(ctx: EvalContext, term: ObsTermCfg) -> np.ndarray:
    """Isaac Lab `projected_gravity_b`: the unit gravity DIRECTION in the root
    frame, so it does not scale with `--gravity`."""
    _accept(term)
    return ctx.sim.projected_gravity_b


# ---------------------------------------------------------------------------
# Joint state
# ---------------------------------------------------------------------------
@observation("joint_pos")
def _joint_pos(ctx: EvalContext, term: ObsTermCfg) -> np.ndarray:
    _accept(term)
    return ctx.sim.joint_pos


@observation("joint_pos_rel")
def _joint_pos_rel(ctx: EvalContext, term: ObsTermCfg) -> np.ndarray:
    _accept(term)
    return ctx.sim.joint_pos - ctx.cfg.default_joint_pos


@observation("joint_vel")
def _joint_vel(ctx: EvalContext, term: ObsTermCfg) -> np.ndarray:
    _accept(term)
    return ctx.sim.joint_vel


@observation("joint_vel_rel")
def _joint_vel_rel(ctx: EvalContext, term: ObsTermCfg) -> np.ndarray:
    """`q̇ - q̇_default`. `default_joint_vel` is all-zero for every shipped robot,
    making this identical to `joint_vel` -- kept as its own entry because the
    config names it, and because a robot whose default velocity is ever
    non-zero must not silently get this one wrong."""
    _accept(term)
    return ctx.sim.joint_vel


# ---------------------------------------------------------------------------
# Actions and commands
# ---------------------------------------------------------------------------
@observation("last_action")
def _last_action(ctx: EvalContext, term: ObsTermCfg) -> np.ndarray:
    """Isaac Lab `env.action_manager.action`: the full RAW action vector across
    every term, pre-scale/offset/clip."""
    params = _accept(term, "action_name")
    action_name = params.get("action_name")
    if action_name is None:
        return ctx.last_action
    for action_term in ctx.cfg.actions.terms:
        if action_term.name == action_name:
            start, stop = action_term.slice
            return ctx.last_action[start:stop]
    raise RunConfigError(
        f"observation term '{term.name}' asks for action_name={action_name!r}, which is "
        f"not an action term: {[t.name for t in ctx.cfg.actions.terms]}"
    )


@observation("generated_commands")
def _generated_commands(ctx: EvalContext, term: ObsTermCfg) -> np.ndarray:
    params = _accept(term, "command_name")
    if "command_name" not in params:
        raise RunConfigError(
            f"observation term '{term.name}' is a generated_commands term with no "
            f"params.command_name, so there is no way to know which command it reads."
        )
    return ctx.commands.value(str(params["command_name"]))


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def assemble(ctx: EvalContext, out: np.ndarray | None = None) -> np.ndarray:
    """Build the policy observation vector for the current state.

    Post-processing order is Isaac Lab's, fixed at
    `managers/observation_manager.py`: compute -> modifier -> noise -> clip ->
    scale. Every `-PLAY` task sets `enable_corruption = False` and no LearningRLlab
    task uses modifiers, so the two middle steps are absent here and the order
    that remains is **clip, then scale**. Backwards, it is silent.

    Terms are written into their declared `slice`, which `runconfig` has
    already proven tiles the vector contiguously.
    """
    total = ctx.cfg.observations.total_dim
    if out is None:
        out = np.empty(total, dtype=np.float64)

    for term in ctx.cfg.observations.terms:
        value = np.asarray(OBSERVATIONS.lookup(term.func)(ctx, term), dtype=np.float64).reshape(-1)
        if value.shape != (term.dim,):
            raise RunConfigError(
                f"observation term '{term.name}' ({term.func}) produced {value.shape[0]} "
                f"values, but config.yaml declares dim={term.dim}."
            )
        if term.clip is not None:
            value = np.clip(value, term.clip[0], term.clip[1])
        if term.scale is not None:
            value = value * term.scale
        start, stop = term.slice
        out[start:stop] = value

    return out
