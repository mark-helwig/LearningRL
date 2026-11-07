"""The `@command` library: one entry per command `type` in `config.yaml`.

A command TERM is what the policy consumes -- it owns the layout of its slice
of the observation vector and the semantics of its fields. What *drives* it is
a command SOURCE (`sources.py`), which is a different question and a
different layer.

Only the terms a run's `config.yaml` declares are instantiated.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from LearningRLsim.registry import COMMANDS, command
from LearningRLsim.runconfig import CommandCfg, RunConfig, RunConfigError

if TYPE_CHECKING:
    from LearningRLsim.context import EvalContext


def wrap_to_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class CommandTerm(ABC):
    """One entry of `commands[]`, instantiated for a specific run."""

    FIELDS: ClassVar[tuple[str, ...]] = ()
    """The names a command source may set, and the YAML keys of a timeline
    waypoint. Order is the term's own; it need not match the obs layout."""

    def __init__(self, cfg: CommandCfg, run: RunConfig) -> None:
        self.cfg = cfg
        self.run = run
        self._fields: dict[str, float] = dict.fromkeys(self.FIELDS, 0.0)

    @property
    @abstractmethod
    def dim(self) -> int:
        """Width of this term's contribution to the observation vector."""

    @abstractmethod
    def value(self) -> np.ndarray:
        """The current command, as the policy sees it."""

    def set_fields(self, fields: dict[str, float]) -> None:
        unknown = sorted(set(fields) - set(self.FIELDS))
        if unknown:
            raise RunConfigError(
                f"command '{self.cfg.name}' ({self.cfg.type}) has no field(s) {unknown}. Fields: {list(self.FIELDS)}"
            )
        self._fields.update({k: float(v) for k, v in fields.items()})

    def update(self, ctx: EvalContext) -> None:
        """Per-policy-step post-processing. Most terms need none."""

    def reset(self) -> None:
        self._fields = dict.fromkeys(self.FIELDS, 0.0)


@command("UniformVelocityCommandCfg")
class UniformVelocityCommand(CommandTerm):
    """`[lin_vel_x, lin_vel_y, ang_vel_z]` in the BASE frame.

    The velocity command of every LearningRLlab task.

    Two behaviours of the Isaac Lab term are reproduced and one is
    deliberately not:

    - **`heading_command`** overwrites `ang_vel_z` every policy step with
      `clip(stiffness * wrap_to_pi(heading_target - heading_w),
      ang_vel_z_range)`. No shipped task sets it.
    - **`rel_standing_envs`** is NOT sampled. It is the fraction of a
      vectorised env's rows told to stand still, which has no meaning for one
      robot; rolling a die against it every reset would make a scripted
      rollout non-reproducible for no gain. Standing is a command source's
      job -- a zeroed waypoint, or a future button.

    `heading_control_stiffness` is not exported, so it comes from a flag
    defaulting to Isaac Lab's own default of 1.0 and is recorded in
    `/meta/run` as assumed.
    """

    FIELDS: ClassVar[tuple[str, ...]] = ("lin_vel_x", "lin_vel_y", "ang_vel_z", "heading")

    def __init__(self, cfg: CommandCfg, run: RunConfig, heading_stiffness: float = 1.0) -> None:
        super().__init__(cfg, run)
        self.heading_stiffness = heading_stiffness
        ang_range = cfg.ranges.get("ang_vel_z")
        self._ang_vel_z_limits = (float(ang_range[0]), float(ang_range[1])) if ang_range is not None else None
        if cfg.heading_command and self._ang_vel_z_limits is None:
            raise RunConfigError(
                f"command '{cfg.name}' sets heading_command but declares no ang_vel_z range "
                f"to clip the heading controller's output to."
            )

    @property
    def dim(self) -> int:
        return 3

    def value(self) -> np.ndarray:
        return np.array([self._fields["lin_vel_x"], self._fields["lin_vel_y"], self._fields["ang_vel_z"]])

    def update(self, ctx: EvalContext) -> None:
        if not self.cfg.heading_command:
            return
        error = wrap_to_pi(self._fields["heading"] - ctx.sim.heading_w)
        lo, hi = self._ang_vel_z_limits
        self._fields["ang_vel_z"] = float(np.clip(self.heading_stiffness * error, lo, hi))


@command("HandPosCommandCfg")
class HandPosCommand(CommandTerm):
    """`[left_active, right_active, left_xyz, right_xyz]`, targets in the YAW-ALIGNED root frame.

    LearningRL-G1-LocoManipulation-v0's hand targets. The policy reads the
    command verbatim, so this is the whole term: the training side's one-step
    velocity advance of the target lives in its reward and markers, never in
    the observation. An inactive hand's target is still observed, so a
    timeline should leave it at a sensible value rather than garbage.

    Nothing is sampled -- not `mode`, not `rel_standing_envs` -- for the same
    reason `UniformVelocityCommand` samples nothing: a scripted rollout must
    be reproducible. All-zero (the `ZeroSource` default) is exactly the
    training side's idle row: no hand active, both targets at the origin.
    """

    FIELDS: ClassVar[tuple[str, ...]] = (
        "left_active",
        "right_active",
        "left_x",
        "left_y",
        "left_z",
        "right_x",
        "right_y",
        "right_z",
    )

    @property
    def dim(self) -> int:
        return 8

    def value(self) -> np.ndarray:
        return np.array([self._fields[name] for name in self.FIELDS])


def build(run: RunConfig, heading_stiffness: float = 1.0) -> dict[str, CommandTerm]:
    """Instantiate every command term the run declares."""
    terms: dict[str, CommandTerm] = {}
    for cfg in run.commands:
        cls = COMMANDS.lookup(cfg.type)
        term = cls(cfg, run, heading_stiffness=heading_stiffness) if cls is UniformVelocityCommand else cls(cfg, run)
        # The observation slice is the contract between this term and the
        # policy. If the widths disagree, every later element of the
        # observation vector is shifted.
        if cfg.obs_slice is not None:
            width = cfg.obs_slice[1] - cfg.obs_slice[0]
            if width != term.dim:
                raise RunConfigError(
                    f"command '{cfg.name}' occupies obs_slice {list(cfg.obs_slice)} "
                    f"({width} wide) but {cfg.type} produces {term.dim} values."
                )
        terms[cfg.name] = term
    return terms
