"""The `@action` library: one entry per action-term `type` in `config.yaml`.

An action term turns its slice of the raw policy output into an actuator
command. Nothing here steps physics; the PD law that consumes these commands
lives in `loop.py`, because it runs at a different rate.

The split mirrors Isaac Lab's: `process_actions` once per policy step,
`apply_actions` once per physics substep. LearningRLsim keeps both even though
today's term is constant within a policy step, so a term that is not stays
correct without restructuring the loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from LearningRLsim.registry import ACTIONS, action
from LearningRLsim.runconfig import ActionTermCfg, RunConfig


@dataclass
class ActuatorCommand:
    """What the action terms collectively ask of the actuators, in `joint_order`.

    Gains live here rather than being read from `config.yaml` at use time, so
    an action term may rewrite them per step.
    """

    joint_pos_target: np.ndarray
    stiffness: np.ndarray
    damping: np.ndarray

    @staticmethod
    def nominal(cfg: RunConfig) -> ActuatorCommand:
        """The reset state: default pose, nominal gains from `config.yaml`."""
        return ActuatorCommand(
            joint_pos_target=cfg.default_joint_pos.copy(),
            stiffness=cfg.actuators.stiffness.copy(),
            damping=cfg.actuators.damping.copy(),
        )


class ActionTerm(ABC):
    """One entry of `actions.terms[]`, instantiated for a specific run."""

    def __init__(self, cfg: ActionTermCfg, run: RunConfig) -> None:
        self.cfg = cfg
        self.run = run
        # Position within `joint_order` of each joint this term drives.
        index_of = {name: i for i, name in enumerate(run.joint_order)}
        self.joint_ids = np.array([index_of[j] for j in cfg.joint_names], dtype=np.int32)

    @property
    def reset_action(self) -> np.ndarray:
        """The raw action this term resets to.

        Only `last_action` ever observes it -- `process` overwrites it before
        anything is applied -- but it is the first thing the policy sees, so a
        wrong value is a wrong first observation.
        """
        return np.zeros(self.cfg.dim)

    @abstractmethod
    def process(self, raw: np.ndarray) -> None:
        """Consume this term's slice of the raw policy output. Once per policy step."""

    @abstractmethod
    def apply(self, command: ActuatorCommand) -> None:
        """Write into the actuator command. Once per physics substep."""


@action("JointPositionAction")
class JointPositionAction(ActionTerm):
    """`processed = clip(raw * scale + offset)`, written as a position target.

    The clip is on the PROCESSED value, in radians -- not on the raw action.
    The locomotion tasks' `(-10, 10)` rad is effectively a NaN guard, and
    applying it to the raw action instead would be a hard saturation of the
    policy output.

    `offset` equals `default_joint_pos` exactly, because the tasks set
    `use_default_offset=True`; it is read from the config rather than assumed
    equal, since another task is free to differ.
    """

    def __init__(self, cfg: ActionTermCfg, run: RunConfig) -> None:
        super().__init__(cfg, run)
        self._scale = cfg.scale if cfg.scale is not None else np.ones(cfg.dim)
        self._offset = cfg.offset if cfg.offset is not None else np.zeros(cfg.dim)
        self._processed = np.zeros(cfg.dim)

    def process(self, raw: np.ndarray) -> None:
        processed = raw * self._scale + self._offset
        if self.cfg.clip is not None:
            processed = np.clip(processed, self.cfg.clip[:, 0], self.cfg.clip[:, 1])
        self._processed = processed

    def apply(self, command: ActuatorCommand) -> None:
        command.joint_pos_target[self.joint_ids] = self._processed


def build(run: RunConfig) -> list[ActionTerm]:
    """Instantiate every action term the run declares, in declaration order."""
    return [ACTIONS.lookup(term.type)(term, run) for term in run.actions.terms]
