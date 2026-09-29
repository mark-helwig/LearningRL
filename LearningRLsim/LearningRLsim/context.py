"""The rollout's mutable state, as the term libraries see it.

Separated from `loop.py` so that `observations.py`, `actions.py` and the
command terms can name the type they receive without importing the loop that
constructs it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from LearningRLsim.commands.sources import CommandManager
    from LearningRLsim.model import RobotModel
    from LearningRLsim.runconfig import RunConfig


@dataclass
class EvalContext:
    cfg: RunConfig
    sim: RobotModel
    commands: CommandManager

    last_action: np.ndarray
    """The previous raw policy output, VERBATIM -- the whole vector across all
    action terms, pre-scale/offset/clip. That is what Isaac Lab's
    `last_action` observation returns (`env.action_manager.action`); with a
    second action term it is wider than the joint count, and feeding it the
    processed joint targets instead would silently shift every later element."""
