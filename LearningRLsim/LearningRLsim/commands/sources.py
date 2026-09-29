"""Command SOURCES: what drives the terms, and the events they raise.

A source answers two separate questions each policy step:

1. **Term values** -- `update()` returns `{term_name: {field: value}}`, which
   the manager hands to the terms. This is what the policy consumes.
2. **Events** -- `events()` returns discrete things that are not term values
   at all: reset the robot, drop a marker into the bag, stop the rollout.
   Modelling these as commands (a magic "reset" value inside the velocity
   vector) is exactly the conflation this split exists to prevent.

`ScriptedSource` and `KeyboardSource` both attach here; a `GamepadSource`
would too, without the manager changing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from LearningRLsim.commands.terms import CommandTerm, build
from LearningRLsim.runconfig import RunConfig, RunConfigError

if TYPE_CHECKING:
    from LearningRLsim.context import EvalContext


@dataclass(frozen=True)
class Event:
    """A discrete instruction from a source, timestamped in sim seconds."""

    kind: str
    t: float
    payload: dict[str, Any] = field(default_factory=dict)


# Events the loop acts on. Anything else is recorded into the bag and
# otherwise ignored, so a source may raise its own without the loop knowing
# about them -- but a typo'd `reest` must not silently mean nothing, which is
# why `ScriptedSource` validates against this set at load.
KNOWN_EVENTS = frozenset({"reset", "mark", "stop", "place_object"})


class CommandSource(ABC):
    """Drives one or more command terms over the course of a rollout."""

    name: str = "unnamed"
    """Recorded in `/meta/run`. A bag that was driven by hand and one driven
    by a timeline are not the same evidence, and nothing else in the file
    distinguishes them."""

    @abstractmethod
    def update(self, t: float, ctx: EvalContext) -> dict[str, dict[str, float]]:
        """Field values per term name, for sim time `t`."""

    def events(self) -> list[Event]:
        """Events raised since the last call. Drained by the loop."""
        return []

    @property
    def duration(self) -> float | None:
        """When this source considers the rollout finished, if it has an opinion."""
        return None

    def validate(self, run: RunConfig, terms: dict[str, CommandTerm]) -> None:
        """Check this source against the terms the run actually declares."""

    def close(self) -> None:
        """Release anything held (a device handle, a thread)."""


class ZeroSource(CommandSource):
    """Every command held at zero -- stand in place.

    The default when no timeline is given: an explicit, reproducible
    stand-still rather than `rel_standing_envs`' per-env die roll, which is
    meaningless for a single robot.
    """

    name = "zero (stand)"

    def update(self, t: float, ctx: EvalContext) -> dict[str, dict[str, float]]:
        return {}


class CommandManager:
    """Owns the terms, and pumps whichever source is attached.

    `value(name)` is what the `generated_commands` observation reads.
    """

    def __init__(
        self,
        run: RunConfig,
        source: CommandSource | None = None,
        heading_stiffness: float = 1.0,
    ) -> None:
        self.run = run
        self.terms = build(run, heading_stiffness=heading_stiffness)
        self.source = source or ZeroSource()
        self.source.validate(run, self.terms)
        self._pending: list[Event] = []

    def value(self, name: str) -> np.ndarray:
        if name not in self.terms:
            raise RunConfigError(
                f"an observation term reads command {name!r}, which config.yaml does not "
                f"declare. Commands: {sorted(self.terms)}"
            )
        return self.terms[name].value()

    def update(self, t: float, ctx: EvalContext) -> None:
        """One policy step: pull the source, then let the terms post-process."""
        for term_name, fields in self.source.update(t, ctx).items():
            if term_name not in self.terms:
                raise RunConfigError(
                    f"command source drives {term_name!r}, which config.yaml does not "
                    f"declare. Commands: {sorted(self.terms)}"
                )
            self.terms[term_name].set_fields(fields)
        for term in self.terms.values():
            term.update(ctx)
        self._pending.extend(self.source.events())

    def drain_events(self) -> list[Event]:
        events, self._pending = self._pending, []
        return events

    def reset(self) -> None:
        for term in self.terms.values():
            term.reset()

    def close(self) -> None:
        self.source.close()
