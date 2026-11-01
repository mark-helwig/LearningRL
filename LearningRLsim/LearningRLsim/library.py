"""Imports every term library for its registration side effects.

One module to import when you want the registries populated -- `runconfig`'s
validation and `eval`'s `--list-terms` both go through here, so neither has to
know which modules exist. Adding a library means adding one import here.

Re-exports the registries themselves so callers need only this module.
"""

from __future__ import annotations

from LearningRLsim import actions as _actions  # noqa: F401  -- registration side effect
from LearningRLsim import observations as _observations  # noqa: F401
from LearningRLsim.commands import motion as _motion  # noqa: F401
from LearningRLsim.commands import terms as _command_terms  # noqa: F401
from LearningRLsim.registry import ACTIONS, COMMANDS, OBSERVATIONS

__all__ = ["ACTIONS", "COMMANDS", "OBSERVATIONS", "summary"]


def summary() -> str:
    """Human-readable listing of everything registered."""
    return "\n".join(
        f"{registry.kind}s ({len(registry)}): {', '.join(registry.names())}"
        for registry in (OBSERVATIONS, ACTIONS, COMMANDS)
    )
