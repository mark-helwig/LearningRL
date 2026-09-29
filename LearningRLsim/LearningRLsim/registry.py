"""Name-keyed registries: `config.yaml` names a term, exactly one entry answers.

The exported `config.yaml` identifies every term by a bare string -- `func:
base_ang_vel`, `type: JointGainAction`, `type: UniformVelocityCommandCfg`.
LearningRLsim holds a library keyed on exactly those strings and instantiates only
what a run's config names. That single mechanism is both invariant 1 (nothing
here reaches into Isaac Lab to ask what a term means) and invariant 3 (nothing
about a task is hardcoded; the config selects).

**An unrecognised name is a loud failure.** Never a silent skip and never a
zero fill: a missing observation term does not make the robot fall over, it
makes it walk slightly badly against a silently wrong observation vector,
which is indistinguishable from the sim2sim gap this harness exists to
measure. `lookup()` raises listing what IS registered, because the realistic
cause is a term added on the LearningRLlab side and not yet ported here.
"""

from __future__ import annotations

from collections.abc import Callable


class RegistryError(KeyError):
    """An unregistered name, or a duplicate registration.

    Subclasses KeyError so `except KeyError` at a call site still catches it,
    but carries a message naming the registry contents. `__str__` is
    overridden because KeyError's repr()s its argument, which mangles the
    multi-line message into one escaped blob.
    """

    def __str__(self) -> str:
        return self.args[0]


class Registry[T]:
    """One namespace of name -> entry, e.g. all observation functions."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._entries: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        """Decorator. `@observations.register("base_ang_vel")`."""

        def decorate(entry: T) -> T:
            if name in self._entries:
                raise RegistryError(
                    f"duplicate {self.kind} registration for {name!r}: already bound to "
                    f"{self._entries[name]!r}. Two entries answering one config name means "
                    f"the config cannot say which it meant."
                )
            self._entries[name] = entry
            return entry

        return decorate

    def lookup(self, name: str) -> T:
        if name not in self._entries:
            raise RegistryError(
                f"config.yaml names the {self.kind} {name!r}, which LearningRLsim does not "
                f"implement.\nRegistered {self.kind}s: {self.names()}\n"
                f"If LearningRLlab grew this term, port it here -- do not drop it: a missing "
                f"term silently shifts every later entry of the observation vector."
            )
        return self._entries[name]

    def names(self) -> list[str]:
        return sorted(self._entries)

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# The three namespaces, and their decorators.
#
# Instances live here rather than beside each library so that runconfig.py can
# validate a config against them without importing the libraries themselves
# (which would be a cycle: a term module wants runconfig's dataclasses).
# `LearningRLsim.library` is the one module that imports all three libraries for
# their registration side effects.
# ---------------------------------------------------------------------------
OBSERVATIONS: Registry = Registry("observation")
ACTIONS: Registry = Registry("action term")
COMMANDS: Registry = Registry("command term")

observation = OBSERVATIONS.register
action = ACTIONS.register
command = COMMANDS.register
