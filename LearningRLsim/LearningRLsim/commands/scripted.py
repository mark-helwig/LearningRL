"""`ScriptedSource`: a YAML timeline of command waypoints. Repeatable by construction.

Phase 1's only source, and the one every later phase is compared against: a
run driven by a file is reproducible, a run driven by a thumb is not.

The timeline is validated against the command terms the run's `config.yaml`
actually declares -- an unknown term name or field is an error, not an
ignored key -- and archived verbatim into the bag's `/meta/run`, so the bag
records what was commanded as well as what happened.

**Interpolation is step-hold by default.** That matches Isaac Lab's
resampling semantics, where a command is constant between resamples. Linear
interpolation is available per-term, never implicit: a ramped velocity
command is a different experiment from a stepped one, and silently choosing
one would make two bags incomparable for a reason nothing recorded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from LearningRLsim.commands.sources import KNOWN_EVENTS, CommandSource, Event
from LearningRLsim.commands.terms import CommandTerm
from LearningRLsim.runconfig import RunConfig

if TYPE_CHECKING:
    from LearningRLsim.context import EvalContext

INTERPOLATIONS = ("step", "linear")


class TimelineError(ValueError):
    """A malformed timeline, or one that does not fit the run's command terms."""


@dataclass(frozen=True)
class Waypoint:
    t: float
    fields: dict[str, float]


@dataclass(frozen=True)
class Track:
    """The waypoints for one command term."""

    term: str
    interpolation: str
    waypoints: tuple[Waypoint, ...]

    def sample(self, t: float) -> dict[str, float]:
        """Field values at sim time `t`.

        Before the first waypoint the first is held; after the last, the last.
        A rollout longer than its timeline coasts on the final command rather
        than snapping to zero, which would look like a deliberate stop.
        """
        points = self.waypoints
        if t <= points[0].t:
            return dict(points[0].fields)
        if t >= points[-1].t:
            return dict(points[-1].fields)

        index = 0
        for i in range(len(points) - 1):
            if points[i].t <= t < points[i + 1].t:
                index = i
                break
        lo, hi = points[index], points[index + 1]
        if self.interpolation == "step":
            return dict(lo.fields)

        span = hi.t - lo.t
        alpha = (t - lo.t) / span
        return {
            key: lo.fields[key] + alpha * (hi.fields.get(key, lo.fields[key]) - lo.fields[key]) for key in lo.fields
        }


BUILTIN_DIR = Path(__file__).resolve().parents[2] / "commands"
"""The timelines shipped with the repo, resolvable by bare name."""


def resolve_timeline(path: str | Path) -> Path:
    """A real path, or the name of a timeline shipped in `commands/`.

    `--commands walk_baseline` beats
    `--commands external/LearningRLsim/commands/walk_baseline.yaml`, which
    is what the in-container cwd would otherwise demand of every invocation.
    """
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    if candidate.parent == Path("."):
        for name in (candidate.name, f"{candidate.name}.yaml", f"{candidate.name}.yml"):
            builtin = BUILTIN_DIR / name
            if builtin.is_file():
                return builtin
    available = sorted(p.stem for p in BUILTIN_DIR.glob("*.y*ml")) if BUILTIN_DIR.is_dir() else []
    raise TimelineError(f"{path}: no such timeline file. Shipped timelines: {available}")


class ScriptedSource(CommandSource):
    def __init__(self, path: str | Path) -> None:
        self.path = resolve_timeline(path)
        with self.path.open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        if not isinstance(raw, dict):
            raise TimelineError(f"{self.path}: top level must be a mapping")

        self.raw = raw
        self.name = str(raw.get("name") or self.path.stem)
        self._duration = float(raw["duration"]) if raw.get("duration") is not None else None

        default_interp = str(raw.get("interpolation", "step"))
        if default_interp not in INTERPOLATIONS:
            raise TimelineError(f"{self.path}: interpolation {default_interp!r} is not one of {list(INTERPOLATIONS)}")

        commands_raw = raw.get("commands") or {}
        if not isinstance(commands_raw, dict):
            raise TimelineError(f"{self.path}.commands: must be a mapping of term name -> waypoints")
        self.tracks = {name: _build_track(name, spec, default_interp, self.path) for name, spec in commands_raw.items()}

        self.all_events = tuple(_build_events(raw.get("events") or [], self.path))
        self._fired = 0
        self._last_t = -1.0

    @property
    def duration(self) -> float | None:
        return self._duration

    def validate(self, run: RunConfig, terms: dict[str, CommandTerm]) -> None:
        unknown = sorted(set(self.tracks) - set(terms))
        if unknown:
            raise TimelineError(
                f"{self.path}: drives command term(s) {unknown}, which this run does not "
                f"declare. {run.path} declares: {sorted(terms)}"
            )
        for name, track in self.tracks.items():
            allowed = set(terms[name].FIELDS)
            for point in track.waypoints:
                bad = sorted(set(point.fields) - allowed)
                if bad:
                    raise TimelineError(
                        f"{self.path}: waypoint at t={point.t} for '{name}' sets {bad}, which "
                        f"{terms[name].cfg.type} has no such field(s). Fields: {sorted(allowed)}"
                    )
        undriven = sorted(set(terms) - set(self.tracks))
        if undriven:
            # Not an error: a command the timeline is silent about holds zero,
            # which is a legitimate "stand" for a velocity command. Said out
            # loud because a typo'd term name lands here.
            print(
                f"[LearningRLsim] timeline {self.name!r} drives none of {undriven}; "
                "they hold zero for the whole rollout."
            )

    def update(self, t: float, ctx: EvalContext) -> dict[str, dict[str, float]]:
        self._last_t = t
        return {name: track.sample(t) for name, track in self.tracks.items()}

    def events(self) -> list[Event]:
        """Every event whose time has passed since the last call.

        Keyed on the timeline's own ordering rather than a search, so an event
        is raised exactly once even if two land inside one policy step.
        """
        due = []
        while self._fired < len(self.all_events) and self.all_events[self._fired].t <= self._last_t:
            due.append(self.all_events[self._fired])
            self._fired += 1
        return due


def _build_track(name: str, spec: Any, default_interp: str, path: Path) -> Track:
    interpolation = default_interp
    if isinstance(spec, dict):
        interpolation = str(spec.get("interpolation", default_interp))
        if interpolation not in INTERPOLATIONS:
            raise TimelineError(
                f"{path}.commands.{name}.interpolation: {interpolation!r} is not one of {list(INTERPOLATIONS)}"
            )
        spec = spec.get("waypoints")

    if not isinstance(spec, list) or not spec:
        raise TimelineError(
            f"{path}.commands.{name}: expected a non-empty list of waypoints (or a mapping with `waypoints:`)"
        )

    waypoints = []
    for entry in spec:
        if not isinstance(entry, dict) or "t" not in entry:
            raise TimelineError(
                f"{path}.commands.{name}: each waypoint is a mapping with a `t` and one or "
                f"more command fields; got {entry!r}"
            )
        waypoints.append(
            Waypoint(
                t=float(entry["t"]),
                fields={k: float(v) for k, v in entry.items() if k != "t"},
            )
        )

    times = [w.t for w in waypoints]
    if times != sorted(times):
        raise TimelineError(f"{path}.commands.{name}: waypoint times are not in increasing order")
    if len(set(times)) != len(times):
        raise TimelineError(f"{path}.commands.{name}: two waypoints share a time")

    # Linear interpolation reads every field off the bracketing pair, so a
    # waypoint that omits one would inherit it from whichever side won --
    # silently turning a ramp into a step for that field alone.
    if interpolation == "linear":
        keys = set(waypoints[0].fields)
        for point in waypoints:
            if set(point.fields) != keys:
                raise TimelineError(
                    f"{path}.commands.{name}: interpolation is linear, so every waypoint must "
                    f"set the same fields; t={point.t} sets {sorted(point.fields)}, expected "
                    f"{sorted(keys)}."
                )

    return Track(term=name, interpolation=interpolation, waypoints=tuple(waypoints))


EVENT_PAYLOADS: dict[str, frozenset[str]] = {
    "reset": frozenset(),
    "stop": frozenset(),
    "mark": frozenset({"label"}),
    "place_object": frozenset({"ahead", "lateral"}),
}
"""What each event kind accepts beside `t` and `event`.

Checked rather than passed through, for the same reason `_accept()` exists in
`observations.py`: a typo'd `ahed: 0.55` would silently fall back to the
scene's default standoff, the sweep would not happen, and every bag in it
would look like a legitimate result.
"""


def _build_events(raw: Any, path: Path) -> list[Event]:
    if not isinstance(raw, list):
        raise TimelineError(f"{path}.events: must be a list")
    events = []
    for entry in raw:
        if not isinstance(entry, dict) or "t" not in entry or "event" not in entry:
            raise TimelineError(f"{path}.events: each entry is a mapping with `t` and `event`; got {entry!r}")
        kind = str(entry["event"])
        if kind not in KNOWN_EVENTS:
            raise TimelineError(f"{path}.events: unknown event {kind!r}. Known: {sorted(KNOWN_EVENTS)}")
        payload = {k: v for k, v in entry.items() if k not in ("t", "event")}
        allowed = EVENT_PAYLOADS[kind]
        bad = sorted(set(payload) - allowed)
        if bad:
            raise TimelineError(
                f"{path}.events: the {kind!r} at t={entry['t']} sets {bad}, which it has no such "
                f"field(s). Fields: {sorted(allowed)}"
            )
        events.append(Event(kind=kind, t=float(entry["t"]), payload=payload))
    events.sort(key=lambda e: e.t)
    return events
