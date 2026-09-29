"""The keyboard command source: drive the robot by hand from the terminal.

Same interface as `ScriptedSource`, so the rollout, the bag and the live
stream do not know the difference -- a hand-driven run produces the same ten
channels as a scripted one, and `/meta/run` records which it was.

**Stepped latch, not hold-to-move.** A terminal in cbreak mode delivers
keypresses, not press/release edges: a held key arrives as the terminal's own
autorepeat, which starts after ~500 ms and repeats at a rate the user's
desktop decides. Driving velocity off that would make the same session feel
different on two machines and make nothing reproducible. So each press steps
its axis by a fixed amount and the command HOLDS until changed -- which is
exactly what a scripted timeline does between waypoints, and means a session
that produced something interesting can be written down as one.

Commands are clipped to the ranges `config.yaml` exports for the term. Driving
a policy outside the envelope it was trained in is not a sim2sim finding, it
is an out-of-distribution query, and the clip keeps the two apart.

No new dependency: `termios` and `select` are standard library, which also
means the whole interactive path runs on a CPU-only host with no container.
"""

from __future__ import annotations

import os
import select
import sys
import termios
import threading
import tty
from typing import TYPE_CHECKING

import numpy as np

from LearningRLsim.commands.sources import CommandSource, Event
from LearningRLsim.commands.terms import CommandTerm, wrap_to_pi
from LearningRLsim.runconfig import RunConfig

if TYPE_CHECKING:
    from LearningRLsim.context import EvalContext

DEFAULT_STEP = 0.1
DEFAULT_YAW_STEP = 0.2
DEFAULT_HEADING_STEP = 0.2

POLL_INTERVAL = 0.05

ARROWS = {"[A": "w", "[B": "s", "[D": "q", "[C": "e"}
"""An arrow key arrives as ESC [ A. Mapped onto the letter bindings rather
than given their own, so there is one set of behaviour to reason about."""

BINDINGS = """
  w / s        forward / back        (lin_vel_x)
  a / d        left / right          (lin_vel_y)
  q / e        turn left / right     (ang_vel_z)
  arrows       same as w/a/s/d, turn on left/right
  space        all commands to zero (stand)
  r            reset the robot to its spawn pose
  b            place the scene's deferred object(s) in front of the robot
  m            drop a marker into the bag
  x            end the rollout
  h            print these bindings
"""


class KeyboardError(RuntimeError):
    """The keyboard cannot be used for this run."""


class KeyboardSource(CommandSource):
    """Reads single keypresses off stdin on a background thread."""

    def __init__(
        self,
        step: float = DEFAULT_STEP,
        yaw_step: float = DEFAULT_YAW_STEP,
        term: str | None = None,
        stream: object | None = None,
    ) -> None:
        self.step = float(step)
        self.yaw_step = float(yaw_step)
        self.name = "keyboard"
        self._term_name = term
        self._stream = stream or sys.stdin

        # Before `fileno()`, which a non-tty stream may not even have.
        if not self._stream.isatty():
            raise KeyboardError(
                "--keyboard needs an interactive terminal, and stdin is not one. "
                "Run it from a real shell: a piped or nohup'd invocation lands here."
            )
        self._fd = self._stream.fileno()

        self._lock = threading.Lock()
        self._fields: dict[str, float] = {}
        self._events: list[Event] = []
        self._limits: dict[str, tuple[float, float]] = {}
        self._heading = False
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._saved: list | None = None
        self._t = 0.0

    # -- CommandSource -------------------------------------------------------
    def validate(self, run: RunConfig, terms: dict[str, CommandTerm]) -> None:
        """Pick the term this drives, and take its clipping ranges."""
        if self._term_name is not None:
            if self._term_name not in terms:
                raise KeyboardError(
                    f"--keyboard-term {self._term_name!r} is not a command of this run: {sorted(terms)}"
                )
            name = self._term_name
        else:
            drivable = [n for n, term in terms.items() if "lin_vel_x" in term.FIELDS]
            if not drivable:
                raise KeyboardError(
                    f"no command term of this run takes a velocity, so there is nothing for the "
                    f"keyboard to drive. Commands: {sorted(terms)}"
                )
            name = drivable[0]

        self._term_name = name
        cfg = terms[name].cfg
        self._heading = bool(cfg.heading_command)
        for field, bounds in (cfg.ranges or {}).items():
            if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                self._limits[field] = (float(bounds[0]), float(bounds[1]))

        driven = ["lin_vel_x", "lin_vel_y", "heading" if self._heading else "ang_vel_z"]
        with self._lock:
            self._fields = dict.fromkeys(driven, 0.0)

    @property
    def command(self) -> dict[str, float]:
        """What this source is currently commanding."""
        with self._lock:
            return dict(self._fields)

    def update(self, t: float, ctx: EvalContext) -> dict[str, dict[str, float]]:
        self._t = t
        if self._thread is None:
            self._start()
        return {self._term_name: self.command}

    def events(self) -> list[Event]:
        with self._lock:
            events, self._events = self._events, []
        return events

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._restore()

    # -- the reader thread ---------------------------------------------------
    def _start(self) -> None:
        self._saved = termios.tcgetattr(self._fd)
        # cbreak, not raw: ISIG survives, so Ctrl-C still interrupts a rollout
        # that has run away -- which is the one key binding nobody should have
        # to look up.
        tty.setcbreak(self._fd)
        print(f"[LearningRLsim] keyboard driving '{self._term_name}':{BINDINGS}")
        self._thread = threading.Thread(target=self._read_loop, name="LearningRLsim-keyboard", daemon=True)
        self._thread.start()

    def _restore(self) -> None:
        if self._saved is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
            self._saved = None

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            if not select.select([self._fd], [], [], POLL_INTERVAL)[0]:
                continue
            key = os.read(self._fd, 1).decode("utf-8", "replace")
            if key == "\x1b":
                key = self._escape_sequence()
            self.press(key)

    def _escape_sequence(self) -> str:
        """An arrow key arrives as ESC [ A; a bare ESC is a bare ESC.

        Read whatever else is already buffered rather than waiting for it: if
        nothing follows within the terminal's own delivery of the sequence,
        the user pressed Escape, which ends the rollout like `x`.
        """
        rest = ""
        while select.select([self._fd], [], [], 0.01)[0]:
            rest += os.read(self._fd, 1).decode("utf-8", "replace")
            if rest.startswith("[") and len(rest) >= 2:
                break
        return ARROWS.get(rest, "x")

    # -- key handling (pure, so it tests without a terminal) -----------------
    def press(self, key: str) -> None:
        """Apply one keypress. Public because it is the whole behaviour."""
        turn = "heading" if self._heading else "ang_vel_z"
        turn_step = DEFAULT_HEADING_STEP if self._heading else self.yaw_step
        deltas = {
            "w": ("lin_vel_x", self.step),
            "s": ("lin_vel_x", -self.step),
            "a": ("lin_vel_y", self.step),
            "d": ("lin_vel_y", -self.step),
            "q": (turn, turn_step),
            "e": (turn, -turn_step),
        }

        with self._lock:
            if key in deltas:
                field, delta = deltas[key]
                self._fields[field] = self._clip(field, self._fields[field] + delta)
                self._echo()
                return
            if key == " ":
                self._fields = dict.fromkeys(self._fields, 0.0)
                self._echo()
                return
            if key == "r":
                self._fields = dict.fromkeys(self._fields, 0.0)
                self._events.append(Event("reset", self._t))
            elif key == "b":
                # The whole of what makes scene objects work in a hand-driven
                # session: the rollout cannot tell this from a timeline
                # waypoint, because it is the same event.
                self._events.append(Event("place_object", self._t))
            elif key == "m":
                self._events.append(Event("mark", self._t, {"label": "keyboard"}))
            elif key in ("x", "\x1b"):
                self._events.append(Event("stop", self._t))
            elif key in ("h", "?"):
                print(BINDINGS)

    def _clip(self, field: str, value: float) -> float:
        if field == "heading":
            # Wrapped, not clipped: a heading is an angle, and clipping one at
            # +pi would make "turn left" stop working halfway round.
            return wrap_to_pi(value)
        lo, hi = self._limits.get(field, (-np.inf, np.inf))
        return float(np.clip(value, lo, hi))

    def _echo(self) -> None:
        """One line per change, carriage-returned in place.

        The terminal is in cbreak so nothing is echoed; without this the robot
        moves and the screen says nothing about why.
        """
        parts = " ".join(f"{name}={value:+.2f}" for name, value in self._fields.items())
        print(f"\r[LearningRLsim] {parts}    ", end="", flush=True)
