"""The keyboard source, driven through `press()` rather than a terminal.

`press` is the whole behaviour -- the reader thread does nothing but turn
bytes into calls to it -- so these run anywhere, with no tty.
"""

from __future__ import annotations

import io

import pytest

from LearningRLsim import runconfig
from LearningRLsim.commands.keyboard import ARROWS, KeyboardError, KeyboardSource
from LearningRLsim.commands.terms import build

pytestmark = pytest.mark.static


class FakeTTY(io.StringIO):
    """A stream that claims to be a terminal. `press()` never touches the fd,
    and `update()` is the only thing that would."""

    def isatty(self):
        return True

    def fileno(self):
        return -1


@pytest.fixture
def driver(minimal_run):
    def make(**kwargs):
        cfg = runconfig.load(minimal_run)
        source = KeyboardSource(stream=FakeTTY(), **kwargs)
        source.validate(cfg, build(cfg))
        return source

    return make


def test_a_press_steps_its_axis_and_the_command_holds(driver):
    source = driver(step=0.1)
    source.press("w")
    source.press("w")
    assert source.command["lin_vel_x"] == pytest.approx(0.2)
    source.press("m")  # an event must not disturb the command
    assert source.command["lin_vel_x"] == pytest.approx(0.2)


def test_the_axes_are_independent_and_signed(driver):
    source = driver(step=0.25, yaw_step=0.5)
    source.press("s")
    source.press("a")
    source.press("e")
    assert source.command == pytest.approx({"lin_vel_x": -0.25, "lin_vel_y": 0.25, "ang_vel_z": -0.5})


def test_space_stands_the_robot_up(driver):
    source = driver()
    for key in "waq":
        source.press(key)
    source.press(" ")
    assert set(source.command.values()) == {0.0}


def test_commands_are_clipped_to_the_exported_ranges(driver):
    """Outside the trained envelope is an out-of-distribution query, not a
    sim2sim finding, and the clip keeps the two apart."""
    source = driver(step=1.0)
    for _ in range(20):
        source.press("w")
    assert source.command["lin_vel_x"] == pytest.approx(1.0)
    for _ in range(40):
        source.press("s")
    assert source.command["lin_vel_x"] == pytest.approx(-1.0)


def test_events_are_raised_in_order_and_drained_once(driver):
    source = driver()
    source.press("r")
    source.press("m")
    source.press("x")
    assert [event.kind for event in source.events()] == ["reset", "mark", "stop"]
    assert source.events() == []


def test_b_places_the_deferred_objects(driver):
    """The whole of what makes scene objects work by hand: the rollout cannot
    tell this from a timeline waypoint, because it is the same event."""
    source = driver()
    source.press("b")
    events = source.events()
    assert [event.kind for event in events] == ["place_object"]
    assert events[0].payload == {}


def test_reset_also_zeroes_the_command(driver):
    """The robot goes back to its spawn pose; a latched 1 m/s would have it
    sprint straight out of the reset."""
    source = driver()
    source.press("w")
    source.press("r")
    assert set(source.command.values()) == {0.0}


def test_an_unbound_key_does_nothing(driver):
    source = driver()
    source.press("z")
    assert set(source.command.values()) == {0.0}
    assert source.events() == []


def test_arrows_map_onto_the_letter_bindings():
    assert set(ARROWS.values()) == {"w", "s", "q", "e"}


def test_a_non_tty_says_what_to_do_about_it():
    with pytest.raises(KeyboardError, match="interactive terminal"):
        KeyboardSource(stream=io.StringIO())


def test_a_run_with_no_velocity_command_is_a_clear_failure(minimal_run):
    cfg = runconfig.load(minimal_run)
    with pytest.raises(KeyboardError, match="nothing for the"):
        KeyboardSource(stream=FakeTTY()).validate(cfg, {})


def test_an_unknown_term_lists_what_the_run_has(minimal_run):
    cfg = runconfig.load(minimal_run)
    source = KeyboardSource(stream=FakeTTY(), term="not_a_command")
    with pytest.raises(KeyboardError, match="not a command of this run"):
        source.validate(cfg, build(cfg))
