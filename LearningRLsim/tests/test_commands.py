"""Command terms, scripted timelines, and the term/source split."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from LearningRLsim import runconfig
from LearningRLsim.commands.scripted import ScriptedSource, TimelineError
from LearningRLsim.commands.sources import CommandManager, ZeroSource
from LearningRLsim.runconfig import RunConfigError

pytestmark = pytest.mark.static


def write_timeline(tmp_path, data, name="timeline.yaml"):
    path = tmp_path / name
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh)
    return path


def test_zero_source_stands(minimal_run):
    cfg = runconfig.load(minimal_run)
    manager = CommandManager(cfg, ZeroSource())
    manager.update(0.0, None)
    np.testing.assert_allclose(manager.value("base_velocity"), [0.0, 0.0, 0.0])


def test_step_hold_is_the_default(tmp_path, minimal_run):
    cfg = runconfig.load(minimal_run)
    path = write_timeline(
        tmp_path,
        {
            "name": "t",
            "duration": 10.0,
            "commands": {
                "base_velocity": [
                    {"t": 0.0, "lin_vel_x": 0.0},
                    {"t": 2.0, "lin_vel_x": 0.8},
                ]
            },
        },
    )
    manager = CommandManager(cfg, ScriptedSource(path))

    for t, expected in ((0.0, 0.0), (1.0, 0.0), (1.999, 0.0), (2.0, 0.8), (9.0, 0.8)):
        manager.update(t, None)
        assert manager.value("base_velocity")[0] == pytest.approx(expected), f"at t={t}"


def test_linear_interpolation_is_opt_in(tmp_path, minimal_run):
    cfg = runconfig.load(minimal_run)
    path = write_timeline(
        tmp_path,
        {
            "name": "t",
            "duration": 10.0,
            "commands": {
                "base_velocity": {
                    "interpolation": "linear",
                    "waypoints": [{"t": 0.0, "lin_vel_x": 0.0}, {"t": 2.0, "lin_vel_x": 0.8}],
                }
            },
        },
    )
    manager = CommandManager(cfg, ScriptedSource(path))
    manager.update(1.0, None)
    assert manager.value("base_velocity")[0] == pytest.approx(0.4)


def test_linear_requires_every_waypoint_to_set_the_same_fields(tmp_path):
    path = write_timeline(
        tmp_path,
        {
            "name": "t",
            "commands": {
                "base_velocity": {
                    "interpolation": "linear",
                    "waypoints": [
                        {"t": 0.0, "lin_vel_x": 0.0, "ang_vel_z": 0.0},
                        {"t": 2.0, "lin_vel_x": 0.8},
                    ],
                }
            },
        },
    )
    with pytest.raises(TimelineError, match="same fields"):
        ScriptedSource(path)


def test_the_last_waypoint_is_held_not_zeroed(tmp_path, minimal_run):
    """A rollout longer than its timeline coasts; snapping to zero would look
    like a commanded stop."""
    cfg = runconfig.load(minimal_run)
    path = write_timeline(
        tmp_path,
        {
            "name": "t",
            "commands": {"base_velocity": [{"t": 0.0, "lin_vel_x": 0.5}]},
        },
    )
    manager = CommandManager(cfg, ScriptedSource(path))
    manager.update(99.0, None)
    assert manager.value("base_velocity")[0] == pytest.approx(0.5)


def test_timeline_driving_an_undeclared_term_is_rejected(tmp_path, minimal_run):
    cfg = runconfig.load(minimal_run)
    path = write_timeline(
        tmp_path,
        {
            "name": "t",
            "commands": {"pose_command": [{"t": 0.0, "x": 1.0}]},
        },
    )
    with pytest.raises(TimelineError, match="pose_command"):
        CommandManager(cfg, ScriptedSource(path))


def test_timeline_setting_an_unknown_field_is_rejected(tmp_path, minimal_run):
    cfg = runconfig.load(minimal_run)
    path = write_timeline(
        tmp_path,
        {
            "name": "t",
            "commands": {"base_velocity": [{"t": 0.0, "lin_vel_z": 1.0}]},
        },
    )
    with pytest.raises(TimelineError, match="lin_vel_z"):
        CommandManager(cfg, ScriptedSource(path))


def test_unknown_event_kind_is_rejected(tmp_path):
    path = write_timeline(
        tmp_path,
        {
            "name": "t",
            "commands": {"base_velocity": [{"t": 0.0, "lin_vel_x": 0.0}]},
            "events": [{"t": 1.0, "event": "reest"}],
        },
    )
    with pytest.raises(TimelineError, match="reest"):
        ScriptedSource(path)


def test_place_object_is_a_known_event(tmp_path, minimal_run):
    cfg = runconfig.load(minimal_run)
    path = write_timeline(
        tmp_path,
        {
            "name": "t",
            "commands": {"base_velocity": [{"t": 0.0, "lin_vel_x": 0.0}]},
            "events": [{"t": 1.0, "event": "place_object", "ahead": 0.7, "lateral": -0.1}],
        },
    )
    manager = CommandManager(cfg, ScriptedSource(path))
    manager.update(1.5, None)
    (event,) = manager.drain_events()
    assert event.kind == "place_object"
    assert event.payload == {"ahead": 0.7, "lateral": -0.1}


def test_an_unknown_event_payload_key_names_the_allowed_ones(tmp_path):
    """A typo'd `ahed` would silently take the scene's default standoff, the
    sweep would not happen, and every bag in it would look legitimate."""
    path = write_timeline(
        tmp_path,
        {
            "name": "t",
            "commands": {"base_velocity": [{"t": 0.0, "lin_vel_x": 0.0}]},
            "events": [{"t": 1.0, "event": "place_object", "ahed": 0.7}],
        },
    )
    with pytest.raises(TimelineError, match=r"\['ahead', 'lateral'\]"):
        ScriptedSource(path)


def test_every_shipped_timeline_parses():
    from LearningRLsim.commands.scripted import BUILTIN_DIR

    for path in sorted(BUILTIN_DIR.glob("*.y*ml")):
        source = ScriptedSource(path)
        assert source.tracks or source.all_events


def test_out_of_order_waypoints_are_rejected(tmp_path):
    path = write_timeline(
        tmp_path,
        {
            "name": "t",
            "commands": {
                "base_velocity": [
                    {"t": 2.0, "lin_vel_x": 0.0},
                    {"t": 1.0, "lin_vel_x": 0.0},
                ]
            },
        },
    )
    with pytest.raises(TimelineError, match="increasing order"):
        ScriptedSource(path)


def test_events_fire_once_each_in_order(tmp_path, minimal_run):
    cfg = runconfig.load(minimal_run)
    path = write_timeline(
        tmp_path,
        {
            "name": "t",
            "commands": {"base_velocity": [{"t": 0.0, "lin_vel_x": 0.0}]},
            "events": [{"t": 1.0, "event": "mark"}, {"t": 2.0, "event": "stop"}],
        },
    )
    manager = CommandManager(cfg, ScriptedSource(path))

    manager.update(0.5, None)
    assert manager.drain_events() == []
    manager.update(1.5, None)
    assert [e.kind for e in manager.drain_events()] == ["mark"]
    manager.update(1.6, None)
    assert manager.drain_events() == []  # not re-raised
    manager.update(2.5, None)
    assert [e.kind for e in manager.drain_events()] == ["stop"]


def test_obs_slice_width_must_match_the_term(tmp_path):
    import copy

    from tests.conftest import MINIMAL, write_run

    config = copy.deepcopy(MINIMAL)
    config["commands"][0]["obs_slice"] = [3, 7]
    cfg = runconfig.load(write_run(tmp_path, config, name="width"))
    with pytest.raises(RunConfigError, match="produces 3 values"):
        CommandManager(cfg)


def test_rel_standing_envs_is_read_but_never_sampled(minimal_run):
    """Parsed for the record, deliberately not acted on: it is a per-env
    fraction across a vectorised env and would make a scripted rollout
    non-reproducible."""
    cfg = runconfig.load(minimal_run)
    assert cfg.commands[0].rel_standing_envs == pytest.approx(0.02)

    manager = CommandManager(cfg, ZeroSource())
    for t in np.linspace(0.0, 5.0, 50):
        manager.update(float(t), None)
        np.testing.assert_allclose(manager.value("base_velocity"), [0.0, 0.0, 0.0])
