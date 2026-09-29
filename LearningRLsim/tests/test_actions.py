"""Action terms: the position map, its clip, and the nominal actuator command."""

from __future__ import annotations

import numpy as np
import pytest

from LearningRLsim import runconfig
from LearningRLsim.actions import ActuatorCommand, JointPositionAction, build
from tests.conftest import HIP_DAMPING, HIP_STIFFNESS

pytestmark = pytest.mark.static


def test_joint_position_action_is_scale_offset_then_clip(minimal_run):
    cfg = runconfig.load(minimal_run)
    term = build(cfg)[0]
    assert isinstance(term, JointPositionAction)

    command = ActuatorCommand.nominal(cfg)
    raw = np.array([1.0, -1.0, 0.0])
    term.process(raw)
    term.apply(command)
    np.testing.assert_allclose(command.joint_pos_target, raw * 0.25 + cfg.default_joint_pos)


def test_position_clip_binds_the_processed_value_not_the_raw_action(minimal_run):
    """The clip is (-10, 10) RADIANS. A raw action of 100 scales to 25 rad and
    is clipped to 10 -- not clipped to 10 first and then scaled to 2.5."""
    cfg = runconfig.load(minimal_run)
    term = build(cfg)[0]
    command = ActuatorCommand.nominal(cfg)

    term.process(np.array([100.0, 0.0, 0.0]))
    term.apply(command)
    assert command.joint_pos_target[0] == pytest.approx(10.0)


def test_reset_action_is_zero_for_a_position_term(minimal_run):
    cfg = runconfig.load(minimal_run)
    np.testing.assert_allclose(build(cfg)[0].reset_action, np.zeros(3))


def test_the_nominal_command_carries_the_exported_gains(minimal_run):
    """Reset gains come straight from `actuators`, in `joint_order`."""
    cfg = runconfig.load(minimal_run)
    command = ActuatorCommand.nominal(cfg)
    np.testing.assert_allclose(command.stiffness[:2], [HIP_STIFFNESS, HIP_STIFFNESS])
    np.testing.assert_allclose(command.damping[:2], [HIP_DAMPING, HIP_DAMPING])
    np.testing.assert_allclose(command.joint_pos_target, cfg.default_joint_pos)


def test_a_second_term_drives_its_own_joints(two_action_terms_run):
    cfg = runconfig.load(two_action_terms_run)
    first, second = build(cfg)
    command = ActuatorCommand.nominal(cfg)
    first.process(np.zeros(3))
    second.process(np.array([1.0, -1.0]))
    first.apply(command)
    second.apply(command)
    # Declaration order: the second term overwrites the hips the first one wrote.
    np.testing.assert_allclose(command.joint_pos_target, [0.0, -0.2, 0.3])
