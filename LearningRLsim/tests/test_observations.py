"""Observation assembly: every term at its declared slice, clip before scale."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from LearningRLsim import runconfig
from LearningRLsim.context import EvalContext
from LearningRLsim.observations import assemble
from LearningRLsim.runconfig import RunConfigError

pytestmark = pytest.mark.static


@dataclass
class FakeSim:
    """The state an observation term reads -- no MuJoCo needed to test layout."""

    joint_pos: np.ndarray
    joint_vel: np.ndarray
    base_ang_vel_b: np.ndarray
    base_lin_vel_b: np.ndarray
    projected_gravity_b: np.ndarray
    heading_w: float = 0.0


class FakeCommands:
    def __init__(self, values):
        self.values = values

    def value(self, name):
        return np.asarray(self.values[name], dtype=np.float64)


def make_ctx(cfg, **overrides):
    n = cfg.num_joints
    sim = FakeSim(
        joint_pos=overrides.get("joint_pos", np.arange(n, dtype=float)),
        joint_vel=overrides.get("joint_vel", np.zeros(n)),
        base_ang_vel_b=overrides.get("base_ang_vel_b", np.array([1.0, 2.0, 3.0])),
        base_lin_vel_b=np.array([0.0, 0.0, 0.0]),
        projected_gravity_b=np.array([0.0, 0.0, -1.0]),
    )
    return EvalContext(
        cfg=cfg,
        sim=sim,
        commands=FakeCommands({"base_velocity": [0.5, 0.0, -0.25]}),
        last_action=overrides.get("last_action", np.arange(cfg.actions.total_dim, dtype=float) * 10.0),
    )


def test_every_term_lands_at_its_declared_slice(minimal_run):
    cfg = runconfig.load(minimal_run)
    ctx = make_ctx(cfg)
    obs = assemble(ctx)

    assert obs.shape == (cfg.observations.total_dim,)
    # base_ang_vel, scale 0.2
    np.testing.assert_allclose(obs[0:3], np.array([1.0, 2.0, 3.0]) * 0.2)
    # generated_commands, scale 1
    np.testing.assert_allclose(obs[3:6], [0.5, 0.0, -0.25])
    # joint_pos_rel = q - default
    np.testing.assert_allclose(obs[6:9], np.arange(3) - cfg.default_joint_pos)
    # last_action, verbatim
    np.testing.assert_allclose(obs[9:12], [0.0, 10.0, 20.0])


def test_last_action_is_the_full_raw_vector_not_the_joint_count(two_action_terms_run):
    """5 here -- never the 3 joints."""
    cfg = runconfig.load(two_action_terms_run)
    term = next(t for t in cfg.observations.terms if t.func == "last_action")
    assert term.dim == cfg.actions.total_dim == 5
    assert term.dim != cfg.num_joints

    ctx = make_ctx(cfg)
    obs = assemble(ctx)
    np.testing.assert_allclose(obs[9:14], np.arange(5) * 10.0)


def test_clip_is_applied_before_scale(tmp_path):
    """Backwards, this is silent: clip-then-scale of 5.0 at clip 2 / scale 10
    gives 20; scale-then-clip gives 2."""
    import copy

    from tests.conftest import MINIMAL, write_run

    config = copy.deepcopy(MINIMAL)
    config["observations"]["terms"][0].update({"clip": [-2.0, 2.0], "scale": 10.0})
    cfg = runconfig.load(write_run(tmp_path, config, name="clip"))

    ctx = make_ctx(cfg, base_ang_vel_b=np.array([5.0, -5.0, 0.5]))
    obs = assemble(ctx)
    np.testing.assert_allclose(obs[0:3], [20.0, -20.0, 5.0])


def test_a_term_returning_the_wrong_width_is_rejected(minimal_run):
    cfg = runconfig.load(minimal_run)
    ctx = make_ctx(cfg)
    ctx.commands = FakeCommands({"base_velocity": [0.5, 0.0]})  # 2, not 3
    with pytest.raises(RunConfigError, match="produced 2 values"):
        assemble(ctx)


def test_an_unhonoured_param_is_rejected_not_ignored(tmp_path):
    """`asset_cfg` with a joint subset would change the term's meaning while
    keeping its shape plausible."""
    import copy

    from tests.conftest import MINIMAL, write_run

    config = copy.deepcopy(MINIMAL)
    config["observations"]["terms"][2]["params"] = {"asset_cfg": {"joint_ids": [0, 1, 2]}}
    cfg = runconfig.load(write_run(tmp_path, config, name="params"))
    with pytest.raises(RunConfigError, match="asset_cfg"):
        assemble(make_ctx(cfg))


def test_generated_commands_without_a_command_name_is_rejected(tmp_path):
    import copy

    from tests.conftest import MINIMAL, write_run

    config = copy.deepcopy(MINIMAL)
    config["observations"]["terms"][1].pop("params")
    cfg = runconfig.load(write_run(tmp_path, config, name="nocmd"))
    with pytest.raises(RunConfigError, match="command_name"):
        assemble(make_ctx(cfg))
