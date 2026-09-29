"""Validation must reject configs that would otherwise produce a robot that
merely walks badly."""

from __future__ import annotations

import copy

import pytest

from LearningRLsim import runconfig
from LearningRLsim.registry import RegistryError
from LearningRLsim.runconfig import RunConfigError
from tests.conftest import MINIMAL, TWO_ACTION_TERMS, write_run

pytestmark = pytest.mark.static


def test_loads_minimal(minimal_run):
    cfg = runconfig.load(minimal_run)
    assert cfg.task == "LearningRL-Test-v0-PLAY"
    assert cfg.num_joints == 3
    assert cfg.observations.total_dim == 12
    assert [t.func for t in cfg.observations.terms] == [
        "base_ang_vel",
        "generated_commands",
        "joint_pos_rel",
        "last_action",
    ]


def test_loads_the_rpo(rpo_run):
    cfg = runconfig.load(rpo_run)
    assert cfg.robot == "roboparty_rpo"
    assert cfg.joint_order[0] == "left_thigh_pitch_joint"


def test_more_actions_than_joints_loads(two_action_terms_run):
    cfg = runconfig.load(two_action_terms_run)
    # The whole point of the trap: more actions than joints.
    assert cfg.actions.total_dim == 5
    assert cfg.num_joints == 3


def test_missing_config_names_the_nesting(tmp_path):
    with pytest.raises(RunConfigError, match="nests one level"):
        runconfig.load(tmp_path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        # A gap: term 2 starts at 4 instead of 3.
        (lambda c: c["observations"]["terms"][1].update({"slice": [4, 7]}), "contiguously"),
        # An overlap.
        (lambda c: c["observations"]["terms"][1].update({"slice": [2, 5]}), "contiguously"),
        # dim disagrees with the slice width.
        (lambda c: c["observations"]["terms"][0].update({"dim": 4}), "spans"),
        # Terms stop short of total_dim.
        (lambda c: c.__setitem__("observations", {**c["observations"], "total_dim": 13}), "total_dim is 13"),
    ],
)
def test_bad_observation_slices_are_rejected(tmp_path, mutate, message):
    config = copy.deepcopy(MINIMAL)
    mutate(config)
    with pytest.raises(RunConfigError, match=message):
        runconfig.load(write_run(tmp_path, config, name="bad"))


def test_action_slices_are_validated_too(tmp_path):
    config = copy.deepcopy(TWO_ACTION_TERMS)
    config["actions"]["terms"][1]["slice"] = [4, 6]
    with pytest.raises(RunConfigError, match="contiguously"):
        runconfig.load(write_run(tmp_path, config, name="bad"))


def test_disagreeing_joint_orders_are_rejected(tmp_path):
    config = copy.deepcopy(MINIMAL)
    config["actuators"]["joint_order"] = list(reversed(config["actuators"]["joint_order"]))
    with pytest.raises(RunConfigError, match="actuators.joint_order differs"):
        runconfig.load(write_run(tmp_path, config, name="bad"))


def test_action_term_naming_an_unknown_joint_is_rejected(tmp_path):
    config = copy.deepcopy(MINIMAL)
    config["actions"]["terms"][0]["joint_names"][2] = "not_a_joint"
    with pytest.raises(RunConfigError, match="not_a_joint"):
        runconfig.load(write_run(tmp_path, config, name="bad"))


def test_inconsistent_rates_are_rejected(tmp_path):
    config = copy.deepcopy(MINIMAL)
    config["decimation"] = 5  # 200/5 = 40 Hz, not the declared 50
    with pytest.raises(RunConfigError, match="policy_frequency_hz"):
        runconfig.load(write_run(tmp_path, config, name="bad"))


def test_history_length_is_rejected_rather_than_flattened(tmp_path):
    config = copy.deepcopy(MINIMAL)
    config["observations"]["terms"][0]["history_length"] = 4
    with pytest.raises(RunConfigError, match="history"):
        runconfig.load(write_run(tmp_path, config, name="bad"))


# -- the loud-failure contract ---------------------------------------------
def test_unknown_observation_func_lists_the_registry(tmp_path):
    config = copy.deepcopy(MINIMAL)
    config["observations"]["terms"][0]["func"] = "base_ang_vel_typo"
    with pytest.raises(RegistryError) as excinfo:
        runconfig.load(write_run(tmp_path, config, name="bad"))
    message = str(excinfo.value)
    assert "base_ang_vel_typo" in message
    # It must say what IS available, not merely that this one is not.
    assert "projected_gravity" in message
    assert "joint_pos_rel" in message


def test_unknown_action_type_lists_the_registry(tmp_path):
    config = copy.deepcopy(MINIMAL)
    config["actions"]["terms"][0]["type"] = "JointVelocityAction"
    with pytest.raises(RegistryError, match="JointPositionAction"):
        runconfig.load(write_run(tmp_path, config, name="bad"))


def test_unknown_command_type_lists_the_registry(tmp_path):
    config = copy.deepcopy(MINIMAL)
    config["commands"][0]["type"] = "NormalVelocityCommandCfg"
    with pytest.raises(RegistryError, match="UniformVelocityCommandCfg"):
        runconfig.load(write_run(tmp_path, config, name="bad"))
