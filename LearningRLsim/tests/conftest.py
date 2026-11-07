"""Synthetic run configs, so the static tests need no export, no MuJoCo and no
`robot_description`.

The actuator numbers are the G1's real exported values; everything else is the
smallest config that exercises the code path.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

# The G1's left/right hip pitch, straight out of a real exported config.yaml.
HIP_STIFFNESS = 40.17923847
HIP_DAMPING = 2.55788977
HIP_ARMATURE = 0.01017752

JOINTS = ["left_hip_pitch_joint", "right_hip_pitch_joint", "left_knee_joint"]

# Every use site gets its OWN list. `copy.deepcopy` memoizes, so three keys
# sharing one list object would still share one copy afterwards -- and a test
# that edits `joint_names` to something unknown would silently edit
# `joint_order` to match, making the config valid again.

MINIMAL = {
    "policy_frequency_hz": 50.0,
    "sim_frequency_hz": 200.0,
    "sim_dt": 0.005,
    "decimation": 4,
    "robot": "unitree_g1",
    "task": "LearningRL-Test-v0-PLAY",
    "run": {"name": "test", "wandb_id": None, "checkpoint": "model_0.pt"},
    "observations": {
        "total_dim": 12,
        "terms": [
            {"name": "base_ang_vel", "func": "base_ang_vel", "dim": 3, "slice": [0, 3], "scale": 0.2, "clip": None},
            {
                "name": "velocity_commands",
                "func": "generated_commands",
                "dim": 3,
                "slice": [3, 6],
                "scale": 1.0,
                "clip": None,
                "params": {"command_name": "base_velocity"},
            },
            {"name": "joint_pos_rel", "func": "joint_pos_rel", "dim": 3, "slice": [6, 9], "scale": 1.0, "clip": None},
            {"name": "last_action", "func": "last_action", "dim": 3, "slice": [9, 12], "scale": 1.0, "clip": None},
        ],
    },
    "commands": [
        {
            "name": "base_velocity",
            "type": "UniformVelocityCommandCfg",
            "obs_slice": [3, 6],
            "ranges": {"lin_vel_x": [-1.0, 1.0], "lin_vel_y": [-1.0, 1.0], "ang_vel_z": [-1.0, 1.0], "heading": None},
            "heading_command": False,
            "rel_standing_envs": 0.02,
            "resampling_time_range": [10.0, 10.0],
        },
    ],
    "actions": {
        "total_dim": 3,
        "terms": [
            {
                "name": "JointPositionAction",
                "type": "JointPositionAction",
                "dim": 3,
                "slice": [0, 3],
                "joint_names": list(JOINTS),
                "scale": 0.25,
                "offset": [-0.1, -0.1, 0.3],
                "clip": [[-10.0, 10.0]] * 3,
            },
        ],
    },
    "default_joint_pos": {"joint_order": list(JOINTS), "values": [-0.1, -0.1, 0.3]},
    "actuators": {
        "joint_order": list(JOINTS),
        "stiffness": [HIP_STIFFNESS, HIP_STIFFNESS, 99.09842778],
        "damping": [HIP_DAMPING, HIP_DAMPING, 6.30880185],
        "armature": [HIP_ARMATURE, HIP_ARMATURE, 0.02510193],
        "friction": [0.0, 0.0, 0.0],
        "effort_limit": [88.0, 88.0, 139.0],
        "velocity_limit": [32.0, 32.0, 20.0],
    },
}

# The same config for the RoboParty RPO: three of its joints, its own robot key.
RPO_JOINTS = ["left_thigh_pitch_joint", "right_thigh_pitch_joint", "left_knee_joint"]
MINIMAL_RPO = copy.deepcopy(MINIMAL)
MINIMAL_RPO["robot"] = "roboparty_rpo"
for _key in ("default_joint_pos", "actuators"):
    MINIMAL_RPO[_key]["joint_order"] = list(RPO_JOINTS)
MINIMAL_RPO["actions"]["terms"][0]["joint_names"] = list(RPO_JOINTS)

# The same config with a second action term appended: two extra action dims, so
# the action vector is wider than the joint count.
TWO_ACTION_TERMS = copy.deepcopy(MINIMAL)
TWO_ACTION_TERMS["actions"]["total_dim"] = 5
TWO_ACTION_TERMS["actions"]["terms"].append(
    {
        "name": "hip_trim",
        "type": "JointPositionAction",
        "dim": 2,
        "slice": [3, 5],
        "joint_names": list(JOINTS[:2]),
        "scale": 0.1,
        "offset": [-0.1, -0.1],
    }
)
TWO_ACTION_TERMS["observations"]["total_dim"] = 14
TWO_ACTION_TERMS["observations"]["terms"][-1].update({"dim": 5, "slice": [9, 14]})

# The same config with LearningRL-G1-LocoManipulation-v0's hand command, appended
# to the observation exactly as the task appends it.
LOCOMANIP = copy.deepcopy(MINIMAL)
LOCOMANIP["task"] = "LearningRL-G1-LocoManipulation-v0-PLAY"
LOCOMANIP["observations"]["total_dim"] = 20
LOCOMANIP["observations"]["terms"].append(
    {
        "name": "hand_track",
        "func": "generated_commands",
        "dim": 8,
        "slice": [12, 20],
        "scale": 1.0,
        "clip": None,
        "params": {"command_name": "hand_track"},
    }
)
LOCOMANIP["commands"].append(
    {
        "name": "hand_track",
        "type": "HandPosCommandCfg",
        "obs_slice": [12, 20],
        "ranges": {
            "pos_x": [0.0, 0.4],
            "pos_y": [-0.45, 0.45],
            "pos_y_left": [0.0, 0.45],
            "pos_y_right": [-0.45, 0.0],
            "pos_z": [0.0, 0.6],
        },
        "rel_standing_envs": 0.1,
        "resampling_time_range": [0.5, 2.0],
    }
)


def write_run(tmp_path: Path, config: dict, name: str = "run") -> Path:
    """Write a config.yaml into a fresh directory and return it."""
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh)
    return run_dir


@pytest.fixture
def minimal_run(tmp_path: Path) -> Path:
    return write_run(tmp_path, MINIMAL)


@pytest.fixture
def rpo_run(tmp_path: Path) -> Path:
    return write_run(tmp_path, MINIMAL_RPO, name="rpo")


@pytest.fixture
def two_action_terms_run(tmp_path: Path) -> Path:
    return write_run(tmp_path, TWO_ACTION_TERMS, name="two_terms")


@pytest.fixture
def locomanip_run(tmp_path: Path) -> Path:
    return write_run(tmp_path, LOCOMANIP, name="locomanip")
