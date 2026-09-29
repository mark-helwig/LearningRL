"""Isaac Lab `ArticulationCfg` for the Unitree G1 (29-DoF, `unitreerobotics/unitree_ros`
rev `7d6075f7f58588b189b940130e3edab3c839b2df`, `g1_29dof_rev_1_0.urdf`).

Requires Isaac Lab (used inside the Isaac Lab container only). Joint names,
actuator-group regex patterns, and physical limits live in `joint_groups.py`
instead, which has NO Isaac Lab dependency and stays readable and diffable on
its own. Do not move that data back into this file.

Spawns directly from the vendored URDF via `RobotUrdfFileCfg` -- no
separate offline USD build step; Isaac Lab converts on the fly at spawn time
and caches under `paths.robot_build_dir("unitree_g1")`.

`G1_CFG` uses `ImplicitActuatorCfg` (solver-side PD), matching upstream's
Newton-supported `Isaac-Velocity-Flat-G1` actuator model.

Actuator-group regexes, effort/velocity limits, and PD gains are defined once
in `joint_groups.py` (from `UNITREE_G1_29DOF_MIMIC_CFG` in
`unitreerobotics/unitree_rl_lab`).
"""

from __future__ import annotations

from isaaclab.actuators import ActuatorBaseCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from robot_description import paths
from robot_description.robots.unitree_g1 import joint_groups as jg
from robot_description.robots.urdf_common import RobotUrdfFileCfg

# Re-exported so callers (and tests) can do
# `from robot_description.robots.unitree_g1 import G1_JOINT_NAMES` without
# knowing about the internal joint_groups module split.
G1_JOINT_NAMES: tuple[str, ...] = jg.G1_JOINT_NAMES

# `paths.urdf_path(...)` is TOTAL (never raises, see paths.py) -- evaluated at
# import time as a plain string field, and LearningRLlab's lint/config-resolution
# tests import this module with no vendor/ checkout present. `usd_dir` is set
# explicitly so Isaac Lab's hash-based conversion cache persists across runs.
_SPAWN_CFG = RobotUrdfFileCfg(
    asset_path=paths.urdf_path("unitree_g1"),
    usd_dir=str(paths.robot_build_dir("unitree_g1")),
)

# Standing pose from UNITREE_G1_29DOF_CFG (not the crouched MIMIC_CFG pose the
# actuator gains below follow -- this repo mixes MIMIC_CFG's gains with the
# non-mimic sibling's pose). Only some joints are addressed explicitly;
# Isaac Lab zero-initializes default_joint_pos for the rest.
_INIT_STATE = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.8),
    joint_pos={
        "left_hip_pitch_joint": -0.1,
        "right_hip_pitch_joint": -0.1,
        ".*_knee_joint": 0.3,
        ".*_ankle_pitch_joint": -0.2,
        ".*_shoulder_pitch_joint": 0.3,
        "left_shoulder_roll_joint": 0.25,
        "right_shoulder_roll_joint": -0.25,
        ".*_elbow_joint": 0.97,
        "left_wrist_roll_joint": 0.15,
        "right_wrist_roll_joint": -0.15,
    },
    joint_vel={".*": 0.0},
)


def _group_dict(patterns: tuple[str, ...], source: dict[str, float]) -> dict[str, float]:
    """Slice a full pattern->value dict (keyed across all five groups) down to
    just the patterns belonging to one actuator group, preserving pattern
    strings as the dict keys (Isaac Lab actuator cfgs accept
    dict[str, float] | float | None for stiffness/damping/armature/limits,
    keyed by the same joint_names_expr-style regex)."""
    return {pattern: source[pattern] for pattern in patterns}


def _build_actuators(actuator_cls: type[ActuatorBaseCfg]) -> dict[str, ActuatorBaseCfg]:
    """Build the five actuator groups (legs/feet/waist/waist_yaw/arms) with
    `actuator_cls`. Any Isaac Lab actuator cfg accepting these keywords works,
    so a variant with a different actuator model reuses the same physical
    limits and PD baseline.
    """
    return {
        "legs": actuator_cls(
            joint_names_expr=list(jg.LEGS_JOINT_EXPR),
            effort_limit=_group_dict(jg.LEGS_JOINT_EXPR, jg.EFFORT_LIMIT),
            velocity_limit=_group_dict(jg.LEGS_JOINT_EXPR, jg.VELOCITY_LIMIT),
            stiffness=_group_dict(jg.LEGS_JOINT_EXPR, jg.STIFFNESS),
            damping=_group_dict(jg.LEGS_JOINT_EXPR, jg.DAMPING),
            armature=_group_dict(jg.LEGS_JOINT_EXPR, jg.ARMATURE),
        ),
        "feet": actuator_cls(
            joint_names_expr=list(jg.FEET_JOINT_EXPR),
            effort_limit=_group_dict(jg.FEET_JOINT_EXPR, jg.EFFORT_LIMIT),
            velocity_limit=_group_dict(jg.FEET_JOINT_EXPR, jg.VELOCITY_LIMIT),
            stiffness=_group_dict(jg.FEET_JOINT_EXPR, jg.STIFFNESS),
            damping=_group_dict(jg.FEET_JOINT_EXPR, jg.DAMPING),
            armature=_group_dict(jg.FEET_JOINT_EXPR, jg.ARMATURE),
        ),
        "waist": actuator_cls(
            joint_names_expr=list(jg.WAIST_JOINT_EXPR),
            effort_limit=_group_dict(jg.WAIST_JOINT_EXPR, jg.EFFORT_LIMIT),
            velocity_limit=_group_dict(jg.WAIST_JOINT_EXPR, jg.VELOCITY_LIMIT),
            stiffness=_group_dict(jg.WAIST_JOINT_EXPR, jg.STIFFNESS),
            damping=_group_dict(jg.WAIST_JOINT_EXPR, jg.DAMPING),
            armature=_group_dict(jg.WAIST_JOINT_EXPR, jg.ARMATURE),
        ),
        "waist_yaw": actuator_cls(
            joint_names_expr=list(jg.WAIST_YAW_JOINT_EXPR),
            effort_limit=_group_dict(jg.WAIST_YAW_JOINT_EXPR, jg.EFFORT_LIMIT),
            velocity_limit=_group_dict(jg.WAIST_YAW_JOINT_EXPR, jg.VELOCITY_LIMIT),
            stiffness=_group_dict(jg.WAIST_YAW_JOINT_EXPR, jg.STIFFNESS),
            damping=_group_dict(jg.WAIST_YAW_JOINT_EXPR, jg.DAMPING),
            armature=_group_dict(jg.WAIST_YAW_JOINT_EXPR, jg.ARMATURE),
        ),
        "arms": actuator_cls(
            joint_names_expr=list(jg.ARMS_JOINT_EXPR),
            effort_limit=_group_dict(jg.ARMS_JOINT_EXPR, jg.EFFORT_LIMIT),
            velocity_limit=_group_dict(jg.ARMS_JOINT_EXPR, jg.VELOCITY_LIMIT),
            stiffness=_group_dict(jg.ARMS_JOINT_EXPR, jg.STIFFNESS),
            damping=_group_dict(jg.ARMS_JOINT_EXPR, jg.DAMPING),
            armature=_group_dict(jg.ARMS_JOINT_EXPR, jg.ARMATURE),
        ),
    }


G1_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=_SPAWN_CFG,
    init_state=_INIT_STATE,
    soft_joint_pos_limit_factor=0.9,
    actuators=_build_actuators(ImplicitActuatorCfg),
)
"""Baseline G1 config, solver-side PD (`ImplicitActuatorCfg`).

PD gains, effort/velocity limits, and the actuator-group split are taken from
`unitreerobotics/unitree_rl_lab`'s `UNITREE_G1_29DOF_MIMIC_CFG` -- real
motor-derived numbers (see `joint_groups.py`'s docstring), not this repo's
own guesses. The initial pose is UNITREE_G1_29DOF_CFG's standing pose
instead (see `_INIT_STATE` above).

Matches upstream's Newton-supported `Isaac-Velocity-Flat-G1` actuator model
shape."""
