"""Pure-Python RPO joint data: names, actuator groups, limits, gains, and delay.

This module deliberately has ZERO Isaac Lab / isaacsim imports, for the same
reason as `unitree_g1/joint_groups.py`: plain joint data here, Isaac Lab cfg
objects in `robot.py`. Do not add an isaaclab import here.

Joint names and the effort/velocity limits are transcribed from `rpo.urdf`'s
`<limit>` tags. Actuator groups, PD gains, armature, and the delay window come
from RoboParty's own `RPO_CFG` (`robolab/robolab/assets/robots/roboparty.py`,
BSD-3-Clause) -- the config they tuned and trained this robot against. Its
per-group `effort_limit_sim`/`velocity_limit_sim` values agree with the URDF
tags exactly, so the two sources do not conflict anywhere.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# The 23 URDF joint names, in the order they appear in
# vendor/rpo_description/urdf/rpo.urdf. Every joint is revolute -- the URDF
# declares no fixed joints at all, so `merge_fixed_joints` has nothing to fold.
# Resolve joints by name, never by a hardcoded index into this tuple.
# ---------------------------------------------------------------------------
RPO_JOINT_NAMES: tuple[str, ...] = (
    # left leg (6)
    "left_thigh_yaw_joint",
    "left_thigh_roll_joint",
    "left_thigh_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # right leg (6)
    "right_thigh_yaw_joint",
    "right_thigh_roll_joint",
    "right_thigh_pitch_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # waist (1)
    "torso_joint",
    # left arm (5)
    "left_arm_pitch_joint",
    "left_arm_roll_joint",
    "left_arm_yaw_joint",
    "left_elbow_pitch_joint",
    "left_elbow_yaw_joint",
    # right arm (5)
    "right_arm_pitch_joint",
    "right_arm_roll_joint",
    "right_arm_yaw_joint",
    "right_elbow_pitch_joint",
    "right_elbow_yaw_joint",
)

assert len(RPO_JOINT_NAMES) == 23, "RPO has 23 revolute joints (F2b) — count drifted"

# ---------------------------------------------------------------------------
# Actuator groups, `re.fullmatch`-style patterns. The four groups must
# PARTITION all 23 joints: every joint matched by exactly one pattern in
# exactly one group. The split is RPO_CFG's, which also happens to group by
# URDF limit class: `legs` is every 120 N*m / 25 rad/s joint (the torso yaw
# included), and the other three are all 27 N*m / 8 rad/s, split by where
# their PD gains differ.
#
# One deviation from RPO_CFG: its legs group writes the torso as `.*torso.*`.
# That is spelled out as `torso_joint` here because `.*torso.*` would silently
# swallow any future torso_roll/torso_pitch joint into the leg gains.
# ---------------------------------------------------------------------------
LEGS_JOINT_EXPR: tuple[str, ...] = (
    ".*_thigh_yaw_joint",
    ".*_thigh_roll_joint",
    ".*_thigh_pitch_joint",
    ".*_knee_joint",
    "torso_joint",
)  # 4 patterns x {left, right} + 1 torso = 9 joints

FEET_JOINT_EXPR: tuple[str, ...] = (
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
)  # 2 patterns x {left, right} = 4 joints

SHOULDERS_JOINT_EXPR: tuple[str, ...] = (
    ".*_arm_pitch_joint",
    ".*_arm_roll_joint",
    ".*_arm_yaw_joint",
)  # 3 patterns x {left, right} = 6 joints

ARMS_JOINT_EXPR: tuple[str, ...] = (
    ".*_elbow_pitch_joint",
    ".*_elbow_yaw_joint",
)  # 2 patterns x {left, right} = 4 joints

ACTUATOR_GROUPS: dict[str, tuple[str, ...]] = {
    "legs": LEGS_JOINT_EXPR,
    "feet": FEET_JOINT_EXPR,
    "shoulders": SHOULDERS_JOINT_EXPR,
    "arms": ARMS_JOINT_EXPR,
}

# ---------------------------------------------------------------------------
# Effort (N*m) / velocity (rad/s) limits. Identical to rpo.urdf's per-joint
# `<limit effort=... velocity=...>` tags and to RPO_CFG's per-group values.
# Only two classes exist: 120/25 below the waist plus the torso, 27/8 above.
# ---------------------------------------------------------------------------
EFFORT_LIMIT: dict[str, float] = {
    ".*_thigh_yaw_joint": 120.0,
    ".*_thigh_roll_joint": 120.0,
    ".*_thigh_pitch_joint": 120.0,
    ".*_knee_joint": 120.0,
    "torso_joint": 120.0,
    ".*_ankle_pitch_joint": 27.0,
    ".*_ankle_roll_joint": 27.0,
    ".*_arm_pitch_joint": 27.0,
    ".*_arm_roll_joint": 27.0,
    ".*_arm_yaw_joint": 27.0,
    ".*_elbow_pitch_joint": 27.0,
    ".*_elbow_yaw_joint": 27.0,
}

VELOCITY_LIMIT: dict[str, float] = {
    ".*_thigh_yaw_joint": 25.0,
    ".*_thigh_roll_joint": 25.0,
    ".*_thigh_pitch_joint": 25.0,
    ".*_knee_joint": 25.0,
    "torso_joint": 25.0,
    ".*_ankle_pitch_joint": 8.0,
    ".*_ankle_roll_joint": 8.0,
    ".*_arm_pitch_joint": 8.0,
    ".*_arm_roll_joint": 8.0,
    ".*_arm_yaw_joint": 8.0,
    ".*_elbow_pitch_joint": 8.0,
    ".*_elbow_yaw_joint": 8.0,
}

# ---------------------------------------------------------------------------
# PD gains (N*m/rad, N*m*s/rad) and rotor armature (kg*m^2), transcribed from
# RPO_CFG. RoboParty's hand-tuned values, not motor-derived -- armature in
# particular is a flat 0.01 placeholder across all 23 joints.
# ---------------------------------------------------------------------------
STIFFNESS: dict[str, float] = {
    ".*_thigh_yaw_joint": 100.0,
    ".*_thigh_roll_joint": 100.0,
    ".*_thigh_pitch_joint": 100.0,
    ".*_knee_joint": 150.0,
    "torso_joint": 150.0,
    ".*_ankle_pitch_joint": 40.0,
    ".*_ankle_roll_joint": 40.0,
    ".*_arm_pitch_joint": 40.0,
    ".*_arm_roll_joint": 40.0,
    ".*_arm_yaw_joint": 40.0,
    ".*_elbow_pitch_joint": 30.0,
    ".*_elbow_yaw_joint": 20.0,
}

DAMPING: dict[str, float] = {
    ".*_thigh_yaw_joint": 3.3,
    ".*_thigh_roll_joint": 3.3,
    ".*_thigh_pitch_joint": 3.3,
    ".*_knee_joint": 5.0,
    "torso_joint": 5.0,
    ".*_ankle_pitch_joint": 2.0,
    ".*_ankle_roll_joint": 2.0,
    ".*_arm_pitch_joint": 2.0,
    ".*_arm_roll_joint": 2.0,
    ".*_arm_yaw_joint": 2.0,
    ".*_elbow_pitch_joint": 1.5,
    ".*_elbow_yaw_joint": 1.0,
}

ARMATURE: dict[str, float] = {pattern: 0.01 for pattern in STIFFNESS}

# ---------------------------------------------------------------------------
# Action-delay window for DelayedPDActuatorCfg, in PHYSICS steps (not control
# steps). RPO_CFG applies the same 0-2 window to all four groups; the actuator
# resamples a per-joint delay in that range on every env reset.
# ---------------------------------------------------------------------------
MIN_DELAY: int = 0
MAX_DELAY: int = 2

# ---------------------------------------------------------------------------
# Standing pose, from RPO_CFG's init_state. Unlisted joints zero-initialize.
# Written per-joint rather than as `.*_` patterns because the arm roll pose is
# mirrored (+0.06 left, -0.06 right) and a shared pattern could not express it.
# ---------------------------------------------------------------------------
NOMINAL_HEIGHT: float = 0.75

NOMINAL_JOINT_POS: dict[str, float] = {
    "left_thigh_pitch_joint": -0.1,
    "left_knee_joint": 0.3,
    "left_ankle_pitch_joint": -0.2,
    "left_arm_pitch_joint": 0.18,
    "left_arm_roll_joint": 0.06,
    "left_elbow_pitch_joint": 0.78,
    "right_thigh_pitch_joint": -0.1,
    "right_knee_joint": 0.3,
    "right_ankle_pitch_joint": -0.2,
    "right_arm_pitch_joint": 0.18,
    "right_arm_roll_joint": -0.06,
    "right_elbow_pitch_joint": 0.78,
}
