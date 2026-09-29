"""Pure-Python G1 joint data: names, actuator groups, and physical limits.

This module deliberately has ZERO Isaac Lab / isaacsim imports: it is plain
joint data, readable and diffable on its own, while `robot.py` is the only
place that turns it into `ArticulationCfg`/`ActuatorCfg` objects. Keep the
split -- do not move Isaac Lab types in here, and do not move this data back
into `robot.py`.

Joint names are transcribed from `g1_29dof_rev_1_0.urdf`. Actuator groups,
effort/velocity limits, and PD gains (stiffness/damping/armature) are taken
from `unitreerobotics/unitree_rl_lab`'s `UNITREE_G1_29DOF_MIMIC_CFG` -- real
motor-derived numbers from Unitree's own sim2real training config. The
five-group split (`legs`/`feet`/`waist`/`waist_yaw`/`arms`) matches that
config's split exactly.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# The 29 URDF joint names, in the order they appear in
# vendor/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf.
#
# NOT independently re-verified against the vendor URDF from this authoring
# machine (Windows, no vendor/ checkout). The ORDER is the conventional
# left-leg/right-leg/waist/left-arm/right-arm declaration order -- confirm it
# against the real URDF the first time this repo runs on Linux, before
# relying on it for `ArticulationCfg.joint_ordering`. Resolve joints by name,
# never by a hardcoded index into this tuple.
# ---------------------------------------------------------------------------
G1_JOINT_NAMES: tuple[str, ...] = (
    # left leg (6)
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # right leg (6)
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # waist (3)
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    # left arm (7)
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    # right arm (7)
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

assert len(G1_JOINT_NAMES) == 29, "G1 has 29 revolute joints (F2a) — count drifted"

# ---------------------------------------------------------------------------
# Actuator groups. Each entry is a list of `re.fullmatch`-style patterns
# (Isaac Lab's `resolve_matching_names` uses `re.fullmatch`, not `re.match` or
# `re.search`). The five groups below must PARTITION all 29 joints: every
# joint matched by exactly one pattern in exactly one group. A joint no group
# claims is a silent limp limb; a joint two groups claim is a hard Isaac Lab
# error at scene-build time.
#
# `waist` splits into two groups here (`waist`: roll+pitch, `waist_yaw`: yaw
# alone), matching UNITREE_G1_29DOF_MIMIC_CFG -- waist_yaw uses the same
# 7520-14 motor as the hip pitch/yaw joints, while waist roll/pitch use the
# smaller 5020 motor, same as the ankles.
# ---------------------------------------------------------------------------
LEGS_JOINT_EXPR: tuple[str, ...] = (
    ".*_hip_pitch_joint",
    ".*_hip_roll_joint",
    ".*_hip_yaw_joint",
    ".*_knee_joint",
)  # 4 patterns x {left, right} = 8 joints

FEET_JOINT_EXPR: tuple[str, ...] = (
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
)  # 2 patterns x {left, right} = 4 joints

WAIST_JOINT_EXPR: tuple[str, ...] = (
    "waist_roll_joint",
    "waist_pitch_joint",
)  # 2 joints, explicit

WAIST_YAW_JOINT_EXPR: tuple[str, ...] = ("waist_yaw_joint",)  # 1 joint, explicit

ARMS_JOINT_EXPR: tuple[str, ...] = (
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_elbow_joint",
    ".*_wrist_roll_joint",
    ".*_wrist_pitch_joint",
    ".*_wrist_yaw_joint",
)  # 7 patterns x {left, right} = 14 joints

ACTUATOR_GROUPS: dict[str, tuple[str, ...]] = {
    "legs": LEGS_JOINT_EXPR,
    "feet": FEET_JOINT_EXPR,
    "waist": WAIST_JOINT_EXPR,
    "waist_yaw": WAIST_YAW_JOINT_EXPR,
    "arms": ARMS_JOINT_EXPR,
}

# ---------------------------------------------------------------------------
# Effort (N*m) / velocity (rad/s) limits, from UNITREE_G1_29DOF_MIMIC_CFG's
# `effort_limit_sim`/`velocity_limit_sim` (`effort_limit`/`velocity_limit` and
# the `_sim`-suffixed names are deprecated aliases of each other in this
# Isaac Lab version).
# ---------------------------------------------------------------------------
EFFORT_LIMIT: dict[str, float] = {
    ".*_hip_yaw_joint": 88.0,
    ".*_hip_roll_joint": 139.0,
    ".*_hip_pitch_joint": 88.0,
    ".*_knee_joint": 139.0,
    ".*_ankle_pitch_joint": 50.0,
    ".*_ankle_roll_joint": 50.0,
    "waist_roll_joint": 50.0,
    "waist_pitch_joint": 50.0,
    "waist_yaw_joint": 88.0,
    ".*_shoulder_pitch_joint": 25.0,
    ".*_shoulder_roll_joint": 25.0,
    ".*_shoulder_yaw_joint": 25.0,
    ".*_elbow_joint": 25.0,
    ".*_wrist_roll_joint": 25.0,
    ".*_wrist_pitch_joint": 5.0,
    ".*_wrist_yaw_joint": 5.0,
}

VELOCITY_LIMIT: dict[str, float] = {
    ".*_hip_yaw_joint": 32.0,
    ".*_hip_roll_joint": 20.0,
    ".*_hip_pitch_joint": 32.0,
    ".*_knee_joint": 20.0,
    ".*_ankle_pitch_joint": 37.0,
    ".*_ankle_roll_joint": 37.0,
    "waist_roll_joint": 37.0,
    "waist_pitch_joint": 37.0,
    "waist_yaw_joint": 32.0,
    ".*_shoulder_pitch_joint": 37.0,
    ".*_shoulder_roll_joint": 37.0,
    ".*_shoulder_yaw_joint": 37.0,
    ".*_elbow_joint": 37.0,
    ".*_wrist_roll_joint": 37.0,
    ".*_wrist_pitch_joint": 22.0,
    ".*_wrist_yaw_joint": 22.0,
}

# ---------------------------------------------------------------------------
# PD gains (N*m/rad and N*m*s/rad) and rotor armature (kg*m^2), derived
# exactly as UNITREE_G1_29DOF_MIMIC_CFG derives them: a per-motor armature
# (measured, from `unitree_actuators.py`) and a critically-damped-at-10Hz formula --
#
#     stiffness = armature * natural_frequency^2
#     damping   = 2 * damping_ratio * armature * natural_frequency
#
# with natural_frequency = 10 Hz and damping_ratio = 2.0 (overdamped, matching
# upstream). Real motor-derived numbers, not hand-tuned guesses.
#
# Suffixes name the motor: 5020 (ankles, waist roll/pitch, most arm joints),
# 7520-14 (hip pitch/yaw, waist yaw), 7520-22 (hip roll, knee), 4010 (wrist
# pitch/yaw). Feet and waist(roll+pitch) each drive a joint through a 2x
# gear stage relative to a single 5020 motor, hence the `2.0 *` factor.
# ---------------------------------------------------------------------------
_ARMATURE_5020 = 0.003609725
_ARMATURE_7520_14 = 0.010177520
_ARMATURE_7520_22 = 0.025101925
_ARMATURE_4010 = 0.00425

_NATURAL_FREQ = 10 * 2.0 * math.pi  # 10 Hz
_DAMPING_RATIO = 2.0

_STIFFNESS_5020 = _ARMATURE_5020 * _NATURAL_FREQ**2
_STIFFNESS_7520_14 = _ARMATURE_7520_14 * _NATURAL_FREQ**2
_STIFFNESS_7520_22 = _ARMATURE_7520_22 * _NATURAL_FREQ**2
_STIFFNESS_4010 = _ARMATURE_4010 * _NATURAL_FREQ**2

_DAMPING_5020 = 2.0 * _DAMPING_RATIO * _ARMATURE_5020 * _NATURAL_FREQ
_DAMPING_7520_14 = 2.0 * _DAMPING_RATIO * _ARMATURE_7520_14 * _NATURAL_FREQ
_DAMPING_7520_22 = 2.0 * _DAMPING_RATIO * _ARMATURE_7520_22 * _NATURAL_FREQ
_DAMPING_4010 = 2.0 * _DAMPING_RATIO * _ARMATURE_4010 * _NATURAL_FREQ

STIFFNESS: dict[str, float] = {
    ".*_hip_pitch_joint": _STIFFNESS_7520_14,
    ".*_hip_roll_joint": _STIFFNESS_7520_22,
    ".*_hip_yaw_joint": _STIFFNESS_7520_14,
    ".*_knee_joint": _STIFFNESS_7520_22,
    ".*_ankle_pitch_joint": 2.0 * _STIFFNESS_5020,
    ".*_ankle_roll_joint": 2.0 * _STIFFNESS_5020,
    "waist_roll_joint": 2.0 * _STIFFNESS_5020,
    "waist_pitch_joint": 2.0 * _STIFFNESS_5020,
    "waist_yaw_joint": _STIFFNESS_7520_14,
    ".*_shoulder_pitch_joint": _STIFFNESS_5020,
    ".*_shoulder_roll_joint": _STIFFNESS_5020,
    ".*_shoulder_yaw_joint": _STIFFNESS_5020,
    ".*_elbow_joint": _STIFFNESS_5020,
    ".*_wrist_roll_joint": _STIFFNESS_5020,
    ".*_wrist_pitch_joint": _STIFFNESS_4010,
    ".*_wrist_yaw_joint": _STIFFNESS_4010,
}

DAMPING: dict[str, float] = {
    ".*_hip_pitch_joint": _DAMPING_7520_14,
    ".*_hip_roll_joint": _DAMPING_7520_22,
    ".*_hip_yaw_joint": _DAMPING_7520_14,
    ".*_knee_joint": _DAMPING_7520_22,
    ".*_ankle_pitch_joint": 2.0 * _DAMPING_5020,
    ".*_ankle_roll_joint": 2.0 * _DAMPING_5020,
    "waist_roll_joint": 2.0 * _DAMPING_5020,
    "waist_pitch_joint": 2.0 * _DAMPING_5020,
    "waist_yaw_joint": _DAMPING_7520_14,
    ".*_shoulder_pitch_joint": _DAMPING_5020,
    ".*_shoulder_roll_joint": _DAMPING_5020,
    ".*_shoulder_yaw_joint": _DAMPING_5020,
    ".*_elbow_joint": _DAMPING_5020,
    ".*_wrist_roll_joint": _DAMPING_5020,
    ".*_wrist_pitch_joint": _DAMPING_4010,
    ".*_wrist_yaw_joint": _DAMPING_4010,
}

ARMATURE: dict[str, float] = {
    ".*_hip_pitch_joint": _ARMATURE_7520_14,
    ".*_hip_roll_joint": _ARMATURE_7520_22,
    ".*_hip_yaw_joint": _ARMATURE_7520_14,
    ".*_knee_joint": _ARMATURE_7520_22,
    ".*_ankle_pitch_joint": 2.0 * _ARMATURE_5020,
    ".*_ankle_roll_joint": 2.0 * _ARMATURE_5020,
    "waist_roll_joint": 2.0 * _ARMATURE_5020,
    "waist_pitch_joint": 2.0 * _ARMATURE_5020,
    "waist_yaw_joint": _ARMATURE_7520_14,
    ".*_shoulder_pitch_joint": _ARMATURE_5020,
    ".*_shoulder_roll_joint": _ARMATURE_5020,
    ".*_shoulder_yaw_joint": _ARMATURE_5020,
    ".*_elbow_joint": _ARMATURE_5020,
    ".*_wrist_roll_joint": _ARMATURE_5020,
    ".*_wrist_pitch_joint": _ARMATURE_4010,
    ".*_wrist_yaw_joint": _ARMATURE_4010,
}
