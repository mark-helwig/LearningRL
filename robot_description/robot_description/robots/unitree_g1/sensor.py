"""Named sensor groups for the G1 (contact sensors, raycasters, etc.).

Body lists only -- the file to edit when a task needs a different set of bodies
sensed. All plumbing (prim paths, backend dispatch, `SceneEntityCfg`
construction) lives in `robots/sensors.py`.

Body names come from `g1_29dof_rev_1_0.urdf` and are matched with
`re.fullmatch`. `merge_fixed_joints=True` (urdf_common.py) leaves exactly 30
addressable rigid bodies -- `pelvis` plus the 29 revolute-joint child links.
Fixed-joint links (`head_link`, `logo_link`, `pelvis_contour_link`,
`.*_rubber_hand`, the IMU/camera mounts) are merged into their parents as
geometry and are NOT addressable here.

Requires Isaac Lab. `unitree_g1/__init__.py` exports these lazily for that
reason -- see its docstring.
"""

from __future__ import annotations

from robot_description.robots.sensors import contact_sensor_group

TERMINATION_CONTACT = contact_sensor_group(
    "g1_termination",
    bodies=[
        "pelvis",
        "waist_yaw_link",
        "waist_roll_link",
        "torso_link",
        ".*_hip_pitch_link",
        ".*_hip_roll_link",
        ".*_hip_yaw_link",
        ".*_knee_link",
        ".*_shoulder_pitch_link",
        ".*_shoulder_roll_link",
        ".*_shoulder_yaw_link",
        ".*_elbow_link",
        ".*_wrist_roll_link",
        ".*_wrist_pitch_link",
        ".*_wrist_yaw_link",
    ],
    exclude_self=True,
    history_length=3,
)
"""Every body except the feet (28 of 30). Non-self contact on any of them is a fall."""

FEET_CONTACT = contact_sensor_group(
    "g1_feet",
    bodies=[".*_ankle_roll_link"],
    history_length=3,
    track_air_time=True,
)
"""The two feet. `track_air_time` feeds the feet_air_time / feet_slide rewards."""
