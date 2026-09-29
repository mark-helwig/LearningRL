"""Named sensor groups for the RPO. Body lists only; plumbing is in `robots/sensors.py`.

Body names come from `rpo.urdf`, matched with `re.fullmatch`. The URDF declares
no fixed joints, so `merge_fixed_joints=True` folds nothing away and all 24
bodies are addressable: `base_link` plus the 23 revolute joints' child links.
That list is cross-checked against upstream's own `RPO_LINKS`.

CAVEAT -- eight of those bodies have no collider. `rpo.urdf` declares
`<collision>` for only 16 of its 24 links, and `collision_from_visuals` is
`False` (robot.py's docstring), so `.*_thigh_yaw_link`, `.*_thigh_roll_link`,
`.*_ankle_pitch_link` and `.*_elbow_yaw_link` spawn with nothing to collide
with. They are still named by `TERMINATION_CONTACT` below -- the group is
defined by which bodies SHOULD terminate an episode, and dropping them would
bake a vendor-URDF gap into this repo's idea of the robot. But a fall that
lands on a bare thigh-yaw housing reports zero force and will not terminate.
Fixing it means adding collision geometry upstream, not flipping
`collision_from_visuals` (which would make every visual mesh a collider).

Requires Isaac Lab. `roboparty_rpo/__init__.py` exports these lazily.
"""

from __future__ import annotations

from robot_description.robots.sensors import contact_sensor_group

TERMINATION_CONTACT = contact_sensor_group(
    "rpo_termination",
    bodies=[
        "base_link",
        "torso_link",
        ".*_thigh_yaw_link",
        ".*_thigh_roll_link",
        ".*_thigh_pitch_link",
        ".*_knee_link",
        ".*_ankle_pitch_link",
        ".*_arm_pitch_link",
        ".*_arm_roll_link",
        ".*_arm_yaw_link",
        ".*_elbow_pitch_link",
        ".*_elbow_yaw_link",
    ],
    exclude_self=True,
    history_length=3,
)
"""Every body except the feet (22 of 24). Non-self contact on any of them is a
fall -- subject to the eight colliderless bodies noted in the module docstring."""

FEET_CONTACT = contact_sensor_group(
    "rpo_feet",
    bodies=[".*_ankle_roll_link"],
    history_length=3,
    track_air_time=True,
)
"""The two feet. `track_air_time` feeds the feet_air_time / feet_slide rewards."""
