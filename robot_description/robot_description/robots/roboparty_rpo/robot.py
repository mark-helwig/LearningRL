# Portions of this file are derived from RoboLab (https://github.com/Roboparty/robolab),
# robolab/assets/robots/roboparty.py, distributed under the following licence:
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Isaac Lab `ArticulationCfg` for the RoboParty ROBOTO_ORIGIN / RPO (23-DoF,
`Roboparty/rpo_description` rev `37aac9ca665e92731444a1618320078e7ba21569`,
`urdf/rpo.urdf`).

Requires Isaac Lab. Joint names, actuator-group regexes, limits, gains, and the
delay window live in the Isaac-Lab-free `joint_groups.py`, same split as
`unitree_g1`. Spawns directly from the vendored URDF via `RobotUrdfFileCfg`,
cached under `paths.robot_build_dir("roboparty_rpo")`.

Ported from RoboParty's own `RPO_CFG` in
`robolab/robolab/assets/robots/roboparty.py` (BSD-3-Clause). Every number --
gains, delay window, standing pose, solver iteration counts -- is theirs; what
changed is the shape, to match this repo's `joint_groups.py` + `robot.py` split.

Two deviations from upstream's spawn config, both forced by `RobotUrdfFileCfg`:

- `replace_cylinders_with_capsules=True` is carried over on the instance
  (upstream sets it too). `rpo.urdf`'s collision geometry is 8 cylinders,
  5 boxes and 3 meshes, not the per-link meshes the Unitree URDFs ship, and a
  cylinder collider is both slower and worse-behaved on contact than a capsule.
- `collision_from_visuals` stays `False`, so the 8 links whose URDF declares no
  `<collision>` at all (`.*_thigh_yaw_link`, `.*_thigh_roll_link`,
  `.*_ankle_pitch_link`, `.*_elbow_yaw_link`) spawn with NO collider. That is
  upstream's behaviour as well. It matters for `sensor.py` -- see its docstring.

Actuator variants, in the order you should reach for them:

- `RPO_CFG` -- `DelayedPDActuatorCfg`, 0-2 physics steps of action delay. The
  default, because it is what RoboParty tuned and trained this robot against.
- `RPO_IMPLICIT_CFG` -- `ImplicitActuatorCfg`, solver-side PD, no delay. The
  cheap and unconditionally stable fallback, and what `unitree_g1` uses as its
  own default.
"""

from __future__ import annotations

from isaaclab.actuators import ActuatorBaseCfg, DelayedPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from robot_description import paths
from robot_description.robots.roboparty_rpo import joint_groups as jg
from robot_description.robots.urdf_common import RobotUrdfFileCfg

RPO_JOINT_NAMES: tuple[str, ...] = jg.RPO_JOINT_NAMES

_SPAWN_CFG = RobotUrdfFileCfg(
    asset_path=paths.urdf_path("roboparty_rpo"),
    usd_dir=str(paths.robot_build_dir("roboparty_rpo")),
    # Instance override, not a fork of RobotUrdfFileCfg (docs/adding-a-robot.md
    # step 5) -- rpo.urdf is the only source here with primitive colliders.
    replace_cylinders_with_capsules=True,
)

_INIT_STATE = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, jg.NOMINAL_HEIGHT),
    joint_pos=dict(jg.NOMINAL_JOINT_POS),
    joint_vel={".*": 0.0},
)


def _group_dict(patterns: tuple[str, ...], source: dict[str, float]) -> dict[str, float]:
    return {pattern: source[pattern] for pattern in patterns}


def _build_actuators(actuator_cls: type[ActuatorBaseCfg], **extra) -> dict[str, ActuatorBaseCfg]:
    """One `actuator_cls` per `jg.ACTUATOR_GROUPS` entry, so both exported cfgs
    share one copy of the limits and PD baseline. `extra` carries the kwargs
    only some actuator classes accept (`min_delay`/`max_delay`).

    Both the model-side (`effort_limit`) and the solver-side
    (`effort_limit_sim`) limit are set from the same URDF-transcribed dict.
    They are equivalent for `ImplicitActuatorCfg`, but diverge for the explicit
    `DelayedPDActuatorCfg`: upstream's `RPO_CFG` sets only the `_sim` pair and
    lets the model-side limit fall back to whatever value survived the URDF ->
    USD conversion. Setting both pins the clip at the URDF's motor rating in
    the actuator model and in the solver, independent of that round-trip.
    """
    return {
        name: actuator_cls(
            joint_names_expr=list(patterns),
            effort_limit=_group_dict(patterns, jg.EFFORT_LIMIT),
            effort_limit_sim=_group_dict(patterns, jg.EFFORT_LIMIT),
            velocity_limit=_group_dict(patterns, jg.VELOCITY_LIMIT),
            velocity_limit_sim=_group_dict(patterns, jg.VELOCITY_LIMIT),
            stiffness=_group_dict(patterns, jg.STIFFNESS),
            damping=_group_dict(patterns, jg.DAMPING),
            armature=_group_dict(patterns, jg.ARMATURE),
            **extra,
        )
        for name, patterns in jg.ACTUATOR_GROUPS.items()
    }


RPO_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=_SPAWN_CFG,
    init_state=_INIT_STATE,
    soft_joint_pos_limit_factor=0.90,
    actuators=_build_actuators(
        DelayedPDActuatorCfg, min_delay=jg.MIN_DELAY, max_delay=jg.MAX_DELAY
    ),
)
"""Baseline RPO config, matching RoboParty's `RPO_CFG`.

`DelayedPDActuatorCfg` runs the PD loop in Python each control step (it
subclasses `IdealPDActuatorCfg`) and holds each joint's action in a ring buffer
for 0-2 PHYSICS steps, resampled per joint on every env reset. That models the
real robot's sensing-to-torque latency, which is what makes a policy trained
here transfer instead of exploiting a zero-latency simulator.

This is the default because it is what upstream tuned against, but it is an
explicit actuator: if the PD loop chatters at your task's `dt`, drop to
`RPO_IMPLICIT_CFG` to tell the delay apart from the integrator before touching
the gains."""

RPO_IMPLICIT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=_SPAWN_CFG,
    init_state=_INIT_STATE,
    soft_joint_pos_limit_factor=0.90,
    actuators=_build_actuators(ImplicitActuatorCfg),
)
"""Same robot, same gains, solver-side PD and NO action delay.

The `G1_CFG` equivalent, and the A/B baseline for `RPO_CFG`: a policy
that trains here but not there is being taught by the delay, not hindered by
it."""
