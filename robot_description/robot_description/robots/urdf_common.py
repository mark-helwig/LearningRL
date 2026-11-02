"""Shared URDF-importer spawn config for every robot in this repo.

Needs Isaac Lab (like each robot's `robot.py`) -- not part of the
Isaac-Lab-free module set documented in repo CLAUDE.md.

Mirrors `unitreerobotics/unitree_rl_lab`'s `UnitreeUrdfFileCfg` pattern, which
is where these defaults were first transcribed from: a `sim_utils.UrdfFileCfg`
subclass with shared importer/physics defaults, so each robot's `robot.py` only
supplies `asset_path` and `usd_dir` (see `paths.robot_build_dir`) plus whatever
it must override on the instance. Converts URDF to USD on the fly at spawn time
(requires Kit running -- CLAUDE.md F1), with its own content-hash-keyed cache --
no separate build step.

Override on the INSTANCE, never by forking this class (docs/adding-a-robot.md
step 5). `roboparty_rpo/robot.py` is the worked example: it flips
`replace_cylinders_with_capsules` because its URDF's collision geometry is
primitives rather than meshes.

`unitree_common.py` re-exports this class as `UnitreeUrdfFileCfg` for callers
that predate the rename.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.utils.configclass import configclass


@configclass
class RobotUrdfFileCfg(sim_utils.UrdfFileCfg):
    func: str = "robot_description.robots.urdf_spawn:spawn_urdf_inertia_fixed"
    # The importer rotates every link's inertia axes; spawn_urdf_inertia_fixed undoes it.
    fix_principal_axes: bool = True
    fix_base: bool = False
    activate_contact_sensors: bool = True
    merge_fixed_joints: bool = True  # collapses logo/head/sensor-mount fixed joints
    collision_type: str = "Convex Hull"  # decomposition is 10-50x slower, unneeded on flat terrain
    # False: every vendor URDF here ships its own <collision> geometry. A robot whose
    # URDF omits it for some links gets no collider on those links -- decide that per
    # robot (see roboparty_rpo/robot.py's docstring), don't flip this default globally.
    collision_from_visuals: bool = False
    # importer-level flag; drives both PhysX and Newton self-collision attributes at conversion
    self_collision: bool = True
    merge_mesh: bool = False  # would destroy the per-link mesh<->link mapping mesh_ops.py relies on
    link_density: float = 0.0  # MUST stay 0.0 -- a density fill would corrupt every link's mass
    robot_type: str = "Humanoid"
    run_asset_transformer: bool = True  # layered USD; False would make physics_variant a no-op
    run_multi_physics_conversion: bool = True  # emits MuJoCo attributes for Newton -- keep True
    physics_variant: str = "physics"
    make_instanceable: bool = True  # memory win at 4096 parallel envs

    # Zero baseline drive -- each robot's ActuatorCfg owns the real PD gains instead.
    joint_drive = sim_utils.UrdfConverterCfg.JointDriveCfg(
        gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
    )
    joint_drive_props = sim_utils.JointDriveBaseCfg(ensure_drives_exist=True)

    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    )
    # Transcribed from unitree_rl_lab's UnitreeUrdfFileCfg. RoboParty's own
    # RPO_CFG independently specifies the same three values, so they are the
    # shared default rather than a Unitree-specific choice.
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=4,
    )
