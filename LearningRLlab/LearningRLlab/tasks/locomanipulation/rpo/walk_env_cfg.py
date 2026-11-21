"""HCRL-RPO-Walk-Flat-v0, ported from HCRLlab (``mh/quiet_walking`` cbfa056): the walk under the RPO locomanip task.

HCRLlab builds it in three layers -- the G1 Walk-v1 cfg (``tasks/locomotion/g1/g1_env_cfg.py``), the RPO swap
(``tasks/locomotion/rpo/rpo_env_cfg.py``) and the flat overrides (``HCRLRPOWalkFlatEnvCfg`` +
``use_flat_terrain``). This file is the result of all three, written out for the RPO directly, so every value
below is the one an HCRL-RPO-Walk-Flat-v0 env ends up with:

- Walk-v1 rewards: a contact-timer gait clock (``gait_phase``) and swing-height profile (``gait_clearance``),
  torso orientation, foot impact/slip, self-collision and normalized joint limits;
- the RPO's robot, feet/termination sensors and joint names (``thigh`` for hip, one ``torso_joint``, no wrist);
- flat ground: clearance measured from the sole (``FOOT_OFFSET``), ``action_rate`` -0.01, a knee contact ends
  the episode, ``joint_deviation_legs`` -0.2 and ``track_ang_vel_z`` 1.5 (the RPO turns on its tilted thigh yaw).

The critic additionally sees the feet's air/contact timers and the stance type, which the gait clock reads.
"""

import math

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg
from isaaclab_physx.physics import PhysxCfg
from robot_description.robots.roboparty_rpo import RPO_CFG, RPO_FEET_CONTACT, RPO_TERMINATION_CONTACT
from robot_description.robots.sensors import ContactSensorGroup

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_tasks.utils import PresetCfg

from LearningRLlab.scenes import FlatSceneCfg

from . import mdp

SWING_TIME = 0.32  # 0.8 s cycle: 40% swing, 60% stance per foot -> 20% double stance
STANCE_TIME = 0.48
DOUBLE_STANCE_TIME = 0.08
MIN_SWING = 0.2  # double stance only counts after a real step
# ankle_roll_link origin above the sole: `rpo.urdf`'s foot mesh bottoms out 0.045 m below it (G1: 0.035 m).
FOOT_OFFSET = 0.045
FEET = ".*_ankle_roll_link"


def below_ankle_free_contact(group: ContactSensorGroup) -> SceneEntityCfg:
    """``group``'s bodies minus the ankles: on flat ground a knee or anything above it touching down is a fall.

    HCRLlab's ``above_knee_contact`` (knees and ankles out, for shins grazing boxes) with ``use_flat_terrain``
    putting ``.*_knee_link`` back.
    """
    return SceneEntityCfg(group.name, body_names=[b for b in group.bodies if "ankle" not in b])


##
# Physics presets
##


@configclass
class PhysicsCfg(PresetCfg):
    default = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    # upstream RoughPhysicsCfg for mesh terrain, minus its 1 cm shape margin (G1 arm links self-contact).
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=512,
            nconmax=100,
            cone="pyramidal",
            impratio=1,
            integrator="implicitfast",
            use_mujoco_contacts=False,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=2_500_000),
        num_substeps=1,
        debug_mode=False,
    )


##
# Scene definition
##


@configclass
class SceneCfg(FlatSceneCfg):
    """Flat ground with the RPO."""

    # F10: RPO_CFG is the DelayedPDActuatorCfg variant RoboParty tuned and trained against
    # (0-2 physics steps of action delay), not the implicit cfg the G1 walk uses.
    robot: ArticulationCfg = RPO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # sensors: attached from robot_description in LearningRLRPOWalkV1EnvCfg.__post_init__


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True, clip={".*": (-10.0, 10.0)}
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 1

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)
        feet_timers = ObsTerm(func=mdp.feet_air_contact_time, params={"sensor_cfg": RPO_FEET_CONTACT.entity_cfg()})
        stance = ObsTerm(func=mdp.stance_type, params={"sensor_cfg": RPO_FEET_CONTACT.entity_cfg()})

        def __post_init__(self) -> None:
            self.history_length = 1

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # the RPO's `torso_link` is, as on the G1, its heaviest link (7.58 kg) and the one carrying the arms
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    view_middle_env = EventTerm(func=mdp.view_middle_env, mode="startup")

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.05, 0.05), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    # turning is bounded by the [-1, 0.2] rad thigh yaw range; weight yaw tracking up rather than free the posture
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    body_orientation = RewTerm(
        func=mdp.body_orientation_exp,
        weight=0.5,
        params={"std": 0.15, "asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
    )
    gait_phase = RewTerm(
        func=mdp.gait_contact_clock,
        weight=1.0,
        params={
            "swing_time": SWING_TIME,
            "double_stance_time": DOUBLE_STANCE_TIME,
            "min_swing": MIN_SWING,
            "stance_time": STANCE_TIME,
            "timing_std": 0.15,
            "sensor_cfg": RPO_FEET_CONTACT.entity_cfg(),
            "command_name": "base_velocity",
        },
    )
    gait_clearance = RewTerm(
        func=mdp.gait_clearance,
        weight=0.5,
        params={
            "swing_time": SWING_TIME,
            "double_stance_time": DOUBLE_STANCE_TIME,
            "min_swing": MIN_SWING,
            "height": 0.15,
            "std": 0.1,
            "foot_offset": FOOT_OFFSET,
            "sensor_cfg": RPO_FEET_CONTACT.entity_cfg(),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET),
            "command_name": "base_velocity",
            # flat ground: swing height from z = 0 (via FOOT_OFFSET), not from the lowest grounded foot
            "stance_relative": False,
        },
    )

    # -- regularization
    foot_impact = RewTerm(
        func=mdp.foot_impact_force,
        weight=-0.05,
        params={"sensor_cfg": RPO_FEET_CONTACT.entity_cfg(), "asset_cfg": SceneEntityCfg("robot")},
    )
    foot_slip = RewTerm(
        func=mdp.foot_slip_exp,
        weight=-0.05,
        params={
            "std": 0.1,
            "sensor_cfg": RPO_FEET_CONTACT.entity_cfg(),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET),
        },
    )
    self_collision = RewTerm(
        func=mdp.self_collision_fraction,
        weight=-0.025,
        params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg(RPO_TERMINATION_CONTACT.name, body_names=".*")},
    )
    joint_limits = RewTerm(
        func=mdp.joint_pos_limits_normalized, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # -- posture. The RPO's names: arms are `arm_pitch/roll/yaw` + 2 elbow joints per side (no wrist), the waist
    # is one `torso_joint`, and `thigh_roll`/`thigh_yaw` are the G1's `hip_roll`/`hip_yaw` (on tilted axes).
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_arm_.*_joint", ".*_elbow_.*_joint"])},
    )
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["torso_joint"])},
    )
    # the tilted thigh yaw is the RPO's turning DoF; -0.5 left in-place yaw at ~0.57 rad/s for a 1.0 command
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_thigh_roll_joint", ".*_thigh_yaw_joint"])},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_external_contact,
        params={"threshold": 1.0, "sensor_cfg": below_ankle_free_contact(RPO_TERMINATION_CONTACT)},
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


##
# Environment configuration
##


@configclass
class LearningRLRPOWalkV1EnvCfg(ManagerBasedRLEnvCfg):
    """HCRL-RPO-Walk-Flat-v0: the Walk-v1 velocity-tracking walk for the RPO, on flat ground."""

    # Simulation settings
    sim: SimulationCfg = SimulationCfg(physics=PhysicsCfg())
    # Viewer
    viewer = ViewerCfg(eye=(2.8, -0.5, 0.7), origin_type="asset_root", env_index=0, asset_name="robot")
    # Scene settings
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material

        # sensors from robot_description; one fresh entity_cfg() per term, since resolution mutates it
        RPO_FEET_CONTACT.attach(self.scene)
        RPO_TERMINATION_CONTACT.attach(self.scene)
        # history 1, physics-rate ticks, and a 1 N contact switch
        for sensor in (self.scene.rpo_feet, self.scene.rpo_termination):
            sensor.history_length = 1
            sensor.update_period = self.sim.dt
        self.scene.rpo_feet.force_threshold = 1.0
