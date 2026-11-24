"""Environment configuration for LearningRL-RPO-LocoManipulation-v0: the HCRL RPO walk plus wrist targets.

The walk is HCRL-RPO-Walk-Flat-v0 (``walk_env_cfg.py``, the Walk-v1 contact-clock gait), and the reaching half is
``LearningRL-G1-LocoManipulation-v0``'s, moved onto the RPO. One policy drives all 23 joints: it walks to the
``base_velocity`` command and puts its wrists where ``hand_track`` asks, so arms and legs are trained together.

Added on top of the walk:

- ``hand_track``: the G1 task's 8-dim command (active mask for ``[left, right]``, then a target per hand in the
  yaw-aligned root frame), appended to the policy and critic observations. The RPO has no hand or wrist link,
  so it names the forearms (``<side>_elbow_yaw_link``) and :class:`RPOHandPosCommand` tracks a point
  ``WRIST_OFFSET`` along each, near the forearm tip.
- **Target ranges** resized to the RPO's arm (shoulder about 0.27 m above and 0.18 m beside the root, about
  0.43 m of reach). They cover holding a box between the forearms at table height in front of the body, and
  lifting it.
- ``hand_tracking``: the G1 task's ``exp(-error / 0.3)`` reward, weight 0 at the start and ramped linearly to
  2.0 over the first 120k env steps (5000 PPO iterations) by the curriculum.
- ``add_forearm_mass``: 0-0.3 kg on each forearm at startup, so the policy also learns to balance with
  something light held in front of it.
- **Stance** (added after model_7999 stood at ~0.71 m with knees at 0.67-0.76 rad, more than double the
  default 0.30, which left carried gifts too low; the walk has no height term and rewards a 0.15 m swing):
  ``base_height`` holds the pelvis near ``STANCE_HEIGHT``, ``gait_clearance`` asks for a ``SWING_HEIGHT``
  swing instead of 0.15 m, and ``joint_deviation_sagittal`` lightly pulls hip pitch, knee and ankle pitch
  back toward the default pose.

Everything else -- rewards and weights (including the walk's ``track_ang_vel_z`` 1.5, not the G1 locomanip
task's 0.5), terminations, flat ground, physics, PPO -- is the walk's.
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from . import mdp
from . import walk_env_cfg as walk

FOREARMS = ["left_elbow_yaw_link", "right_elbow_yaw_link"]


##
# MDP settings
##


@configclass
class CommandsCfg(walk.CommandsCfg):
    """The walk's velocity command plus ``hand_track`` on the RPO's wrist points.

    Declared here, at class level, rather than patched in ``__post_init__``: the cfg's
    ``class_type`` string is only made resolvable while the cfg instance is constructed.
    """

    hand_track = mdp.HandPosCommandCfg(
        asset_name="robot",
        # left first, then right: HandPosCommand checks the prefixes
        body_names=list(FOREARMS),
        mode="random",
        resampling_time_range=(0.5, 2.0),
        rel_standing_envs=0.1,
        debug_vis=True,
        # yaw-aligned root frame, metres. x: from beside the hip to a forearm's length past a
        # table edge; y: each wrist on its own side, from a 0.2 m box's faces out to arms-wide;
        # z: from arms hanging at the hips to a lifted box held at chest height.
        ranges=mdp.HandPosCommandCfg.Ranges(
            pos_x=(0.0, 0.40),
            pos_y=(-0.35, 0.35),
            pos_y_left=(0.0, 0.35),
            pos_y_right=(-0.35, 0.0),
            pos_z=(-0.15, 0.35),
        ),
    )


@configclass
class ObservationsCfg(walk.ObservationsCfg):
    """The walk's observations with the hand command appended to both groups."""

    @configclass
    class PolicyCfg(walk.ObservationsCfg.PolicyCfg):
        hand_track = ObsTerm(func=mdp.generated_commands, params={"command_name": "hand_track"})

    @configclass
    class CriticCfg(walk.ObservationsCfg.CriticCfg):
        hand_track = ObsTerm(func=mdp.generated_commands, params={"command_name": "hand_track"})

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


STANCE_HEIGHT = 0.73
"""Pelvis target (m). The default pose stands at 0.747 m; a little lower leaves knee bend for stepping."""

SWING_HEIGHT = 0.09
"""Swing-foot clearance target (m) for ``gait_clearance``; the walk's 0.15 m is a lot for a ~0.55 m leg."""


@configclass
class RewardsCfg(walk.RewardsCfg):
    """The walk's rewards plus hand tracking and a taller stance."""

    # starts at 0; CurriculumCfg.hand_tracking owns the weight from then on
    hand_tracking = RewTerm(
        func=mdp.hand_pos_tracking_reward, weight=0.0, params={"command_name": "hand_track", "std": 0.3}
    )
    # the G1 walk's height term and weight, at the RPO's height
    base_height = RewTerm(func=mdp.base_height_l2, weight=-10.0, params={"target_height": STANCE_HEIGHT})
    # light: the knees still have to bend to step
    joint_deviation_sagittal = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_thigh_pitch_joint", ".*_knee_joint", ".*_ankle_pitch_joint"]
            )
        },
    )


@configclass
class EventCfg(walk.EventCfg):
    """The walk's events plus a light load on each forearm."""

    add_forearm_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=list(FOREARMS)),
            "mass_distribution_params": (0.0, 0.3),
            "operation": "add",
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    hand_tracking = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.hand_tracking.weight",
            "modify_fn": mdp.linear_ramp,
            "modify_params": {"start_step": 0, "end_step": 120_000, "start_value": 0.0, "end_value": 2.0},
        },
    )


##
# Environment configuration
##


@configclass
class LearningRLRPOLocoManipEnvCfg(walk.LearningRLRPOWalkV1EnvCfg):
    """The HCRL RPO walk (``LearningRLRPOWalkV1EnvCfg``), with wrist-position targets to track while walking."""

    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.rewards.gait_clearance.params["height"] = SWING_HEIGHT


@configclass
class LearningRLRPOLocoManipEnvCfg_PLAY(LearningRLRPOLocoManipEnvCfg):
    """Playback variant of ``LearningRLRPOLocoManipEnvCfg``, per the ``-PLAY`` rule in ``CLAUDE.md``."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 100
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0

        self.observations.policy.enable_corruption = False
