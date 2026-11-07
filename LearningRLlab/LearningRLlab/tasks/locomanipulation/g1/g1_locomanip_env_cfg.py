"""Environment configuration for LearningRL-G1-LocoManipulation-v0: the G1 walk, plus reaching with either hand.

A port of hcrl_isaaclab's ``G1-LocoManipulation-Walking-v0`` (branch ``G1-locomanipulation``,
``b9820f1``). The walk half of that task is what ``LearningRL-G1-Walk-v0`` was ported from, so this
subclasses the walk and adds only the reaching half:

- ``hand_track``: an 8-dim command (active mask for ``[left, right]``, then a target for each hand)
  sampled in the robot's yaw-aligned root frame, appended to the policy and critic observations;
- ``hand_tracking``: an ``exp(-error / 0.3)`` reward, weight 0 at the start and ramped linearly to 2.0
  over the first 120k env steps (5000 PPO iterations) by the curriculum.

Two settings of the original are restored over the walk's own: flat ground (the original's terrain
generator held only flat tiles) and ``track_ang_vel_z`` at 0.5. Everything else -- sensors, physics,
events, terminations, the other rewards -- is the walk's.
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils.configclass import configclass

from LearningRLlab.scenes import FLAT_TERRAIN_CFG
from LearningRLlab.tasks.locomotion.g1 import g1_env_cfg as walk

from . import mdp

##
# Scene definition
##


@configclass
class SceneCfg(walk.SceneCfg):
    """The walk's scene on a flat ground plane."""

    terrain = FLAT_TERRAIN_CFG


##
# MDP settings
##


@configclass
class CommandsCfg(walk.CommandsCfg):
    """The walk's velocity command plus the hand targets."""

    hand_track = mdp.HandPosCommandCfg(
        asset_name="robot",
        body_names=["left_wrist_yaw_link", "right_wrist_yaw_link"],
        mode="random",
        resampling_time_range=(0.5, 2.0),
        rel_standing_envs=0.1,
        debug_vis=True,
        ranges=mdp.HandPosCommandCfg.Ranges(
            pos_x=(0.0, 0.4),
            pos_y=(-0.45, 0.45),
            # each hand reaches out to its own side only (+y is the robot's left)
            pos_y_left=(0.0, 0.45),
            pos_y_right=(-0.45, 0.0),
            pos_z=(0.0, 0.6),
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


@configclass
class RewardsCfg(walk.RewardsCfg):
    """The walk's rewards plus hand tracking."""

    # starts at 0; CurriculumCfg.hand_tracking owns the weight from then on
    hand_tracking = RewTerm(
        func=mdp.hand_pos_tracking_reward, weight=0.0, params={"command_name": "hand_track", "std": 0.3}
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
class LearningRLG1LocoManipEnvCfg(walk.LearningRLG1WalkEnvCfg):
    """``LearningRLG1WalkEnvCfg`` on flat ground, with hand-position targets to track while walking."""

    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # the walk raises the spawn 5 cm to clear its boxes; on flat ground it would only drop the robot
        self.events.reset_base.params["pose_range"] = {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)}
        # the original's weight; the walk has since doubled it
        self.rewards.track_ang_vel_z.weight = 0.5


@configclass
class LearningRLG1LocoManipEnvCfg_PLAY(LearningRLG1LocoManipEnvCfg):
    """Playback variant of ``LearningRLG1LocoManipEnvCfg``, per the ``-PLAY`` rule in ``CLAUDE.md``."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 100
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0

        self.observations.policy.enable_corruption = False
