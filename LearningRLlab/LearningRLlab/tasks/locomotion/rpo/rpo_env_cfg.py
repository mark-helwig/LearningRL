"""Environment configuration for LearningRL-RPO-Walk-v0: LearningRL-G1-Walk-v0 with the RoboParty RPO swapped in.

The RPO shares almost no joint names with the G1, so three
``joint_deviation_*`` terms have to be repointed rather than left to resolve: the RPO calls the
hip joints ``thigh``, has one ``torso_joint`` where the G1 has three ``waist_*_joint``, and calls
the arm chain ``arm_*``/``elbow_*`` with no wrist at all. Every one of those G1 regexes matches
ZERO RPO joints, and ``SceneEntityCfg`` raises on a pattern that matches nothing, so this is not
an optimization -- the env will not build without it.

Two targets are also retuned off the RPO's own geometry rather than carried over (see
``__post_init__``). Everything else -- MDP terms, weights, terrain, physics, PPO -- is the G1 walk's.
"""

from robot_description.robots.roboparty_rpo import RPO_CFG, RPO_FEET_CONTACT, RPO_TERMINATION_CONTACT
from robot_description.robots.roboparty_rpo.joint_groups import NOMINAL_HEIGHT

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from LearningRLlab.tasks.locomotion.g1.g1_env_cfg import LearningRLG1WalkEnvCfg

# Swing-foot target for `feet_clearance`, in world z. The reward reads `body_pos_w` of
# `.*ankle_roll.*` directly, so the number is the FOOT BODY ORIGIN's height, not the sole's.
# `rpo.urdf`'s foot mesh bottoms out 0.045 m below that origin, against the G1's 0.035 m
# (contact spheres at z -0.03, radius 0.005), so carrying the G1's 0.10 over would ask the RPO
# for 1 cm less real ground clearance. 0.11 keeps the G1's 0.065 m sole-above-ground target.
FEET_CLEARANCE_TARGET = 0.11


@configclass
class LearningRLRPOWalkEnvCfg(LearningRLG1WalkEnvCfg):
    """``LearningRLG1WalkEnvCfg`` with the RPO's robot, sensors, joint names, and standing height."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # F10: RPO_CFG is the DelayedPDActuatorCfg variant RoboParty tuned and trained against
        # (0-2 physics steps of action delay), not the implicit cfg the G1 walk uses.
        # RPO_IMPLICIT_CFG is the A/B baseline if the explicit PD chatters at this task's dt.
        self.scene.robot = RPO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # swap the G1 sensors attached by the parent for the RPO's, with the same settings
        self.scene.g1_feet = None
        self.scene.g1_termination = None
        RPO_FEET_CONTACT.attach(self.scene)
        RPO_TERMINATION_CONTACT.attach(self.scene)
        for sensor in (self.scene.rpo_feet, self.scene.rpo_termination):
            sensor.history_length = 1
            sensor.update_period = self.sim.dt
        self.scene.rpo_feet.force_threshold = 1.0

        self.rewards.gait.params["sensor_cfg"] = RPO_FEET_CONTACT.entity_cfg()
        self.rewards.feet_slide.params["sensor_cfg"] = RPO_FEET_CONTACT.entity_cfg()
        self.rewards.undesired_contacts.params["sensor_cfg"] = RPO_TERMINATION_CONTACT.entity_cfg()

        # -- joint names. None of the three G1 regex sets below match any RPO joint (see module
        # docstring). Spelled to cover exactly the same body parts the G1 terms cover.
        # arms: 3 shoulder-equivalents (`arm_pitch/roll/yaw`) + 2 elbow joints per side. The RPO
        # has no wrist, so the G1's `.*_wrist_.*` has no counterpart. `.*_arm_.*_joint` and
        # `.*_elbow_.*_joint` between them match all 10 upper-body joints and nothing else.
        self.rewards.joint_deviation_arms.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_arm_.*_joint", ".*_elbow_.*_joint"]
        )
        # waist: one yaw joint, spelled out rather than `torso.*` so a future torso_roll/pitch
        # doesn't join silently (joint_groups.py makes the same choice for the actuator group).
        self.rewards.joint_deviation_waists.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=["torso_joint"])
        # legs: the RPO's `thigh_yaw`/`thigh_roll` are the G1's `hip_yaw`/`hip_roll`. Their axes
        # are the tilted pair (-0.5, 0, -0.866) and (0.866, 0, -0.5), not world yaw/roll, but they
        # are the same two out-of-sagittal-plane DoFs this term exists to keep near neutral.
        self.rewards.joint_deviation_legs.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_thigh_roll_joint", ".*_thigh_yaw_joint"]
        )

        # -- heights, both G1 geometry in the parent. Forward kinematics on rpo.urdf in the
        # nominal pose puts the sole on the ground with base_link at 0.747 m, 3 mm under
        # NOMINAL_HEIGHT; the G1's 0.78 target is likewise its own 0.785 m standing height.
        self.rewards.base_height.params["target_height"] = NOMINAL_HEIGHT
        self.rewards.feet_clearance.params["target_height"] = FEET_CLEARANCE_TARGET

        # NOT overridden, deliberately: `add_base_mass` and `base_external_force_torque` both name
        # `torso_link`, which the RPO has under that exact spelling -- and, as on the G1, it is the
        # heaviest link (7.58 kg) and the one carrying the arms.


@configclass
class LearningRLRPOWalkEnvCfg_PLAY(LearningRLRPOWalkEnvCfg):
    """Playback variant of ``LearningRLRPOWalkEnvCfg``, per the ``-PLAY`` rule in ``CLAUDE.md``."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 100
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0

        self.observations.policy.enable_corruption = False
