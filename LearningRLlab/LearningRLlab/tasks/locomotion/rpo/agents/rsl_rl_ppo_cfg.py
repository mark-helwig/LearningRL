"""PPO runner configuration for LearningRL-RPO-Walk-v0.

LearningRL-G1-Walk-v0's, under its own experiment name and wandb project.
"""

from isaaclab.utils.configclass import configclass

from LearningRLlab.tasks.locomotion.g1.agents.rsl_rl_ppo_cfg import LearningRLG1WalkPPORunnerCfg


@configclass
class LearningRLRPOWalkPPORunnerCfg(LearningRLG1WalkPPORunnerCfg):
    experiment_name = "learningrl_rpo_walk"
    wandb_project = "RPO-locomotion"
