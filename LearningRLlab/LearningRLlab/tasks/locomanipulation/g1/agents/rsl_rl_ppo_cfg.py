"""PPO runner configuration for LearningRL-G1-LocoManipulation-v0.

LearningRL-G1-Walk-v0's, under its own experiment name and wandb project. The original task's
``G1WalkingBootstrapPPORunnerCfg`` was the same PPO, minus the left/right mirror augmentation;
``HandPosCommand.mirror_map`` is what lets the mirror cover the hand command.
"""

from isaaclab.utils.configclass import configclass

from LearningRLlab.tasks.locomotion.g1.agents.rsl_rl_ppo_cfg import LearningRLG1WalkPPORunnerCfg


@configclass
class LearningRLG1LocoManipPPORunnerCfg(LearningRLG1WalkPPORunnerCfg):
    experiment_name = "learningrl_g1_locomanipulation"
    run_name = "locomanip"
    wandb_project = "G1-locomanipulation"
