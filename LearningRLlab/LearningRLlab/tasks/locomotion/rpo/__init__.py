"""Gym registration for the LearningRL-RPO-Walk-v0 locomotion task."""

import gymnasium as gym

from . import agents

gym.register(
    id="LearningRL-RPO-Walk-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rpo_env_cfg:LearningRLRPOWalkEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LearningRLRPOWalkPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="LearningRL-RPO-Walk-v0-PLAY",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rpo_env_cfg:LearningRLRPOWalkEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LearningRLRPOWalkPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
