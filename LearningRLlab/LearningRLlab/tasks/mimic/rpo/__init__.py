"""Gym registration for the LearningRL-RPO-BeyondMimic-v0 motion-tracking task."""

import gymnasium as gym

from . import agents

gym.register(
    id="LearningRL-RPO-BeyondMimic-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rpo_beyondmimic_env_cfg:LearningRLRPOBeyondMimicEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LearningRLRPOBeyondMimicPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="LearningRL-RPO-BeyondMimic-v0-PLAY",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rpo_beyondmimic_env_cfg:LearningRLRPOBeyondMimicEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LearningRLRPOBeyondMimicPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
