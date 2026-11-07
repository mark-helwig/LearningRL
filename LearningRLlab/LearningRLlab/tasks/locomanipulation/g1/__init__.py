"""Gym registration for the LearningRL-G1-LocoManipulation-v0 task."""

import gymnasium as gym

from . import agents

gym.register(
    id="LearningRL-G1-LocoManipulation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_locomanip_env_cfg:LearningRLG1LocoManipEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LearningRLG1LocoManipPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="LearningRL-G1-LocoManipulation-v0-PLAY",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_locomanip_env_cfg:LearningRLG1LocoManipEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LearningRLG1LocoManipPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
