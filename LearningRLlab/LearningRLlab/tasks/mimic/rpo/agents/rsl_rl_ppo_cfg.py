"""PPO runner configuration for LearningRL-RPO-BeyondMimic-v0. Every field is set explicitly (see CLAUDE.md)."""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class LearningRLRPOBeyondMimicPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 15_000
    save_interval = 200

    experiment_name = "learningrl_rpo_beyondmimic"
    run_name = "beyondmimic"

    # F4: rsl_rl reads the wandb entity from WANDB_USERNAME, not WANDB_ENTITY.
    # One wandb project per robot ("<ROBOT>-locomotion"); a motion-tracking run is a
    # different task from walking, so it gets its own project rather than diluting that one.
    logger = "wandb"
    wandb_project = "RPO-beyondmimic"

    obs_groups = {"actor": ["policy"], "critic": ["critic"]}

    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=None,
    )
