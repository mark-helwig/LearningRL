"""PPO runner configuration for LearningRL-G1-Walk-v0. Every field is set explicitly (see CLAUDE.md)."""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg

from LearningRLlab.LearningRL_mdp import symmetry


@configclass
class LearningRLG1WalkPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50_000
    save_interval = 1000

    experiment_name = "learningrl_g1_walk"
    run_name = "walk"

    # F4: rsl_rl reads the wandb entity from WANDB_USERNAME, not WANDB_ENTITY.
    # One wandb project per robot ("<ROBOT>-locomotion"), so runs compare like for like.
    logger = "wandb"
    wandb_project = "G1-locomotion"

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
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # left/right mirror doubles every minibatch; the mirror loss is still logged as "symmetry"
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=False,
            data_augmentation_func=symmetry.compute_symmetric_states,
            mirror_loss_coeff=0.0,
        ),
    )
