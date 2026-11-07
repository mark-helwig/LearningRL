# LearningRLlab

The Isaac Lab extension for LearningRL: locomotion task definitions, PPO
configs, and train/play wrappers around Isaac Lab's rsl_rl scripts. Training
runs on the TA's workstation; see the repo-root `README.md` for the student
workflow.

## Layout

```
LearningRLlab/                    <- THE PACKAGE (nested)
├── __init__.py                   no omni.* imports at module scope
├── export.py                     writes config.yaml beside policy.onnx for sim2sim
├── scenes.py, terrain_generator.py   the terrain the walks train on
├── LearningRL_mdp/               MDP terms shared across tasks (the left/right symmetry)
├── tasks/locomanipulation/
│   └── g1/                       LearningRL-G1-LocoManipulation-v0 (+ -PLAY): the G1 walk + hand targets
│       ├── g1_locomanip_env_cfg.py   subclasses the G1 walk; adds the command, obs, reward, curriculum
│       ├── mdp/                  HandPosCommand(Cfg), hand_pos_tracking_reward, linear_ramp
│       └── agents/rsl_rl_ppo_cfg.py
└── tasks/locomotion/
    ├── g1/                       LearningRL-G1-Walk-v0 (+ -PLAY): the base walk
    │   ├── g1_env_cfg.py         scene, commands, observations, rewards, terminations
    │   ├── mdp/                  task-local reward and event terms
    │   └── agents/rsl_rl_ppo_cfg.py
    └── rpo/                      LearningRL-RPO-Walk-v0 (+ -PLAY): the G1 walk, RPO swapped in
        ├── rpo_env_cfg.py        subclasses the G1 walk; overrides only what the RPO needs
        └── agents/rsl_rl_ppo_cfg.py
scripts/                          train.py, play.py, list_envs.py
tests/                            static tests: ruff gate, isaaclab_assets ban, -PLAY rule
```

## Train / play (inside the Isaac Lab container)

Run from `/workspace/isaaclab`; log paths are CWD-relative.

```bash
./isaaclab.sh -p source/LearningRLlab/scripts/train.py --task LearningRL-RPO-Walk-v0 --viz kit --enable_cameras physics=newton_mjwarp
./isaaclab.sh -p source/LearningRLlab/scripts/play.py --task LearningRL-RPO-Walk-v0-PLAY --viz kit --enable_cameras physics=newton_mjwarp --video
python source/LearningRLlab/scripts/list_envs.py
```

`play.py` also exports `policy.onnx`, `policy.onnx.data` and `config.yaml` into
`logs/rsl_rl/<experiment>/<run>/<run>/` -- the directory LearningRLsim replays.

## Lint / test

```bash
ruff format --check . && ruff check .
pytest -m static tests/     # ruff + AST scans; no Isaac Lab needed
```
