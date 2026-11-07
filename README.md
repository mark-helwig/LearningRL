# LearningRL

Reinforcement-learning locomotion for humanoid robots, built on
[Isaac Lab](https://isaac-sim.github.io/IsaacLab). You design the task -- the
rewards, observations, terminations and PPO settings -- for a walking robot.
The TA trains it on a GPU workstation and sends you back the trained policy,
which you replay and inspect on your own laptop in MuJoCo.

You never need a GPU, Docker, or Isaac Sim yourself.

## What's in the repo

```
LearningRL/
├── LearningRLlab/        Isaac Lab tasks: the environments you edit and the TA trains
├── robot_description/    robot configs (joints, gains, limits, contact sensors)
├── LearningRLsim/        MuJoCo replay of a trained policy -- runs on any CPU-only machine
└── gitman.yml            fetches the robots' meshes into robot_description/vendor/
```

Two robots ship with three tasks:

| Task id | Robot |
|---|---|
| `LearningRL-RPO-Walk-v0` | RoboParty RPO (ROBOTO_ORIGIN), 23 joints -- **the one you work on** |
| `LearningRL-G1-Walk-v0` | Unitree G1, 29 joints -- the base task the RPO walk inherits from |
| `LearningRL-G1-LocoManipulation-v0` | Unitree G1 -- the G1 walk, plus reaching hand targets with either or both hands |

Each has a `-PLAY` twin (e.g. `LearningRL-RPO-Walk-v0-PLAY`) that the TA uses to
record a video and export the policy.

## Where you edit

**`LearningRLlab/LearningRLlab/tasks/locomotion/rpo/`** is your task:

- `rpo_env_cfg.py` -- `LearningRLRPOWalkEnvCfg`. Change rewards, observations,
  terminations, commands and events in its `__post_init__`, after
  `super().__post_init__()` (a class-level override would be overwritten).
- `agents/rsl_rl_ppo_cfg.py` -- the PPO hyperparameters (network size,
  learning rate, entropy, iterations, ...).

**The RPO walk inherits from the G1 walk.** `LearningRLRPOWalkEnvCfg` subclasses
`LearningRLG1WalkEnvCfg` in `LearningRLlab/LearningRLlab/tasks/locomotion/g1/g1_env_cfg.py`,
so every reward term, its weight, the observation set, the terrain and the
physics start out as the G1's. Read that file first: `RewardsCfg` lists every
reward term and weight, `ObservationsCfg` what the policy sees, and
`TerminationsCfg` when an episode ends. Custom reward terms live in
`g1/mdp/rewards.py`. Override what you want to change from the RPO file --
for example:

```python
self.rewards.feet_slide.weight = -0.5
self.rewards.base_height.params["target_height"] = 0.72
self.rewards.energy = None  # remove a term
```

Anything else about the reward functions Isaac Lab ships is in its
[`isaaclab.envs.mdp`](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html)
module. To add a new reward term, write it in the task's own `mdp/` package
rather than editing Isaac Lab.

**The robot itself** is in
`robot_description/robot_description/robots/roboparty_rpo/`: `joint_groups.py`
holds joint names, PD gains, limits and the standing pose; `robot.py` builds the
Isaac Lab config from them; `sensor.py` names which bodies count as feet and
which end an episode on contact.

## Checks you can run locally

In a Python 3.12 virtual environment, from the repo root:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install ruff pytest gitman
gitman install                                     # robot meshes into robot_description/vendor
pip install -e robot_description -e LearningRLsim

ruff check . && ruff format --check .              # the same lint the TA runs
pytest robot_description/tests LearningRLsim/tests
pytest -m static LearningRLlab/tests               # static checks on the tasks (ruff, AST rules)
```

Run the LearningRLlab tests as their own `pytest` command: `LearningRLlab/tests`
and `robot_description/tests` each have a `test_linting.py`, and pytest refuses
to collect two modules with the same name in one run. Actually building and
training an environment needs Isaac Lab and a GPU -- that is what the TA runs.

## Submitting

1. Create a branch named `student/<your-name>` and push your changes to it.
   **Never push to `main`.**
2. Make sure the checks above pass.
3. Tell the TA which commit to train (`git rev-parse --short HEAD`).

The TA's training run is named after your branch and commit, so every result
traces back to exactly the code that produced it.

## Receiving results

The TA sends you a `.tar.gz` named after the run (which carries your branch
and commit, e.g. `2026-10-01_12-00-00_student_alice-1a2b3c4.tar.gz`), containing three files:
`config.yaml`, `policy.onnx` and `policy.onnx.data`. Extract it anywhere and
replay it:

```bash
mkdir my_run && tar -xzf <archive>.tar.gz -C my_run
learningrl-sim my_run --commands walk_baseline
```

`walk_baseline` stands for 5 s, then walks forward, backward, left,
right, and turns both ways. Without `--commands` the robot just stands. For a
`LearningRL-G1-LocoManipulation-v0` run, `--commands reach_baseline` reaches with
each hand mode while standing, then while walking and turning.

The rollout prints a summary (distance walked, whether it fell) and writes an
MCAP bag to `my_run/sim2sim/`. To look at it:

- **Watch live:** add `--live` and, in [Foxglove](https://foxglove.dev/download)
  desktop, "Open connection" to `ws://127.0.0.1:8765`.
- **Drive it yourself:** add `--keyboard` (`w`/`a`/`s`/`d` to walk, `q`/`e` to
  turn, space to stop, `r` to reset, `x` to quit).
- **Inspect the bag in 3D:** generate a Foxglove layout for it, then in
  Foxglove "Import from file..." the layout and open the bag:
  ```bash
  python LearningRLsim/tools/foxglove_layout.py my_run/sim2sim/<bag>.mcap \
      --urdf "$PWD/robot_description/vendor/rpo_description/urdf/rpo.urdf"
  ```

MuJoCo is not the simulator the policy trained in, and the RPO's actuator model
(a delayed PD controller) is simplified to a plain PD controller here, so expect
some difference from the training videos. A policy that only works in one
simulator is worth knowing about. See [`LearningRLsim/README.md`](LearningRLsim/README.md)
for every option.

## Licences

- The code in this repository is MIT-licensed; see [`LICENSE`](LICENSE).
- **Third-party notices.**
  `robot_description/robot_description/robots/roboparty_rpo/robot.py` is derived
  from RoboParty's [RoboLab](https://github.com/Roboparty/robolab) under the
  BSD-3-Clause licence, whose notice is reproduced at the top of that file.
- The RPO's URDF and meshes come from
  [`Roboparty/rpo_description`](https://github.com/Roboparty/rpo_description)
  under CERN-OHL-W-2.0, and the G1's from
  [`unitreerobotics/unitree_ros`](https://github.com/unitreerobotics/unitree_ros)
  under BSD-3-Clause. Neither is redistributed here: `gitman install` fetches
  them onto your machine.
