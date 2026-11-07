# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Overview

`LearningRL` is a monorepo of three Python sub-projects for humanoid
locomotion RL on Isaac Lab `v3.0.0-beta2.patch1` (Newton / MuJoCo-Warp physics):

| Sub-project | Package | Role | Needs |
|---|---|---|---|
| `LearningRLlab/` | `LearningRLlab` | Isaac Lab tasks, PPO cfgs, train/play wrappers, sim2sim export | Isaac Lab + GPU |
| `robot_description/` | `robot_description` | Robot `ArticulationCfg`s, contact-sensor groups, validation | Isaac Lab for `robots.*`; the rest is pure Python |
| `LearningRLsim/` | `LearningRLsim` | MuJoCo replay of an exported policy | CPU only -- **never** Isaac Lab |

Students edit tasks (and possibly robot configs) on `student/<name>` branches;
training happens elsewhere and returns an exported run directory
(`config.yaml` + `policy.onnx` + `policy.onnx.data`) that `LearningRLsim` replays.

Tasks: `LearningRL-G1-Walk-v0` (Unitree G1, 29-DoF), `LearningRL-RPO-Walk-v0`
(RoboParty RPO, 23-DoF) and `LearningRL-G1-LocoManipulation-v0` (the G1 walk plus
hand-position targets), each with a `-PLAY` twin. The RPO walk and the
locomanipulation task both subclass the G1 walk.

## Key Commands

```bash
# repo root, Python 3.12 venv
gitman install                                   # vendor assets -> robot_description/vendor/
pip install -e robot_description -e LearningRLsim
ruff check . && ruff format --check .
pytest robot_description/tests LearningRLsim/tests
pytest -m static LearningRLlab/tests             # separately: two test_linting.py basenames collide

# replay (= python -m LearningRLsim.eval, which the training side calls)
learningrl-sim <run-dir> --commands walk_baseline [--spawn-height 0.75]

# training side, inside the Isaac Lab container, from /workspace/isaaclab (F3)
./isaaclab.sh -p source/LearningRLlab/scripts/train.py --task LearningRL-RPO-Walk-v0 --viz kit --enable_cameras physics=newton_mjwarp
./isaaclab.sh -p source/LearningRLlab/scripts/play.py --task LearningRL-RPO-Walk-v0-PLAY --viz kit --enable_cameras physics=newton_mjwarp --video
python source/LearningRLlab/scripts/list_envs.py [--filter X] [--plain]
```

Install order on the training side: `robot_description` first, then
`LearningRLlab`, then `LearningRLsim --no-deps` (LearningRLlab imports
`robot_description` without declaring it -- pip would look for it on PyPI).

## Repository Structure

```
LearningRL/
├── gitman.yml                       rpo_description + unitree_ros @ locked SHAs -> robot_description/vendor
├── LearningRLlab/
│   ├── LearningRLlab/               THE package (nested layout) -- no omni.* imports at module scope
│   │   ├── export.py                config.yaml writer for sim2sim
│   │   ├── scenes.py, terrain_generator.py
│   │   ├── LearningRL_mdp/          shared MDP terms (symmetry.py)
│   │   ├── tasks/locomotion/{g1,rpo}/   env cfg, mdp/, agents/rsl_rl_ppo_cfg.py, gym registration
│   │   └── tasks/locomanipulation/g1/   the G1 walk + HandPosCommand, its reward and weight ramp
│   ├── scripts/                     train.py, play.py, list_envs.py
│   └── tests/                       static: ruff gate, isaaclab_assets ban, -PLAY rule
├── robot_description/
│   ├── robot_description/           THE package (nested layout, F6)
│   │   ├── paths.py                 repo/vendor/build paths; urdf_path() never raises
│   │   ├── robots/{unitree_g1,roboparty_rpo}/   joint_groups.py, robot.py, sensor.py, robot.yaml
│   │   └── tools/                   robot_schema.py, mesh_ops.py, validate.py
│   ├── docs/                        vendor-assets.md, adding-a-robot.md
│   ├── vendor/                      gitman checkout -- gitignored
│   └── build/                       on-the-fly USD cache -- gitignored
└── LearningRLsim/
    ├── LearningRLsim/               eval.py (THE entrypoint), runconfig.py, registry.py, observations.py,
    │                                actions.py, model.py, policy.py, loop.py, commands/, data/
    ├── commands/                    scripted timelines (stand, walk_baseline, reach_baseline), by bare name
    ├── scenes/                      scene objects (playground), by bare name
    ├── layouts/                     Foxglove layout template
    ├── tools/foxglove_layout.py     layout generator for a bag -- not packaged
    └── tests/                       a package (tests.conftest imports)
```

## LearningRLlab

Each task is a manager-based RL env: a `ManagerBasedRLEnvCfg` subclass, an
`RslRlOnPolicyRunnerCfg` subclass, and two `gym.register` calls (the task and
its `-PLAY` twin) with `isaaclab.envs:ManagerBasedRLEnv` as entry point.
Training reuses upstream's rsl_rl scripts; LearningRLlab only makes its tasks
visible to them.

### Key patterns
- **Overrides go in `__post_init__`, after `super().__post_init__()`** --
  upstream cfgs assign fields unconditionally there, so a class-level override
  is silently clobbered.
- **The PPO cfg sets every field explicitly**, so the file is a complete record
  of the run's hyperparameters.
- **One wandb project per robot and task family** (`G1-locomotion`,
  `RPO-locomotion`, `G1-locomanipulation`), set by `wandb_project` in the runner
  cfg. A runner cfg that subclasses another robot's or family's must override it.
- **The `-PLAY` rule:** every task ships a `<task-id>-PLAY` twin -- its own env
  cfg subclass with a literal `scene.num_envs = 100` and a robot-tracking viewer
  (`viewer.origin_type = "asset_root"`, `viewer.asset_name`). Same agent cfg.
  Enforced by `tests/test_play_variant.py`.
- **Robot cfgs come from `robot_description`**, never `isaaclab_assets`
  (`tests/test_no_isaaclab_assets.py`).
- **Joint/body names are a task concern; which bodies are sensed is an asset
  concern.** The RPO names its hips `thigh`, has one `torso_joint` and no wrist,
  so `rpo_env_cfg.py` repoints the G1's `joint_deviation_*` terms -- a regex that
  matches nothing raises at env build.
- **MDP term placement:** task-local `mdp/` first; move (never copy) to
  `LearningRL_mdp/` once a second task needs it. A subclassing task's `mdp/`
  re-exports its parent's (`locomanipulation/g1/mdp` does `from ...locomotion.g1.mdp import *`).
- **Mirror augmentation covers every policy input.** `LearningRL_mdp/symmetry.py`
  raises on a term it has no rule for. A command whose layout is not
  `(vx, vy, wz)` declares `mirror_map = (perm, sign)` on its term class
  (`HandPosCommand` swaps the hands and negates y).
- **Never modify Isaac Lab source** -- copy a term into the task's `mdp/` and
  change it there.

### Pinned Isaac Lab facts
IDs are cited from code comments -- update those when renumbering.

- **F1 -- `./isaaclab.sh -i` won't install this extension.** Both packages need
  `pip install -e`, `robot_description` first.
- **F2 -- upstream's train/play scripts can't see LearningRLlab tasks.**
  `scripts/train.py`/`play.py` import `LearningRLlab.tasks`, then load
  `train_rsl_rl.py`/`play_rsl_rl.py` by file path.
- **F3 -- the log directory is CWD-relative**
  (`logs/rsl_rl/<experiment_name>/<timestamp>_<run_name>`). Run from
  `/workspace/isaaclab`.
- **F4 -- rsl_rl passes `entity=$WANDB_USERNAME` to `wandb.init()`**, overriding
  `WANDB_ENTITY`. The wandb run name is the log directory's basename.
- **F5 -- a field set on a `PresetCfg` wrapper is silently discarded.** Name the
  variant explicitly (`sim.physics.newton_mjwarp.solver_cfg.nconmax`) or replace
  the whole entry.
- **F6 -- a task's `mdp/__init__.py` re-exports upstream's leaf modules first,
  then its own**, never the `isaaclab.envs.mdp` package itself (it binds lazily
  and drags in `pxr`, see F9).
- **F7 -- the physics preset defaults to PhysX.** Pass `physics=newton_mjwarp`.
- **F8 -- asset/sensor data are `ProxyArray`s, and quaternions are
  `(x, y, z, w)`.** Read via `.torch` (or `.warp`).
- **F9 -- nothing on the env-cfg import path may import `pxr`.** The cfg is
  imported during Hydra task resolution, before Kit exists; a second USD crashes
  Kit startup. Check a cfg with
  `python -c "import sys,importlib;importlib.import_module('<cfg module>');print([m for m in sys.modules if m.startswith('pxr')])"`
  -- it must print `[]`.
- **`--viz kit` needs `--enable_cameras`**, or Kit's visualizer fails on
  `No module named 'omni.kit.viewport'` while still exiting 0. `--video` sets it
  implicitly.

### Sim2sim export
`play.py` writes `<run_dir>/<run_name>/{policy.onnx, policy.onnx.data, config.yaml}`
by default (`--no-export` skips it, `--robot-name` overrides the sniffed robot
key). `export.py` builds the yaml -- the contract `LearningRLsim` consumes.

- Upstream `play_rsl_rl.py` has no hook between `runner.load()` and its rollout,
  so `play.py` wraps `OnPolicyRunner.load` and `export_policy_to_onnx`.
- rsl-rl emits `policy.onnx` **plus** a `policy.onnx.data` sidecar holding the
  weights -- both must travel together.
- Actuator gains come from the cfg, never the live tensors.
- Observation corruption is dropped by contract: every `-PLAY` cfg sets
  `observations.policy.enable_corruption = False`.
- **Not exported:** the robot's init height, terrain friction, gravity, and the
  actuator model. See LearningRLsim below.

## robot_description

Each robot is a `robot.yaml` (source URDF/MJCF plus validation expectations)
and a `robot.py` that builds `ArticulationCfg`s on `RobotUrdfFileCfg`, which
converts URDF to USD at spawn time. Joint data lives in an Isaac-Lab-free
`joint_groups.py`, sensed-body lists in `sensor.py`.

### Key patterns
- **G1 gains and limits come from `unitree_rl_lab`'s
  `UNITREE_G1_29DOF_MIMIC_CFG`**; `isaaclab_assets` encodes a different gear variant.
- **`G1_CFG` is implicit PD; `RPO_CFG` is `DelayedPDActuatorCfg`** (F10), the
  actuator RoboParty trained against, with `RPO_IMPLICIT_CFG` as the fallback.
- **Actuator groups partition every joint exactly once.** Zero groups is a limp
  limb, two is a hard error. Not test-enforced -- check by hand.
- **No hand-edited USD.** Fix the URDF, `robot.yaml` or `RobotUrdfFileCfg`.
- **`robot.yaml` is strict** -- an unknown key is a hard error.
- **Isaac-Lab-free modules:** `paths.py`, `tools/robot_schema.py`,
  `tools/mesh_ops.py`, each robot's `joint_groups.py`. A robot package's
  `__init__.py` imports `robot.py` eagerly, so `robot_description.robots.<robot>`
  needs Isaac Lab.
- **`paths.urdf_path()` never raises** (it runs at import time); `validate.py`
  Level A is the loud check. `REPO_ROOT` is `parents[1]` of `paths.py`, so
  `VENDOR_DIR` resolves to `robot_description/vendor/` inside this monorepo.

### Resolved facts
- **F1 -- URDF conversion requires Kit.** Spawning a robot needs Kit running.
  `paths.mjcf_path()` hands the vendor MJCF straight to MuJoCo, no Kit.
- **F1b -- the MJCF is a second declared source** (`source.mjcf_path`), not a
  suffix swap: right for the G1, wrong for the RPO (`urdf/rpo.urdf` vs
  `mjcf/rpo.xml`).
- **F2a -- G1 vendor is `unitreerobotics/unitree_ros`** (BSD-3-Clause, relative
  mesh paths). `g1_29dof_rev_1_0.urdf`: 29 revolute joints, 33.341 kg, root `pelvis`.
- **F2b -- RPO vendor is `Roboparty/rpo_description`** (CERN-OHL-W-2.0).
  `urdf/rpo.urdf`: 23 revolute joints, zero fixed joints, 33.757 kg, root
  `base_link`, MJCF at `mjcf/rpo.xml`.
- **F3 -- gitman `sparse_paths` only shrinks the working tree**; the `--mirror`
  cache stays full size. Never `--depth=1` (can't check out a locked SHA).
- **F4 -- `gitman lock` rewrites `gitman.yml` and drops comments**, so pin
  rationale lives in `robot_description/README.md`.
- **F6 -- nested package layout.** Don't flatten or remap via `package-dir`.
- **F7 -- a zero-gain `joint_drive` needs `ensure_drives_exist=True`**, or Newton
  spawns the robot unactuated.
- **F8 -- the converted USD nests bodies by kinematic chain**; with
  `merge_fixed_joints=True` the G1 has 30 rigid bodies.
- **F9 -- `exclude_self=True` separates self-contact, Newton only.**
- **F10 -- `RPO_CFG` is explicit** (`DelayedPDActuatorCfg` subclasses
  `IdealPDActuatorCfg`), so `effort_limit` and `effort_limit_sim` are set from
  the same dict.
- **F11 -- 8 of `rpo.urdf`'s 24 links have no `<collision>`**, so a contact
  sensor on them reads zero. Do not flip `collision_from_visuals`.

### Known issues
- `pytest -m sim` fails `test_groups_partition_all_bodies[unitree_g1]`: both
  `.*_ankle_pitch_link`s are in no G1 sensor group. The RPO passes it.
- `validate.py unitree_g1` fails A3 (`imu_in_torso`, `imu_in_pelvis`,
  `d435_link`, `mid360_link` near-massless but not in `massless_links`), B1
  (MuJoCo resolves the URDF's meshes to `meshes/meshes/...`) and D4
  (`robot.yaml`'s `nominal_joint_pos` no longer matches `robot.py`).
  `validate.py roboparty_rpo --strict` passes Levels A, B and D.
- `validate.py --sim` (Level E) doesn't launch Kit, so it fails with
  `No module named 'omni.physics'` before spawning anything.

## LearningRLsim

**Never imports Isaac Lab, Isaac Sim, Newton, warp or torch -- including
transitively** (`tests/test_import_purity.py`). It must run on a CPU-only
machine from `pip install -e`, which is why its `pyproject.toml` declares real
dependencies (`mujoco`, `onnxruntime`, `numpy`, `pyyaml`, `mcap`, `zstandard`,
`foxglove-websocket`). On the training side it runs with plain `python`, never
`./isaaclab.sh -p`, which would put Isaac Lab on its path.

**The run directory is the only input.** Never reach back into a training-time
config, a task class, or a wandb run. `config.yaml`'s `robot` keys into
`robot_description`; LearningRLsim vendors no assets.

### Name-keyed registries
`config.yaml` names things (`func: base_ang_vel`, `type: JointPositionAction`,
`type: UniformVelocityCommandCfg`). LearningRLsim keeps registries keyed on
those strings and instantiates only what the config names. **An unrecognised
name is a loud failure listing what is registered**, and an exported param a
term does not honour is an error (`_accept()` in `observations.py`) -- a silently
wrong observation vector walks slightly badly rather than failing.

### Facts worth keeping
- **`joint_order` is not MuJoCo's order.** Remap by name; getting it wrong gives
  a robot that stands then falls plausibly.
- **`actions.total_dim` can exceed the joint count** with more than one action
  term; `last_action` is the full raw vector.
- **Free-joint velocity frames are mixed:** `qvel[0:3]` is world-frame linear,
  `qvel[3:6]` body-frame angular. Pinned against the G1's pelvis gyro.
- **The vendor MJCFs declare no armature/friction/timestep defaults**;
  `model.py` writes `sim_dt`, `actuators.armature` and `actuators.friction` from
  `config.yaml` into the model.
- **MuJoCo enums don't compare reliably against numpy arrays across versions**
  -- `model.py` compares plain ints. The training image pins mujoco 3.8.1; a
  standalone venv gets a newer one, so run the CPU-only tests after touching
  anything that reads a MuJoCo type array.
- **`mj_step` leaves `xpos` one physics step behind `qpos`**;
  `refresh_link_poses()` runs before `/tf`. The same lag remains in the
  observation path (`base_rot_w` reads `xmat`) -- deliberately not changed.
- **`python -m`/`python -c` from the repo root breaks imports:** the root's
  `LearningRLsim/` and `robot_description/` directories (no `__init__.py`) win
  as namespace packages over the editable installs' meta-path finders, and
  `LearningRLsim/commands/` masquerades as `LearningRLsim.commands`. Hence the
  `learningrl-sim` console script, and `cwd=HERE` in `test_import_purity.py`.
  pytest is unaffected (it prepends each sub-project to `sys.path`).
- **A payload is JSON-encoded before `write()` returns**, on the rollout thread,
  since the loop rewrites its numpy arrays in place.
- **Foxglove:** `/tf` must carry the schema name `foxglove.FrameTransforms`;
  `scene.meshUpAxis` must be `"z_up"`; the URDF layer takes an absolute path, so
  a layout is only valid on the machine whose checkout it names (`--urdf`).

### Physics the exporter does not record
Taken from flags and written into `/meta/run` marked **assumed**:

| Not exported | Default | Flag |
|---|---|---|
| initial root height | 0.8 (G1); **pass 0.75 for the RPO** | `--spawn-height` |
| terrain friction | 1.0 | `--friction` |
| gravity | -9.81 | `--gravity` |
| `heading_control_stiffness` | 1.0 | `--heading-stiffness` |

**The actuator model is not exported either.** LearningRLsim implements the
`IdealPDActuator` law with no delay. The RPO trains with `DelayedPDActuatorCfg`
(0-2 physics steps of delay), so an RPO replay shows a **numeric divergence**
from training, not a visual one -- don't read it as a policy finding.

## Coding Standards

- **ruff only** -- each sub-project's `pyproject.toml` holds its config
  (line length 120 for LearningRLlab/LearningRLsim, 100 for robot_description).
  Not upstream's black/flake8/isort.
- Mark tests `static` when they need no Isaac Lab, GPU, MuJoCo model or
  `vendor/`.
- Keep comments to two lines max.
- Physical numbers (masses, limits, gains) are transcribed from a named source,
  never estimated.
- No defensive scaffolding for failures that cannot occur.
