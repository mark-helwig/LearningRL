# robot_description

Robot configs for LearningRL. Owns per-robot Isaac Lab configs, named
contact-sensor groups, mesh utilities, and model validation. Every robot config
is defined here end to end -- **Isaac Lab's built-in Unitree assets
(`isaaclab_assets`) are never used.**

Robots spawn directly from their vendored URDF via Isaac Lab's `UrdfFileCfg`
(see `robot_description/robots/urdf_common.py`), which converts to USD on
the fly at spawn time and caches the result -- there is no separate offline
"convert to USD" step to run first.

The same package also serves LearningRLsim on a CPU-only machine: its
Isaac-Lab-free modules (`paths.py`, `tools/robot_schema.py`,
`tools/mesh_ops.py`, each robot's `joint_groups.py`) import without Isaac Lab,
and `paths.mjcf_path()` hands the vendor MJCF to MuJoCo.

## What lives where

```
robot_description/            <- THE PACKAGE (nested; see the root CLAUDE.md)
├── paths.py                  repo-root/vendor/USD-cache path resolution
├── robots/
│   ├── urdf_common.py        shared RobotUrdfFileCfg spawn config
│   ├── unitree_g1/           robot.py + robot.yaml (Unitree G1, 29-DoF)
│   └── roboparty_rpo/        same layout (RoboParty RPO, 23-DoF; ported from RoboParty's RPO_CFG)
└── tools/
    ├── robot_schema.py       strict robot.yaml loader
    ├── mesh_ops.py           mesh reference auditing + STL stats
    └── validate.py           Levels A, B, D, E validation CLI
docs/                         vendor comparison, adding a robot
tests/                        static (no Isaac Lab) tests + sim tests
vendor/                       gitman-managed vendor checkouts, GITIGNORED
build/                        on-the-fly USD conversion cache, GITIGNORED
```

## Setup

The vendor assets are fetched by the **repo-root** `gitman.yml`, which places
them at `robot_description/vendor/` -- the path `paths.py` resolves.

```bash
# from the LearningRL repo root
gitman install
pip install -e robot_description
python -m robot_description.tools.validate roboparty_rpo --strict   # optional sanity check
```

On the training side this package must be installed **before** LearningRLlab,
because LearningRLlab imports robot configs from here
(`from robot_description.robots.roboparty_rpo import RPO_CFG`).

## Vendor pins

**`unitreerobotics/unitree_ros`** @ `ccfc6fd8430a17ba3dacef9a1e2faf64ff3b0aee`
(BSD-3-Clause), sparse path `robots/g1_description`, for the G1. Why this repo
and not one of the alternatives: see `docs/vendor-assets.md`.

**`Roboparty/rpo_description`** @ `37aac9ca665e92731444a1618320078e7ba21569`,
sparse paths `urdf`, `meshes`, `mjcf`, for the RPO. Two notes:

- *Why this repo, not the one the robot is announced from.* RoboParty's
  headline repo is `Roboparty/roboto_origin`, a snapshot-aggregation monorepo
  that vendors its own submodules -- including OrangePi/Armbian firmware build
  trees, which put it at **~1.9 GB**. `rpo_description` is the 57 MB standalone
  source of the same assets, and its `urdf/rpo.urdf` is byte-identical to the
  monorepo's copy.
- *Licence.* `rpo_description` is **CERN-OHL-W-2.0**, a weakly-reciprocal
  hardware licence. `vendor/` is gitignored and this repo redistributes none of
  it -- gitman fetches the meshes on each machine.

The RPO's `robot.py` is a separate matter: it is ported from RoboParty's
`robolab` (**BSD-3-Clause**), and carries that licence notice at its top.

### gitman notes

- `gitman lock` strips comments from `gitman.yml` (jacebrowning/gitman#170),
  so pin rationale lives here instead of in the YAML file.
- **Never `gitman install --force`** or **`-c` / `--clean`** -- both run
  `git clean -f -d -x` on the checkouts.

## Two things that must stay exactly right

1. **The actuator regexes partition every joint.** Each joint is claimed by
   exactly one actuator group in the robot's `joint_groups.py`. Zero groups is
   a silent limp limb; two groups is a hard Isaac Lab error at scene-build
   time. Check it by eye against the vendor URDF's joint list whenever the
   patterns change.
2. **G1 actuator groups, limits, and PD gains come from
   `unitreerobotics/unitree_rl_lab`'s `UNITREE_G1_29DOF_MIMIC_CFG`**, not
   `isaaclab_assets`, which encodes a different (22.5/22.5) gear variant. See
   `joint_groups.py`'s module docstring.

## Conventions

- **ruff only** -- `ruff format` / `ruff check`, no black/isort/flake8.
- No hand-edited USD, ever -- including the USD Isaac Lab's `UrdfFileCfg`
  generates on the fly under `build/<robot>/`. If it's wrong, fix the source
  URDF (via `gitman`), `robot.yaml`, or `urdf_common.py`'s importer
  options, and let it re-convert.
- `robot.yaml` is strict: an unknown key is a hard error, not a warning.
- See the repo-root `CLAUDE.md` for the resolved-facts list (F1-F11) this
  package's design depends on.
