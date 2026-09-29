# Adding a robot

Six steps, in order. `robots/roboparty_rpo/` is the most recent worked
example, and the one to copy if your robot arrives with an upstream Isaac Lab
config you want to preserve rather than author fresh.

## 1. Pick and pin a vendor source

Find a URDF or MJCF for the robot: BSD-3-Clause-compatible license, no Git
LFS, relative mesh references (so `source.ros_package_paths` can stay `[]`).
Add it to the repo-root `gitman.yml`'s `sources:` (F4 in the root CLAUDE.md), run
`gitman lock`, and record the *why* in `README.md`'s "Vendor pins" section
(`gitman lock` strips comments from `gitman.yml` itself, and appends
`default_group`/`groups` keys you should strip back out). Document
alternatives considered in `docs/vendor-assets.md`.

Prefer the smallest repo that carries the source description files. A robot's
headline repo is often a monorepo that vendors firmware and toolchains beside
the assets; `sparse_paths` will not save you from it, since it only shrinks the
working tree and never the `--mirror` cache (F3). The RPO is the worked case:
57 MB standalone versus 1.9 GB monorepo for a byte-identical URDF.

If the only viable source carries a non-BSD-compatible licence, that is a
judgement call to make explicitly and write down (see `docs/vendor-assets.md`'s
"The CERN-OHL-W-2.0 licence"), not a rule to silently skip.

## 2. Transcribe the ground-truth numbers

From the vendor URDF/MJCF itself -- never `isaaclab_assets` or any other
secondary source:

- The full joint name list, in URDF declaration order.
- Per-joint effort and velocity limits, from each `<limit>` tag.
- Total mass and root link name.

Put these in a new `robots/<robot>/joint_groups.py`, mirroring
`unitree_g1/joint_groups.py`'s shape and its zero-Isaac-Lab-import property
(see that module's docstring and the root CLAUDE.md's Isaac-Lab-free module list).

## 3. Design the actuator groups

Partition every joint into actuator groups using `re.fullmatch`-style
patterns. The partition must be **total and disjoint** -- every joint claimed
by exactly one group -- zero groups is a silent limp limb, two is a hard
Isaac Lab error at scene-build time. Check the partition against the vendor
URDF's joint list by hand; the per-group joint counts in the comments next to
each `*_JOINT_EXPR` are what make that checkable later. If the robot has a
`sensor.py`, add its groups to `tests/test_sensor_groups.py`.

## 4. Write `robot.yaml`

Follow the schema in `robot_schema.py` (read the dataclasses -- they're the
source of truth). Only names the source file and what `validate.py` should
expect. Start with `enabled: false` until Level A/B/D validation passes.

## 5. Write `robot.py`

Spawn from `urdf_common.RobotUrdfFileCfg`
(`asset_path=paths.urdf_path("<robot>")`,
`usd_dir=str(paths.robot_build_dir("<robot>"))`) -- override importer options
on the instance rather than forking the class. Export `<PREFIX>_CFG` and
`<PREFIX>_JOINT_NAMES` (built from `joint_groups.py`, never invented
independently). Mirror `unitree_g1/robot.py`'s `_build_actuators(actuator_cls)`
pattern. PD gains belong here (`ActuatorCfg.stiffness`/`damping`), never in
`robot.yaml` or `urdf_common.py`.

Which actuator class `<PREFIX>_CFG` uses is a per-robot decision, and the two
established answers are:

- **Author the robot yourself** -- `<PREFIX>_CFG` is `ImplicitActuatorCfg`
  (solver-side PD, cheap and unconditionally stable). The G1.
- **Port an upstream config** -- `<PREFIX>_CFG` is whatever upstream tuned and
  trained against, because that is the number that has actually been validated
  on hardware, and `<PREFIX>_IMPLICIT_CFG` is the implicit fallback. RPO, whose
  upstream uses `DelayedPDActuatorCfg` (0-2 physics steps of action delay).

Always export an implicit variant either way. Explicit actuators run the PD
loop in Python each control step and can chatter at a task's `dt`; the
implicit cfg is how you tell an integrator problem apart from an actuator-model
one.

## 6. Validate

```bash
python -m robot_description.tools.validate <robot> --strict
```

Work through Level A failures first (cheapest, most informative), then B
(self-collision), then D (`robot.py` wiring). Only flip `enabled: true` once
`validate.py <robot> --strict` exits 0. Level E (`--sim`) is optional and
manual -- the first point anything actually spawns the robot -- not required
to merge.

## A note on inertial parameters specifically

Link inertias and actuator parameters (especially rotor armature) are
modelling choices, not fixed facts. `RobotUrdfFileCfg.link_density` must stay
`0.0`: a nonzero density silently fills in missing mass/inertia from mesh
geometry, corrupting the dynamics invisibly. Missing `<inertial>` data is a data problem to solve explicitly
(allowlist via `validate.massless_links` if genuinely massless, or measure
the real numbers) -- never a density fill.
