# Vendor asset comparison

Why each vendor source won. `unitreerobotics/unitree_ros` covers the G1;
`Roboparty/rpo_description` covers the RPO. Both are pinned in the repo-root
`gitman.yml`, which checks them out under `robot_description/vendor/`.

# Part 1 -- G1

## Candidates considered

| Repo | G1 | Meshes | LFS | License | Verdict |
|---|---|---|---|---|---|
| `unitreerobotics/unitree_ros` | `g1_29dof_rev_1_0.urdf`, 29 revolute joints, 33.341 kg | Relative refs, no `package://` needed | None -- ordinary git blobs | BSD-3-Clause | **Chosen** |
| `unitreerobotics/unitree_model` | G1 present | Ships **pre-built USD**, not source URDF | N/A | Unclear | Rejected |
| `unitreerobotics/unitree_rl_gym` | G1 present (older revision) | N/A | N/A | Unclear | Rejected |

## Why `unitree_ros` won

- **Source URDF, not pre-built USD.** A robot config needs to control inertial
  and actuator parameters (`merge_fixed_joints`, `link_density`, joint drive
  gains); a pre-converted USD gives no leverage over how that conversion
  happened. A hard requirement, not a preference.
- **No Git LFS** -- simplifies `gitman` sparse-checkout, no bandwidth budget to manage.
- **Relative mesh references**, never `package://` URIs -- `robot.yaml`'s
  `source.ros_package_paths` stays `[]`.
- **Ships URDF and MJCF side by side** -- the MJCF is what LearningRLsim runs.
- **Licence compatible** (BSD-3-Clause).

## Why the alternatives lost

- **`unitree_model`** ships pre-built USD, not source description files --
  nothing to feed a URDF/MJCF converter, no way to audit or vary the
  inertial parameters that produced it.
- **`unitree_rl_gym`** carries an older G1 revision.

# Part 2 -- RPO (RoboParty ROBOTO_ORIGIN)

## Candidates considered

| Repo | Size | Contents | LFS | License | Verdict |
|---|---|---|---|---|---|
| `Roboparty/rpo_description` | 57 MB | `urdf/rpo.urdf`, 23 revolute joints, 33.757 kg, 24 STL meshes (~30 MB), plus MJCF | None | CERN-OHL-W-2.0 | **Chosen** |
| `Roboparty/roboto_origin` | **1.9 GB** | The announced repo. Same `rpo.urdf`, byte-identical, under `modules/rpo_description/` | None | Per-module; see its `LICENSES.md` | Rejected -- size |
| `Roboparty/robolab` | 62 MB | Same URDF + meshes under `data/robots/roboparty/rpo/`, plus ~50 MB of motion-capture `.pkl`/`.npz` | None | BSD-3-Clause | Rejected -- redundant payload |

## Why `rpo_description` won

- **Byte-identical URDF** to the monorepo's copy, so nothing is lost by not
  pinning `roboto_origin`.
- **`roboto_origin` is a snapshot-aggregation monorepo.** It vendors its own
  submodules in-tree -- `roboparty_firmware` alone carries the full
  OrangePi/Armbian `orangepi-build` toolchain. `sparse_paths` is cone-mode and
  only shrinks the working tree; the `--mirror` cache stays full size (F3), so
  pinning it would cost ~2 GB of disk to obtain one 25 KB URDF.
- **Source URDF, not pre-built USD** -- the same hard requirement that decided
  Part 1.
- **No Git LFS**, **relative mesh references** (`../meshes/*.STL`, never
  `package://`), so `source.ros_package_paths` stays `[]`.
- **`robolab` lost on payload, not licence.** It is where the RPO's `robot.py`
  is ported from, but as an *asset* source it bundles ~50 MB of motion-capture
  data nothing here consumes. The description assets inside it originate from
  `rpo_description` regardless.

## The CERN-OHL-W-2.0 licence

`rpo_description` is CERN-OHL-W-2.0 -- a weakly-reciprocal **hardware**
licence, which is the natural fit for what a URDF is (a description of a
physical design). `vendor/` is gitignored, `gitman` fetches it per-machine, and
nothing derived from those meshes is redistributed by this repo. Publishing
RPO meshes, a converted USD, or a rendered dataset would have to clear
CERN-OHL-W-2.0's reciprocity terms first. The licence covers the **assets**;
the RPO's `robot.py` is ported from `robolab` under BSD-3-Clause.

## The missing-collider caveat

`rpo.urdf` declares `<collision>` for only **16 of its 24 links**. These eight
have none at all:

```
left/right_thigh_yaw_link    left/right_thigh_roll_link
left/right_ankle_pitch_link  left/right_elbow_yaw_link
```

With `collision_from_visuals=False` (the `RobotUrdfFileCfg` default, and what
upstream's own `RPO_CFG` does) they spawn with no collider: they cannot collide
and a contact sensor on them always reads zero. `roboparty_rpo/sensor.py` still
names them in `TERMINATION_CONTACT`, on the grounds that the group should say
which bodies *ought* to end an episode rather than encode a vendor-URDF gap --
but a fall landing on a bare thigh-yaw housing will not terminate.

Flipping `collision_from_visuals` is not the fix: it would promote all 24
visual meshes to colliders, replacing the URDF's deliberate primitive colliders
(8 cylinders, 5 boxes, 3 meshes) with full-detail geometry everywhere. The real
fix is collision geometry added upstream.

What `roboparty_rpo/robot.py` *does* carry over from upstream is
`replace_cylinders_with_capsules=True`, since a capsule collider is both faster
and better-behaved on contact than a cylinder.
