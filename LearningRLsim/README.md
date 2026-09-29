# LearningRLsim

MuJoCo sim2sim replay for policies trained with LearningRLlab.

It takes an exported run directory -- an ONNX policy plus a `config.yaml`
describing the full observation/action/actuator pipeline that produced it --
and replays it in MuJoCo on an ordinary CPU-only machine, where you can watch
it, drive it, and take the data away as an MCAP bag.

LearningRLsim imports **no** Isaac Lab, Isaac Sim, Newton, warp or torch. It needs
no GPU and no Docker: a laptop is enough.

## Install

From the LearningRL repo root, in a Python 3.12 virtual environment:

```bash
gitman install                                   # fetches robot assets into robot_description/vendor
pip install -e robot_description -e LearningRLsim
```

`robot_description` must be installed too: LearningRLsim finds each robot's
MuJoCo model through it, keyed by `config.yaml`'s `robot` field.

## Run

```bash
learningrl-sim <run-dir>                              # stands still for 10 s
learningrl-sim <run-dir> --commands walk_baseline     # a scripted walk timeline
learningrl-sim <run-dir> --spawn-height 0.75 ...      # for an RPO policy (see below)
learningrl-sim <run-dir> --live                       # watch it in Foxglove as it runs
learningrl-sim <run-dir> --keyboard                   # drive it: w/a/s/d, q/e, space, r, m, x
learningrl-sim --list-terms                           # every observation/action/command it implements
```

`<run-dir>` holds `config.yaml`, `policy.onnx` and `policy.onnx.data`; keep
the last two side by side (`policy.onnx.data` holds the weights).

Without `--commands` every command holds zero -- the robot stands. The
timelines in [`commands/`](commands/) (`stand`, `walk_baseline`) resolve by bare
name, and `--scene playground` adds a few objects to walk into.

`learningrl-sim` is the installed command for `python -m LearningRLsim.eval`. Prefer
it: `python -m` from the repo root (or anywhere with a `LearningRLsim/` or
`robot_description/` directory in it) imports that directory instead of the
installed package and fails with a confusing `ImportError`.

### Spawn height

The exported `config.yaml` does not record the robot's initial height, so
`--spawn-height` defaults to the G1's 0.8 m. **For an RPO policy pass
`--spawn-height 0.75`** (`RPO_CFG`'s standing height).

### Expect some divergence on the RPO

The RPO trains with a delayed-PD actuator (0-2 physics steps of action delay).
LearningRLsim implements an ideal PD controller with no delay, so an RPO policy
runs against a slightly different actuator than it trained with. Expect the
numbers to differ somewhat from training; the gait should still look the same.

## What a run produces

An MCAP bag at `<run-dir>/sim2sim/<utc>_<seed>.mcap`, openable in Foxglove or
PlotJuggler: the policy's observation vector split by term, its raw output,
the joint targets and gains, torque with a saturation flag, per-foot contact,
the commanded velocity, full sim state, one frame per link for a 3D view, and a
`/meta/run` record carrying `config.yaml` verbatim plus the sha256 of the
policy and the MuJoCo model. `--bag-rate control` thins the physics-rate
channels; `--no-bag` skips the file.

## Watching it in Foxglove

**Live:** `--live` serves the same channels over a Foxglove WebSocket while the
rollout runs. In Foxglove desktop, "Open connection" to `ws://127.0.0.1:8765`.
If that port is taken, pass `--live-port` or set `LEARNINGRL_FOXGLOVE_PORT`.

**After the fact:** generate a layout for a bag, then import it in Foxglove
("Import from file...") and open the bag:

```bash
python LearningRLsim/tools/foxglove_layout.py <bag.mcap> --urdf <absolute path to the robot's .urdf>
```

The URDF is under `robot_description/vendor/`: `unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf`
for the G1, `rpo_description/urdf/rpo.urdf` for the RPO. `--urdf` is only
needed when the path recorded in the bag does not exist on your machine -- e.g.
a bag from the TA's machine.

`--keyboard` drives the velocity command from the terminal: each press steps
one axis and the command holds until changed (`w`/`s`, `a`/`d`, `q`/`e`, or
the arrows), space stands the robot still, `r` resets, `m` drops a marker into
the bag and `x` ends the rollout. It needs a real terminal. Both `--live` and
`--keyboard` pace the rollout to wall clock (`--speed` scales it,
`--no-realtime` turns it off).

## Tests

```bash
pytest LearningRLsim/tests              # the MuJoCo tests need `gitman install` first
pytest -m static LearningRLsim/tests    # no MuJoCo model needed
```
