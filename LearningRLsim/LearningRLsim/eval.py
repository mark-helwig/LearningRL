"""`python -m LearningRLsim.eval <run-dir> [args...]` -- the one entrypoint.

The contract is fixed: the training side's sim2sim wrapper calls exactly this.

`<run-dir>` is interpreted relative to the current directory: either an
extracted results archive, or a training run's export directory,
`logs/rsl_rl/<task>/<run-name>/<run-name>`.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from LearningRLsim import __version__, library, runconfig
from LearningRLsim.commands.keyboard import KeyboardError, KeyboardSource
from LearningRLsim.commands.scripted import ScriptedSource, TimelineError
from LearningRLsim.commands.sources import CommandManager, CommandSource
from LearningRLsim.data.live_sink import DEFAULT_HOST, DEFAULT_PORT, LiveSink
from LearningRLsim.data.mcap_sink import McapSink
from LearningRLsim.data.recorder import Recorder
from LearningRLsim.data.transforms import LinkTransforms
from LearningRLsim.loop import DEFAULT_FOOT_BODIES, Pacer, Rollout, parse_patterns
from LearningRLsim.model import ModelError, RobotModel, SimOverrides, resolve_mjcf, resolve_urdf
from LearningRLsim.objects import ObjectError, Scene, load_scene
from LearningRLsim.policy import Policy, PolicyError
from LearningRLsim.runconfig import RunConfigError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m LearningRLsim.eval",
        description="Replay a LearningRLlab-exported policy in MuJoCo and record the rollout.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        help="exported run directory: config.yaml + policy.onnx + policy.onnx.data",
    )
    parser.add_argument(
        "--duration", type=float, default=None, help="rollout length in seconds; defaults to the timeline's, else 10"
    )
    parser.add_argument(
        "--commands",
        type=Path,
        default=None,
        help="scripted command timeline (YAML). Without one, every command "
        "holds zero -- an explicit, reproducible stand",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="recorded in the bag and its filename; Phase 1's rollout is "
        "deterministic, so it does not yet change the result",
    )

    physics = parser.add_argument_group(
        "assumed physics",
        "The exporter records none of these, so each is an "
        "assumption. Every one is written into /meta/run marked as assumed rather than read, "
        "so a divergence is never ambiguous about which of them it might be measuring.",
    )
    physics.add_argument(
        "--spawn-height",
        type=float,
        default=SimOverrides.spawn_height,
        help="world z of the root body at reset (G1_CFG.init_state.pos[2]; pass 0.75 for the RPO)",
    )
    physics.add_argument(
        "--friction",
        type=float,
        default=SimOverrides.friction,
        help="sliding friction on every geom (FLAT_TERRAIN_CFG's 1.0)",
    )
    physics.add_argument("--gravity", type=float, default=SimOverrides.gravity)
    physics.add_argument(
        "--sim-rate",
        type=float,
        default=None,
        help="re-run the physics at this rate instead of config.yaml's sim_frequency_hz. The "
        "policy rate is fixed by training and never moves, so decimation absorbs the change; "
        "a rate that is not an integer multiple of it is rejected. This is deliberately NOT a "
        "sim2sim comparison any more -- it is a different plant from the one the policy trained "
        "against -- but it is how you see contact content above the trained rate's Nyquist",
    )
    physics.add_argument(
        "--heading-stiffness",
        type=float,
        default=1.0,
        help="only used by a heading_command term; Isaac Lab's own default",
    )

    scene = parser.add_argument_group(
        "scene",
        "Objects to put in the world beside the robot. They are injected into the model via "
        "mujoco.MjSpec after the MJCF is read, so they are NOT in the file whose sha256 the bag "
        "records -- /meta/run's `scene` key is the only other statement of what was in the scene.",
    )
    scene.add_argument(
        "--scene",
        type=Path,
        default=None,
        help="a scene YAML, or the bare name of one shipped in scenes/ -- resolved exactly as "
        "--commands is from commands/",
    )
    scene.add_argument(
        "--object-ahead",
        type=float,
        default=None,
        help="override every deferred object's standoff, in metres, without editing either YAML",
    )

    bag = parser.add_argument_group("recording")
    bag.add_argument("--no-bag", action="store_true", help="run without writing an MCAP file")
    bag.add_argument("--bag", type=Path, default=None, help="bag path; default <run-dir>/sim2sim/<utc>_<seed>.mcap")
    bag.add_argument(
        "--bag-rate",
        choices=("physics", "control"),
        default="physics",
        help="rate for the physics-rate channels: full 200 Hz, or thinned to the "
        "policy rate. Full is what you want when chasing a sim2sim gap and "
        "dead weight when you are just checking it walks",
    )

    live = parser.add_argument_group(
        "watching and driving",
        "Streaming and keyboard control both pace the rollout to wall clock, since neither "
        "means anything against a 15 s rollout that finishes in 2 s. The bag is written either "
        "way and at its own rate.",
    )
    live.add_argument(
        "--live",
        action="store_true",
        help="serve the rollout to Foxglove over a WebSocket as it runs; connect with "
        "'Open connection' rather than opening a file",
    )
    live.add_argument(
        "--live-port",
        type=int,
        default=int(os.environ.get("LEARNINGRL_FOXGLOVE_PORT", DEFAULT_PORT)),
        help="two concurrent runs on one machine collide here; LEARNINGRL_FOXGLOVE_PORT sets it too",
    )
    live.add_argument(
        "--live-host",
        default=DEFAULT_HOST,
        help="loopback by default; pass 0.0.0.0 to offer the stream to the network",
    )
    live.add_argument(
        "--keyboard",
        action="store_true",
        help="drive the velocity command from the terminal (w/a/s/d, q/e, space, r, b, m, x). "
        "Needs a tty, and cannot be combined with --commands",
    )
    live.add_argument("--key-step", type=float, default=0.1, help="m/s added per keypress")
    live.add_argument("--key-yaw-step", type=float, default=0.2, help="rad/s added per turn keypress")
    live.add_argument(
        "--keyboard-term",
        default=None,
        help="which command term the keyboard drives; default is the first one taking a velocity",
    )
    live.add_argument(
        "--realtime",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="hold sim time to wall time. On by default with --live or --keyboard, off otherwise",
    )
    live.add_argument("--speed", type=float, default=1.0, help="wall-clock multiplier when pacing")

    parser.add_argument(
        "--foot-bodies",
        type=str,
        default=",".join(DEFAULT_FOOT_BODIES),
        help="comma-separated body-name regexes for contact logging",
    )
    parser.add_argument(
        "--min-base-height",
        type=float,
        default=0.4,
        help="below this, the robot counts as fallen and the rollout stops",
    )
    parser.add_argument(
        "--fail-on-fall",
        action="store_true",
        help="exit non-zero if the robot falls (for scripted use). By default a "
        "fall is reported and exits 0 -- the bag is still the deliverable",
    )

    parser.add_argument("--list-terms", action="store_true", help="print every registered term and exit")
    parser.add_argument("--list-bodies", action="store_true", help="print the MuJoCo model's body names and exit")
    return parser


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(path: Path, *args: str) -> str | None:
    """Run a read-only git query in `path`. None on any failure.

    `-c safe.directory` because a checkout owned by another uid (a bind mount,
    a shared machine) is refused as "dubious ownership" otherwise. Scoped to
    this one invocation rather than written into the user's config, since it
    is a read.
    """
    try:
        out = subprocess.run(
            ["git", "-c", f"safe.directory={path}", "-C", str(path), *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _git_sha(path: Path) -> str | None:
    """HEAD of the repo containing `path`, `-dirty` if the tree is modified.

    Provenance, never required -- but a bare SHA next to an edited working
    tree would name a commit that is not what actually ran.
    """
    sha = _git(path, "rev-parse", "HEAD")
    if sha is None:
        return None
    return f"{sha}-dirty" if _git(path, "status", "--porcelain") else sha


def _scene_meta(sim: RobotModel, scene: Scene | None) -> dict | None:
    """The scene YAML verbatim plus each object as resolved, or None.

    Parallel to `timeline`, and for the same reason: the bag records what was
    asked for as well as what happened. The note matters -- `sha256.mjcf` is
    the only other record of the plant, and the objects are not in that file.
    """
    if scene is None:
        return None
    return {
        "name": scene.name,
        "path": str(scene.path),
        "raw": scene.raw,
        "objects": [{**obj.as_meta(), "source": "assumed"} for obj in sim.objects],
        "note": (
            "Objects are injected through mujoco.MjSpec after the MJCF is read, so they are NOT "
            "in the file whose sha256 is recorded above."
        ),
    }


def _meta(
    cfg,
    sim: RobotModel,
    run_dir: Path,
    source: CommandSource | None,
    seed: int,
    urdf: Path | None,
    scene: Scene | None = None,
    commands: CommandManager | None = None,
) -> dict:
    import LearningRLsim

    try:
        import robot_description

        rd_root = Path(robot_description.__file__).resolve().parents[1]
    except ImportError:
        rd_root = None

    return {
        "learningrlsim_version": __version__,
        "task": cfg.task,
        "robot": cfg.robot or "",
        "run_dir": str(run_dir.resolve()),
        "mjcf": str(sim.mjcf),
        # Viewer geometry, not physics -- `tools/foxglove_layout.py` reads
        # this to point a 3D panel at the meshes for the links `/tf` carries.
        "urdf": str(urdf) if urdf else None,
        "seed": seed,
        "config": cfg.raw,
        "sha256": {
            "policy.onnx": _sha256(run_dir / "policy.onnx"),
            "policy.onnx.data": _sha256(run_dir / "policy.onnx.data"),
            "mjcf": _sha256(sim.mjcf),
        },
        "git": {
            "LearningRLsim": _git_sha(Path(LearningRLsim.__file__).resolve().parents[1]),
            "robot_description": _git_sha(rd_root) if rd_root else None,
        },
        "motion": {
            name: {
                "file": str(term.motion.file),
                "sha256": _sha256(term.motion.file),
                "frames": term.num_frames,
                "fps": term.fps,
                "policy_frequency_hz": cfg.policy_frequency_hz,
            }
            for name, term in commands.terms.items()
            if hasattr(term, "tracking")
        }
        if commands
        else {},
        "command_source": source.name if source else "zero (stand)",
        "timeline": getattr(source, "raw", None),
        "scene": _scene_meta(sim, scene),
        "physics": {
            **sim.overrides.as_meta(),
            # Read from config.yaml, unlike the three above -- stated
            # explicitly so the distinction is legible in the bag itself.
            # `config` below is the exported YAML verbatim, so when --sim-rate
            # moved these two they disagree with it on purpose: the override is
            # marked as such, and carries what it displaced.
            "sim_dt": _rate_meta(cfg, "sim_dt"),
            "decimation": _rate_meta(cfg, "decimation"),
            "armature": {"value": cfg.actuators.armature.tolist(), "source": "read"},
            "joint_friction": {"value": cfg.actuators.friction.tolist(), "source": "read"},
            "obs_normalization": {
                "value": False,
                "source": "assumed",
                "note": "LearningRLlab's runner sets obs_normalization=False, but config.yaml "
                "does not record it, so LearningRLsim applies no normalizer.",
            },
            "clip_actions": {
                "value": None,
                "source": "assumed",
                "note": "RslRlVecEnvWrapper.step applies clip_actions before the action "
                "manager when it is set; config.yaml does not record it.",
            },
        },
        "joint_order": list(cfg.joint_order),
    }


def _rate_meta(cfg: runconfig.RunConfig, key: str) -> dict[str, Any]:
    """One rate field, saying whether --sim-rate displaced what was exported."""
    exported = cfg.raw.get(key)
    value = getattr(cfg, key)
    if exported == value:
        return {"value": value, "source": "read"}
    return {
        "value": value,
        "source": "override",
        "exported": exported,
        "note": "--sim-rate re-ran the physics off the exported rate; the policy rate is unchanged.",
    }


def _scene_objects(scene: Scene | None, ahead: float | None):
    """The scene's objects, with `--object-ahead` folded into the deferred ones."""
    from dataclasses import replace

    if scene is None:
        if ahead is not None:
            raise ObjectError("--object-ahead overrides a deferred object's standoff, but no --scene was given")
        return ()
    if ahead is None:
        return scene.objects
    return tuple(replace(obj, ahead=ahead) if obj.deferred else obj for obj in scene.objects)


def motion_duration(commands: CommandManager) -> float | None:
    """Length of the longest reference clip among the run's command terms, if any has one.

    A tracking policy is only asked to follow the clip, so a rollout that outlasts it would
    hold the last frame, and one that ends early never reaches the clip's end.
    """
    lengths = [term.duration for term in commands.terms.values() if hasattr(term, "tracking")]
    return max(lengths) if lengths else None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_terms:
        print(library.summary())
        return 0

    if args.run_dir is None:
        build_parser().print_usage(sys.stderr)
        print("error: run_dir is required (unless --list-terms)", file=sys.stderr)
        return 2

    run_dir = Path(args.run_dir)
    overrides = SimOverrides(spawn_height=args.spawn_height, friction=args.friction, gravity=args.gravity)

    if args.keyboard and args.commands:
        build_parser().print_usage(sys.stderr)
        print(
            "error: --keyboard and --commands are two command sources and a rollout has one. "
            "Run the timeline, or drive it by hand.",
            file=sys.stderr,
        )
        return 2

    try:
        cfg = runconfig.load(run_dir)
        if args.sim_rate is not None:
            cfg = runconfig.at_sim_rate(cfg, args.sim_rate)
        mjcf = resolve_mjcf(cfg)
        urdf = resolve_urdf(cfg)
        scene = load_scene(args.scene) if args.scene else None
        objects = _scene_objects(scene, args.object_ahead)
        sim = RobotModel(mjcf, cfg, overrides, objects=objects)

        if args.list_bodies:
            import mujoco

            for bid in range(sim.model.nbody):
                print(mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_BODY, bid))
            return 0

        policy = Policy(run_dir / "policy.onnx", cfg)
        source: CommandSource | None = None
        if args.commands:
            source = ScriptedSource(args.commands)
        elif args.keyboard:
            source = KeyboardSource(step=args.key_step, yaw_step=args.key_yaw_step, term=args.keyboard_term)
        commands = CommandManager(cfg, source, heading_stiffness=args.heading_stiffness)
    except (RunConfigError, ModelError, PolicyError, TimelineError, KeyboardError, ObjectError) as exc:
        print(f"[LearningRLsim] {exc}", file=sys.stderr)
        return 1

    duration = args.duration
    if duration is None:
        if args.keyboard:
            # A hand-driven rollout ends when the driver ends it: `x`, a fall,
            # or Ctrl-C. A default length would just cut them off mid-turn.
            duration = math.inf
        else:
            duration = source.duration if source and source.duration else motion_duration(commands) or 10.0

    sinks = []
    bag_path = None
    if not args.no_bag:
        bag_path = args.bag or (
            run_dir / "sim2sim" / f"{datetime.now(UTC).strftime('%Y-%m-%dT%H-%M-%SZ')}_{args.seed}.mcap"
        )
        sinks.append(McapSink(bag_path, physics_decimation=(cfg.decimation if args.bag_rate == "control" else 1)))
    if args.live:
        # Control rate for everything: twelve channels of 200 Hz float arrays
        # saturate the socket and tell a watcher nothing the 50 Hz view does
        # not. The bag keeps whatever --bag-rate asked for, independently.
        sinks.append(LiveSink(port=args.live_port, host=args.live_host, physics_decimation=cfg.decimation))

    interactive = args.live or args.keyboard
    pacer = Pacer(args.speed) if (args.realtime if args.realtime is not None else interactive) else None

    print(f"[LearningRLsim] {cfg.task}  robot={cfg.robot}  policy={run_dir / 'policy.onnx'}")
    print(
        f"[LearningRLsim] {cfg.sim_frequency_hz:g} Hz physics / {cfg.policy_frequency_hz:g} Hz policy, "
        f"{'until stopped' if math.isinf(duration) else f'{duration:g} s'}, "
        f"commands={source.name if source else 'zero (stand)'}"
    )
    for obj in objects:
        where = (
            f"at {list(obj.pos)}"
            if obj.spawn == "at_start"
            else f"deferred, {obj.ahead:g} m ahead / {obj.lateral:g} m across on `place_object`"
        )
        print(f"[LearningRLsim]   object {obj.name}: {obj.shape} size={list(obj.size)} {obj.mass:g} kg, {where}")

    rollout = Rollout(
        cfg,
        sim,
        policy,
        commands,
        recorder=Recorder(sinks),
        foot_bodies=parse_patterns(args.foot_bodies),
        min_base_height=args.min_base_height,
        transforms=LinkTransforms(sim, urdf),
        pacer=pacer,
    )

    interrupted = False
    try:
        with rollout.recorder as recorder:
            recorder.record("/meta/run", 0.0, _meta(cfg, sim, run_dir, source, args.seed, urdf, scene, commands))
            result = rollout.run(duration)
    except KeyboardInterrupt:
        # The bag is already finalised by the recorder's __exit__; say so
        # instead of printing a traceback over a run that went fine.
        interrupted = True
        result = rollout.result
    finally:
        commands.close()

    result.bag_path = str(bag_path) if bag_path else None
    print("\n[LearningRLsim] rollout interrupted" if interrupted else "[LearningRLsim] rollout complete")
    print(result.summary())

    if result.fell and args.fail_on_fall:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
