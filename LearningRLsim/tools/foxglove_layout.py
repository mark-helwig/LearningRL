"""Write a Foxglove layout that renders a bag's robot in 3D.

    python tools/foxglove_layout.py <bag.mcap> [-o <layout.json>]

Then in Foxglove: the layout menu -> "Import from file...", open the bag, and
the 3D panel is already pointed at the right URDF.

**Why this is generated rather than committed as a fixed file.** Foxglove's
URDF layer takes an absolute path -- there is no `package://` resolver and no
notion of a workspace -- so a layout is only valid on the machine whose
checkout it names. Reading the path out of the bag's own `/meta/run` instead
of hardcoding one means the layout is right for whatever robot and whatever
checkout produced that bag. Generated layouts are gitignored for that reason;
`layouts/sim2sim_template.json` is the tracked copy -- the same panels with an
empty URDF path, for someone who wants the arrangement without a bag in hand.

**A bag written on another machine (or in a container) records that
machine's path**, which need not exist where Foxglove runs. There is no mapping
between the two that this repo could know, so the path is resolved in this
order, and the tool
refuses to emit a layout naming a file that is not there rather than leaving
someone to wonder why the 3D panel is empty:

1. `--urdf`, when given -- taken on trust, since it names a path on the
   machine Foxglove runs on, which need not be the one generating the layout;
2. the bag's recorded path, if it exists on THIS filesystem;
3. `robot_description.urdf_path(robot)`, if the package is importable here.

The 3D panel works off `/tf`, which carries one frame per URDF link straight
from MuJoCo (`LearningRLsim/data/transforms.py`). The URDF layer's `controlMode` is
Foxglove's default `"transforms"` -- it places each link's meshes from that
tree and runs no kinematics of its own -- so the only thing it needs from the
URDF is geometry.

Two deliberate choices in the generated panel, both easy to flip in the UI:

- **Display frame `world`, follow mode `follow-none`.** The camera stays put
  while the robot walks out of frame, which is the tradeoff for a ground plane
  that stays on the ground. Setting the display frame to `pelvis` chases the
  robot instead, at the cost of the grid riding up to hip height -- fine for
  studying a gait, misleading for judging whether a foot is in contact.
- **Mesh paths are resolved relative to the URDF.** Foxglove builds
  `file://<filePath>` as the base URL, so the vendor's relative
  `meshes/*.STL` refs resolve without any package mapping.

Reads the bag with `mcap`, which LearningRLsim already depends on, and imports
nothing from the package.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mcap.reader import make_reader

META_TOPIC = "/meta/run"
TF_TOPIC = "/tf"
JOINT_STATES_TOPIC = "/joint/states"


def read_meta(bag: Path) -> dict[str, Any]:
    with bag.open("rb") as fh:
        for _, _, message in make_reader(fh).iter_messages(topics=[META_TOPIC]):
            return json.loads(message.data)
    raise SystemExit(f"{bag}: no {META_TOPIC} message -- not a LearningRLsim bag, or written before it existed.")


def count_topics(bag: Path, topics: tuple[str, ...]) -> dict[str, int]:
    """How many messages each topic carried. Zero is the interesting answer.

    Worth its own pass over the bag, because both failures it catches are
    silent and does not look like a missing channel. With no `/tf`, Foxglove
    draws every link at the render origin, so you get a heap of disconnected
    parts that never moves -- which reads as a broken URDF or a broken policy,
    not a stale bag.

    Every channel is registered at open, so a topic present in the summary but
    absent from the messages is exactly the stale-bag case -- counting
    messages is the only thing that distinguishes them.
    """
    counts = dict.fromkeys(topics, 0)
    with bag.open("rb") as fh:
        for _, channel, _ in make_reader(fh).iter_messages(topics=list(topics)):
            counts[channel.topic] += 1
    return counts


def resolve_urdf(meta: dict[str, Any], override: Path | None) -> Path:
    """The URDF this machine should load, or a refusal naming what was tried."""
    if override is not None:
        # NOT existence-checked, deliberately. The common case for --urdf is
        # generating in the container a layout for the Foxglove on the host,
        # where the path is correct and simply not visible from here.
        if not override.is_file():
            print(f"[LearningRLsim] note: {override} is not visible from here; taking it on trust.")
        return override

    recorded = meta.get("urdf")
    if recorded and Path(recorded).is_file():
        return Path(recorded)

    robot = meta.get("robot")
    if robot:
        try:
            from robot_description.paths import urdf_path

            local = Path(urdf_path(robot))
            if local.is_file():
                return local
        except ImportError:
            pass

    raise SystemExit(
        f"cannot locate a URDF for robot {robot!r} on this machine.\n"
        f"  the bag records:  {recorded or '(nothing)'}\n"
        f"  which is {'a container path, not a host one' if recorded else 'absent'}.\n"
        f"Pass --urdf <path to the .urdf on this machine>. Inside the stack that is\n"
        f"  <checkout>/workspace/robot_description/vendor/.../<robot>.urdf"
    )


def _plot(title: str, paths: list[tuple[str, str]]) -> dict[str, Any]:
    palette = ["#4e98e2", "#f5774d", "#a0b563", "#c07ad6", "#e2b04e"]
    return {
        "title": title,
        "legendDisplay": "floating",
        "showPlotValuesInLegend": True,
        "isSynced": True,
        "xAxisVal": "timestamp",
        "paths": [
            {
                "value": value,
                "label": label,
                "enabled": True,
                "timestampMethod": "receiveTime",
                "color": palette[i % len(palette)],
                "id": f"{title}-{i}",
            }
            for i, (value, label) in enumerate(paths)
        ],
    }


def _stack(panel_ids: list[str]) -> Any:
    """Right-nested column splits with even shares -- so adding a panel is
    appending to a list rather than rewriting a nested tree by hand."""
    if len(panel_ids) == 1:
        return panel_ids[0]
    return {
        "first": panel_ids[0],
        "second": _stack(panel_ids[1:]),
        "direction": "column",
        "splitPercentage": 100.0 / len(panel_ids),
    }


def build_layout(meta: dict[str, Any]) -> dict[str, Any]:
    urdf = meta.get("urdf")
    robot = meta.get("robot") or "robot"

    urdf_layer = {
        "instanceId": "learningrlsim-urdf",
        "layerId": "foxglove.Urdf",
        "label": robot,
        "visible": True,
        "order": 2,
        # `filePath` is desktop-only; Foxglove turns it into `file://<path>`
        # and resolves the URDF's relative mesh refs against it.
        "sourceType": "filePath",
        "filePath": urdf or "",
        "url": "",
        "parameter": "",
        "topic": "",
        "framePrefix": "",
        # Place each link at the pose the sim actually had, from `/tf`, rather
        # than having Foxglove re-derive it by FK. Exact, and independent of
        # whether the URDF's joint model still matches the MJCF's. The bag
        # also carries `/joint/states`, so switching this to "jointStates"
        # (with the topic below) is a live A/B with no re-run.
        "controlMode": "transforms",
        "jointStatesTopic": JOINT_STATES_TOPIC,
        "displayMode": "visual",
        "fallbackColor": "#b8c4cc",
        "showAxis": False,
        "axisScale": 1,
        "showOutlines": True,
        "opacity": 1,
    }

    panel_3d = {
        "cameraState": {
            "distance": 4.5,
            "perspective": True,
            "phi": 72,
            "target": [0, 0, 0],
            "targetOffset": [0, 0, 0.7],
            "targetOrientation": [0, 0, 0, 1],
            "thetaOffset": 45,
            "fovy": 45,
            "near": 0.01,
            "far": 5000,
        },
        "followMode": "follow-none",
        "followTf": "world",
        "scene": {
            # THE setting that matters. Foxglove defaults `meshUpAxis` to
            # "y_up" and rotates every loaded mesh +90 degrees about X to
            # match; a URDF's meshes are Z-up by ROS convention, so leaving
            # the default silently rotates all 39 of them inside their own
            # link frames. The robot then has every link in exactly the right
            # place with the wrong orientation, which reads as a quaternion
            # bug in the producer and is nothing of the kind. It is a mesh
            # loader setting, so it survives switching `controlMode` -- that
            # both encodings look equally wrong is the tell.
            "meshUpAxis": "z_up",
            # 39 link frames would otherwise each draw an axis triad and a
            # text label on top of the meshes they belong to.
            "transforms": {"showLabel": False, "axisScale": 0, "enablePreloading": True},
        },
        "transforms": {},
        "topics": {},
        "layers": {
            "learningrlsim-grid": {
                "instanceId": "learningrlsim-grid",
                "layerId": "foxglove.Grid",
                "label": "Ground",
                "visible": True,
                "frameLocked": True,
                "order": 1,
                "size": 20,
                "divisions": 20,  # 1 m cells
                "lineWidth": 1,
                "color": "#3f4b57",
                "position": [0, 0, 0],
                "rotation": [0, 0, 0],
            },
            "learningrlsim-urdf": urdf_layer,
        },
        "publish": {
            "type": "point",
            "poseTopic": "/move_base_simple/goal",
            "pointTopic": "/clicked_point",
            "poseEstimateTopic": "/initialpose",
            "poseEstimateXDeviation": 0.5,
            "poseEstimateYDeviation": 0.5,
            "poseEstimateThetaDeviation": 0.26179939,
        },
        "synchronize": False,
        "imageMode": {},
    }

    command = (meta.get("config", {}).get("commands") or [{}])[0].get("name", "base_velocity")
    tracking = _plot(
        "velocity tracking (command vs actual)",
        [
            (f"/cmd/velocity.commands.{command}[0]", "cmd vx"),
            ("/sim/state.base_lin_vel_b[0]", "vx"),
            (f"/cmd/velocity.commands.{command}[2]", "cmd wz"),
            ("/sim/state.base_ang_vel_b[2]", "wz"),
        ],
    )
    contact = _plot(
        "foot contact (world-z GRF)",
        [("/sim/contact.normal_force[0]", "left"), ("/sim/contact.normal_force[1]", "right")],
    )
    height = _plot("base height", [("/sim/state.base_height", "z")])

    panels = {
        "Plot!tracking": tracking,
        "Plot!contact": contact,
        "Plot!height": height,
    }
    return {
        "configById": {"3D!learningrlsim": panel_3d, **panels},
        "globalVariables": {},
        "userNodes": {},
        "playbackConfig": {"speed": 1},
        "layout": {
            "first": "3D!learningrlsim",
            "second": _stack(list(panels)),
            "direction": "row",
            "splitPercentage": 62,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python tools/foxglove_layout.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("bag", type=Path, help="a LearningRLsim MCAP bag; its /meta/run names the URDF")
    parser.add_argument(
        "--urdf",
        type=Path,
        default=None,
        help="override the URDF the 3D panel loads. Needed when the bag was written on another "
        "machine or in a container, since the recorded path is that machine's",
    )
    parser.add_argument("-o", "--output", type=Path, default=None, help="default: layouts/<robot>_sim2sim.json")
    args = parser.parse_args(argv)

    meta = read_meta(args.bag)
    meta["urdf"] = str(resolve_urdf(meta, args.urdf))

    counts = count_topics(args.bag, (TF_TOPIC,))
    transforms = counts[TF_TOPIC]
    if transforms == 0:
        raise SystemExit(
            f"{args.bag} carries no {TF_TOPIC} messages, so a 3D panel cannot place the "
            f"robot's links.\nFoxglove would draw every link at the origin -- a heap of "
            f"parts that never moves.\nThe bag predates the {TF_TOPIC} channel; re-run the "
            f"rollout to get one that has it:\n"
            f"  learningrl-sim {meta.get('run_dir', '<run-dir>')} --commands <timeline>"
        )

    output = args.output or (
        Path(__file__).resolve().parents[1] / "layouts" / f"{meta.get('robot') or 'robot'}_sim2sim.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_layout(meta), indent=2) + "\n", encoding="utf-8")

    print(f"[LearningRLsim] wrote {output}")
    print(f"[LearningRLsim]   robot {meta.get('robot')}  urdf {meta.get('urdf')}")
    print(f"[LearningRLsim]   {transforms} {TF_TOPIC} messages in {args.bag.name}")
    print(f"[LearningRLsim] open THIS bag -- {args.bag}")
    print("[LearningRLsim] Foxglove: layout menu -> Import from file..., then open the bag.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
