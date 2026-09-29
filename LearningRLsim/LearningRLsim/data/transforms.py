"""Link poses for a 3D viewer, as `foxglove.FrameTransforms`.

Foxglove renders a URDF by attaching each link's visual meshes to a coordinate
frame **named after that link** -- its URDF layer's default `controlMode` is
`"transforms"`, meaning it places every link from the transform tree and never
runs forward kinematics itself. So the bag has to carry one frame per link.

**The poses come straight from MuJoCo, not from FK on `joint_pos`.** `xpos` /
`xquat` are the world pose of every body after `mj_step`, which is the exact
state the rollout was in; reimplementing FK here would add a second kinematics
path that could disagree with the one being evaluated. That is the whole
reason this file is short.

Two things make this work, and both are checked rather than assumed:

- **MJCF body frames coincide with URDF link frames.** Verified for the G1:
  all 29 non-root bodies have byte-identical parent-relative origins in the
  two files (the MJCF writes the rotation as a quaternion where the URDF
  writes rpy). This is a property of a vendor that hand-authored the MJCF from
  the URDF, so `tests/test_transforms.py` pins it rather than trusting it.
- **The URDF has links MuJoCo does not.** MuJoCo merges a fixed joint's child
  into its parent, so the G1's MJCF has 30 bodies against the URDF's 39 links
  -- the missing nine include the head and both hands, i.e. exactly the parts
  whose absence is most obvious on screen. `urdf_static_frames` recovers them
  by composing the fixed-joint chain back up to the nearest real body.

**The tree mirrors the robot's kinematics, and must.** MuJoCo hands us a world
pose per body, so the cheapest thing to publish would be a flat `world -> link`
fan. Foxglove renders that correctly only on its happy path; the moment it
falls back -- any URDF link without a frame -- it walks the URDF's JOINT chain
asking the scene tree for each `parent_link <- child_link` relative transform,
and places a child at the joint's static origin, with **no joint angle**, for
every lookup that misses. The result is a robot whose links are each in about
the right place and every one of them rotated wrong, which reads as a
quaternion bug rather than a tree-shape one. Publishing `parent -> child`
makes those lookups single edges, so the fallback degrades to the correct
pose instead of the rest pose. It also stops a viewer drawing one line from
the world origin to all 39 links.

Every transform is re-sent each control step rather than split into a
`/tf_static` written once. One message is then one complete pose of the robot,
which costs ~nine redundant transforms per step (zstd absorbs bytes that never
change) and buys not having to depend on a viewer preloading the whole topic
before a seek resolves.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT_FRAME = "world"
"""Frame id of the MuJoCo world. The G1's URDF declares a `world` link too --
orphaned, with no joint to `pelvis` and no visual -- so the two agree by name
and nothing is rendered at the origin either way."""

# Quaternions here are (x, y, z, w) -- Foxglove's order, and the order these
# values are written out in. MuJoCo's `xquat` is (w, x, y, z) and is converted
# at the one place it enters, below.
_IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])


def _quat_from_rpy(rpy: tuple[float, float, float]) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw -> (x, y, z, w). R = Rz(y) Ry(p) Rx(r)."""
    roll, pitch, yaw = (v / 2.0 for v in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ]
    )


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    u, w = q[:3], q[3]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def _quat_inverse(q: np.ndarray) -> np.ndarray:
    """Conjugate. These are unit quaternions, so it is also the inverse."""
    return np.array([-q[0], -q[1], -q[2], q[3]])


def _relative(parent: tuple[np.ndarray, np.ndarray], child: tuple[np.ndarray, np.ndarray]):
    """`parent^-1 . child` -- the child's pose expressed in the parent's frame."""
    (pp, pq), (cp, cq) = parent, child
    inv = _quat_inverse(pq)
    return _quat_rotate(inv, cp - pp), _quat_multiply(inv, cq)


def _stamp(t: float) -> dict[str, int]:
    """Rounded exactly as `Recorder.record` rounds it, so a timestamp inside a
    message cannot drift from that message's own log_time."""
    t_ns = int(round(t * 1e9))
    return {"sec": t_ns // 1_000_000_000, "nsec": t_ns % 1_000_000_000}


def joint_states_message(sim, t: float) -> dict[str, Any]:
    """`foxglove.JointStates`: the robot's pose as joint angles.

    The second of the two encodings this module writes. A viewer that runs its
    own forward kinematics needs only these plus the root link's frame, so it
    never has to match a frame per link -- a different failure surface from
    `/tf`, which is the entire reason both exist.

    Only the actuated joints appear, because they are the only ones the config
    names. A fixed joint has no angle to report and a viewer takes it from the
    URDF.
    """
    return {
        "timestamp": _stamp(t),
        "joints": [
            {"name": name, "position": float(position), "velocity": float(velocity)}
            for name, position, velocity in zip(sim.cfg.joint_order, sim.joint_pos, sim.joint_vel)
        ],
    }


@dataclass(frozen=True)
class StaticFrame:
    """A URDF link MuJoCo merged away, as a constant offset from a real body."""

    parent: str
    child: str
    translation: np.ndarray
    rotation: np.ndarray


def urdf_static_frames(urdf: Path, dynamic: Collection[str]) -> list[StaticFrame]:
    """Every URDF link NOT in `dynamic`, as an offset from its nearest ancestor that is.

    `dynamic` is the set of MuJoCo body names -- the links whose pose the sim
    publishes directly. A link outside it is reachable only through fixed
    joints (that is why MuJoCo merged it), so walking up the chain and
    composing the origins gives its pose exactly, once, forever.

    A link whose chain reaches a non-fixed joint, or runs out of joints
    entirely, is skipped: it is not rigidly attached to anything the sim
    publishes, so there is no constant offset to emit. That covers the G1's
    orphaned `world` link without special-casing it.
    """
    root = ET.parse(urdf).getroot()

    parents: dict[str, tuple[str, str, np.ndarray, np.ndarray]] = {}
    for joint in root.findall("joint"):
        origin = joint.find("origin")
        xyz = (origin.get("xyz") if origin is not None else None) or "0 0 0"
        rpy = (origin.get("rpy") if origin is not None else None) or "0 0 0"
        parents[joint.find("child").get("link")] = (
            joint.find("parent").get("link"),
            joint.get("type", ""),
            np.array([float(v) for v in xyz.split()]),
            _quat_from_rpy(tuple(float(v) for v in rpy.split())),
        )

    dynamic = set(dynamic)
    frames: list[StaticFrame] = []
    for element in root.findall("link"):
        link = element.get("name")
        if link in dynamic:
            continue
        translation, rotation, node = np.zeros(3), _IDENTITY.copy(), link
        while (entry := parents.get(node)) is not None:
            parent, joint_type, offset, turn = entry
            if joint_type != "fixed":
                break
            translation = _quat_rotate(turn, translation) + offset
            rotation = _quat_multiply(turn, rotation)
            if parent in dynamic:
                frames.append(StaticFrame(parent, link, translation, rotation))
                break
            node = parent
    return frames


def _transform(stamp: dict[str, int], parent: str, child: str, p: np.ndarray, q: np.ndarray) -> dict[str, Any]:
    return {
        "timestamp": stamp,
        "parent_frame_id": parent,
        "child_frame_id": child,
        "translation": {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])},
        "rotation": {"x": float(q[0]), "y": float(q[1]), "z": float(q[2]), "w": float(q[3])},
    }


class LinkTransforms:
    """Builds the `/tf` payload: every MuJoCo body, plus the URDF's fixed extras."""

    def __init__(self, sim, urdf: Path | None = None, root_frame: str = ROOT_FRAME) -> None:
        self.sim = sim
        self.root_frame = root_frame
        # A missing URDF is not fatal: the 29 articulated links still render,
        # and the bag is still the deliverable. It costs the head and hands.
        self.static = urdf_static_frames(Path(urdf), sim.body_names) if urdf else []

    def message(self, t: float) -> dict[str, Any]:
        stamp = _stamp(t)

        # `xpos` is one physics step stale on return from `mj_step`; without
        # this the robot is drawn 5 ms behind the plots beside it.
        self.sim.refresh_link_poses()
        positions, quaternions = self.sim.body_pos_w, self.sim.body_quat_w
        # MuJoCo's (w, x, y, z) -> Foxglove's (x, y, z, w). The one place the
        # two conventions meet.
        world = {name: (positions[i], np.roll(quaternions[i], -1)) for i, name in enumerate(self.sim.body_names)}

        transforms = []
        for name, parent in zip(self.sim.body_names, self.sim.body_parents):
            if parent is None or parent not in world:
                # A child of the MuJoCo worldbody: the floating base. Its pose
                # IS the world pose, so it anchors the whole tree.
                p, q = world[name]
                transforms.append(_transform(stamp, self.root_frame, name, p, q))
            else:
                p, q = _relative(world[parent], world[name])
                transforms.append(_transform(stamp, parent, name, p, q))
        transforms.extend(_transform(stamp, f.parent, f.child, f.translation, f.rotation) for f in self.static)

        # Scene objects hang off the WORLD, not off a robot link -- they are
        # not part of the kinematic chain, and `/sim/scene`'s primitives are
        # frame-locked to these names. Read from `qpos` like the object
        # channel, not from `xpos`, so the two cannot disagree by a step.
        for index, name in enumerate(self.sim.object_body_names):
            position, quaternion = self.sim.object_pose_w(index)
            transforms.append(_transform(stamp, self.root_frame, name, position, np.roll(quaternion, -1)))
        return {"transforms": transforms}
