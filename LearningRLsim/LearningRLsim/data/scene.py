"""Scene objects as `foxglove.SceneUpdate` primitives -- what the 3D panel draws.

One entity per object, each holding a single primitive with an **identity
pose**, `frame_locked`, hung on the frame `/tf` publishes for that object. So
`/tf` alone carries the motion and the primitive never needs a world pose:
there is exactly one statement of where the object is, and it is the same one
the robot's links use.

**Foxglove's primitive `size` is the FULL extent; MuJoCo's box `size` is
half-extents.** Getting that wrong draws an object at half its real width,
which the robot then walks through -- geometry that looks deliberate and is a
factor of two. The doubling happens here, in this one place. `/sim/object`'s
`half_extents` and `/meta/run`'s `size` both stay in MuJoCo's convention, and
`tests/test_data.py` pins the doubling.

MuJoCo-free: the geometry comes off `ObjectSpec`, so this is static-testable.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from LearningRLsim.objects import ObjectSpec

_IDENTITY_POSE = {
    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
}

# Which primitive list a shape lands in. A cylinder additionally takes
# `bottom_scale`/`top_scale`, which are 1.0 for a true cylinder rather than a
# cone -- omitting them defaults them to 0 in some builds and draws nothing.
_PRIMITIVE = {"box": "cubes", "sphere": "spheres", "cylinder": "cylinders"}


def _stamp(t: float) -> dict[str, int]:
    t_ns = int(round(t * 1e9))
    return {"sec": t_ns // 1_000_000_000, "nsec": t_ns % 1_000_000_000}


def _primitive(obj: ObjectSpec) -> dict[str, Any]:
    x, y, z = obj.half_extents
    primitive: dict[str, Any] = {
        "pose": _IDENTITY_POSE,
        "size": {"x": 2.0 * x, "y": 2.0 * y, "z": 2.0 * z},
        "color": {"r": obj.rgba[0], "g": obj.rgba[1], "b": obj.rgba[2], "a": obj.rgba[3]},
    }
    if obj.shape == "cylinder":
        primitive["bottom_scale"] = 1.0
        primitive["top_scale"] = 1.0
    return primitive


def scene_message(objects: Iterable[ObjectSpec], t: float) -> dict[str, Any]:
    """One `foxglove.SceneUpdate` describing every object's geometry."""
    stamp = _stamp(t)
    entities = []
    for obj in objects:
        entity: dict[str, Any] = {
            "timestamp": stamp,
            "frame_id": obj.name,
            "id": obj.name,
            "frame_locked": True,
            "cubes": [],
            "spheres": [],
            "cylinders": [],
        }
        entity[_PRIMITIVE[obj.shape]].append(_primitive(obj))
        entities.append(entity)
    return {"deletions": [], "entities": entities}
