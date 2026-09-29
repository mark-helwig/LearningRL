"""Scene objects: the declarative spec, and the YAML that names them.

MuJoCo-free on purpose. `model.py` is what injects these into the plant, and
it imports `mujoco` at module scope; keeping the spec and the parser here is
what lets `pytest -m static` cover them on a host with no MuJoCo at all.

**Sizes are MuJoCo's convention throughout** -- half-extents for a box, radius
for a sphere, `(radius, half_length)` for a cylinder. That convention holds in
`/meta/run` and on `/sim/object` too; the one place it is doubled into a full
extent is `data/scene.py`, which builds the drawable primitive Foxglove wants.

**A deferred object waits ON THE FLOOR, far away**, rather than frozen in
mid-air or with its contacts disabled. Parked that way it is genuinely at rest
with no special-casing anywhere in the loop, and `mj_resetData` restores it
from `qpos0` for free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SHAPES = ("box", "sphere", "cylinder")
SPAWNS = ("at_start", "deferred")

SIZE_ARITY = {"box": 3, "sphere": 1, "cylinder": 2}
"""How many numbers `size` takes per shape: box (x, y, z) half-extents,
sphere (radius,), cylinder (radius, half_length)."""

MIN_STAGING_DISTANCE = 5.0
"""How far a deferred object's parking spot must be from the origin, metres.

The robot spawns at the origin and walks; a staging point it could reach
before the `place_object` event would make the experiment happen twice, the
first time by accident."""


class ObjectError(ValueError):
    """A malformed object spec, or a scene file that does not parse."""


@dataclass(frozen=True)
class ObjectSpec:
    """One rigid body to inject into the scene, with its own freejoint."""

    name: str
    shape: str = "box"
    size: tuple[float, ...] = (0.12, 0.12, 0.12)
    mass: float = 2.0
    friction: float = 0.6
    rgba: tuple[float, float, float, float] = (0.92, 0.41, 0.20, 1.0)
    spawn: str = "at_start"
    pos: tuple[float, float, float] = (2.0, 0.0, 0.12)
    """`at_start` only: where it sits in the world from t=0."""
    ahead: float = 0.55
    """`deferred` only: metres in front of the base when it is placed."""
    lateral: float = 0.0
    """`deferred` only: metres across the base's heading, +left."""
    staging: tuple[float, float] = (0.0, -25.0)
    """`deferred` only: the (x, y) it rests at until placed."""

    def __post_init__(self) -> None:
        if not self.name:
            raise ObjectError("an object needs a name; it is what every channel and event keys on")
        if self.shape not in SHAPES:
            raise ObjectError(f"{self.name}: shape {self.shape!r} is not one of {list(SHAPES)}")
        if self.spawn not in SPAWNS:
            raise ObjectError(f"{self.name}: spawn {self.spawn!r} is not one of {list(SPAWNS)}")

        arity = SIZE_ARITY[self.shape]
        if len(self.size) != arity:
            raise ObjectError(
                f"{self.name}: a {self.shape} takes {arity} size value(s) "
                f"({_SIZE_MEANING[self.shape]}), got {list(self.size)}"
            )
        if any(v <= 0 for v in self.size):
            raise ObjectError(f"{self.name}: every size must be positive, got {list(self.size)}")
        if self.mass <= 0:
            raise ObjectError(f"{self.name}: mass must be positive, got {self.mass}")
        if self.friction < 0:
            raise ObjectError(f"{self.name}: friction must not be negative, got {self.friction}")
        if len(self.rgba) != 4:
            raise ObjectError(f"{self.name}: rgba takes four values, got {list(self.rgba)}")

        if self.spawn == "at_start" and self.pos[2] < self.half_height:
            raise ObjectError(
                f"{self.name}: pos z={self.pos[2]} is below its own half-height {self.half_height}, so it "
                f"would start inside the floor and be launched by the solver rather than by the robot."
            )
        if self.spawn == "deferred" and math.hypot(*self.staging) < MIN_STAGING_DISTANCE:
            raise ObjectError(
                f"{self.name}: staging {list(self.staging)} is {math.hypot(*self.staging):.1f} m from the "
                f"origin, inside the {MIN_STAGING_DISTANCE:g} m the robot could walk to before it is placed."
            )

    @property
    def half_height(self) -> float:
        """Distance from the body frame down to the floor when it rests flat."""
        if self.shape == "box":
            return float(self.size[2])
        if self.shape == "sphere":
            return float(self.size[0])
        return float(self.size[1])  # cylinder: half_length, and MuJoCo's is z-aligned

    @property
    def half_extents(self) -> tuple[float, float, float]:
        """The shape's bounding half-extents, (x, y, z) -- what a viewer draws from."""
        if self.shape == "box":
            return (float(self.size[0]), float(self.size[1]), float(self.size[2]))
        if self.shape == "sphere":
            r = float(self.size[0])
            return (r, r, r)
        r, half = float(self.size[0]), float(self.size[1])
        return (r, r, half)

    @property
    def deferred(self) -> bool:
        return self.spawn == "deferred"

    @property
    def initial_pos(self) -> tuple[float, float, float]:
        """Where the body sits at reset: in place, or parked at staging."""
        if self.spawn == "at_start":
            return (float(self.pos[0]), float(self.pos[1]), float(self.pos[2]))
        return (float(self.staging[0]), float(self.staging[1]), self.half_height)

    def as_meta(self) -> dict[str, Any]:
        """The declared parameters, JSON-clean, for `/meta/run`."""
        return {
            "name": self.name,
            "shape": self.shape,
            "size": [float(v) for v in self.size],
            "half_extents": list(self.half_extents),
            "mass": float(self.mass),
            "friction": float(self.friction),
            "rgba": [float(v) for v in self.rgba],
            "spawn": self.spawn,
            "pos": [float(v) for v in self.pos],
            "ahead": float(self.ahead),
            "lateral": float(self.lateral),
            "staging": [float(v) for v in self.staging],
            "initial_pos": list(self.initial_pos),
        }


_SIZE_MEANING = {
    "box": "x, y, z half-extents",
    "sphere": "radius",
    "cylinder": "radius, half_length",
}

_OBJECT_KEYS = frozenset(
    {"name", "shape", "size", "mass", "friction", "rgba", "spawn", "pos", "ahead", "lateral", "staging"}
)
_SCENE_KEYS = frozenset({"name", "objects"})


@dataclass(frozen=True)
class Scene:
    """A parsed scene file: its objects, plus the YAML verbatim for the bag."""

    path: Path
    name: str
    objects: tuple[ObjectSpec, ...]
    raw: dict[str, Any] = field(default_factory=dict)


BUILTIN_DIR = Path(__file__).resolve().parents[1] / "scenes"
"""The scenes shipped with the repo, resolvable by bare name -- `commands/`'s
sibling, and resolved by exactly the same rule."""


def resolve_scene(path: str | Path) -> Path:
    """A real path, or the name of a scene shipped in `scenes/`."""
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    if candidate.parent == Path("."):
        for name in (candidate.name, f"{candidate.name}.yaml", f"{candidate.name}.yml"):
            builtin = BUILTIN_DIR / name
            if builtin.is_file():
                return builtin
    available = sorted(p.stem for p in BUILTIN_DIR.glob("*.y*ml")) if BUILTIN_DIR.is_dir() else []
    raise ObjectError(f"{path}: no such scene file. Shipped scenes: {available}")


def load_scene(path: str | Path) -> Scene:
    """Parse a scene YAML. An unknown key is an error, as everywhere else here."""
    resolved = resolve_scene(path)
    with resolved.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ObjectError(f"{resolved}: top level must be a mapping")

    unknown = sorted(set(raw) - _SCENE_KEYS)
    if unknown:
        raise ObjectError(f"{resolved}: unknown key(s) {unknown}. A scene takes {sorted(_SCENE_KEYS)}.")

    entries = raw.get("objects")
    if not isinstance(entries, list) or not entries:
        raise ObjectError(f"{resolved}.objects: expected a non-empty list of object mappings")

    objects = tuple(_build_object(entry, resolved) for entry in entries)
    names = [obj.name for obj in objects]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ObjectError(f"{resolved}: object name(s) {duplicates} appear twice; names key every channel and event")

    return Scene(path=resolved, name=str(raw.get("name") or resolved.stem), objects=objects, raw=raw)


def _build_object(entry: Any, path: Path) -> ObjectSpec:
    if not isinstance(entry, dict) or "name" not in entry:
        raise ObjectError(f"{path}.objects: each entry is a mapping with at least a `name`; got {entry!r}")
    unknown = sorted(set(entry) - _OBJECT_KEYS)
    if unknown:
        raise ObjectError(
            f"{path}.objects.{entry['name']}: unknown key(s) {unknown}. An object takes {sorted(_OBJECT_KEYS)}."
        )
    fields: dict[str, Any] = {"name": str(entry["name"])}
    for key in ("shape", "spawn"):
        if key in entry:
            fields[key] = str(entry[key])
    for key in ("mass", "friction", "ahead", "lateral"):
        if key in entry:
            fields[key] = float(entry[key])
    for key in ("size", "rgba", "pos", "staging"):
        if key in entry:
            fields[key] = tuple(float(v) for v in entry[key])
    return ObjectSpec(**fields)
