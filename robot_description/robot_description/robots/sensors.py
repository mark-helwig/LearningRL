"""Shared sensor-config helpers for every robot in this repo.

Needs Isaac Lab (like each robot's `robot.py`) -- NOT part of the
Isaac-Lab-free module set documented in repo CLAUDE.md.

Each robot's `robots/<robot>/sensor.py` declares named sensor groups by body
name and nothing else; this module turns a group into the `SensorBaseCfg` +
`SceneEntityCfg` pair a training env actually needs. Contact sensors are the
only kind implemented so far -- raycasters, IMUs and frame transformers belong
here too when they arrive, which is why this module is named for sensors
generally.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg

# Every robot in this repo spawns at this prim path (see each robot.py's
# ArticulationCfg.prim_path); groups resolve their bodies underneath it.
DEFAULT_ROOT_PRIM = "{ENV_REGEX_NS}/Robot"


@dataclass(frozen=True)
class ContactSensorGroup:
    """A named set of bodies plus the contact sensor that senses them.

    `bodies` are `re.fullmatch`-style patterns exactly as written in the
    robot's `sensor.py`. `sensor_cfg` is the template scene entity; `attach()`
    deep-copies it per scene so two env cfgs (a task and its `-PLAY` twin)
    never share one mutable cfg object.
    """

    name: str
    bodies: tuple[str, ...]
    sensor_cfg: ContactSensorCfg

    def entity_cfg(self, key: str | None = None) -> SceneEntityCfg:
        """A NEW `SceneEntityCfg` naming this group's bodies on scene entity `key`.

        Call this once per MDP term. `SceneEntityCfg.resolve` rewrites `body_names`
        and `body_ids` in place, so handing the same instance to two terms makes the
        second one re-resolve an already-resolved object and raise "Both 'body_names'
        and 'body_ids' are specified, and are not consistent."
        """
        return SceneEntityCfg(key or self.name, body_names=list(self.bodies))

    def attach(self, scene_cfg: Any, key: str | None = None) -> SceneEntityCfg:
        """Add the sensor to `scene_cfg` under `key` and return one `SceneEntityCfg`.

        The return value is safe for a single term; use `entity_cfg()` per term when
        more than one references this group.
        """
        key = key or self.name
        setattr(scene_cfg, key, copy.deepcopy(self.sensor_cfg))
        return self.entity_cfg(key)


def contact_sensor_group(
    name: str,
    bodies: Sequence[str],
    *,
    root_prim: str = DEFAULT_ROOT_PRIM,
    prim_expr: str | None = None,
    exclude_self: bool = False,
    **sensor_kwargs: Any,
) -> ContactSensorGroup:
    """Build a `ContactSensorGroup` from a list of body-name patterns.

    Args:
        name: Scene entity key the group attaches under, e.g. `"g1_termination"`.
        bodies: `re.fullmatch`-style body patterns. Resolved against the live
            scene by `SceneEntityCfg`, which raises on a pattern matching nothing.
        root_prim: Prim path the robot spawns at.
        prim_expr: Leaf expression appended to `root_prim` to form the sensor's
            `prim_path`, overriding the default below.
        exclude_self: Filter against the robot's own bodies, so self-contact lands in
            `force_matrix_w` and can be subtracted from `net_forces_w`. Newton only.
        **sensor_kwargs: Passed through to `ContactSensorCfg`
            (`history_length`, `track_air_time`, ...).

    A single-pattern group narrows `prim_path` to that pattern; a multi-pattern
    group falls back to `.*` (whole robot) because a `(a|b)` alternation is a
    valid PhysX leaf regex but NOT valid fnmatch, and the Newton backend matches
    via fnmatch (`newton_manager.add_contact_sensor`). Over-matching is harmless:
    `SceneEntityCfg.body_names` narrows to the group, so correctness never
    depends on the prim_path narrowing.

    Emits `isaaclab.sensors.ContactSensorCfg` -- the backend-dispatching factory
    -- deliberately, NOT a `PresetCfg`: a field assigned on a `PresetCfg` wrapper
    is silently discarded at preset-resolution time.
    """
    if not bodies:
        raise ValueError(f"contact sensor group {name!r} has no bodies")

    patterns = tuple(bodies)
    if prim_expr is None:
        prim_expr = patterns[0] if len(patterns) == 1 else ".*"
    if exclude_self:
        sensor_kwargs["filter_prim_paths_expr"] = [f"{root_prim}/.*"]

    return ContactSensorGroup(
        name=name,
        bodies=patterns,
        sensor_cfg=ContactSensorCfg(prim_path=f"{root_prim}/{prim_expr}", **sensor_kwargs),
    )
