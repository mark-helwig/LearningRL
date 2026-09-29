"""The scene-object spec and the scene files, with no MuJoCo anywhere.

`objects.py` is deliberately MuJoCo-free so that this whole file is static:
the injection into the plant is `model.py`'s and is tested there, against a
real robot.
"""

from __future__ import annotations

import json

import pytest
import yaml

from LearningRLsim.objects import BUILTIN_DIR, ObjectError, ObjectSpec, load_scene, resolve_scene

pytestmark = pytest.mark.static


def write_scene(tmp_path, data, name="scene.yaml"):
    path = tmp_path / name
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh)
    return path


def test_an_at_start_object_stays_where_it_was_put():
    obj = ObjectSpec(name="crate", pos=(2.0, 0.5, 0.12))
    assert obj.initial_pos == (2.0, 0.5, 0.12)
    assert not obj.deferred


def test_a_deferred_object_waits_on_the_floor_at_staging():
    """Parked at rest rather than frozen mid-air or with contacts off, so it
    needs no special-casing in the loop and `mj_resetData` restores it."""
    obj = ObjectSpec(name="puck", spawn="deferred", staging=(0.0, -25.0))
    assert obj.initial_pos == (0.0, -25.0, obj.half_height)


@pytest.mark.parametrize(
    ("shape", "size", "half_height", "half_extents"),
    [
        ("box", (0.1, 0.2, 0.3), 0.3, (0.1, 0.2, 0.3)),
        ("sphere", (0.15,), 0.15, (0.15, 0.15, 0.15)),
        ("cylinder", (0.18, 0.25), 0.25, (0.18, 0.18, 0.25)),
    ],
)
def test_each_shape_rests_on_its_own_half_height(shape, size, half_height, half_extents):
    obj = ObjectSpec(name="o", shape=shape, size=size, pos=(1.0, 0.0, half_height))
    assert obj.half_height == pytest.approx(half_height)
    assert obj.half_extents == pytest.approx(half_extents)


@pytest.mark.parametrize(
    ("shape", "size"),
    [("box", (0.1, 0.2)), ("sphere", (0.1, 0.2)), ("cylinder", (0.1,))],
)
def test_size_arity_is_checked_per_shape(shape, size):
    with pytest.raises(ObjectError, match="size value"):
        ObjectSpec(name="o", shape=shape, size=size)


def test_an_unknown_shape_lists_the_valid_ones():
    with pytest.raises(ObjectError, match=r"\['box', 'sphere', 'cylinder'\]"):
        ObjectSpec(name="o", shape="pyramid")


def test_an_unknown_spawn_mode_lists_the_valid_ones():
    with pytest.raises(ObjectError, match=r"\['at_start', 'deferred'\]"):
        ObjectSpec(name="o", spawn="later")


def test_a_massless_object_is_rejected():
    with pytest.raises(ObjectError, match="mass must be positive"):
        ObjectSpec(name="o", mass=0.0)


def test_an_object_starting_inside_the_floor_is_rejected():
    """It would be launched by the solver rather than by the robot, which
    reads as a collision and is nothing of the kind."""
    with pytest.raises(ObjectError, match="below its own half-height"):
        ObjectSpec(name="o", size=(0.12, 0.12, 0.12), pos=(1.0, 0.0, 0.05))


def test_staging_too_close_to_the_origin_is_rejected():
    """The robot spawns at the origin and walks; a staging point it could
    reach would make the placement happen twice, the first time by accident."""
    with pytest.raises(ObjectError, match="from the origin"):
        ObjectSpec(name="o", spawn="deferred", staging=(1.0, 1.0))


def test_as_meta_is_json_round_trippable():
    """It lands in `/meta/run`, which is JSON."""
    meta = ObjectSpec(name="puck", spawn="deferred").as_meta()
    assert json.loads(json.dumps(meta)) == meta
    assert meta["size"] == [0.12, 0.12, 0.12]  # MuJoCo's HALF-extents, unconverted


def test_every_shipped_scene_parses():
    for path in sorted(BUILTIN_DIR.glob("*.y*ml")):
        scene = load_scene(path)
        assert scene.objects
        assert scene.name


def test_the_playground_holds_one_deferred_object():
    scene = load_scene("playground")
    assert [obj.name for obj in scene.objects if obj.deferred] == ["puck"]


def test_a_bare_stem_resolves_against_the_shipped_scenes():
    assert resolve_scene("playground") == BUILTIN_DIR / "playground.yaml"


def test_a_missing_scene_lists_what_is_shipped():
    with pytest.raises(ObjectError, match="playground"):
        resolve_scene("not_a_scene")


def test_an_unknown_object_key_is_rejected(tmp_path):
    """A typo'd key must not silently fall back to the default -- the same
    reasoning as the timelines' and `robot.yaml`'s."""
    path = write_scene(tmp_path, {"name": "s", "objects": [{"name": "o", "shpae": "box"}]})
    with pytest.raises(ObjectError, match="shpae"):
        load_scene(path)


def test_an_unknown_scene_key_is_rejected(tmp_path):
    path = write_scene(tmp_path, {"name": "s", "object": [], "objects": [{"name": "o"}]})
    with pytest.raises(ObjectError, match="object"):
        load_scene(path)


def test_two_objects_may_not_share_a_name(tmp_path):
    """Names key `/sim/object`, `/tf` and the scene primitives."""
    path = write_scene(tmp_path, {"name": "s", "objects": [{"name": "o"}, {"name": "o"}]})
    with pytest.raises(ObjectError, match="twice"):
        load_scene(path)


def test_an_empty_scene_is_rejected(tmp_path):
    path = write_scene(tmp_path, {"name": "s", "objects": []})
    with pytest.raises(ObjectError, match="non-empty list"):
        load_scene(path)
