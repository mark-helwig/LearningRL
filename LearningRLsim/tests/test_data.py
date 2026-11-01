"""The recording layer: schemas, decimation, and a round-trippable bag."""

from __future__ import annotations

import json

import numpy as np
import pytest

from LearningRLsim.data import channels
from LearningRLsim.data.encoding import jsonable
from LearningRLsim.data.mcap_sink import McapSink
from LearningRLsim.data.recorder import Recorder, Sink

pytestmark = pytest.mark.static


def test_every_channel_has_a_schema_and_a_rate():
    assert len(channels.ALL) == 13
    for channel in channels.ALL:
        assert channel.topic.startswith("/")
        assert channel.schema["type"] == "object"
        assert channel.schema["properties"]
        assert channel.description
    assert len(channels.BY_TOPIC) == len(channels.ALL)


def test_actuator_cmd_carries_the_live_gains():
    """The bag records the PD law the policy actually ran against."""
    properties = channels.ACTUATOR_CMD.schema["properties"]
    assert "stiffness" in properties
    assert "damping" in properties


def test_sim_contact_keeps_its_original_two_fields():
    """`force` is strictly additive. Every bag written before it existed
    carries `normal_force` and `in_contact`, and a bag from either side of
    this change has to stay comparable to the other."""
    properties = channels.SIM_CONTACT.schema["properties"]
    assert properties["normal_force"]["description"] == "world-z component of the net external force"
    assert properties["in_contact"]["description"] == "1 where normal_force exceeds the threshold"
    assert properties["force"]["items"]["type"] == "array"


def test_sim_object_is_recorded_at_physics_rate():
    """An impact transient is over within one control step, so control rate
    would miss it."""
    assert channels.SIM_OBJECT.rate is channels.Rate.PHYSICS
    for key in ("pos_w", "quat_w", "lin_vel_w", "speed", "force_w", "half_extents", "placed", "shape"):
        assert key in channels.SIM_OBJECT.schema["properties"], key


def test_sim_scene_is_a_foxglove_scene_update_at_control_rate():
    """Foxglove dispatches on the SCHEMA NAME. Control rate rather than ONCE
    because the 3D panel rebuilds entity state on seek, so a lone message at
    t=0 risks nothing being drawn when you scrub into the middle of a bag."""
    assert channels.SIM_SCENE.schema_name == "foxglove.SceneUpdate"
    assert channels.SIM_SCENE.rate is channels.Rate.CONTROL


def test_the_primitive_is_drawn_at_full_extent():
    """MuJoCo's box `size` is HALF-extents and Foxglove's `size` is the full
    one. Getting it wrong draws an object at half its width, which the robot
    then walks through -- geometry that looks deliberate and is a factor of
    two, so it gets a test rather than a comment."""
    from LearningRLsim.data.scene import scene_message
    from LearningRLsim.objects import ObjectSpec

    obj = ObjectSpec(name="crate", size=(0.12, 0.12, 0.12))
    entity = scene_message([obj], 1.5)["entities"][0]

    assert entity["frame_id"] == "crate"  # hung on /tf's frame, not posed itself
    assert entity["frame_locked"] is True
    assert entity["timestamp"] == {"sec": 1, "nsec": 500_000_000}

    cube = entity["cubes"][0]
    assert cube["size"] == {"x": 0.24, "y": 0.24, "z": 0.24}
    assert cube["pose"]["position"] == {"x": 0.0, "y": 0.0, "z": 0.0}
    assert cube["pose"]["orientation"] == {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}


def test_each_shape_lands_in_its_own_primitive_list():
    from LearningRLsim.data.scene import scene_message
    from LearningRLsim.objects import ObjectSpec

    objects = [
        ObjectSpec(name="b", shape="box", size=(0.1, 0.1, 0.1), pos=(1.0, 0.0, 0.1)),
        ObjectSpec(name="s", shape="sphere", size=(0.2,), pos=(1.0, 0.0, 0.2)),
        ObjectSpec(name="c", shape="cylinder", size=(0.3, 0.4), pos=(1.0, 0.0, 0.4)),
    ]
    entities = scene_message(objects, 0.0)["entities"]
    assert len(entities[0]["cubes"]) == 1
    assert entities[1]["spheres"][0]["size"] == {"x": 0.4, "y": 0.4, "z": 0.4}
    cylinder = entities[2]["cylinders"][0]
    assert cylinder["size"] == {"x": 0.6, "y": 0.6, "z": 0.8}
    # 1.0 both ends is a cylinder; a default of 0 draws a cone, or nothing.
    assert cylinder["bottom_scale"] == cylinder["top_scale"] == 1.0


def test_meta_run_carries_provenance():
    properties = channels.META_RUN.schema["properties"]
    for key in ("config", "sha256", "git", "physics", "timeline", "joint_order", "urdf"):
        assert key in properties, key


class ListSink(Sink):
    def __init__(self, physics_decimation=1):
        self.messages = []
        self.physics_decimation = physics_decimation

    def open(self, chans):
        self.opened = chans

    def write(self, channel, t_ns, payload):
        self.messages.append((channel.topic, t_ns, payload))

    def close(self):
        self.closed = True


def test_physics_decimation_thins_only_physics_channels():
    sink = ListSink(physics_decimation=4)
    with Recorder([sink]) as recorder:
        for step in range(8):
            recorder.tick_physics()
            recorder.record("/sim/state", step * 0.005, {"t": step})
            if step % 4 == 0:
                recorder.record("/policy/obs", step * 0.005, {"t": step})

    physics = [m for m in sink.messages if m[0] == "/sim/state"]
    control = [m for m in sink.messages if m[0] == "/policy/obs"]
    assert len(physics) == 2  # steps 0 and 4
    assert len(control) == 2  # never thinned


def test_each_sink_decimates_independently():
    """The file wants 200 Hz torque; a socket stutters on it. One rollout,
    two rates, and a sink must never take the torque of a step whose contact
    it dropped."""
    bag, live = ListSink(physics_decimation=1), ListSink(physics_decimation=4)
    with Recorder([bag, live]) as recorder:
        for step in range(8):
            recorder.tick_physics()
            recorder.record("/sim/state", step * 0.005, {"t": step})
            recorder.record("/actuator/torque", step * 0.005, {"t": step})

    assert len(bag.messages) == 16
    assert len(live.messages) == 4
    assert [m[2]["t"] for m in live.messages] == [0, 0, 4, 4]


def test_timestamps_are_sim_time_in_nanoseconds():
    sink = ListSink()
    with Recorder([sink]) as recorder:
        recorder.tick_physics()
        recorder.record("/sim/state", 1.5, {"t": 1.5})
    assert sink.messages[0][1] == 1_500_000_000


def test_a_failing_sink_does_not_end_the_rollout():
    class Broken(Sink):
        def open(self, chans):
            pass

        def write(self, channel, t_ns, payload):
            raise RuntimeError("disk full")

        def close(self):
            pass

    good = ListSink()
    with Recorder([Broken(), good]) as recorder:
        recorder.tick_physics()
        recorder.record("/sim/state", 0.0, {"t": 0.0})
    assert len(good.messages) == 1


def test_non_finite_floats_become_null_not_invalid_json():
    """A diverging rollout produces NaN; `json.dumps` would emit bare `NaN`,
    which no JSON reader accepts."""
    encoded = json.dumps(jsonable({"v": np.array([1.0, np.nan, np.inf])}))
    assert json.loads(encoded) == {"v": [1.0, None, None]}


def test_numpy_payloads_round_trip_through_a_real_bag(tmp_path):
    from mcap.reader import make_reader

    path = tmp_path / "out.mcap"
    with Recorder([McapSink(path)]) as recorder:
        recorder.tick_physics()
        recorder.record(
            "/sim/state",
            0.25,
            {"t": 0.25, "joint_pos": np.arange(3, dtype=float), "base_height": np.float64(0.8)},
        )

    with path.open("rb") as fh:
        reader = make_reader(fh)
        summary = reader.get_summary()
        # Every channel is registered up front, so a bag is self-describing
        # even for topics that never carried a message.
        assert {c.topic for c in summary.channels.values()} == {c.topic for c in channels.ALL}
        assert all(summary.schemas[c.schema_id].encoding == "jsonschema" for c in summary.channels.values())

        messages = list(reader.iter_messages(topics=["/sim/state"]))
        assert len(messages) == 1
        _, _, message = messages[0]
        assert message.log_time == message.publish_time == 250_000_000
        assert json.loads(message.data)["joint_pos"] == [0.0, 1.0, 2.0]
