"""The `/tf` link frames a 3D viewer renders the URDF from.

The static half needs nothing: quaternion conventions and the fixed-joint
walker, against a synthetic URDF. The bottom half needs MuJoCo and the vendor
checkout, and pins the one assumption the whole approach rests on -- that a
MuJoCo body frame IS the URDF link frame of the same name.
"""

from __future__ import annotations

import numpy as np
import pytest

from LearningRLsim.data import channels
from LearningRLsim.data.transforms import _quat_from_rpy, _quat_multiply, _quat_rotate, urdf_static_frames

SYNTHETIC_URDF = """<?xml version="1.0"?>
<robot name="test">
  <link name="world"/>
  <link name="base"/>
  <link name="upper"/>
  <link name="sensor"/>
  <link name="tip"/>
  <link name="floating"/>
  <joint name="hinge" type="revolute">
    <parent link="base"/><child link="upper"/>
    <origin xyz="0 0 0.5" rpy="0 0 0"/>
  </joint>
  <joint name="to_sensor" type="fixed">
    <parent link="upper"/><child link="sensor"/>
    <origin xyz="0.1 0 0" rpy="0 0 1.5707963267948966"/>
  </joint>
  <joint name="to_tip" type="fixed">
    <parent link="sensor"/><child link="tip"/>
    <origin xyz="0.2 0 0" rpy="0 0 0"/>
  </joint>
</robot>
"""


@pytest.fixture
def synthetic_urdf(tmp_path):
    path = tmp_path / "test.urdf"
    path.write_text(SYNTHETIC_URDF, encoding="utf-8")
    return path


@pytest.mark.static
def test_tf_is_registered_under_foxglove_s_own_schema_name():
    """Foxglove renders off the SCHEMA NAME. A schema called `tf` draws nothing."""
    assert channels.TF.schema_name == "foxglove.FrameTransforms"
    assert channels.TF.rate is channels.Rate.CONTROL
    # Every other channel keeps the derived name.
    assert channels.SIM_STATE.schema_name == "sim.state"


@pytest.mark.static
def test_rpy_to_quaternion_is_the_urdf_fixed_axis_convention():
    """R = Rz(yaw) Ry(pitch) Rx(roll), applied to a vector."""
    q = _quat_from_rpy((0.0, 0.0, np.pi / 2))
    assert np.allclose(_quat_rotate(q, np.array([1.0, 0.0, 0.0])), [0.0, 1.0, 0.0], atol=1e-12)

    q = _quat_from_rpy((np.pi / 2, 0.0, 0.0))
    assert np.allclose(_quat_rotate(q, np.array([0.0, 1.0, 0.0])), [0.0, 0.0, 1.0], atol=1e-12)

    # Composition order: rotating by a then b must equal rotating by a*b.
    a, b = _quat_from_rpy((0.3, -0.2, 0.5)), _quat_from_rpy((-0.1, 0.4, 0.2))
    v = np.array([0.3, -0.7, 0.2])
    assert np.allclose(_quat_rotate(_quat_multiply(a, b), v), _quat_rotate(a, _quat_rotate(b, v)))


@pytest.mark.static
def test_fixed_joint_chains_collapse_onto_the_nearest_simulated_body(synthetic_urdf):
    """MuJoCo merges fixed children away; every one must come back as a constant."""
    frames = {f.child: f for f in urdf_static_frames(synthetic_urdf, {"base", "upper"})}

    # `sensor` hangs one fixed joint off a body the sim publishes.
    assert frames["sensor"].parent == "upper"
    assert np.allclose(frames["sensor"].translation, [0.1, 0.0, 0.0])

    # `tip` is two fixed joints deep, so its offset is the COMPOSED chain --
    # the yaw at `sensor` turns tip's own +x offset into +y.
    assert frames["tip"].parent == "upper"
    assert np.allclose(frames["tip"].translation, [0.1, 0.2, 0.0], atol=1e-12)


@pytest.mark.static
def test_links_not_rigidly_attached_to_a_simulated_body_are_skipped(synthetic_urdf):
    """A frame that would need kinematics is not a static frame, and an
    orphaned link (the G1's `world`) has nothing to attach to at all."""
    children = {f.child for f in urdf_static_frames(synthetic_urdf, {"base"})}
    assert "world" not in children  # no joint names it as a child
    assert "floating" not in children  # declared, never jointed
    assert "upper" not in children  # reachable only through a revolute joint
    assert "sensor" not in children  # ... and so is everything past it


# -- the assumption everything else rests on --------------------------------

pytest.importorskip("mujoco")
pytest.importorskip("robot_description")


@pytest.fixture(scope="module")
def g1_frames():
    """Parent-relative body origins from the vendor MJCF, and link origins
    from the vendor URDF, for the same robot."""
    import xml.etree.ElementTree as ET

    import mujoco
    from robot_description.paths import mjcf_path, urdf_path

    model = mujoco.MjModel.from_xml_path(str(mjcf_path("unitree_g1")))
    mjcf = {}
    for bid in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        parent = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.body_parentid[bid]))
        mjcf[name] = (parent, model.body_pos[bid].copy(), np.roll(model.body_quat[bid].copy(), -1))

    root = ET.parse(urdf_path("unitree_g1")).getroot()
    urdf = {}
    for joint in root.findall("joint"):
        origin = joint.find("origin")
        xyz = (origin.get("xyz") if origin is not None else None) or "0 0 0"
        rpy = (origin.get("rpy") if origin is not None else None) or "0 0 0"
        urdf[joint.find("child").get("link")] = (
            joint.find("parent").get("link"),
            np.array([float(v) for v in xyz.split()]),
            _quat_from_rpy(tuple(float(v) for v in rpy.split())),
        )
    return mjcf, urdf


def test_every_mujoco_body_frame_is_the_urdf_link_frame_of_the_same_name(g1_frames):
    """The reason `/tf` can publish `data.xpos` verbatim and have Foxglove hang
    the URDF's meshes on it. True because Unitree hand-authored the MJCF from
    the URDF -- a property of this vendor, not a guarantee, so it is pinned.

    If this ever fails for a new robot, the fix is a per-link correction in
    `transforms.py`, NOT a tolerance here: a rotated link frame renders a limb
    pointing the wrong way while every number in the bag stays correct.
    """
    mjcf, urdf = g1_frames
    for name, (parent, pos, quat) in mjcf.items():
        if name == "pelvis":
            continue  # the free-joint root: its parent is the world, not a link
        assert name in urdf, f"{name} is a MuJoCo body with no URDF joint"
        urdf_parent, urdf_pos, urdf_quat = urdf[name]
        assert parent == urdf_parent, name
        assert np.allclose(pos, urdf_pos, atol=1e-9), name
        # A quaternion and its negation are the same rotation.
        assert np.allclose(quat, urdf_quat, atol=1e-6) or np.allclose(quat, -urdf_quat, atol=1e-6), name


def test_the_urdf_links_mujoco_merged_away_are_recovered(g1_frames):
    """Nine on the G1, and they are the visible ones: head, both hands, the
    pelvis shell. A bag without them renders a decapitated robot."""
    from robot_description.paths import urdf_path

    mjcf, _ = g1_frames
    recovered = {f.child for f in urdf_static_frames(urdf_path("unitree_g1"), set(mjcf))}
    assert {"head_link", "left_rubber_hand", "right_rubber_hand", "pelvis_contour_link"} <= recovered


@pytest.fixture(scope="module")
def g1_sim():
    """A real G1 `RobotModel`, driven by a synthetic three-joint config -- enough
    to step the physics, which is all these need."""
    import copy
    import tempfile
    from pathlib import Path

    from LearningRLsim import runconfig
    from LearningRLsim.model import RobotModel, SimOverrides, resolve_mjcf
    from tests.conftest import MINIMAL, write_run

    config = copy.deepcopy(MINIMAL)
    joints = ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint"]
    for key in ("default_joint_pos", "actuators"):
        config[key]["joint_order"] = joints
    config["actions"]["terms"][0]["joint_names"] = joints

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = write_run(Path(tmp), config, name="g1")
        cfg = runconfig.load(run_dir)
        yield RobotModel(resolve_mjcf(cfg), cfg, SimOverrides())


def test_link_poses_lag_one_physics_step_until_refreshed(g1_sim):
    """`mj_step` forwards, THEN integrates -- so `xpos` describes the state
    before the integration and `qpos` the state after it. Drawing the robot
    from unrefreshed poses puts it 5 ms behind the plots beside it."""
    g1_sim.reset()
    pelvis = g1_sim.body_names.index("pelvis")
    g1_sim.step(np.zeros(g1_sim.cfg.num_joints))

    lag = g1_sim.base_pos_w - g1_sim.body_pos_w[pelvis]
    assert np.any(lag != 0.0), "no lag at all would mean mj_step's ordering changed"
    assert np.allclose(lag, g1_sim.base_lin_vel_w * g1_sim.model.opt.timestep, atol=1e-15)

    g1_sim.refresh_link_poses()
    assert np.allclose(g1_sim.base_pos_w, g1_sim.body_pos_w[pelvis], atol=1e-15)


def test_the_tf_message_agrees_with_sim_state_exactly(g1_sim):
    """The 3D view and the plotted numbers must describe the same instant."""
    from LearningRLsim.data.transforms import LinkTransforms
    from LearningRLsim.model import resolve_urdf

    g1_sim.reset()
    g1_sim.step(np.zeros(g1_sim.cfg.num_joints))

    message = LinkTransforms(g1_sim, resolve_urdf(g1_sim.cfg)).message(g1_sim.time)
    pelvis = next(t for t in message["transforms"] if t["child_frame_id"] == "pelvis")
    assert pelvis["parent_frame_id"] == "world"  # the floating base anchors the tree

    translation = [pelvis["translation"][k] for k in "xyz"]
    assert np.allclose(translation, g1_sim.base_pos_w, atol=1e-15)
    # /tf is (x, y, z, w); /sim/state's base_quat_w is MuJoCo's (w, x, y, z).
    rotation = [pelvis["rotation"][k] for k in ("w", "x", "y", "z")]
    assert np.allclose(rotation, g1_sim.base_quat_w, atol=1e-15)


def test_the_tree_mirrors_the_kinematic_chain(g1_sim):
    """Not a flat `world -> link` fan. Foxglove's URDF fallback looks up each
    `parent_link <- child_link` edge; when those are sibling lookups through
    the world it misses and places the link at its joint's REST pose, which
    presents as every link being rotated wrong."""
    from LearningRLsim.data.transforms import LinkTransforms

    g1_sim.reset()
    message = LinkTransforms(g1_sim).message(0.0)
    parent_of = {t["child_frame_id"]: t["parent_frame_id"] for t in message["transforms"]}

    assert parent_of["pelvis"] == "world"
    assert parent_of["left_knee_link"] == "left_hip_yaw_link"
    assert parent_of["torso_link"] == "waist_roll_link"
    assert parent_of["right_elbow_link"] == "right_shoulder_yaw_link"
    # Exactly one link hangs off the world: the floating base.
    assert [c for c, p in parent_of.items() if p == "world"] == ["pelvis"]


def test_relative_transforms_reproduce_the_world_poses(g1_sim):
    """Composing the chain must give back MuJoCo's own `xpos`/`xquat` -- the
    tree is a re-parenting, not a re-derivation."""
    import numpy as np

    from LearningRLsim.data.transforms import LinkTransforms, _quat_multiply, _quat_rotate

    g1_sim.reset()
    g1_sim.step(np.zeros(g1_sim.cfg.num_joints))
    frames = {t["child_frame_id"]: t for t in LinkTransforms(g1_sim).message(g1_sim.time)["transforms"]}

    def world(name):
        t = frames[name]
        p = np.array([t["translation"][k] for k in "xyz"])
        q = np.array([t["rotation"][k] for k in "xyzw"])
        if t["parent_frame_id"] == "world":
            return p, q
        pp, pq = world(t["parent_frame_id"])
        return pp + _quat_rotate(pq, p), _quat_multiply(pq, q)

    for i, name in enumerate(g1_sim.body_names):
        p, q = world(name)
        assert np.allclose(p, g1_sim.body_pos_w[i], atol=1e-12), name
        expected = np.roll(g1_sim.body_quat_w[i], -1)
        assert np.allclose(q, expected, atol=1e-12) or np.allclose(q, -expected, atol=1e-12), name


@pytest.mark.static
def test_joint_states_is_registered_under_foxglove_s_own_schema_name():
    assert channels.JOINT_STATES.schema_name == "foxglove.JointStates"
    assert channels.JOINT_STATES.rate is channels.Rate.CONTROL


def test_joint_states_carries_every_actuated_joint_by_name(g1_sim):
    """Foxglove's URDF layer matches these against the URDF's JOINT names, not
    its link names -- a different lookup from the one `/tf` needs, which is why
    both encodings are written."""
    import numpy as np

    from LearningRLsim.data.transforms import joint_states_message

    g1_sim.reset()
    g1_sim.step(np.zeros(g1_sim.cfg.num_joints))
    joints = joint_states_message(g1_sim, g1_sim.time)["joints"]

    assert [j["name"] for j in joints] == list(g1_sim.cfg.joint_order)
    assert np.allclose([j["position"] for j in joints], g1_sim.joint_pos, atol=1e-15)
    assert np.allclose([j["velocity"] for j in joints], g1_sim.joint_vel, atol=1e-15)


def test_a_scene_object_gets_a_world_parented_frame(g1_sim):
    """Objects hang off the WORLD, not off a robot link -- they are not part
    of the kinematic chain, and `/sim/scene`'s primitives are frame-locked to
    these names. The object-free fixture above keeps its `exactly one link
    hangs off the world` assertion; this is the other half of it.
    """
    from LearningRLsim.data.transforms import LinkTransforms
    from LearningRLsim.model import RobotModel, SimOverrides, resolve_mjcf
    from LearningRLsim.objects import ObjectSpec

    box = ObjectSpec(name="crate", size=(0.12, 0.12, 0.12), pos=(2.0, 0.5, 0.12))
    sim = RobotModel(resolve_mjcf(g1_sim.cfg), g1_sim.cfg, SimOverrides(), objects=(box,))
    sim.reset()

    transforms = LinkTransforms(sim).message(0.0)["transforms"]
    parent_of = {t["child_frame_id"]: t["parent_frame_id"] for t in transforms}
    assert parent_of["crate"] == "world"
    assert sorted(c for c, p in parent_of.items() if p == "world") == ["crate", "pelvis"]

    # Straight off `qpos`, so `/tf` and `/sim/object` cannot disagree by a step.
    frame = next(t for t in transforms if t["child_frame_id"] == "crate")
    assert [frame["translation"][k] for k in "xyz"] == pytest.approx(box.pos)
    assert [frame["rotation"][k] for k in "xyzw"] == pytest.approx([0.0, 0.0, 0.0, 1.0])
