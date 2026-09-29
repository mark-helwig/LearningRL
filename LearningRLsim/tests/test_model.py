"""Model loading and the sim2sim frame conventions.

NOT static: needs MuJoCo, `robot_description` and its vendor checkout.
"""

from __future__ import annotations

import numpy as np
import pytest

from LearningRLsim import runconfig
from tests.conftest import MINIMAL, MINIMAL_RPO, write_run

# Before importing LearningRLsim.model, which imports mujoco at module scope: these
# guards must run at COLLECTION time, or `pytest -m static` on a CPU-only host
# with no MuJoCo dies collecting this file instead of skipping it.
mujoco = pytest.importorskip("mujoco")
pytest.importorskip("robot_description")

from LearningRLsim.model import ModelError, RobotModel, SimOverrides, resolve_mjcf  # noqa: E402
from LearningRLsim.objects import ObjectSpec  # noqa: E402


@pytest.fixture(scope="module")
def g1():
    """The real G1, driven by a synthetic config whose joint_order is the
    MJCF's -- so this exercises the remap without needing an exported run."""
    import copy
    import tempfile
    from pathlib import Path

    config = copy.deepcopy(MINIMAL)
    joints = [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
    ]
    for key in ("default_joint_pos", "actuators"):
        config[key]["joint_order"] = joints
    config["actions"]["terms"][0]["joint_names"] = joints

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = write_run(Path(tmp), config, name="g1")
        cfg = runconfig.load(run_dir)
        yield RobotModel(resolve_mjcf(cfg), cfg, SimOverrides())


def test_joints_are_remapped_by_name(g1):
    for i, name in enumerate(g1.cfg.joint_order):
        assert mujoco.mj_id2name(g1.model, mujoco.mjtObj.mjOBJ_JOINT, g1.joint_ids[i]) == name


def test_actuators_are_matched_on_their_target_joint(g1):
    for i, jid in enumerate(g1.joint_ids):
        aid = g1.actuator_ids[i]
        # int(), not the bare enum: see the constants at the top of model.py.
        assert int(g1.model.actuator_trntype[aid]) == int(mujoco.mjtTrn.mjTRN_JOINT)
        assert g1.model.actuator_trnid[aid, 0] == jid


def test_config_armature_and_friction_reach_the_model(g1):
    """The vendor MJCF declares neither, so leaving MuJoCo's zeros would
    compare the policy against a different plant."""
    np.testing.assert_allclose(g1.model.dof_armature[g1.dof_adr], g1.cfg.actuators.armature)
    np.testing.assert_allclose(g1.model.dof_frictionloss[g1.dof_adr], g1.cfg.actuators.friction)
    assert g1.model.opt.timestep == pytest.approx(g1.cfg.sim_dt)


def test_reset_puts_the_robot_upright_at_spawn_height(g1):
    g1.reset()
    assert g1.base_pos_w[2] == pytest.approx(g1.overrides.spawn_height)
    np.testing.assert_allclose(g1.base_quat_w, [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(g1.joint_pos, g1.cfg.default_joint_pos)
    np.testing.assert_allclose(g1.projected_gravity_b, [0.0, 0.0, -1.0], atol=1e-12)


def test_angular_velocity_is_body_frame_and_matches_the_pelvis_gyro(g1):
    """THE sim2sim trap. `qvel[3:6]` is already body-frame; rotating it too
    produces a robot that walks almost right.

    Pinned against the MJCF's own gyro sensor, which is defined in the site
    frame -- an independent statement of the same quantity.
    """
    gyro_id = mujoco.mj_name2id(g1.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu-pelvis-angular-velocity")
    adr = g1.model.sensor_adr[gyro_id]

    rng = np.random.default_rng(0)
    for _ in range(5):
        quat = rng.normal(size=4)
        g1.data.qpos[g1.base_qpos_adr + 3 : g1.base_qpos_adr + 7] = quat / np.linalg.norm(quat)
        g1.data.qvel[g1.base_dof_adr : g1.base_dof_adr + 6] = rng.normal(size=6)
        mujoco.mj_forward(g1.model, g1.data)

        gyro = g1.data.sensordata[adr : adr + 3]
        np.testing.assert_allclose(g1.base_ang_vel_b, gyro, atol=1e-9)
        # And the wrong version really is different, so the test can fail.
        assert not np.allclose(g1.base_rot_w.T @ g1.base_ang_vel_b, gyro, atol=1e-6)

    g1.reset()


def test_linear_velocity_is_rotated_out_of_the_world_frame(g1):
    """`qvel[0:3]` is world-frame, unlike the angular half. Yaw the base 90
    degrees and a world +x velocity must read as body -y."""
    g1.reset()
    half = np.pi / 4  # 90 deg yaw -> quat (cos45, 0, 0, sin45)
    g1.data.qpos[g1.base_qpos_adr + 3 : g1.base_qpos_adr + 7] = [np.cos(half), 0.0, 0.0, np.sin(half)]
    g1.data.qvel[g1.base_dof_adr : g1.base_dof_adr + 3] = [1.0, 0.0, 0.0]
    mujoco.mj_forward(g1.model, g1.data)

    np.testing.assert_allclose(g1.base_lin_vel_w, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(g1.base_lin_vel_b, [0.0, -1.0, 0.0], atol=1e-12)
    g1.reset()


def test_a_joint_the_model_lacks_is_reported_with_the_models_joints(tmp_path):
    import copy

    config = copy.deepcopy(MINIMAL)
    joints = ["left_hip_pitch_joint", "left_hip_roll_joint", "not_a_real_joint"]
    for key in ("default_joint_pos", "actuators"):
        config[key]["joint_order"] = joints
    config["actions"]["terms"][0]["joint_names"] = joints

    cfg = runconfig.load(write_run(tmp_path, config, name="bad"))
    with pytest.raises(ModelError, match="not_a_real_joint"):
        RobotModel(resolve_mjcf(cfg), cfg, SimOverrides())


def test_the_rpo_loads_remaps_and_stands_at_its_own_height(tmp_path):
    """The RPO through the same path as the G1. Its init height is 0.75, not the
    G1's 0.8 default, and the exporter does not record it -- hence the override."""
    cfg = runconfig.load(write_run(tmp_path, MINIMAL_RPO, name="rpo"))
    sim = RobotModel(resolve_mjcf(cfg), cfg, SimOverrides(spawn_height=0.75))
    for i, name in enumerate(cfg.joint_order):
        assert mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_JOINT, sim.joint_ids[i]) == name
        assert sim.model.actuator_trnid[sim.actuator_ids[i], 0] == sim.joint_ids[i]
    sim.reset()
    assert sim.base_pos_w[2] == pytest.approx(0.75)
    np.testing.assert_allclose(sim.joint_pos, cfg.default_joint_pos)


def test_g1_joint_order_happens_to_be_the_mjcfs():
    """Documented luck, not contract -- but worth pinning, because if the
    vendor reorders them the remap is what keeps working and this test is what
    says the free ride ended."""
    from robot_description.paths import mjcf_path

    model = mujoco.MjModel.from_xml_path(mjcf_path("unitree_g1"))
    hinges = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        for j in range(model.njnt)
        if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE)
    ]
    assert len(hinges) == 29
    assert hinges[0] == "left_hip_pitch_joint"


# -- scene objects ----------------------------------------------------------
BOX = ObjectSpec(name="crate", shape="box", size=(0.12, 0.12, 0.12), mass=2.0, friction=0.6, pos=(2.0, 0.0, 0.12))
PUCK = ObjectSpec(name="puck", shape="cylinder", size=(0.1, 0.05), spawn="deferred", staging=(0.0, -25.0))


@pytest.fixture(scope="module")
def g1_cfg():
    """The same synthetic three-joint G1 config the `g1` fixture uses."""
    import copy
    import tempfile
    from pathlib import Path

    config = copy.deepcopy(MINIMAL)
    joints = ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint"]
    for key in ("default_joint_pos", "actuators"):
        config[key]["joint_order"] = joints
    config["actions"]["terms"][0]["joint_names"] = joints

    with tempfile.TemporaryDirectory() as tmp:
        yield runconfig.load(write_run(Path(tmp), config, name="g1"))


@pytest.fixture(scope="module")
def g1_objects(g1_cfg):
    return RobotModel(resolve_mjcf(g1_cfg), g1_cfg, SimOverrides(), objects=(BOX, PUCK))


def test_mjspec_load_is_the_same_plant(g1_cfg):
    """THE test that licenses making MjSpec the only load path.

    `model.py` no longer forks on whether objects were asked for, so every
    baseline bag now comes out of the object runs' own code path. If the two
    loaders disagreed anywhere, the difference would present as a policy
    difference and nothing in a rollout would say otherwise.
    """
    path = resolve_mjcf(g1_cfg)
    direct = mujoco.MjModel.from_xml_path(str(path))
    through_spec = RobotModel(path, g1_cfg, SimOverrides()).model

    for attr in ("nq", "nv", "nbody", "ngeom", "njnt", "nu"):
        assert getattr(direct, attr) == getattr(through_spec, attr), attr
    for attr in ("body_mass", "body_inertia", "geom_size", "jnt_range", "qpos0"):
        np.testing.assert_array_equal(getattr(direct, attr), getattr(through_spec, attr), err_msg=attr)
    # These two are written by `_apply_config_to_model`/`_apply_overrides`, so
    # compare the spec-loaded model against the same treatment applied by hand.
    direct.geom_friction[:, 0] = SimOverrides.friction
    np.testing.assert_array_equal(direct.geom_friction, through_spec.geom_friction)
    direct.dof_armature[:] = 0.0
    through_spec.dof_armature[:] = 0.0
    np.testing.assert_array_equal(direct.dof_armature, through_spec.dof_armature)


def test_injected_objects_leave_the_base_lookup_alone(g1_objects):
    """Each object carries its own freejoint, so the raw count is 3. The base
    lookup subtracts exactly the bodies `_add_object` created."""
    free_type = int(mujoco.mjtJoint.mjJNT_FREE)
    free = [j for j in range(g1_objects.model.njnt) if int(g1_objects.model.jnt_type[j]) == free_type]
    assert len(free) == 3
    assert mujoco.mj_id2name(g1_objects.model, mujoco.mjtObj.mjOBJ_BODY, g1_objects.base_body_id) == "pelvis"


def test_object_bodies_are_not_offered_as_robot_links(g1_objects):
    """`body_names` is consumed by `urdf_static_frames` and `LinkTransforms`
    as the ROBOT's link frames; a crate in there is offered to the URDF as a
    link."""
    assert "crate" not in g1_objects.body_names
    assert "puck" not in g1_objects.body_names
    assert g1_objects.object_body_names == ["crate", "puck"]


def test_terrain_friction_does_not_reach_an_object(g1_cfg):
    """`--friction` reproduces FLAT_TERRAIN_CFG, which is the terrain's. A
    sweep of one must not silently move the other."""
    sim = RobotModel(resolve_mjcf(g1_cfg), g1_cfg, SimOverrides(friction=0.35), objects=(BOX, PUCK))
    box_geom, puck_geom = sim.object_geom_ids
    assert sim.model.geom_friction[box_geom, 0] == pytest.approx(BOX.friction)
    assert sim.model.geom_friction[puck_geom, 0] == pytest.approx(PUCK.friction)

    others = [g for g in range(sim.model.ngeom) if g not in sim.object_geom_ids]
    np.testing.assert_allclose(sim.model.geom_friction[others, 0], 0.35)


def test_reset_returns_a_deferred_object_to_staging(g1_objects):
    """Which is what re-arms placement after a mid-rollout reset."""
    g1_objects.reset()
    np.testing.assert_allclose(g1_objects.object_pose_w(1)[0], PUCK.initial_pos)

    g1_objects.place_object(1, (1.0, 0.5, PUCK.half_height))
    g1_objects.reset()
    np.testing.assert_allclose(g1_objects.object_pose_w(1)[0], PUCK.initial_pos)
    np.testing.assert_allclose(g1_objects.object_pose_w(0)[0], BOX.pos)


def test_place_object_moves_it_and_zeroes_its_velocity(g1_objects):
    g1_objects.reset()
    g1_objects.data.qvel[g1_objects.object_dof_adr[0] : g1_objects.object_dof_adr[0] + 6] = 1.0

    g1_objects.place_object(0, (1.5, -0.25, 0.12))
    np.testing.assert_allclose(g1_objects.object_pose_w(0)[0], [1.5, -0.25, 0.12])
    np.testing.assert_allclose(g1_objects.object_pose_w(0)[1], [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(g1_objects.object_vel_w(0), 0.0)
    g1_objects.reset()


def test_place_object_does_not_forward_the_model(g1_objects):
    """Calling `mj_forward` would refresh `xmat` mid-policy-step and close,
    for that one step only, the deliberate one-step orientation lag in the
    observation path -- at exactly the instant under study."""
    g1_objects.reset()
    before = g1_objects.data.xpos[g1_objects.object_body_ids[0]].copy()
    g1_objects.place_object(0, (5.0, 5.0, 0.12))
    np.testing.assert_array_equal(g1_objects.data.xpos[g1_objects.object_body_ids[0]], before)
    g1_objects.reset()


def test_every_shape_compiles_and_rests_on_the_floor(g1_cfg):
    objects = (
        ObjectSpec(name="b", shape="box", size=(0.1, 0.1, 0.1), pos=(2.0, 0.0, 0.1)),
        ObjectSpec(name="s", shape="sphere", size=(0.2,), pos=(2.0, 1.0, 0.2)),
        ObjectSpec(name="c", shape="cylinder", size=(0.15, 0.3), pos=(2.0, -1.0, 0.3)),
    )
    sim = RobotModel(resolve_mjcf(g1_cfg), g1_cfg, SimOverrides(), objects=objects)
    sim.reset()
    for index, obj in enumerate(objects):
        assert sim.object_pose_w(index)[0][2] == pytest.approx(obj.half_height)

    # Settle: resting on the floor means not falling through it.
    for _ in range(200):
        mujoco.mj_step(sim.model, sim.data)
    for index, obj in enumerate(objects):
        assert sim.object_pose_w(index)[0][2] == pytest.approx(obj.half_height, abs=2e-3)


def test_an_object_named_after_a_robot_body_is_a_clear_failure(g1_cfg):
    with pytest.raises(ModelError, match="pelvis"):
        RobotModel(
            resolve_mjcf(g1_cfg),
            g1_cfg,
            SimOverrides(),
            objects=(ObjectSpec(name="pelvis", pos=(2.0, 0.0, 0.12)),),
        )
