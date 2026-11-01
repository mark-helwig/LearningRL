"""The motion command: a reference clip as a command term, and starting from its first frame.

The first half is static -- synthetic clips, no MuJoCo. The second half is NOT: it loads the real RPO
model and LearningRLlab's shipped clip, which is the only independent check that the clip's stored body
and joint order, and its wxyz quaternions, mean what the exporter says they mean.
"""

from __future__ import annotations

import copy
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from LearningRLsim import runconfig
from LearningRLsim.commands.sources import CommandManager
from LearningRLsim.runconfig import RunConfigError
from tests.conftest import JOINTS, MINIMAL, MINIMAL_RPO, write_run

FRAMES = 5
CLIP_JOINTS = [JOINTS[2], JOINTS[0], JOINTS[1]]  # NOT joint_order: the term has to reorder
CLIP_BODIES = ["pelvis", "torso_link"]
REPO_CLIP = Path(__file__).resolve().parents[2] / "LearningRLlab" / "data" / "motions" / "rpo" / "yundong1.npz"


def write_clip(path: Path, joints: int = 3, bodies: int = 2) -> None:
    frame = np.arange(FRAMES, dtype=float)[:, None]
    np.savez(
        path,
        fps=np.array([50]),
        joint_pos=frame + np.arange(joints) / 10.0,
        joint_vel=-frame - np.arange(joints) / 10.0,
        body_pos_w=np.tile(frame[:, :, None] * [1.0, 0.1, 0.0] + [0.0, 0.0, 0.8], (1, bodies, 1)),
        body_quat_w=np.tile([1.0, 0.0, 0.0, 0.0], (FRAMES, bodies, 1)),
        body_lin_vel_w=np.tile([1.0, 0.0, 0.0], (FRAMES, bodies, 1)),
        body_ang_vel_w=np.tile([0.0, 0.0, 2.0], (FRAMES, bodies, 1)),
    )


def motion_config(**overrides) -> dict:
    """MINIMAL, with its velocity command replaced by a 6-wide motion command."""
    config = copy.deepcopy(MINIMAL)
    config["observations"]["terms"][1].update(
        {"name": "motion", "dim": 6, "slice": [3, 9], "params": {"command_name": "motion"}}
    )
    config["observations"]["terms"][2].update({"slice": [9, 12]})
    config["observations"]["terms"][3].update({"slice": [12, 15]})
    config["observations"]["total_dim"] = 15
    config["commands"] = [
        {
            "name": "motion",
            "type": "MotionCommandCfg",
            "obs_slice": [3, 9],
            "resampling_time_range": [1.0e9, 1.0e9],
            "motion_file": "motion.npz",
            "motion_joint_names": list(CLIP_JOINTS),
            "motion_body_names": list(CLIP_BODIES),
            "anchor_body_name": "torso_link",
        }
    ]
    config["commands"][0].update(overrides)
    return config


@pytest.fixture
def motion_run(tmp_path: Path) -> Path:
    run_dir = write_run(tmp_path, motion_config(), name="motion")
    write_clip(run_dir / "motion.npz")
    return run_dir


def ctx_at(time: float) -> SimpleNamespace:
    return SimpleNamespace(sim=SimpleNamespace(time=time))


@pytest.mark.static
def test_motion_cfg_resolves_the_clip_beside_config_yaml(motion_run):
    cfg = runconfig.load(motion_run)
    motion = cfg.command("motion").motion
    assert motion.file == motion_run / "motion.npz"
    assert motion.anchor_body_name == "torso_link"


@pytest.mark.static
def test_a_missing_clip_is_a_config_error(tmp_path):
    run_dir = write_run(tmp_path, motion_config(), name="no_clip")
    with pytest.raises(RunConfigError, match="motion_file"):
        runconfig.load(run_dir)


@pytest.mark.static
def test_a_motion_command_without_a_clip_is_refused(tmp_path):
    config = motion_config()
    del config["commands"][0]["motion_file"]
    run_dir = write_run(tmp_path, config, name="old_export")
    with pytest.raises(RunConfigError, match="no motion_file"):
        CommandManager(runconfig.load(run_dir))


@pytest.mark.static
def test_value_is_the_clips_joints_in_joint_order(motion_run):
    manager = CommandManager(runconfig.load(motion_run))
    manager.update(0.0, ctx_at(0.0))
    pos, vel = manager.value("motion")[:3], manager.value("motion")[3:]
    # the clip stores [joints[2], joints[0], joints[1]] as columns 0, 1, 2
    np.testing.assert_allclose(pos, [0.1, 0.2, 0.0])
    np.testing.assert_allclose(vel, [-0.1, -0.2, -0.0])


@pytest.mark.static
def test_one_frame_per_policy_step_and_the_last_frame_holds(motion_run):
    manager = CommandManager(runconfig.load(motion_run))
    term = manager.terms["motion"]
    for sim_time, frame in ((0.0, 0), (0.02, 1), (0.04, 2), (0.08, 4), (5.0, 4)):
        manager.update(sim_time, ctx_at(sim_time))
        assert term.frame == frame, f"at t={sim_time}"


@pytest.mark.static
def test_update_is_idempotent_at_a_given_time(motion_run):
    """The rollout updates once at t=0 before the first observation; that must not advance the clip."""
    manager = CommandManager(runconfig.load(motion_run))
    for _ in range(3):
        manager.update(0.0, ctx_at(0.0))
    assert manager.terms["motion"].frame == 0


@pytest.mark.static
def test_duration_is_the_clip_length_at_the_policy_rate(motion_run):
    term = CommandManager(runconfig.load(motion_run)).terms["motion"]
    assert term.duration == pytest.approx((FRAMES - 1) / 50.0)


@pytest.mark.static
def test_initial_state_is_the_first_frame_in_joint_order(motion_run):
    state = CommandManager(runconfig.load(motion_run)).initial_state()
    np.testing.assert_allclose(state.joint_pos, [0.1, 0.2, 0.0])
    np.testing.assert_allclose(state.base_pos_w, [0.0, 0.0, 0.8])
    np.testing.assert_allclose(state.base_quat_w, [1.0, 0.0, 0.0, 0.0])


@pytest.mark.static
def test_a_robot_joint_the_clip_lacks_is_an_error(tmp_path):
    run_dir = write_run(tmp_path, motion_config(motion_joint_names=JOINTS[:2] + ["not_a_joint"]), name="bad_joints")
    write_clip(run_dir / "motion.npz")
    with pytest.raises(RunConfigError, match="no column"):
        CommandManager(runconfig.load(run_dir))


@pytest.mark.static
def test_a_velocity_run_has_no_initial_state(minimal_run):
    assert CommandManager(runconfig.load(minimal_run)).initial_state() is None


# ---------------------------------------------------------------------------
# Against the real RPO model and the shipped clip. NOT static.
# ---------------------------------------------------------------------------
mujoco = pytest.importorskip("mujoco")
pytest.importorskip("robot_description")

from LearningRLsim.model import RobotModel, SimOverrides, resolve_mjcf  # noqa: E402


@pytest.fixture(scope="module")
def rpo_clip_run(tmp_path_factory):
    if not REPO_CLIP.is_file():
        pytest.skip(f"{REPO_CLIP} is not in this checkout")
    clip = np.load(REPO_CLIP)
    body_names = list(MOTION_BODY_NAMES)
    joints = [name.replace("_link", "_joint") for name in body_names[1:]]
    robot_joints = joints[::-1]  # reversed: the term must reorder by name, not trust the clip's order

    config = copy.deepcopy(MINIMAL_RPO)
    n = len(joints)
    config["default_joint_pos"] = {"joint_order": list(robot_joints), "values": [0.0] * n}
    config["actuators"] = {
        "joint_order": list(robot_joints),
        "stiffness": [20.0] * n,
        "damping": [1.0] * n,
        "armature": [0.01] * n,
        "friction": [0.0] * n,
        "effort_limit": [100.0] * n,
        "velocity_limit": [20.0] * n,
    }
    config["actions"] = {
        "total_dim": n,
        "terms": [
            {
                "name": "JointPositionAction",
                "type": "JointPositionAction",
                "dim": n,
                "slice": [0, n],
                "joint_names": list(robot_joints),
                "scale": 0.25,
                "offset": [0.0] * n,
            }
        ],
    }
    config["observations"] = {
        "total_dim": 2 * n,
        "terms": [
            {
                "name": "motion",
                "func": "generated_commands",
                "dim": 2 * n,
                "slice": [0, 2 * n],
                "scale": 1.0,
                "clip": None,
                "params": {"command_name": "motion"},
            }
        ],
    }
    config["commands"] = [
        {
            "name": "motion",
            "type": "MotionCommandCfg",
            "obs_slice": [0, 2 * n],
            "motion_file": "motion.npz",
            "motion_joint_names": joints,
            "motion_body_names": body_names,
            "anchor_body_name": "torso_link",
        }
    ]
    run_dir = write_run(tmp_path_factory.mktemp("rpo_motion"), config, name="rpo_motion")
    shutil.copy2(REPO_CLIP, run_dir / "motion.npz")
    return run_dir, clip


# The order `yundong1.npz` stores its bodies in; LearningRLlab's rpo_beyondmimic_env_cfg.py declares the same list.
MOTION_BODY_NAMES = [
    "base_link", "left_thigh_yaw_link", "right_thigh_yaw_link", "torso_link",
    "left_thigh_roll_link", "right_thigh_roll_link", "left_arm_pitch_link", "right_arm_pitch_link",
    "left_thigh_pitch_link", "right_thigh_pitch_link", "left_arm_roll_link", "right_arm_roll_link",
    "left_knee_link", "right_knee_link", "left_arm_yaw_link", "right_arm_yaw_link",
    "left_ankle_pitch_link", "right_ankle_pitch_link", "left_elbow_pitch_link", "right_elbow_pitch_link",
    "left_ankle_roll_link", "right_ankle_roll_link", "left_elbow_yaw_link", "right_elbow_yaw_link",
]  # fmt: skip


def test_the_first_frame_puts_every_body_where_the_clip_says(rpo_clip_run):
    """MuJoCo forward kinematics from the clip's joints and root, against the clip's own body poses.

    The clip carries no names, so this is what proves its body order, its joint order and its wxyz
    quaternions are what the exporter declares: a wrong one puts bodies centimetres off.
    """
    run_dir, clip = rpo_clip_run
    cfg = runconfig.load(run_dir)
    sim = RobotModel(resolve_mjcf(cfg), cfg, SimOverrides())
    manager = CommandManager(cfg)
    sim.reset()
    sim.set_state(manager.initial_state())

    for name, ref_pos, ref_quat in zip(MOTION_BODY_NAMES, clip["body_pos_w"][0], clip["body_quat_w"][0], strict=True):
        body = sim.body_names.index(name)
        np.testing.assert_allclose(sim.body_pos_w[body], ref_pos, atol=2e-3, err_msg=name)
        assert abs(float(np.dot(sim.body_quat_w[body], ref_quat))) == pytest.approx(1.0, abs=1e-3), name


def test_tracking_error_is_zero_on_the_first_frame(rpo_clip_run):
    run_dir, _ = rpo_clip_run
    cfg = runconfig.load(run_dir)
    sim = RobotModel(resolve_mjcf(cfg), cfg, SimOverrides())
    manager = CommandManager(cfg)
    sim.reset()
    sim.set_state(manager.initial_state())

    metrics = manager.terms["motion"].tracking(SimpleNamespace(sim=sim))
    assert metrics["joint_rmse"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["anchor_pos_error"] < 2e-3
    assert metrics["anchor_ori_error"] < 2e-3


def test_initial_angular_velocity_is_rotated_into_the_body_frame(rpo_clip_run):
    run_dir, _ = rpo_clip_run
    cfg = runconfig.load(run_dir)
    sim = RobotModel(resolve_mjcf(cfg), cfg, SimOverrides())
    state = CommandManager(cfg).initial_state()
    state = type(state)(**{**state.__dict__, "base_ang_vel_w": np.array([0.3, -0.2, 0.5])})
    sim.reset()
    sim.set_state(state)
    np.testing.assert_allclose(sim.base_ang_vel_b, sim.base_rot_w.T @ [0.3, -0.2, 0.5], atol=1e-9)
