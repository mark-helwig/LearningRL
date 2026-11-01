"""`MotionCommandCfg`: the reference clip a mimic policy is asked to track.

Reproduces LearningRLlab's `mdp/commands.py` `MotionCommand` against MuJoCo state. What the policy
sees is `[joint_pos, joint_vel]` of the clip's CURRENT frame, in `joint_order`; the clip advances
ONE frame per policy step, whatever its stored `fps` -- that is how the training env plays it, so a
clip whose fps differs from the policy rate was time-warped in training and is time-warped here.

Two things differ from the training command on purpose:

- **No resampling.** Training starts each episode at a failure-weighted random frame and
  restarts at the clip's end. A replay starts at frame 0 and ends at the last frame; it does not
  hold, loop or fall back to a nominal pose.
- **No noise.** Training perturbs the starting pose and joint positions; the replay starts exactly
  on the clip's first frame.

The frame is a pure function of sim time, so `update` is idempotent: the rollout calls it once at
t=0 before the first observation, which must NOT advance the clip, and a scripted `reset` event
winds sim time back to 0 and the clip with it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from LearningRLsim.commands.terms import CommandTerm, InitialState
from LearningRLsim.registry import command
from LearningRLsim.runconfig import CommandCfg, RunConfig, RunConfigError

if TYPE_CHECKING:
    from LearningRLsim.context import EvalContext

_REQUIRED = ("fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


def quat_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Smallest rotation between two unit quaternions of the same order, in radians."""
    return float(2.0 * np.arccos(np.clip(abs(float(np.dot(a, b))), 0.0, 1.0)))


@command("MotionCommandCfg")
class MotionCommand(CommandTerm):
    """`[joint_pos_ref, joint_vel_ref]` for the clip's current frame."""

    def __init__(self, cfg: CommandCfg, run: RunConfig) -> None:
        super().__init__(cfg, run)
        if cfg.motion is None:
            raise RunConfigError(
                f"command '{cfg.name}' is a {cfg.type} but config.yaml gives it no motion_file. "
                f"LearningRLlab's exporter writes one; an older run directory has to be re-exported."
            )
        self.motion = cfg.motion
        data = np.load(self.motion.file)
        missing = [key for key in _REQUIRED if key not in data]
        if missing:
            raise RunConfigError(f"{self.motion.file}: missing array(s) {missing}")

        self.fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        missing_joints = sorted(set(run.joint_order) - set(self.motion.joint_names))
        if missing_joints:
            raise RunConfigError(f"{self.motion.file}: no column for robot joint(s) {missing_joints}")
        order = [self.motion.joint_names.index(name) for name in run.joint_order]
        self.joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)[:, order]
        self.joint_vel = np.asarray(data["joint_vel"], dtype=np.float64)[:, order]
        self.body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float64)
        self.body_quat_w = np.asarray(data["body_quat_w"], dtype=np.float64)  # (w, x, y, z), as MuJoCo
        self.body_lin_vel_w = np.asarray(data["body_lin_vel_w"], dtype=np.float64)
        self.body_ang_vel_w = np.asarray(data["body_ang_vel_w"], dtype=np.float64)

        self.num_frames = self.joint_pos.shape[0]
        # the clip's first body is the robot's root, which is the floating base
        self.root_index = 0
        if self.motion.anchor_body_name not in self.motion.body_names:
            raise RunConfigError(f"{cfg.name}: anchor body {self.motion.anchor_body_name!r} is not a clip body")
        self.anchor_index = self.motion.body_names.index(self.motion.anchor_body_name)
        self._frame = 0

    @property
    def dim(self) -> int:
        return 2 * self.joint_pos.shape[1]

    @property
    def frame(self) -> int:
        return self._frame

    @property
    def duration(self) -> float:
        """Sim seconds from the first frame to the last, at the policy rate."""
        return (self.num_frames - 1) / self.run.policy_frequency_hz

    def value(self) -> np.ndarray:
        return np.concatenate([self.joint_pos[self._frame], self.joint_vel[self._frame]])

    def update(self, ctx: EvalContext) -> None:
        step = int(round(ctx.sim.time * self.run.policy_frequency_hz))
        self._frame = min(step, self.num_frames - 1)

    def reset(self) -> None:
        super().reset()
        self._frame = 0

    def initial_state(self) -> InitialState:
        return InitialState(
            joint_pos=self.joint_pos[0],
            joint_vel=self.joint_vel[0],
            base_pos_w=self.body_pos_w[0, self.root_index],
            base_quat_w=self.body_quat_w[0, self.root_index],
            base_lin_vel_w=self.body_lin_vel_w[0, self.root_index],
            base_ang_vel_w=self.body_ang_vel_w[0, self.root_index],
        )

    def tracking(self, ctx: EvalContext) -> dict[str, float]:
        """How far the robot is from the clip's current frame.

        The anchor comparison is in the world frame, which is meaningful because the
        rollout starts at the clip's own first-frame root pose. Training's termination
        compares only the anchor's HEIGHT and tilt (`bad_anchor_pos_z_only`, `bad_anchor_ori`);
        the full position error is reported too, since drift along the ground is how a
        tracking policy usually degrades.
        """
        sim = ctx.sim
        sim.refresh_link_poses()
        body = sim.body_names.index(self.motion.anchor_body_name)
        pos = sim.body_pos_w[body]
        quat = sim.body_quat_w[body]
        ref_pos = self.body_pos_w[self._frame, self.anchor_index]
        ref_quat = self.body_quat_w[self._frame, self.anchor_index]
        return {
            "frame": float(self._frame),
            "joint_rmse": float(np.sqrt(np.mean((sim.joint_pos - self.joint_pos[self._frame]) ** 2))),
            "anchor_pos_error": float(np.linalg.norm(pos - ref_pos)),
            "anchor_height_error": float(abs(pos[2] - ref_pos[2])),
            "anchor_ori_error": quat_angle(quat, ref_quat),
        }
