"""Motion command: replays a reference clip and tracks the robot against it.

Adapted from RoboParty's ``robolab`` (BSD-3-Clause, see ``NOTICE.md``). Changes: the clip's body and
joint order are declared in the cfg and reordered to the robot's by name, quaternions are converted
from the clip's wxyz to Isaac Lab's xyzw (F8), and a clip that ends is always resampled.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import (
    convert_quat,
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

from .commands_cfg import MotionCommandCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionLoader:
    """A reference clip on ``device``, with joints and bodies in the robot's order."""

    def __init__(
        self,
        motion_file: str,
        motion_joint_names: Sequence[str],
        motion_body_names: Sequence[str],
        robot_joint_names: Sequence[str],
        robot_body_names: Sequence[str],
        device: str = "cpu",
    ):
        data = np.load(motion_file)
        self.fps = int(data["fps"].item())

        joint_ids = [list(motion_joint_names).index(name) for name in robot_joint_names]
        body_ids = [list(motion_body_names).index(name) for name in robot_body_names]

        def load(key: str, ids: list[int]) -> torch.Tensor:
            return torch.tensor(data[key], dtype=torch.float32, device=device)[:, ids]

        self.joint_pos = load("joint_pos", joint_ids)
        self.joint_vel = load("joint_vel", joint_ids)
        # all bodies, in the robot's order; MotionCommand slices out the tracked ones
        self.body_pos_w = load("body_pos_w", body_ids)
        self.body_quat_w = convert_quat(load("body_quat_w", body_ids), to="xyzw")
        self.body_lin_vel_w = load("body_lin_vel_w", body_ids)
        self.body_ang_vel_w = load("body_ang_vel_w", body_ids)
        self.time_step_total = self.joint_pos.shape[0]


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        self.motion = MotionLoader(
            self.cfg.motion_file,
            self.cfg.motion_joint_names,
            self.cfg.motion_body_names,
            self.robot.joint_names,
            self.robot.body_names,
            device=self.device,
        )
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 3] = 1.0

        # one bin per second of clip, for the failure-weighted start-frame sampling
        self.bin_count = int(self.motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        for name in (
            "error_anchor_pos",
            "error_anchor_rot",
            "error_anchor_lin_vel",
            "error_anchor_ang_vel",
            "error_body_pos",
            "error_body_rot",
            "error_body_lin_vel",
            "error_body_ang_vel",
            "error_joint_pos",
            "error_joint_vel",
            "sampling_entropy",
            "sampling_top1_prob",
            "sampling_top1_bin",
        ):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps][:, self.body_indexes] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps][:, self.body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps][:, self.body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps][:, self.body_indexes]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps, self.robot_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, self.robot_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps, self.robot_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps, self.robot_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos.torch

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel.torch

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w.torch[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w.torch[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w.torch[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w.torch[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w.torch[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w.torch[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w.torch[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w.torch[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1), 0, self.bin_count - 1
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        # bins where episodes failed recently are sampled more often; a uniform floor keeps the rest alive
        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),  # non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)

        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)

        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.motion.time_step_total - 1)
        ).long()

        entropy = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = entropy / math.log(self.bin_count)
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def _resample_command(self, env_ids: Sequence[int] | slice):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        if len(env_ids) == 0:
            return
        self._adaptive_sampling(env_ids)

        # the robot's root is the articulation's first body
        root_pos = self.motion.body_pos_w[self.time_steps, 0] + self._env.scene.env_origins
        root_ori = self.motion.body_quat_w[self.time_steps, 0].clone()
        root_lin_vel = self.motion.body_lin_vel_w[self.time_steps, 0].clone()
        root_ang_vel = self.motion.body_ang_vel_w[self.time_steps, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits.torch[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_position_to_sim_index(position=joint_pos[env_ids], env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_pose_to_sim_index(
            root_pose=torch.cat([root_pos[env_ids], root_ori[env_ids]], dim=-1), env_ids=env_ids
        )
        self.robot.write_root_velocity_to_sim_index(
            root_velocity=torch.cat([root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1), env_ids=env_ids
        )
        # Pose writes invalidate the backend's FK cache. Reading the anchor here
        # refreshes kinematics without stepping physics or advancing the motion.
        self.refresh_body_reference(env_ids)

    def refresh_body_reference(self, env_ids: Sequence[int] | None = None):
        """Align the reference bodies to the robot's anchor (yaw and xy only) for the current frame."""
        if env_ids is not None and len(env_ids) == 0:
            return
        selection = slice(None) if env_ids is None else env_ids
        num_bodies = len(self.cfg.body_names)
        anchor_pos = self.anchor_pos_w[selection, None, :].expand(-1, num_bodies, -1)
        anchor_quat = self.anchor_quat_w[selection, None, :].expand(-1, num_bodies, -1)
        robot_anchor_quat = self.robot_anchor_quat_w[selection, None, :].expand(-1, num_bodies, -1)
        aligned_pos = self.robot_anchor_pos_w[selection, None, :].expand(-1, num_bodies, -1).clone()
        aligned_pos[..., 2] = anchor_pos[..., 2]
        delta_ori = yaw_quat(quat_mul(robot_anchor_quat, quat_inv(anchor_quat)))
        self.body_quat_relative_w[selection] = quat_mul(delta_ori, self.body_quat_w[selection])
        self.body_pos_relative_w[selection] = aligned_pos + quat_apply(
            delta_ori, self.body_pos_w[selection] - anchor_pos
        )

    def _update_command(self):
        self.time_steps += 1
        env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
        self._resample_command(env_ids)

        self.refresh_body_reference()

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis and not hasattr(self, "current_anchor_visualizer"):
            self.current_anchor_visualizer = VisualizationMarkers(
                self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
            )
            self.goal_anchor_visualizer = VisualizationMarkers(
                self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
            )
            self.current_body_visualizers = []
            self.goal_body_visualizers = []
            for name in self.cfg.body_names:
                self.current_body_visualizers.append(
                    VisualizationMarkers(
                        self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                    )
                )
                self.goal_body_visualizers.append(
                    VisualizationMarkers(
                        self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                    )
                )

        if hasattr(self, "current_anchor_visualizer"):
            self.current_anchor_visualizer.set_visibility(debug_vis)
            self.goal_anchor_visualizer.set_visibility(debug_vis)
            for current, goal in zip(self.current_body_visualizers, self.goal_body_visualizers):
                current.set_visibility(debug_vis)
                goal.set_visibility(debug_vis)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])
