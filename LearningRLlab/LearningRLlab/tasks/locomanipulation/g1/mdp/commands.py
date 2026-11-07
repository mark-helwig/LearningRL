"""The two-hand position command for LearningRL-G1-LocoManipulation-v0.

Loaded only through ``HandPosCommandCfg.class_type``, after Kit starts: this module imports
``CommandTerm`` and ``VisualizationMarkers``, which reach ``pxr`` (F9).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands_cfg import HandPosCommandCfg

_MODES = {"left": 0, "right": 1, "both": 2, "symmetric": 3}


class HandPosCommand(CommandTerm):
    """Position targets for the left and right hands, held in the robot's yaw-aligned root frame.

    The command is ``(num_envs, 8)``: ``[0:2]`` is the active mask for ``[left, right]`` (0/1),
    ``[2:5]`` the left target and ``[5:8]`` the right target. An inactive hand keeps its sampled
    target; only an idle env (``rel_standing_envs``) zeroes the whole row.

    Targets are absolute positions relative to the root, so they follow the walking body. The
    reward and the markers move them one env step along the commanded base velocity
    (:meth:`target_pos_b`), so a hand is asked to keep pace with the body instead of trailing it.
    """

    cfg: HandPosCommandCfg

    mirror_map = ([1, 0, 5, 6, 7, 2, 3, 4], [1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0])
    """Left/right mirror of the command for ``LearningRL_mdp.symmetry``: swap the hands, negate y."""

    def __init__(self, cfg: HandPosCommandCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_ids, body_names = self.robot.find_bodies(cfg.body_names, preserve_order=True)
        # the command layout, the per-hand y ranges and `mirror_map` all assume [left, right]
        if len(body_names) != 2 or not (body_names[0].startswith("left_") and body_names[1].startswith("right_")):
            raise ValueError(f"HandPosCommand needs one left_ then one right_ body; {cfg.body_names} -> {body_names}")

        self._command = torch.zeros(self.num_envs, 8, device=self.device)
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "HandPosCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tBodies: {self.cfg.body_names}\n"
        msg += f"\tMode: {self.cfg.mode}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """Active mask and both targets. Shape is (num_envs, 8)."""
        return self._command

    @property
    def active_hand_mask(self) -> torch.Tensor:
        """Whether each of ``[left, right]`` has a target. Shape is (num_envs, 2), bool."""
        return self._command[:, 0:2] > 0.5

    @property
    def pos_b(self) -> torch.Tensor:
        """Targets for ``[left, right]`` in the yaw-aligned root frame. Shape is (num_envs, 2, 3)."""
        return self._command[:, 2:8].view(self.num_envs, 2, 3)

    """
    Frames
    """

    def target_pos_b(self) -> torch.Tensor:
        """``pos_b`` moved one env step along the commanded base velocity. Shape is (num_envs, 2, 3)."""
        base_vel = self._env.command_manager.get_command(self.cfg.velocity_command_name)
        target = self.pos_b.clone()
        target[:, :, :2] += base_vel[:, None, :2] * self._env.step_dt
        return target

    def hand_pos_b(self) -> torch.Tensor:
        """Current hand body origins in the yaw-aligned root frame. Shape is (num_envs, 2, 3)."""
        root_pos, yaw = self._yaw_frame()
        hand_pos_w = self.robot.data.body_pos_w.torch[:, self.body_ids]
        return math_utils.quat_apply_inverse(yaw, hand_pos_w - root_pos)

    def _to_world(self, pos_b: torch.Tensor) -> torch.Tensor:
        root_pos, yaw = self._yaw_frame()
        return root_pos + math_utils.quat_apply(yaw, pos_b)

    def _yaw_frame(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Root position (num_envs, 1, 3) and yaw quaternion repeated per hand (num_envs, 2, 4)."""
        root_pos = self.robot.data.root_pos_w.torch.unsqueeze(1)
        yaw = math_utils.yaw_quat(self.robot.data.root_quat_w.torch)
        return root_pos, yaw.unsqueeze(1).expand(-1, 2, -1)

    """
    Implementation specific functions
    """

    def _update_metrics(self) -> None:
        # mean error over the active hands; 0 for an idle env
        active = self.active_hand_mask.float()
        error = torch.linalg.norm(self.hand_pos_b() - self.target_pos_b(), dim=-1)
        self.metrics["position_error"] = (error * active).sum(dim=1) / active.sum(dim=1).clamp(min=1.0)

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        n = len(env_ids)
        ranges = self.cfg.ranges
        r = torch.empty(n, device=self.device)
        cmd = torch.zeros(n, 8, device=self.device)

        if self.cfg.mode == "random":
            mode = torch.randint(0, 4, (n,), device=self.device)
        else:
            mode = torch.full((n,), _MODES[self.cfg.mode], device=self.device)
        cmd[:, 0] = (mode != _MODES["right"]).float()
        cmd[:, 1] = (mode != _MODES["left"]).float()

        pos = cmd[:, 2:8].view(n, 2, 3)
        # one x and one z per env, shared by both hands; y per hand
        pos[:, :, 0] = r.uniform_(*ranges.pos_x).unsqueeze(1)
        pos[:, :, 2] = r.uniform_(*ranges.pos_z).unsqueeze(1)
        pos[:, 0, 1] = r.uniform_(*(ranges.pos_y_left or ranges.pos_y))
        pos[:, 1, 1] = r.uniform_(*(ranges.pos_y_right or ranges.pos_y))
        # symmetric: the right target is the left one mirrored across the sagittal plane
        symmetric = mode == _MODES["symmetric"]
        pos[symmetric, 1] = pos[symmetric, 0] * torch.tensor([1.0, -1.0, 1.0], device=self.device)

        idle = r.uniform_(0.0, 1.0) < self.cfg.rel_standing_envs
        cmd[idle] = 0.0
        self._command[env_ids] = cmd

    def _update_command(self) -> None:
        pass

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer = VisualizationMarkers(self.cfg.goal_pos_visualizer_cfg)
                self.current_pos_visualizer = VisualizationMarkers(self.cfg.current_pos_visualizer_cfg)
            self.goal_pos_visualizer.set_visibility(True)
            self.current_pos_visualizer.set_visibility(True)
        elif hasattr(self, "goal_pos_visualizer"):
            self.goal_pos_visualizer.set_visibility(False)
            self.current_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event) -> None:
        if not self.robot.is_initialized:
            return
        # marker 1 is the hidden sphere, so an inactive hand shows nothing
        marker_indices = (~self.active_hand_mask).int().view(-1)
        self.goal_pos_visualizer.visualize(
            self._to_world(self.target_pos_b()).view(-1, 3), marker_indices=marker_indices
        )
        self.current_pos_visualizer.visualize(
            self.robot.data.body_pos_w.torch[:, self.body_ids].reshape(-1, 3), marker_indices=marker_indices
        )
