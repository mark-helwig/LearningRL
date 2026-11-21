"""Walk-v1 reward terms, ported verbatim from HCRLlab (`mh/quiet_walking` cbfa056, tasks/locomotion/g1/mdp/rewards.py).

The contact-timer gait clock and its companions behind HCRL-RPO-Walk-Flat-v0, the walk this task now builds on.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


def _is_moving(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    return torch.norm(env.command_manager.get_command(command_name), dim=1) > threshold


def _step_double_stance(sensor: ContactSensor, body_ids, double_stance_time: float, min_swing: float) -> torch.Tensor:
    """Both feet down, within ``double_stance_time`` of a touchdown that ended a swing of at least ``min_swing``."""
    contact = sensor.data.current_contact_time.torch[:, body_ids]
    newest = contact.argmin(dim=1, keepdim=True)
    landed_swing = sensor.data.last_air_time.torch[:, body_ids].gather(1, newest).squeeze(1)
    return (contact.min(dim=1).values <= double_stance_time) & (landed_swing >= min_swing)


def gait_contact_clock(
    env: ManagerBasedRLEnv,
    swing_time: float,
    double_stance_time: float,
    min_swing: float,
    stance_time: float,
    timing_std: float,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Mean of two parts, each in [0, 1]. Alternation: single support pays 1 per step once the stance foot has been
    down ``double_stance_time`` (0 for an early liftoff), decaying past ``swing_time`` of air; double support pays 1
    within ``double_stance_time`` of a touchdown ending a ``min_swing`` swing; flight 0.
    Timing: exp kernels
    on each foot's last completed swing and stance against ``swing_time`` and ``stance_time``. Standing pays both
    feet down."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    data = sensor.data
    air = data.current_air_time.torch[:, sensor_cfg.body_ids]
    contact = data.current_contact_time.torch[:, sensor_cfg.body_ids]
    in_contact = contact > 0
    n_down = in_contact.sum(dim=1)
    swing_air = (air * ~in_contact).sum(dim=1)
    overrun = (swing_air - swing_time).clamp(min=0.0) / (0.25 * swing_time)
    stance_down = (contact * in_contact).sum(dim=1)
    single = torch.exp(-torch.square(overrun)) * (stance_down >= double_stance_time)
    double = _step_double_stance(sensor, sensor_cfg.body_ids, double_stance_time, min_swing).float()
    alternation = torch.where(n_down == 1, single, torch.where(n_down == 2, double, torch.zeros_like(single)))
    last_swing = data.last_air_time.torch[:, sensor_cfg.body_ids]
    last_stance = data.last_contact_time.torch[:, sensor_cfg.body_ids]
    timing = 0.5 * (
        torch.exp(-torch.square(last_swing - swing_time) / timing_std**2)
        + torch.exp(-torch.square(last_stance - stance_time) / timing_std**2)
    )
    moving = 0.5 * (alternation + timing.mean(dim=1))
    return torch.where(_is_moving(env, command_name, command_threshold), moving, (n_down == 2).float())


def gait_clearance(
    env: ManagerBasedRLEnv,
    swing_time: float,
    double_stance_time: float,
    min_swing: float,
    height: float,
    std: float,
    foot_offset: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    command_threshold: float = 0.1,
    stance_relative: bool = False,
) -> torch.Tensor:
    """Exp kernel on foot height minus ``height * sin(pi * min(air / swing_time, 1))``, averaged over airborne feet.
    Double support pays 1 in the same window as ``gait_contact_clock``, else 0. Standing: both feet, 0 m target.
    ``foot_offset`` is the flat-foot body height. ``stance_relative`` measures height from the lowest grounded foot
    (rough terrain) and pays 1 when standing."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    air = sensor.data.current_air_time.torch[:, sensor_cfg.body_ids]
    progress = (air / swing_time).clamp(max=1.0)
    moving = _is_moving(env, command_name, command_threshold)
    target = height * torch.sin(math.pi * progress) * moving.unsqueeze(1)
    contact = sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids]
    foot_z = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 2]
    if stance_relative:
        grounded_z = torch.where(contact > 0, foot_z, torch.full_like(foot_z, torch.inf)).min(dim=1).values
        foot_z = foot_z - torch.where(torch.isinf(grounded_z), foot_z.min(dim=1).values, grounded_z).unsqueeze(1)
    else:
        foot_z = foot_z - foot_offset
    kernel = torch.exp(-torch.square(foot_z - target) / std**2)
    airborne = (contact == 0).float()
    swing = (kernel * airborne).sum(dim=1) / airborne.sum(dim=1).clamp(min=1.0)
    double = _step_double_stance(sensor, sensor_cfg.body_ids, double_stance_time, min_swing).float()
    swing = torch.where(airborne.sum(dim=1) == 0, double, swing)
    standing = torch.ones_like(swing) if stance_relative else kernel.mean(dim=1)
    return torch.where(moving, swing, standing)


def body_orientation_exp(env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Exp kernel on the tilt of one body: the xy part of gravity in its frame, ``sin^2`` of the tilt angle."""
    asset: Articulation = env.scene[asset_cfg.name]
    quat = asset.data.body_quat_w.torch[:, asset_cfg.body_ids[0]]
    gravity_b = quat_apply_inverse(quat, asset.data.GRAVITY_VEC_W.torch)
    return torch.exp(-torch.sum(torch.square(gravity_b[:, :2]), dim=1) / std**2)


class foot_impact_force(ManagerTermBase):
    """Rise in each foot's normal force since the last policy step while in contact, in body weights, summed over
    feet. Unloading is free. The first ``settle_steps`` steps pay 0, since the spawn drop is not the policy's."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._prev_fz: torch.Tensor | None = None

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
        contact_threshold: float = 1.0,
        settle_steps: int = 10,
    ) -> torch.Tensor:
        sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        asset: Articulation = env.scene[asset_cfg.name]
        fz = sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids, 2]
        if self._prev_fz is None:
            self._prev_fz = fz.clone()
        body_weight = asset.data.body_mass.torch.sum(dim=-1) * -env.sim.cfg.gravity[2]
        rise = torch.clamp(fz - self._prev_fz, min=0.0) * (fz > contact_threshold)
        self._prev_fz = fz.clone()
        return rise.sum(dim=1) / body_weight * (env.episode_length_buf >= settle_steps)


def foot_slip_exp(
    env: ManagerBasedRLEnv, std: float, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Saturating penalty ``1 - exp(-|v_xy|^2 / std^2)`` per foot in contact, summed over feet."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    in_contact = sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids] > 0
    speed_sq = torch.sum(torch.square(asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2]), dim=-1)
    return torch.sum((1.0 - torch.exp(-speed_sq / std**2)) * in_contact, dim=1)


def self_collision_fraction(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Fraction of sensed bodies touching another body of the robot. Needs an ``exclude_self=True`` sensor."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    self_force = sensor.data.force_matrix_w.torch[:, sensor_cfg.body_ids].sum(dim=2)
    return (torch.linalg.norm(self_force, dim=-1) > threshold).float().mean(dim=1)


def joint_pos_limits_normalized(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Distance past the soft position limits, each joint divided by its soft range, summed over joints."""
    asset: Articulation = env.scene[asset_cfg.name]
    limits = asset.data.soft_joint_pos_limits.torch[:, asset_cfg.joint_ids]
    joint_pos = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
    over = (limits[..., 0] - joint_pos).clamp(min=0.0) + (joint_pos - limits[..., 1]).clamp(min=0.0)
    return torch.sum(over / (limits[..., 1] - limits[..., 0]), dim=1)
