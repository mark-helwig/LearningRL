"""Walk-v1 critic observations and termination, ported from HCRLlab (`mh/quiet_walking` cbfa056).

``feet_air_contact_time`` is tasks/locomotion/g1/mdp/observations.py; ``stance_type`` is HCRL_mdp/observations.py
with ``stance_label`` (HCRL_mdp/contact_estimator.py) inlined; ``illegal_external_contact`` is HCRL_mdp/terminations.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


def feet_air_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """``[air time, contact time]`` per sensed foot, the timers ``gait_contact_clock`` reads. Shape is (N, 2 * feet)."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air = sensor.data.current_air_time.torch[:, sensor_cfg.body_ids]
    contact = sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids]
    return torch.stack((air, contact), dim=-1).flatten(1)


def stance_type(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """The true stance type, one-hot over the contact-mask classes (bit i = foot i down). Shape is (N, 2**feet)."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_contact = sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids] > 0
    bits = 1 << torch.arange(in_contact.shape[-1], device=in_contact.device)
    return F.one_hot((in_contact.long() * bits).sum(-1), 2 ** in_contact.shape[-1]).float()


def illegal_external_contact(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """``mdp.illegal_contact``, ignoring contact between the robot's own bodies.

    Needs a sensor built with ``exclude_self=True`` (Newton only). Current step only:
    Newton keeps no ``force_matrix_w`` history.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    data = contact_sensor.data
    if data.force_matrix_w is None:
        raise RuntimeError(f"sensor {sensor_cfg.name!r} has no force matrix; build it with exclude_self=True")
    external = data.net_forces_w.torch - data.force_matrix_w.torch.sum(dim=2)
    return torch.any(torch.linalg.norm(external[:, sensor_cfg.body_ids], dim=-1) > threshold, dim=1)
