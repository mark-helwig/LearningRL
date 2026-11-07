"""Left/right mirror augmentation for rsl_rl's ``RslRlSymmetryCfg``, generic over our bipeds.

The mirror is the reflection y -> -y in the base frame. Left/right joints pair by their ``left_``/``right_``
prefix; each joint's sign comes from the robot's source URDF, never from its name, so a new robot needs no
edits here as long as its URDF is mirror-built. Observation terms are matched by their ``func``, so an
unknown term, action term, or asymmetric default pose raises instead of silently training on a bad mirror.
A command term whose layout is not ``(vx, vy, wz)`` declares its own ``mirror_map = (perm, sign)``.

Nothing here imports Isaac Lab at module scope: the agent cfg imports this before Kit starts (F9).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from tensordict import TensorDict

    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states", "mirror_joint_name", "urdf_mirror_signs"]

# Base-frame vectors: polar ones negate y, the angular velocity (a pseudovector) negates x and z.
_OBS_SIGNS: dict[str, tuple[float, ...]] = {
    "base_lin_vel": (1.0, -1.0, 1.0),
    "base_ang_vel": (-1.0, 1.0, -1.0),
    "projected_gravity": (1.0, -1.0, 1.0),
    "generated_commands": (1.0, -1.0, -1.0),  # (vx, vy, wz) from UniformVelocityCommand, unless `mirror_map`
}
_JOINT_OBS = ("joint_pos_rel", "joint_vel_rel", "joint_pos", "joint_vel")


def mirror_joint_name(name: str) -> str:
    """Swap a ``left_``/``right_`` prefix; center-line joints map to themselves."""
    if name.startswith("left_"):
        return "right_" + name[5:]
    if name.startswith("right_"):
        return "left_" + name[6:]
    return name


def urdf_mirror_signs(urdf_path: str) -> dict[str, float]:
    """Per-joint sign ``s`` with ``q_mirror(partner) = s * q(joint)``, for every non-fixed URDF joint.

    With mirror-image parent frames, reflecting a rotation ``q`` about axis ``a`` gives a rotation ``-q`` about
    ``M a`` (``M = diag(1, -1, 1)``), so ``s = -(a_partner . M a)``: +1 for pitch-like axes, -1 for roll/yaw-like.
    """
    joints = {}
    for joint in ET.parse(urdf_path).getroot().iter("joint"):
        if joint.get("type") == "fixed":
            continue
        joints[joint.get("name")] = (
            _vec(joint.find("axis"), "xyz", (1.0, 0.0, 0.0)),
            _vec(joint.find("origin"), "rpy", (0.0, 0.0, 0.0)),
        )

    signs = {}
    for name, (axis, rpy) in joints.items():
        partner_axis, partner_rpy = joints[mirror_joint_name(name)]
        # only orientation matters for the sign: the partner's frame must be this one's with roll and yaw negated
        if not all(abs(a - b) < 1e-3 for a, b in zip(partner_rpy, (-rpy[0], rpy[1], -rpy[2]))):
            raise ValueError(f"{urdf_path}: joint frames of {name} and {mirror_joint_name(name)} are not mirror images")
        dot = partner_axis[0] * axis[0] - partner_axis[1] * axis[1] + partner_axis[2] * axis[2]
        if abs(abs(dot) - 1.0) > 1e-3:
            raise ValueError(f"{urdf_path}: axes of {name} and {mirror_joint_name(name)} are not mirror images")
        signs[name] = -round(dot)
    return signs


def _vec(element: ET.Element | None, key: str, default: tuple[float, float, float]) -> tuple[float, ...]:
    if element is None or element.get(key) is None:
        return default
    return tuple(float(v) for v in element.get(key).split())


@torch.no_grad()
def compute_symmetric_states(
    env,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """``RslRlSymmetryCfg.data_augmentation_func``: returns ``[original; left/right mirror]`` along the batch."""
    maps = _mirror_maps(env.unwrapped)
    obs_aug = None
    if obs is not None:
        obs_aug = obs.repeat(2)
        batch_size = obs.batch_size[0]
        for group in obs.keys():
            perm, sign = maps[group]
            obs_aug[group][batch_size:] = obs[group][:, perm] * sign
    actions_aug = None
    if actions is not None:
        perm, sign = maps["actions"]
        actions_aug = torch.cat((actions, actions[:, perm] * sign))
    return obs_aug, actions_aug


def _mirror_maps(env: ManagerBasedRLEnv) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """``{obs group | "actions": (perm, sign)}`` with ``mirrored = x[:, perm] * sign``, built once per env."""
    if hasattr(env, "_learningrl_mirror_maps"):
        return env._learningrl_mirror_maps

    from isaaclab.envs.mdp.actions.joint_actions import JointAction

    robot = env.scene["robot"]
    signs = urdf_mirror_signs(robot.cfg.spawn.asset_path)
    all_joints = list(robot.joint_names)

    def joint_map(names: list[str], signed: bool = True) -> tuple[list[int], list[float]]:
        perm = [names.index(mirror_joint_name(n)) for n in names]
        return perm, [signs[n] if signed else 1.0 for n in names]

    # joint_pos_rel and the default-offset position action are only mirror-exact about a mirrored default pose
    default = robot.data.default_joint_pos.torch[0].cpu()
    perm, sign = joint_map(all_joints)
    if not torch.allclose(default[perm] * torch.tensor(sign, dtype=default.dtype), default, atol=1e-5):
        raise ValueError("default joint pose is not left/right symmetric; the mirror would be biased")

    action_perm, action_sign = [], []
    for term_name in env.action_manager.active_terms:
        term = env.action_manager.get_term(term_name)
        names = getattr(term, "joint_names", None) or term._joint_names
        # joint targets (position/velocity/effort) flip with the joint; other terms declare `mirror_signed`
        if not isinstance(term, JointAction) and not hasattr(term, "mirror_signed"):
            raise NotImplementedError(f"no mirror rule for action term '{term_name}' ({type(term).__name__})")
        p, s = joint_map(list(names), signed=isinstance(term, JointAction) or term.mirror_signed)
        offset = len(action_perm)
        action_perm += [offset + i for i in p]
        action_sign += s

    maps = {"actions": (action_perm, action_sign)}
    obs_manager = env.observation_manager
    for group, term_names in obs_manager.active_terms.items():
        group_perm, group_sign = [], []
        for term_name, term_cfg, dims in zip(
            term_names, obs_manager._group_obs_term_cfgs[group], obs_manager.group_obs_term_dim[group]
        ):
            func = term_cfg.func.__name__
            command = term_cfg.params.get("command_name") if func == "generated_commands" else None
            mirror_map = getattr(env.command_manager.get_term(command), "mirror_map", None) if command else None
            if mirror_map is not None:
                p, s = list(mirror_map[0]), list(mirror_map[1])
            elif func in _OBS_SIGNS:
                p, s = list(range(3)), list(_OBS_SIGNS[func])
            elif func in _JOINT_OBS:
                asset_cfg = term_cfg.params.get("asset_cfg")
                joint_ids = slice(None) if asset_cfg is None else asset_cfg.joint_ids
                names = all_joints if isinstance(joint_ids, slice) else [all_joints[i] for i in joint_ids]
                p, s = joint_map(names)
            elif func == "last_action" and term_cfg.params.get("action_name") is None:
                p, s = action_perm, action_sign
            else:
                raise NotImplementedError(f"no mirror rule for observation term '{group}/{term_name}' ({func})")
            # a flattened history is history_length back-to-back copies of the term
            copies, width = max(term_cfg.history_length, 1), len(p)
            if dims[0] != copies * width:
                raise ValueError(f"observation term '{group}/{term_name}' is {dims[0]} wide, expected {copies * width}")
            offset = len(group_perm)
            group_perm += [offset + h * width + i for h in range(copies) for i in p]
            group_sign += s * copies
        maps[group] = (group_perm, group_sign)

    env._learningrl_mirror_maps = {
        key: (torch.tensor(p, device=env.device), torch.tensor(s, dtype=torch.float32, device=env.device))
        for key, (p, s) in maps.items()
    }
    return env._learningrl_mirror_maps
