"""Sim2sim export: a run-named folder holding ``policy.onnx`` and a ``config.yaml``
that fully describes the policy's interface.

The yaml is the contract for the MuJoCo sim2sim repo, so everything it needs to drive
the ONNX graph is derived from the *live* env -- regexes expanded, ``__post_init__``
applied -- rather than re-read from the task's source. Ordering is deliberate:
frequencies and robot name at the top, actuators at the bottom.

Actuator gains are read from the cfg rather than the live articulation on purpose; see
``_actuator_section``. ``_obs_section`` drops observation corruption.

Nothing here may be imported on the env-cfg path -- it touches Articulation, hence
``isaaclab.sim``, hence ``pxr`` (F9). Scripts import it after AppLauncher.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import yaml

from isaaclab.utils.string import resolve_matching_names_values

__all__ = ["build_config", "resolve_export_dir", "write_export"]


##
# Scalars
##


def _tensor(value: Any) -> Any:
    """Unwrap a ``ProxyArray`` to its torch view; asset data is never a plain tensor (F8)."""
    return value.torch if hasattr(value, "torch") else value


def _py(value: Any) -> Any:
    """Coerce tensors/arrays/tuples into plain YAML-safe Python.

    Nesting is preserved one level deep so ``(lo, hi)`` pairs -- action clips, command
    ranges -- survive as pairs instead of flattening into an ambiguous run of numbers.
    """
    value = _tensor(value)
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.ndim == 0:
            return round(float(value), 8)
        if value.ndim == 1:
            return [round(float(v), 8) for v in value]
        return [_py(row) for row in value]
    if isinstance(value, (tuple, list)):
        return [_py(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _py(v) for k, v in value.items()}
    if isinstance(value, float):
        return round(value, 8)
    if isinstance(value, (int, bool, str)) or value is None:
        return value
    return str(value)


def _per_joint(spec: Any, joint_names: Sequence[str]) -> list[float] | None:
    """Expand an actuator-cfg value (dict of regex->value, bare scalar, or None) over ``joint_names``."""
    if spec is None:
        return None
    if isinstance(spec, dict):
        out: list[float | None] = [None] * len(joint_names)
        index_list, _, values = resolve_matching_names_values(spec, list(joint_names))
        for idx, value in zip(index_list, values):
            out[idx] = round(float(value), 8)
        return None if any(v is None for v in out) else out  # type: ignore[return-value]
    return [round(float(spec), 8)] * len(joint_names)


##
# Sections
##


def _robot_name(articulation) -> str | None:
    """Best-effort robot key, matching robot_description's per-robot directory name."""
    spawn = getattr(articulation.cfg, "spawn", None)
    usd_dir = getattr(spawn, "usd_dir", None)
    if usd_dir:
        return Path(str(usd_dir)).name
    asset_path = getattr(spawn, "asset_path", None) or getattr(spawn, "usd_path", None)
    return Path(str(asset_path)).parent.name if asset_path else None


def _obs_section(env) -> dict[str, Any]:
    """Policy-group observation layout: term order, per-term slice and scale.

    Corruption terms are omitted by contract -- every ``-PLAY`` cfg sets
    ``enable_corruption = False``, so the exported policy's input is the clean signal.
    """
    manager = env.observation_manager
    names = manager.active_terms["policy"]
    dims = manager.group_obs_term_dim["policy"]
    # No public accessor exposes the resolved term cfgs, only their names and dims.
    cfgs = manager._group_obs_term_cfgs["policy"]

    terms, offset = [], 0
    for name, dim, cfg in zip(names, dims, cfgs):
        width = int(torch.tensor(dim).prod().item())
        entry: dict[str, Any] = {
            "name": name,
            "func": getattr(cfg.func, "__name__", str(cfg.func)),
            "dim": width,
            "slice": [offset, offset + width],
            "scale": _py(cfg.scale) if cfg.scale is not None else 1.0,
            "clip": _py(cfg.clip),
        }
        if cfg.params:
            entry["params"] = _py(cfg.params)
        if getattr(cfg, "history_length", 0) > 1:
            entry["history_length"] = int(cfg.history_length)
        terms.append(entry)
        offset += width

    return {"total_dim": offset, "terms": terms}


def _command_section(env, obs_terms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Command terms with their *training* sampling ranges, plus where each lands in the obs vector."""
    manager = getattr(env, "command_manager", None)
    if manager is None:
        return []

    # generated_commands terms name their command in params; that is the obs<->command link.
    obs_slice_of = {
        term.get("params", {}).get("command_name"): list(term["slice"])
        for term in obs_terms
        if term["func"] == "generated_commands"
    }

    out = []
    for name in manager.active_terms:
        cfg = manager.get_term(name).cfg
        entry: dict[str, Any] = {"name": name, "type": type(cfg).__name__}
        if name in obs_slice_of:
            entry["obs_slice"] = obs_slice_of[name]

        ranges = getattr(cfg, "ranges", None)
        if ranges is not None:
            # Field names and arity vary by command type (Normal* uses mean/std, not lo/hi pairs).
            entry["ranges"] = {k: _py(v) for k, v in vars(ranges).items() if not k.startswith("_")}
        for extra in ("heading_command", "rel_standing_envs", "resampling_time_range"):
            if hasattr(cfg, extra):
                entry[extra] = _py(getattr(cfg, extra))
        out.append(entry)
    return out


def _action_section(env) -> dict[str, Any]:
    """Action-term layout. Terms are described by whatever public attributes they expose,
    so a task with a custom term exports without changes here."""
    manager = env.action_manager
    terms, offset = [], 0
    for name in manager.active_terms:
        term = manager.get_term(name)
        width = int(term.action_dim)
        entry: dict[str, Any] = {
            "name": name,
            "type": type(term).__name__,
            "dim": width,
            "slice": [offset, offset + width],
        }
        # Upstream JointAction resolves names into a private attr and exposes no property.
        joint_names = getattr(term, "joint_names", None) or getattr(term, "_joint_names", None)
        if joint_names is not None:
            entry["joint_names"] = list(joint_names)
        for attr in ("_scale", "_offset", "_clip"):
            if (value := getattr(term, attr, None)) is not None:
                entry[attr.lstrip("_")] = _py(value[0] if getattr(value, "ndim", 0) > 1 else value)
        terms.append(entry)
        offset += width

    return {"total_dim": offset, "terms": terms}


def _actuator_section(articulation) -> dict[str, Any]:
    """Per-joint actuator parameters in articulation joint order, from the *cfg* not the
    live tensors: an action term may overwrite ``actuator.stiffness`` in place, so the live
    values are scratch state."""
    joint_names = list(articulation.joint_names)
    section: dict[str, Any] = {"joint_order": joint_names}

    fields = ("stiffness", "damping", "armature", "friction", "effort_limit", "velocity_limit")
    resolved: dict[str, list[float | None]] = {f: [None] * len(joint_names) for f in fields}

    for actuator in articulation.actuators.values():
        cfg = actuator.cfg
        group_names = list(actuator.joint_names)
        group_indices = [joint_names.index(n) for n in group_names]
        for field in fields:
            values = _per_joint(getattr(cfg, field, None), group_names)
            if values is None:
                # Not declared in the cfg (inherited from USD); fall back to the live tensor.
                live = getattr(actuator, field, None)
                if live is None:
                    continue
                flat = _py(live[0] if live.ndim > 1 else live)
                values = flat if isinstance(flat, list) else [flat] * len(group_names)
            for slot, value in zip(group_indices, values):
                resolved[field][slot] = value

    for field in fields:
        if any(v is not None for v in resolved[field]):
            section[field] = resolved[field]
    return section


##
# Entry points
##


def resolve_wandb_id(run_dir: str, logs_root: str | None = None) -> str | None:
    """Reverse-lookup of the wandb run id for ``run_dir``.

    Each wandb run records its rsl-rl log directory under a top-level ``log_dir`` key in
    its ``files/config.yaml``, so this is an exact match rather than a timestamp guess.
    Only the trailing ``<experiment_name>/<run_name>`` is compared: the recorded path is
    the container's (``/workspace/isaaclab/logs/...``), which differs from the host's.
    """
    root = Path(logs_root) if logs_root else Path(run_dir).parents[2] / "wandb" / "wandb"
    if not root.is_dir():
        return None
    target = Path(run_dir).parts[-2:]
    for candidate in sorted(root.glob("run-*/files/config.yaml"), reverse=True):
        try:
            with open(candidate) as handle:
                data = yaml.safe_load(handle) or {}
        except (OSError, yaml.YAMLError):
            continue
        entry = data.get("log_dir")
        logged = entry.get("value") if isinstance(entry, dict) else entry
        if logged and Path(str(logged)).parts[-2:] == target:
            return candidate.parents[1].name.rsplit("-", 1)[-1]
    return None


def resolve_export_dir(resume_path: str) -> Path:
    """``<run_dir>/<run_name>/`` -- one level below the run dir, so videos and
    checkpoints sit one above it. ``run_name`` is also the wandb run name (F4)."""
    run_dir = Path(resume_path).parent
    return run_dir / run_dir.name


def build_config(env, *, task: str, resume_path: str, robot_name: str | None = None) -> dict[str, Any]:
    """Assemble the sim2sim config for a live env and the checkpoint it was loaded from."""
    run_dir = Path(resume_path).parent
    articulation = env.scene[env.action_manager.get_term(env.action_manager.active_terms[0]).cfg.asset_name]

    observations = _obs_section(env)
    decimation = int(env.cfg.decimation)
    sim_dt = float(env.cfg.sim.dt)

    config: dict[str, Any] = {
        "policy_frequency_hz": round(1.0 / (sim_dt * decimation), 6),
        "sim_frequency_hz": round(1.0 / sim_dt, 6),
        "sim_dt": sim_dt,
        "decimation": decimation,
        "robot": robot_name or _robot_name(articulation),
        "task": task,
        "run": {
            "name": run_dir.name,
            "wandb_id": resolve_wandb_id(str(run_dir)),
            "checkpoint": Path(resume_path).name,
        },
        "observations": observations,
        "commands": _command_section(env, observations["terms"]),
        "actions": _action_section(env),
        "default_joint_pos": {
            "joint_order": list(articulation.joint_names),
            "values": _py(articulation.data.default_joint_pos[0]),
        },
        "actuators": _actuator_section(articulation),
    }
    return config


class _ConfigDumper(yaml.SafeDumper):
    """Block style throughout, except short scalar lists -- ``(lo, hi)`` bounds and
    ``[start, stop]`` slices stay on one line, while per-joint arrays get one entry per
    line so a value can be matched against ``joint_order`` by counting rows."""


def _represent_list(dumper: yaml.SafeDumper, data: list) -> Any:
    inline = len(data) <= 2 and all(isinstance(v, (int, float, str)) for v in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=inline)


_ConfigDumper.add_representer(list, _represent_list)


def write_export(env, *, task: str, resume_path: str, robot_name: str | None = None) -> Path:
    """Write ``config.yaml`` next to the ONNX and return the export directory."""
    export_dir = resolve_export_dir(resume_path)
    export_dir.mkdir(parents=True, exist_ok=True)
    config = build_config(env, task=task, resume_path=resume_path, robot_name=robot_name)
    with open(export_dir / "config.yaml", "w") as handle:
        yaml.dump(config, handle, Dumper=_ConfigDumper, sort_keys=False, default_flow_style=False)
    return export_dir
