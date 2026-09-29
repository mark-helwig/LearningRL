"""Parse and hard-validate the `config.yaml` written by LearningRLlab's exporter.

Everything task-specific enters LearningRLsim through this file and nowhere else
(invariant 3). The dataclasses below mirror the exported schema one-for-one;
LearningRLsim never reaches back into a LearningRLlab task class to ask what a run meant.

**Validation is eager and total.** Every check here is one that would
otherwise surface as a robot that walks slightly wrong rather than as an
error: an observation slice off by one, a `joint_order` that disagrees with
an action term's `joint_names`, a term name nothing implements. Those are
indistinguishable from the sim2sim gap this harness measures, so they are
rejected at load with a message naming the offending key.

Conditional keys (the exporter omits, rather than nulls, most optional
fields) are read through `.get()` with the documented default; the ones that
are always present are read through `_require` so a schema drift fails loudly
instead of defaulting.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml


class RunConfigError(ValueError):
    """A malformed, inconsistent, or unimplementable run config.

    Always names the run directory and the offending key.
    """


def _require(data: dict[str, Any], key: str, where: str) -> Any:
    if key not in data:
        raise RunConfigError(f"{where}: missing required key '{key}'")
    return data[key]


def _as_slice(raw: Any, where: str) -> tuple[int, int]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise RunConfigError(f"{where}.slice: expected a [start, stop] pair, got {raw!r}")
    start, stop = int(raw[0]), int(raw[1])
    if stop < start:
        raise RunConfigError(f"{where}.slice: [{start}, {stop}] is decreasing")
    return start, stop


def _as_pair(raw: Any, where: str) -> tuple[float, float] | None:
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise RunConfigError(f"{where}: expected a [lo, hi] pair or null, got {raw!r}")
    return float(raw[0]), float(raw[1])


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ObsTermCfg:
    """One entry of `observations.terms[]`.

    `func` is the dispatch key, NOT `name`: the two coincide for the
    locomotion tasks, but a task inheriting upstream's naming (`joint_pos` /
    `actions`) calls the same functions `joint_pos_rel` / `last_action`.
    Dispatching on `name` would silently fail to find a term that is in fact
    implemented.
    """

    name: str
    func: str
    dim: int
    slice: tuple[int, int]
    scale: np.ndarray | None
    clip: tuple[float, float] | None
    params: dict[str, Any] = field(default_factory=dict)
    history_length: int = 1

    @staticmethod
    def from_dict(data: dict[str, Any], where: str) -> ObsTermCfg:
        name = _require(data, "name", where)
        where = f"{where}[{name}]"
        dim = int(_require(data, "dim", where))
        scale = data.get("scale")
        return ObsTermCfg(
            name=str(name),
            func=str(_require(data, "func", where)),
            dim=dim,
            slice=_as_slice(_require(data, "slice", where), where),
            # A scalar scale broadcasts; a per-element list must match `dim`,
            # since a short list would broadcast-fail only for some dims.
            scale=_scale_array(scale, dim, where),
            clip=_as_pair(data.get("clip"), f"{where}.clip"),
            params=dict(data.get("params") or {}),
            # Exported only when > 1, and no LearningRLlab task uses history today.
            # Read it anyway so the unimplemented case fails in validate()
            # rather than being silently flattened to a single frame.
            history_length=int(data.get("history_length", 1)),
        )


def _scale_array(raw: Any, dim: int, where: str) -> np.ndarray | None:
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(dim, float(arr))
    if arr.shape != (dim,):
        raise RunConfigError(f"{where}.scale: has {arr.shape} entries for a dim-{dim} term")
    return arr


@dataclass(frozen=True)
class ObsGroupCfg:
    total_dim: int
    terms: tuple[ObsTermCfg, ...]


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ActionTermCfg:
    """One entry of `actions.terms[]`.

    Most fields are exported only when the corresponding private attribute on
    the Isaac Lab term is not None, so nearly all of them are optional here.
    `type` is the dispatch key.
    """

    name: str
    type: str
    dim: int
    slice: tuple[int, int]
    joint_names: tuple[str, ...]
    scale: np.ndarray | None
    offset: np.ndarray | None
    clip: np.ndarray | None  # (dim, 2), on the PROCESSED value

    @staticmethod
    def from_dict(data: dict[str, Any], where: str) -> ActionTermCfg:
        name = _require(data, "name", where)
        where = f"{where}[{name}]"
        dim = int(_require(data, "dim", where))
        joint_names = tuple(str(j) for j in (data.get("joint_names") or ()))

        clip_raw = data.get("clip")
        clip = None
        if clip_raw is not None:
            clip = np.asarray(clip_raw, dtype=np.float64)
            if clip.ndim == 1:  # one shared pair
                clip = np.tile(clip.reshape(1, 2), (dim, 1))
            if clip.shape != (dim, 2):
                raise RunConfigError(f"{where}.clip: expected ({dim}, 2) bounds, got {clip.shape}")

        return ActionTermCfg(
            name=str(name),
            type=str(_require(data, "type", where)),
            dim=dim,
            slice=_as_slice(_require(data, "slice", where), where),
            joint_names=joint_names,
            scale=_scale_array(data.get("scale"), dim, where),
            offset=_scale_array(data.get("offset"), dim, where),
            clip=clip,
        )


@dataclass(frozen=True)
class ActionGroupCfg:
    total_dim: int
    terms: tuple[ActionTermCfg, ...]


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CommandCfg:
    """One entry of `commands[]`.

    `obs_slice` is present only when some policy observation term actually
    consumes this command (`func: generated_commands` with a matching
    `params.command_name`); a command the policy never sees still needs to be
    driven, so `obs_slice` is optional rather than required.

    `rel_standing_envs` is exported but deliberately NOT acted on -- it is a
    per-env sampling fraction across a vectorised env and is meaningless for
    one robot. Standing is an explicit command-source mode instead.
    """

    name: str
    type: str
    obs_slice: tuple[int, int] | None
    ranges: dict[str, Any]
    heading_command: bool
    rel_standing_envs: float | None
    resampling_time_range: tuple[float, float] | None

    @staticmethod
    def from_dict(data: dict[str, Any], where: str) -> CommandCfg:
        name = _require(data, "name", where)
        where = f"{where}[{name}]"
        obs_slice = data.get("obs_slice")
        return CommandCfg(
            name=str(name),
            type=str(_require(data, "type", where)),
            obs_slice=_as_slice(obs_slice, where) if obs_slice is not None else None,
            ranges=dict(data.get("ranges") or {}),
            heading_command=bool(data.get("heading_command", False)),
            rel_standing_envs=(float(data["rel_standing_envs"]) if data.get("rel_standing_envs") is not None else None),
            resampling_time_range=_as_pair(data.get("resampling_time_range"), f"{where}.resampling_time_range"),
        )


# ---------------------------------------------------------------------------
# Actuators
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ActuatorCfg:
    """Per-joint actuator parameters, every array indexed by `joint_order`."""

    joint_order: tuple[str, ...]
    stiffness: np.ndarray
    damping: np.ndarray
    armature: np.ndarray
    friction: np.ndarray
    effort_limit: np.ndarray
    velocity_limit: np.ndarray


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RunConfig:
    policy_frequency_hz: float
    sim_frequency_hz: float
    sim_dt: float
    decimation: int
    robot: str | None
    task: str
    run: dict[str, Any]
    observations: ObsGroupCfg
    commands: tuple[CommandCfg, ...]
    actions: ActionGroupCfg
    joint_order: tuple[str, ...]
    default_joint_pos: np.ndarray
    actuators: ActuatorCfg
    raw: dict[str, Any]
    """The parsed YAML verbatim, re-emitted into the bag's `/meta/run` so a bag
    is self-contained: a sim2sim result whose policy interface you cannot
    recover from the file is an anecdote."""
    path: Path

    @property
    def num_joints(self) -> int:
        return len(self.joint_order)

    def command(self, name: str) -> CommandCfg:
        for cmd in self.commands:
            if cmd.name == name:
                return cmd
        raise RunConfigError(
            f"{self.path}: no command term named {name!r}; declared: {[c.name for c in self.commands]}"
        )


def load(run_dir: str | Path) -> RunConfig:
    """Load `<run_dir>/config.yaml` and validate it. Raises `RunConfigError`."""
    run_dir = Path(run_dir)
    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        raise RunConfigError(
            f"{run_dir}: no config.yaml. Expected an exported run directory "
            f"(config.yaml + policy.onnx + policy.onnx.data) as written by "
            f"LearningRLlab's scripts/play.py. Note the layout nests one level: "
            f"logs/rsl_rl/<experiment>/<run>/<run>/."
        )

    with config_path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise RunConfigError(f"{config_path}: top level must be a mapping")

    where = str(config_path)
    cfg = _build(raw, where, config_path)
    _validate(cfg)
    return cfg


def _build(raw: dict[str, Any], where: str, path: Path) -> RunConfig:
    obs_raw = _require(raw, "observations", where)
    act_raw = _require(raw, "actions", where)
    djp_raw = _require(raw, "default_joint_pos", where)
    actuators_raw = _require(raw, "actuators", where)

    joint_order = tuple(str(j) for j in _require(djp_raw, "joint_order", f"{where}.default_joint_pos"))

    return RunConfig(
        policy_frequency_hz=float(_require(raw, "policy_frequency_hz", where)),
        sim_frequency_hz=float(_require(raw, "sim_frequency_hz", where)),
        sim_dt=float(_require(raw, "sim_dt", where)),
        decimation=int(_require(raw, "decimation", where)),
        robot=(str(raw["robot"]) if raw.get("robot") is not None else None),
        task=str(_require(raw, "task", where)),
        run=dict(raw.get("run") or {}),
        observations=ObsGroupCfg(
            total_dim=int(_require(obs_raw, "total_dim", f"{where}.observations")),
            terms=tuple(
                ObsTermCfg.from_dict(t, f"{where}.observations.terms")
                for t in _require(obs_raw, "terms", f"{where}.observations")
            ),
        ),
        commands=tuple(CommandCfg.from_dict(c, f"{where}.commands") for c in (raw.get("commands") or [])),
        actions=ActionGroupCfg(
            total_dim=int(_require(act_raw, "total_dim", f"{where}.actions")),
            terms=tuple(
                ActionTermCfg.from_dict(t, f"{where}.actions.terms")
                for t in _require(act_raw, "terms", f"{where}.actions")
            ),
        ),
        joint_order=joint_order,
        default_joint_pos=np.asarray(_require(djp_raw, "values", f"{where}.default_joint_pos"), dtype=np.float64),
        actuators=ActuatorCfg(
            joint_order=tuple(str(j) for j in _require(actuators_raw, "joint_order", f"{where}.actuators")),
            **{
                key: np.asarray(_require(actuators_raw, key, f"{where}.actuators"), dtype=np.float64)
                for key in (
                    "stiffness",
                    "damping",
                    "armature",
                    "friction",
                    "effort_limit",
                    "velocity_limit",
                )
            },
        ),
        raw=raw,
        path=path,
    )


def _validate_slices(terms: tuple[Any, ...], total_dim: int, what: str, where: str) -> None:
    """Slices must tile [0, total_dim) contiguously, in declaration order.

    The vector is built by concatenating terms in order and `slice` is
    authoritative, so a gap, an overlap or an out-of-order entry means the
    policy would be fed a differently-laid-out vector than it was trained on
    -- which does not raise anywhere downstream.
    """
    cursor = 0
    for term in terms:
        start, stop = term.slice
        if start != cursor:
            raise RunConfigError(
                f"{where}: {what} term '{term.name}' starts at {start}, but the previous "
                f"term ended at {cursor}. Terms must tile the vector contiguously in "
                f"declaration order."
            )
        if stop - start != term.dim:
            raise RunConfigError(
                f"{where}: {what} term '{term.name}' declares dim={term.dim} but its slice "
                f"[{start}, {stop}] spans {stop - start}."
            )
        cursor = stop
    if cursor != total_dim:
        raise RunConfigError(f"{where}: {what} terms end at {cursor}, but total_dim is {total_dim}.")


def _validate(cfg: RunConfig) -> None:
    # Imported here, not at module scope: the term libraries want these
    # dataclasses, so a top-level import would be a cycle.
    from LearningRLsim import library

    where = str(cfg.path)

    _validate_slices(cfg.observations.terms, cfg.observations.total_dim, "observation", where)
    _validate_slices(cfg.actions.terms, cfg.actions.total_dim, "action", where)

    # Rates must agree with each other: decimation is what the loop actually
    # runs on, and the two frequencies are derived, so a disagreement means
    # one of the three is not what the run used.
    expected_sim_hz = 1.0 / cfg.sim_dt
    if abs(expected_sim_hz - cfg.sim_frequency_hz) > 1e-6 * expected_sim_hz:
        raise RunConfigError(
            f"{where}: sim_dt={cfg.sim_dt} implies {expected_sim_hz} Hz, but "
            f"sim_frequency_hz is {cfg.sim_frequency_hz}."
        )
    expected_policy_hz = cfg.sim_frequency_hz / cfg.decimation
    if abs(expected_policy_hz - cfg.policy_frequency_hz) > 1e-6 * expected_policy_hz:
        raise RunConfigError(
            f"{where}: sim_frequency_hz/decimation = {expected_policy_hz} Hz, but "
            f"policy_frequency_hz is {cfg.policy_frequency_hz}."
        )

    # joint_order is one ordering, published twice. If the two copies ever
    # disagree, every actuator array is indexed against the wrong names.
    if cfg.actuators.joint_order != cfg.joint_order:
        raise RunConfigError(
            f"{where}: actuators.joint_order differs from default_joint_pos.joint_order. "
            f"Every per-joint array is indexed by one ordering; two disagreeing copies "
            f"means the gains belong to different joints than the offsets."
        )
    if cfg.default_joint_pos.shape != (len(cfg.joint_order),):
        raise RunConfigError(
            f"{where}.default_joint_pos: {cfg.default_joint_pos.shape[0]} values for {len(cfg.joint_order)} joints."
        )
    for key in ("stiffness", "damping", "armature", "friction", "effort_limit", "velocity_limit"):
        arr = getattr(cfg.actuators, key)
        if arr.shape != (len(cfg.joint_order),):
            raise RunConfigError(f"{where}.actuators.{key}: {arr.shape[0]} values for {len(cfg.joint_order)} joints.")

    known = set(cfg.joint_order)
    for term in cfg.actions.terms:
        unknown = [j for j in term.joint_names if j not in known]
        if unknown:
            raise RunConfigError(
                f"{where}: action term '{term.name}' names joint(s) {unknown} that are not in joint_order."
            )
        if term.joint_names and len(term.joint_names) != term.dim:
            raise RunConfigError(
                f"{where}: action term '{term.name}' has dim={term.dim} but {len(term.joint_names)} joint_names."
            )

    # Every named func/type must be implemented. Checked last so that a config
    # which is also structurally broken reports the structural fault first.
    for term in cfg.observations.terms:
        library.OBSERVATIONS.lookup(term.func)
        if term.history_length > 1:
            raise RunConfigError(
                f"{where}: observation term '{term.name}' has history_length="
                f"{term.history_length}. LearningRLsim assembles a single frame per term; "
                f"history buffers are not implemented."
            )
    for term in cfg.actions.terms:
        library.ACTIONS.lookup(term.type)
    for cmd in cfg.commands:
        library.COMMANDS.lookup(cmd.type)


def at_sim_rate(cfg: RunConfig, sim_rate_hz: float) -> RunConfig:
    """The same run, stepped at `sim_rate_hz` instead of the exported rate.

    The policy rate is a property of the trained policy and does not move, so
    the whole change lands on `decimation` -- which is what the loop actually
    counts. A non-integer ratio is rejected rather than rounded: a rounded
    decimation silently runs the policy at a rate it was not trained at, which
    presents as a gait change and reads as a finding.

    Raising the rate is not a sim2sim measurement. It is a different plant, run
    to see contact content above the exported rate's Nyquist; `/meta/run` marks
    both fields `override` so a bag can never be mistaken for the other thing.
    """
    if sim_rate_hz <= 0:
        raise RunConfigError(f"--sim-rate must be positive, got {sim_rate_hz}.")

    ratio = sim_rate_hz / cfg.policy_frequency_hz
    decimation = int(round(ratio))
    if decimation < 1 or abs(ratio - decimation) > 1e-9 * max(1.0, ratio):
        raise RunConfigError(
            f"--sim-rate {sim_rate_hz:g} Hz is not an integer multiple of the policy's "
            f"{cfg.policy_frequency_hz:g} Hz ({ratio:.6g}x). Pick a multiple -- "
            f"e.g. {cfg.policy_frequency_hz * max(1, decimation):g} Hz."
        )

    return replace(
        cfg,
        sim_frequency_hz=float(sim_rate_hz),
        sim_dt=1.0 / float(sim_rate_hz),
        decimation=decimation,
    )
