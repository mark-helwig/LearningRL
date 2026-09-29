"""Strict schema + loader for `robots/<robot>/robot.yaml`.

No Isaac Lab dependency -- only `yaml`/`dataclasses`/stdlib -- so this module,
and `tests/test_robot_schema.py`, run on Windows with no vendor/ checkout.

Strictness is deliberate: **an unknown key is a hard error**, not a warning.
A typo'd key (`slef_collision`, `mesh_bugdet_mb`) that silently fell back to a
default would be indistinguishable from a correctly-set key that happens to
match the default -- exactly the kind of mistake that only shows up much
later, as a wrong mass or DoF at spawn time.

`robot.yaml` only describes where the source URDF/MJCF lives and what
`validate.py` should expect from it (DoF, mass, nominal pose, mesh budget,
...) -- it does NOT configure a conversion step. There isn't one anymore:
`robots/<robot>/robot.py` spawns directly from the source file via Isaac
Lab's `UrdfFileCfg`, which converts to USD on the fly at spawn time. Importer
options that used to live here (`merge_fixed_joints`, `collision_type`,
`joint_drive` gains, ...) now live as ordinary Python defaults on
`robot_description.robots.urdf_common.RobotUrdfFileCfg` instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal

import yaml

SCHEMA_VERSION = 1

SourceFormat = Literal["urdf", "mjcf"]


class RobotYamlError(ValueError):
    """Raised for any malformed, unknown-key, or semantically inconsistent
    robot.yaml. Always includes the offending robot/path in the message."""


def _reject_unknown_keys(data: dict[str, Any], known: set[str], where: str) -> None:
    unknown = set(data.keys()) - known
    if unknown:
        raise RobotYamlError(
            f"{where}: unknown key(s) {sorted(unknown)} -- robot.yaml is strict, "
            f"known keys are {sorted(known)}. If this is a new option, add it to "
            f"robot_schema.py first."
        )


def _require(data: dict[str, Any], key: str, where: str) -> Any:
    if key not in data or data[key] is None:
        raise RobotYamlError(f"{where}: missing required key '{key}'")
    return data[key]


def _reject_non_vendor_relative_path(path: str, where: str) -> None:
    """Every source path is relative to `vendor/<vendor>/` and POSIX-style.

    PurePosixPath, not platform Path: robot.yaml paths are always POSIX-style,
    and `Path("/x").is_absolute()` is FALSE on Windows -- using the platform
    Path would silently pass this check when authoring on Windows. Also check
    PureWindowsPath/backslash explicitly, since PurePosixPath treats
    "C:/vendor/x.urdf" as merely relative.
    """
    if PurePosixPath(path).is_absolute() or PureWindowsPath(path).is_absolute() or "\\" in path:
        raise RobotYamlError(
            f"{where}: must be a relative, forward-slash path under "
            f"vendor/<vendor>/, got the absolute or Windows-style path {path!r}"
        )


# ---------------------------------------------------------------------------
# source
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SourceCfg:
    format: SourceFormat = "urdf"
    vendor: str = ""  # required, no real default
    path: str = ""  # required, no real default
    mjcf_path: str | None = None
    ros_package_paths: list[str] = field(default_factory=list)

    @staticmethod
    def from_dict(data: dict[str, Any], where: str) -> SourceCfg:
        known = {"format", "vendor", "path", "mjcf_path", "ros_package_paths"}
        _reject_unknown_keys(data, known, f"{where}.source")

        fmt = data.get("format") or "urdf"
        if fmt not in ("urdf", "mjcf"):
            raise RobotYamlError(f"{where}.source.format: must be 'urdf' or 'mjcf', got {fmt!r}")

        vendor = _require(data, "vendor", f"{where}.source")
        path = _require(data, "path", f"{where}.source")
        _reject_non_vendor_relative_path(path, f"{where}.source.path")

        # Optional second source file: the MuJoCo XML, when `path` is the URDF.
        # Separate key rather than a suffix swap on `path`, because the two live
        # in unrelated subtrees for some vendors -- rpo_description has
        # urdf/rpo.urdf next to mjcf/rpo.xml. When format is already "mjcf",
        # `path` IS the MJCF and this key is redundant; reject it rather than
        # let two keys disagree about the same file.
        mjcf_path = data.get("mjcf_path")
        if mjcf_path is not None:
            if fmt == "mjcf":
                raise RobotYamlError(
                    f"{where}.source.mjcf_path: redundant when format is 'mjcf' -- "
                    f"`path` is already the MJCF. Remove one of the two."
                )
            mjcf_path = str(mjcf_path)
            _reject_non_vendor_relative_path(mjcf_path, f"{where}.source.mjcf_path")

        ros_package_paths = data.get("ros_package_paths") or []
        if not isinstance(ros_package_paths, list):
            raise RobotYamlError(f"{where}.source.ros_package_paths: must be a list")

        return SourceCfg(
            format=fmt,
            vendor=vendor,
            path=path,
            mjcf_path=mjcf_path,
            ros_package_paths=list(ros_package_paths),
        )


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TotalMassCfg:
    value: float = 0.0
    rel_tol: float = 0.01

    @staticmethod
    def from_dict(data: dict[str, Any], where: str) -> TotalMassCfg:
        known = {"value", "rel_tol"}
        _reject_unknown_keys(data, known, f"{where}.validate.total_mass")
        value = _require(data, "value", f"{where}.validate.total_mass")
        return TotalMassCfg(
            value=float(value),
            rel_tol=float(data.get("rel_tol", 0.01)),
        )


@dataclass(frozen=True)
class ValidateCfg:
    expected_dof: int = 0
    total_mass: TotalMassCfg = field(default_factory=TotalMassCfg)
    min_link_mass: float = 1e-4
    massless_links: list[str] = field(default_factory=list)
    nominal_joint_pos: dict[str, float] = field(default_factory=dict)
    joint_limit_margin: float = 0.02
    allow_self_collision_pairs: list[list[str]] = field(default_factory=list)
    mesh_budget_mb: float | None = None

    @staticmethod
    def from_dict(data: dict[str, Any], where: str) -> ValidateCfg:
        known = {
            "expected_dof",
            "total_mass",
            "min_link_mass",
            "massless_links",
            "nominal_joint_pos",
            "joint_limit_margin",
            "allow_self_collision_pairs",
            "mesh_budget_mb",
        }
        _reject_unknown_keys(data, known, f"{where}.validate")

        expected_dof = _require(data, "expected_dof", f"{where}.validate")
        total_mass_data = _require(data, "total_mass", f"{where}.validate")
        if not isinstance(total_mass_data, dict):
            raise RobotYamlError(f"{where}.validate.total_mass: must be a mapping")

        return ValidateCfg(
            expected_dof=int(expected_dof),
            total_mass=TotalMassCfg.from_dict(total_mass_data, where),
            min_link_mass=float(data.get("min_link_mass", 1e-4)),
            massless_links=list(data.get("massless_links") or []),
            nominal_joint_pos={
                str(k): float(v) for k, v in (data.get("nominal_joint_pos") or {}).items()
            },
            joint_limit_margin=float(data.get("joint_limit_margin", 0.02)),
            allow_self_collision_pairs=[
                list(pair) for pair in (data.get("allow_self_collision_pairs") or [])
            ],
            mesh_budget_mb=(
                float(data["mesh_budget_mb"]) if data.get("mesh_budget_mb") is not None else None
            ),
        )


# ---------------------------------------------------------------------------
# top-level robot.yaml
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RobotConfig:
    schema_version: int
    robot: str
    enabled: bool
    source: SourceCfg
    validate: ValidateCfg

    @staticmethod
    def from_dict(data: dict[str, Any], where: str) -> RobotConfig:
        top_known = {"schema_version", "robot", "enabled", "source", "validate"}
        _reject_unknown_keys(data, top_known, where)

        schema_version = _require(data, "schema_version", where)
        if schema_version != SCHEMA_VERSION:
            raise RobotYamlError(
                f"{where}: schema_version={schema_version!r}, this loader only "
                f"understands {SCHEMA_VERSION}. Reject rather than misread a "
                f"future schema."
            )

        robot = _require(data, "robot", where)

        source_data = _require(data, "source", where)
        if not isinstance(source_data, dict):
            raise RobotYamlError(f"{where}.source: must be a mapping")

        validate_data = _require(data, "validate", where)
        if not isinstance(validate_data, dict):
            raise RobotYamlError(f"{where}.validate: must be a mapping")

        return RobotConfig(
            schema_version=int(schema_version),
            robot=str(robot),
            enabled=bool(data.get("enabled", True)),
            source=SourceCfg.from_dict(source_data, where),
            validate=ValidateCfg.from_dict(validate_data, where),
        )


def load_robot_yaml(path: str | Path) -> RobotConfig:
    """Load and strictly validate one `robot.yaml`.

    Also asserts `robot` equals the parent directory name (e.g.
    `robots/unitree_g1/robot.yaml` must have `robot: unitree_g1`) -- this
    catches a copy-pasted robot.yaml that forgot to update `robot`, which
    would otherwise silently misattribute validation results to the wrong
    robot.
    """
    path = Path(path)
    where = str(path)
    if not path.is_file():
        raise RobotYamlError(f"{where}: file not found")

    # yaml.YAMLError (syntax errors) is NOT a RobotYamlError -- left unwrapped it
    # would break paths.urdf_path()'s "never raises" guarantee (it only catches
    # RobotYamlError/OSError) and turn a clean CLI error into a raw traceback.
    try:
        with path.open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise RobotYamlError(f"{where}: invalid YAML syntax: {exc}") from exc

    if not isinstance(raw, dict):
        raise RobotYamlError(f"{where}: top level must be a mapping, got {type(raw).__name__}")

    cfg = RobotConfig.from_dict(raw, where)

    expected_robot = path.parent.name
    if cfg.robot != expected_robot:
        raise RobotYamlError(
            f"{where}: robot={cfg.robot!r} does not match parent directory name {expected_robot!r}"
        )

    return cfg


def discover_robot_yamls(robots_dir: str | Path) -> list[Path]:
    """`robots/*/robot.yaml`, sorted for deterministic ordering."""
    robots_dir = Path(robots_dir)
    return sorted(robots_dir.glob("*/robot.yaml"))
