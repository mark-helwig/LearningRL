"""Filesystem layout: repo root, vendor checkout, and the on-the-fly USD cache.

`urdf_path()` is deliberately TOTAL -- it never raises. `UrdfFileCfg.asset_path`
is evaluated at module import time, so a raise on a missing vendor checkout
would break `import robot_description.robots.unitree_g1` itself -- exactly
what LearningRLlab's lint/config-resolution tests do with no simulator running, on
any machine that hasn't run `gitman install` yet. The loud failure on a
genuinely missing source file belongs in `validate.py` (Level A) instead;
`urdf_path()` only warns.

`build_dir()`/`robot_build_dir()` name the directory `UrdfFileCfg.usd_dir` is
pointed at, so Isaac Lab's own content-hash-keyed conversion cache persists
across runs instead of using a fresh random `/tmp` directory every spawn.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Repo root: this file lives at <repo_root>/robot_description/paths.py -- the
# package is nested (see repo CLAUDE.md / spec F6: no [tool.setuptools]
# package-dir remap), so the repo root is one level ABOVE this file's package,
# i.e. parents[1] of this file, not parents[0].
REPO_ROOT: Path = Path(__file__).resolve().parents[1]
PACKAGE_ROOT: Path = Path(__file__).resolve().parent

VENDOR_DIR: Path = REPO_ROOT / "vendor"


def build_dir() -> Path:
    """Root of the on-the-fly USD conversion cache (`UrdfFileCfg.usd_dir`).

    `ROBOT_DESCRIPTION_BUILD` if set (and non-empty) and not the sentinel
    empty string, else `<repo_root>/build`. That pair -- the env var name and
    the default -- is the published interface; do not change either.
    """
    env_value = os.environ.get("ROBOT_DESCRIPTION_BUILD")
    if env_value:
        return Path(env_value).resolve()
    return (REPO_ROOT / "build").resolve()


def robot_build_dir(robot: str) -> Path:
    """Per-robot subdirectory under `build_dir()`, e.g. `build/unitree_g1/`.

    Passed as `UrdfFileCfg.usd_dir` so repeated spawns reuse the same
    directory and Isaac Lab's built-in content-hash cache (writes
    `.asset_hash` under `usd_dir`, skips reconversion when the source file
    and converter config are unchanged) actually has something stable to
    compare against.
    """
    return build_dir() / robot


def _robot_yaml_path(robot: str) -> Path:
    return PACKAGE_ROOT / "robots" / robot / "robot.yaml"


def urdf_path(robot: str) -> str:
    """Resolve the source URDF/MJCF path for `robot`, as an absolute string.

    TOTAL: never raises. Reads `source.vendor` + `source.path` out of
    `robots/<robot>/robot.yaml` and returns `str(VENDOR_DIR / vendor / path)`.
    If `robot.yaml` can't be read, falls back to a guessed
    `VENDOR_DIR/<robot>.urdf` path and logs a warning; the file need not
    exist yet (e.g. before `gitman install`) since this runs at import time.
    """
    robot_yaml_path = _robot_yaml_path(robot)
    try:
        # Imported here, not at module scope: robot_schema.py has no Isaac
        # Lab dependency (same as this module), but keeping the import local
        # avoids any import-order surprise for callers that only need
        # build_dir()/robot_build_dir() and never touch robot.yaml.
        from robot_description.tools.robot_schema import RobotYamlError, load_robot_yaml

        cfg = load_robot_yaml(robot_yaml_path)
    except (RobotYamlError, OSError) as exc:
        logger.warning(
            "robot_description: could not read %s to resolve '%s's source URDF (%s). "
            "Returning a guessed path, which does not exist. Run `gitman install "
            "--skip-changes` and check robots/%s/robot.yaml.",
            robot_yaml_path,
            robot,
            exc,
            robot,
        )
        return str((VENDOR_DIR / f"{robot}.urdf").resolve())

    return str((VENDOR_DIR / cfg.source.vendor / cfg.source.path).resolve())


def mjcf_path(robot: str) -> str:
    """Resolve the source MuJoCo XML path for `robot`, as an absolute string.

    NOT total, unlike `urdf_path()`: nothing evaluates this at module import
    time -- the only consumer is LearningRLsim, which calls it from inside its own
    entrypoint -- so a missing declaration should raise here rather than hand
    back a guessed path that fails later as a MuJoCo parse error.

    Reads `source.mjcf_path` (or `source.path`, when `source.format` is
    already `mjcf`) out of `robots/<robot>/robot.yaml`. Deliberately a
    separate key rather than a suffix swap on `urdf_path()`: for
    `roboparty_rpo` the two files live in unrelated subtrees
    (`urdf/rpo.urdf` vs `mjcf/rpo.xml`), so `.with_suffix(".xml")` would be
    right for the G1 and silently wrong there.

    The returned path is NOT checked for existence -- the caller is expected
    to `is_file()` it and report in its own terms (see LearningRLsim's model.py).
    """
    from robot_description.tools.robot_schema import RobotYamlError, load_robot_yaml

    robot_yaml_path = _robot_yaml_path(robot)
    if not robot_yaml_path.is_file():
        available = sorted(p.parent.name for p in (PACKAGE_ROOT / "robots").glob("*/robot.yaml"))
        raise RobotYamlError(
            f"robot_description has no robot named {robot!r} (looked for {robot_yaml_path}). "
            f"Available: {available}"
        )

    cfg = load_robot_yaml(robot_yaml_path)
    relative = cfg.source.path if cfg.source.format == "mjcf" else cfg.source.mjcf_path
    if relative is None:
        raise RobotYamlError(
            f"{robot_yaml_path}: robot {robot!r} declares no MuJoCo model. Add "
            f"`source.mjcf_path: <path under vendor/{cfg.source.vendor}/>` to its "
            f"robot.yaml, or vendor an MJCF for it first."
        )

    return str((VENDOR_DIR / cfg.source.vendor / relative).resolve())
