"""Spawn function for `RobotUrdfFileCfg`: upstream's URDF spawn, made correct.

Referenced by import string from `urdf_common.py` and resolved only at spawn time, after Kit is up,
so nothing here imports Isaac Lab or `pxr` at module scope.
"""

from __future__ import annotations

import os
import shutil


def spawn_urdf_inertia_fixed(prim_path, cfg, translation=None, orientation=None, **kwargs):
    from isaaclab.sim import converters
    from isaaclab.sim.spawners.from_files import spawn_from_urdf

    from robot_description.tools.usd_inertia_fix import fix_principal_axes

    # Converts on a cache miss, so the USD exists to patch; upstream's spawn below hits the cache.
    loader = converters.UrdfConverter(cfg)

    # A fresh conversion lands in `<name>_N/`, but every later cache hit resolves to `<name>/`,
    # the oldest conversion. Promote the fresh one so the cached path is always the current asset.
    base = os.path.join(loader.usd_dir, cfg.usd_file_name)
    if os.path.abspath(loader.usd_path) != os.path.abspath(base):
        shutil.rmtree(os.path.dirname(base), ignore_errors=True)
        os.rename(os.path.dirname(loader.usd_path), os.path.dirname(base))

    if cfg.fix_principal_axes:
        fix_principal_axes(base)
    return spawn_from_urdf(prim_path, cfg, translation, orientation, **kwargs)
