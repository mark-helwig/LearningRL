"""Configuration for :class:`HandPosCommand`.

Kept apart from ``commands.py``, as upstream keeps ``commands_cfg.py`` apart from its command
implementations: the env cfg imports this module before Kit starts, so it may not import
``CommandTerm``/``VisualizationMarkers`` (which reach ``pxr``, F9). ``class_type`` is a string
for the same reason.
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils.configclass import configclass

# index 0 draws an active hand, index 1 hides an inactive one (see `HandPosCommand._debug_vis_callback`)
_GOAL_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/Command/goal_hand_pos",
    markers={
        "hand": sim_utils.SphereCfg(
            radius=0.03, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))
        ),
        "hidden": sim_utils.SphereCfg(radius=0.001, visible=False),
    },
)
_CURRENT_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/Command/curr_hand_pos",
    markers={
        "hand": sim_utils.SphereCfg(
            radius=0.025, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0))
        ),
        "hidden": sim_utils.SphereCfg(radius=0.001, visible=False),
    },
)


@configclass
class HandPosCommandCfg(CommandTermCfg):
    """Configuration for the two-hand position command generator."""

    class_type: type | str = "LearningRLlab.tasks.locomanipulation.g1.mdp.commands:HandPosCommand"

    asset_name: str = MISSING
    """Name of the robot in the scene."""

    body_names: list[str] = MISSING
    """The two hand bodies, left first then right. Their origins are the tracked points."""

    rel_standing_envs: float = 0.0
    """Probability that a resample leaves both hands idle (mask and targets all zero)."""

    velocity_command_name: str = "base_velocity"
    """The base velocity command whose one-step displacement is added to the hand targets."""

    mode: Literal["random", "left", "right", "both", "symmetric"] = "random"
    """Which hands are active. ``"random"`` draws one of the other four per env at every resample;
    ``"symmetric"`` activates both hands with the right target mirrored from the left."""

    @configclass
    class Ranges:
        """Uniform ranges for the targets, as absolute positions in the robot's yaw-aligned root frame (m)."""

        pos_x: tuple[float, float] = MISSING
        """Forward reach, shared by both hands."""

        pos_y: tuple[float, float] = MISSING
        """Lateral reach for any hand without its own range below."""

        pos_y_left: tuple[float, float] | None = None
        """Lateral reach for the left hand. ``None`` falls back to ``pos_y``."""

        pos_y_right: tuple[float, float] | None = None
        """Lateral reach for the right hand. ``None`` falls back to ``pos_y``."""

        pos_z: tuple[float, float] = MISSING
        """Height above the root, shared by both hands."""

    ranges: Ranges = MISSING
    """Distribution ranges for the targets."""

    goal_pos_visualizer_cfg: VisualizationMarkersCfg = _GOAL_MARKER_CFG
    """Goal markers: green spheres."""

    current_pos_visualizer_cfg: VisualizationMarkersCfg = _CURRENT_MARKER_CFG
    """Current-hand markers: blue spheres."""
