"""Config for the motion command, apart from ``commands.py`` so the env cfg imports before Kit starts (F9)."""

from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.configclass import configclass


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    # a string, so MotionCommand (-> Articulation -> pxr) isn't imported before Kit launches
    class_type: type | str = "{DIR}.commands:MotionCommand"

    asset_name: str = MISSING

    motion_file: str = MISSING
    """``.npz`` with ``fps``, ``joint_pos``, ``joint_vel`` and ``body_{pos,quat,lin_vel,ang_vel}_w``."""
    motion_body_names: list[str] = MISSING
    """Body order of the ``body_*_w`` arrays in ``motion_file``."""
    motion_joint_names: list[str] = MISSING
    """Joint order of the ``joint_*`` arrays in ``motion_file``."""
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING
    """The bodies the rewards, observations and debug markers track."""

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
