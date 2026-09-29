"""Reusable scene pieces for LearningRLlab tasks.

Subclass a scene cfg and add the task's robot/sensors, or pull individual pieces into a custom scene.
"""

from dataclasses import MISSING

import numpy as np
import trimesh

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

##
# Sub-terrains
##


@configclass
class BoxesTerrainCfg(SubTerrainBaseCfg):
    """Configuration for :func:`boxes_terrain`."""

    function: str = "{DIR}.scenes:boxes_terrain"

    box_width: float = MISSING
    """Side length of each square box (in m). Must divide the sub-terrain size."""

    height_range: tuple[float, float] = MISSING
    """Max box offset above/below the ground (in m) at difficulty 0 and 1."""


def boxes_mesh(x: np.ndarray, y: np.ndarray, box_width: float, max_height: float) -> trimesh.Trimesh:
    """One 1 m tall box per (x, y) center, its top face at a uniform random height in +-max_height."""
    template = trimesh.creation.box((box_width, box_width, 1.0))
    heights = np.random.uniform(-max_height, max_height, x.size)
    centers = np.stack((x.ravel(), y.ravel(), heights - 0.5), axis=1)
    vertices = (template.vertices[None] + centers[:, None]).reshape(-1, 3)
    faces = (template.faces[None] + len(template.vertices) * np.arange(len(centers))[:, None, None]).reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    # gray shade by height, so +-5 cm steps are readable in the viewer
    shade = np.repeat(np.interp(heights, (-max_height, max_height), (70, 180)), len(template.vertices))
    mesh.visual.vertex_colors = np.stack((shade, shade, shade, np.full_like(shade, 255)), axis=1).astype(np.uint8)
    return mesh


def boxes_terrain(difficulty: float, cfg: BoxesTerrainCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Grid of square boxes, each raised or lowered by a uniform random height; no spawn platform."""
    max_height = cfg.height_range[0] + difficulty * (cfg.height_range[1] - cfg.height_range[0])
    num_x, num_y = (round(size / cfg.box_width) for size in cfg.size)
    if not np.allclose((num_x * cfg.box_width, num_y * cfg.box_width), cfg.size):
        raise ValueError(f"Sub-terrain size {cfg.size} is not a multiple of box_width {cfg.box_width}.")

    xx, yy = np.meshgrid((np.arange(num_x) + 0.5) * cfg.box_width, (np.arange(num_y) + 0.5) * cfg.box_width)
    mesh = boxes_mesh(xx, yy, cfg.box_width, max_height)
    # spawn origin at the tallest possible box top, so robots never start inside one
    origin = np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1], max_height])
    return [mesh], origin


##
# Terrains
##

GROUND_MATERIAL_CFG = sim_utils.RigidBodyMaterialCfg(
    friction_combine_mode="multiply",
    restitution_combine_mode="multiply",
    static_friction=1.0,
    dynamic_friction=1.0,
)

FLAT_TERRAIN_CFG = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="plane",
    terrain_generator=None,
    max_init_terrain_level=None,
    collision_group=-1,
    physics_material=GROUND_MATERIAL_CFG,
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2)),
    debug_vis=False,
)

# 0.5 m boxes at a fixed +-5 cm; small tiles give 1024 spawn origins spread over a 64 x 64 m field
BOXES_TERRAIN_CFG = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
        # a string, so TerrainGenerator isn't imported before Kit launches (task cfgs are)
        class_type="LearningRLlab.terrain_generator:BoxesTerrainGenerator",
        size=(2.0, 2.0),
        border_width=20.0,
        num_rows=32,
        num_cols=32,
        use_cache=False,
        sub_terrains={"boxes": BoxesTerrainCfg(proportion=1.0, box_width=0.5, height_range=(0.05, 0.05))},
    ),
    max_init_terrain_level=None,
    collision_group=-1,
    physics_material=GROUND_MATERIAL_CFG,
    # None so the per-box vertex colors show instead of a bound material
    visual_material=None,
    debug_vis=False,
)

##
# Lights
##

SKY_LIGHT_CFG = AssetBaseCfg(
    prim_path="/World/skyLight",
    spawn=sim_utils.DomeLightCfg(
        intensity=750.0,
        texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
    ),
)

##
# Scenes
##


@configclass
class FlatSceneCfg(InteractiveSceneCfg):
    """Flat ground plane under a sky dome light; subclass it and add the robot."""

    terrain = FLAT_TERRAIN_CFG
    sky_light = SKY_LIGHT_CFG


@configclass
class BoxesSceneCfg(InteractiveSceneCfg):
    """``BOXES_TERRAIN_CFG`` under a sky dome light; subclass it and add the robot."""

    terrain = BOXES_TERRAIN_CFG
    sky_light = SKY_LIGHT_CFG
