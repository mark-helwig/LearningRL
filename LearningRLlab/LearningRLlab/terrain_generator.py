"""Terrain generators, referenced by string from ``scenes.py`` so they load only after Kit launches."""

import numpy as np

from isaaclab.terrains import TerrainGenerator

from LearningRLlab.scenes import BoxesTerrainCfg, boxes_mesh


class BoxesTerrainGenerator(TerrainGenerator):
    """Terrain generator whose border ring is boxes at the hardest height, instead of upstream's flat plane."""

    def _add_terrain_border(self):
        boxes_cfg = next(cfg for cfg in self.cfg.sub_terrains.values() if isinstance(cfg, BoxesTerrainCfg))
        width, border = boxes_cfg.box_width, self.cfg.border_width
        num_border = round(border / width)
        if not np.isclose(num_border * width, border):
            raise ValueError(f"border_width {border} is not a multiple of box_width {width}.")
        # sub-terrains span [0, num_rows * size_x] x [0, num_cols * size_y] here; the ring surrounds that
        inner_x, inner_y = self.cfg.num_rows * self.cfg.size[0], self.cfg.num_cols * self.cfg.size[1]
        xs = (np.arange(-num_border, round(inner_x / width) + num_border) + 0.5) * width
        ys = (np.arange(-num_border, round(inner_y / width) + num_border) + 0.5) * width
        xx, yy = np.meshgrid(xs, ys)
        ring = (xx < 0) | (xx > inner_x) | (yy < 0) | (yy > inner_y)
        if ring.any():
            self.terrain_meshes.append(boxes_mesh(xx[ring], yy[ring], width, max(boxes_cfg.height_range)))
