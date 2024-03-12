import os
import torch
import functools
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
from operator import itemgetter
from tqdm import trange
from collections import defaultdict

from nr3d_lib.fmt import log
from nr3d_lib.config import ConfigDict
from nr3d_lib.models.attributes import *
from nr3d_lib.models.utils import batchify_query
from nr3d_lib.models.spatial import AABBSpace, ForestBlockSpace
from nr3d_lib.plot import create_camera_frustum_o3d

from app.resources import load_scenes_and_assets
from app.resources.observers import Lidar, Camera
from app.visible_grid import VisibleGrid, voxel_coords_to_voxel_indices
from app.visualizer.utils import initialize_visualizer, create_voxels_geometry

target_class_name = 'Street'

def create_visible_grid_geometry(visible_grid: VisibleGrid, colors = [.3, .7, 1.],
                                 mesh: bool = True, only_blidx = None):
    if only_blidx is not None:
        voxel_aabb = [
            visible_grid.get_voxel_aabb_in_world(voxel_indices, blidx)
            for blidx, voxel_indices in visible_grid.voxels_in_block.items() if blidx == only_blidx
        ]
    else:
        voxel_aabb = [
            visible_grid.get_voxel_aabb_in_world(voxel_indices, blidx)
            for blidx, voxel_indices in visible_grid.voxels_in_block.items()
        ]
    voxel_aabb = [
        torch.cat([item[0] for item in voxel_aabb], 0),
        torch.cat([item[1] for item in voxel_aabb], 0)
    ]
    return create_voxels_geometry(*voxel_aabb, colors, mesh)


def visualize_visible_grid(visible_grid: VisibleGrid, title: str):
    visualizer = initialize_visualizer(title, has_sidebar=True)

    line_mat = o3d.visualization.rendering.MaterialRecord()
    line_mat.line_width = 1.5
    line_mat.shader = "unlitLine"
    unlit_mat = o3d.visualization.rendering.MaterialRecord()
    unlit_mat.point_size = 1.5
    unlit_mat.shader = "defaultUnlit"
    lit_mat = o3d.visualization.rendering.MaterialRecord()
    lit_mat.shader = "defaultLit"

    visualizer.widget3d.scene.add_geometry(f"{target_class_name}.visible_grid",
                                           create_visible_grid_geometry(visible_grid), lit_mat)
    scene_aabb = visualizer.widget3d.scene.bounding_box
    visualizer.widget3d.setup_camera(60.0, scene_aabb, scene_aabb.get_center())
    visualizer.app.run()


def main(args: ConfigDict, device: torch.device = torch.device('cuda')):
    scene_bank, *_ = load_scenes_and_assets(**args, class_name_list=[target_class_name])
    scene = scene_bank[0]
    bg_node = scene.get_drawable_groups_by_class_name(target_class_name)[0]
    space = bg_node.model.space

    line_mat = o3d.visualization.rendering.MaterialRecord()
    line_mat.line_width = 1
    line_mat.shader = "unlitLine"
    unlit_mat = o3d.visualization.rendering.MaterialRecord()
    unlit_mat.shader = "defaultUnlit"
    lit_mat = o3d.visualization.rendering.MaterialRecord()
    lit_mat.shader = "defaultLit"

    visible_grid = VisibleGrid.load(args.visible_grid, space)
    geo_visible_grid1 = create_visible_grid_geometry(visible_grid)
    
    visible_grid.build_accel().postprocess(args.postprocess)
    geo_visible_grid2 = create_visible_grid_geometry(visible_grid, [1., .5, 0.])
    geo_visible_grid3 = create_visible_grid_geometry(visible_grid, [0., 0., 0.], False)

    visualizer = initialize_visualizer("Visualize visible grid", has_sidebar=True)
    #visualizer.widget3d.scene.add_geometry(f"{target_class_name}.visible_grid1", geo_visible_grid1, lit_mat)
    visualizer.widget3d.scene.add_geometry(f"{target_class_name}.visible_grid2", geo_visible_grid2, lit_mat)
    #visualizer.widget3d.scene.add_geometry(f"{target_class_name}.visible_grid3", geo_visible_grid3, line_mat)

    highlight_voxel_aabb = visible_grid.get_voxel_aabb_in_world(
        voxel_coords_to_voxel_indices(torch.tensor([[2,2,2]], device=device), visible_grid.grid_size),
        19)
    geo_highlight_voxel_aabb = create_voxels_geometry(*highlight_voxel_aabb, [1., 0., 0.], True)
    #visualizer.widget3d.scene.add_geometry(f"{target_class_name}.highlight_voxel", geo_highlight_voxel_aabb, lit_mat)
    if isinstance(space, AABBSpace):
        num_blocks = 1
    elif isinstance(space, ForestBlockSpace):
        num_blocks = space.block_ks.shape[0]
    
    for i in range(num_blocks):
        grid_aabb = visible_grid.get_grid_aabb_in_world(i)
        grid_aabb = [item.cpu().numpy() for item in grid_aabb]
        geo_grid_aabb = o3d.geometry.AxisAlignedBoundingBox(*grid_aabb)
        geo_grid_aabb.color = [0.0, 1.0, 0.5]
        visualizer.widget3d.scene.add_geometry(f"{target_class_name}.grid_{i}", geo_grid_aabb, line_mat)

    scene_aabb = visualizer.widget3d.scene.bounding_box
    visualizer.widget3d.setup_camera(60.0, scene_aabb, scene_aabb.get_center())
    visualizer.app.run()

if __name__ == "__main__":
    from nr3d_lib.config import BaseConfig

    bc = BaseConfig()
    bc.parser.add_argument("--load_pt", type=str, default=None, help="Typically unnecessary as the final or latest ckpt is loaded automatically. \n"\
        "Only specify the ckpt file path if indeed a non-final or non-latest ckpt needs to be loaded.")
    bc.parser.add_argument("--visible_grid", type=str, required=True)
    bc.parser.add_argument("--postprocess", type=str, default="close")
    main(bc.parse())
