"""
@file   utils.py
@author Nianchen Deng, Shanghai AI Lab
@brief  
"""
import torch
import open3d as o3d
import open3d.visualization.gui as gui

from nr3d_lib.config import ConfigDict
from nr3d_lib.models.attributes import *


def initialize_visualizer(title: str, width: int = 1024, height: int = 768, has_sidebar: bool = False):

    app = gui.Application.instance
    app.initialize()
    w = app.create_window(title, width, height)
    em = w.theme.font_size

    if has_sidebar:
        sidebar = gui.Vert(.5 * em, gui.Margins(em, em, em, em))
        w.add_child(sidebar)

    # Create scene widget
    widget3d = gui.SceneWidget()
    widget3d.scene = o3d.visualization.rendering.Open3DScene(w.renderer)
    w.add_child(widget3d)

    # Register layout callback
    def on_layout_callback(layout_context):
        win_rect = w.content_rect
        if has_sidebar:
            sidebar_preferred_size = sidebar.calc_preferred_size(
                layout_context, gui.Widget.Constraints())
            sidebar_width = max(sidebar_preferred_size.width, 15 * em)
            sidebar.frame = gui.Rect(win_rect.x, win_rect.y, sidebar_width, win_rect.height)
            widget3d.frame = gui.Rect(win_rect.x + sidebar_width, win_rect.y,
                                      win_rect.width - sidebar_width, win_rect.height)
        else:
            widget3d.frame = win_rect

    w.set_on_layout(on_layout_callback)

    ret = ConfigDict(app=app, window=w, widget3d=widget3d)
    if has_sidebar:
        ret["sidebar"] = sidebar
    return ret


def create_voxels_geometry(min_corners: torch.Tensor, max_corners: torch.Tensor,
                           colors: Union[torch.Tensor, List[float]], mesh: bool = False):
    n_voxels = min_corners.shape[0]
    weights = min_corners.new_tensor([
        [0., 0., 0.], [0., 0., 1.], [0., 1., 0.], [0., 1., 1.],
        [1., 0., 0.], [1., 0., 1.], [1., 1., 0.], [1., 1., 1.]
    ])  # (8, 3)
    corners = min_corners[:, None] * (1. - weights) + max_corners[:, None] * weights
    corners = corners.reshape(-1, 3)
    offsets = torch.arange(0, n_voxels * 8, 8, device=corners.device)[:, None, None]
    if mesh:
        voxel_triangles = min_corners.new_tensor([
            [0, 1, 2], [5, 4, 7], [0, 4, 1], [6, 2, 7], [0, 2, 4], [3, 1, 7],
            [3, 2, 1], [6, 7, 4], [5, 1, 4], [3, 7, 2], [6, 4, 2], [5, 7, 1]
        ], dtype=torch.long)  # (12, 3)
        triangles = voxel_triangles + offsets  # (N, 12, 3)
        triangles = triangles.reshape(-1, 3)
        geo = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(corners.cpu().numpy()),
            o3d.utility.Vector3iVector(triangles.cpu().numpy()))
        geo.compute_triangle_normals()
    else:
        voxel_lines = min_corners.new_tensor([
            (0, 1), (1, 3), (3, 2), (2, 0),
            (4, 5), (5, 7), (7, 6), (6, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ], dtype=torch.long)  # (12, 2)
        lines = voxel_lines + offsets  # (N, 12, 2)
        lines = lines.reshape(-1, 2)
        geo = o3d.geometry.LineSet(o3d.utility.Vector3dVector(corners.cpu().numpy()),
                                   o3d.utility.Vector2iVector(lines.cpu().numpy()))
    if isinstance(colors, torch.Tensor):
        colors = colors[:, None].expand(-1, 8, -1).reshape(-1, 3)
        geo.colors = colors.cpu().numpy()
    else:
        geo.paint_uniform_color(colors)
    return geo
