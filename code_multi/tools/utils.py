"""
@file   utils.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Utilities for scene rendering.
"""

import cv2
import numpy as np
from typing import Dict, Tuple

import torch

from app.resources import SceneNode
from app.resources.observers import Camera
from nr3d_lib.plot.plot_basic import choose_opposite_color

BBOX_LINES = [[1, 2], [1, 3], [2, 4], [3, 4], [5, 6], [
    5, 7], [6, 8], [7, 8], [1, 5], [2, 6], [3, 7], [4, 8]]

# NOTE: [scale * default_cube] is objects' annotated real size.
#       [obj.space.aabb] is always larger than this value; (helpful for volume rendering's sampling)
#       For example, if aabb is set with the default bounding_size=2.0, then aabb is in range [-1,1]
DEFAULT_CUBE = torch.tensor(
    [
        [-0.5, -0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [-0.5, 0.5, 0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5]
    ], dtype=torch.float
)


def proj(mat_3x3: torch.Tensor, xyz: torch.Tensor):
    uvd = (mat_3x3 * xyz.unsqueeze(-2)).sum(-1)
    uv = uvd[..., 0:2] / torch.abs(uvd[..., 2:3]).clamp(1e-5)
    return uv[...,0], uv[...,1], uvd[..., 2]


def draw_box(
    img: np.ndarray, 
    obj: SceneNode, 
    cam: Camera, 
    oind: int = None, 
    cind: int = None, 
    # Draw configs
    inplace=True,
    thickness: int = 2,
    fontscale: float = 1.,
    # Optional colormap configs
    nolabel=False,
    use_class_cmap: bool = True,
    instance_id_map: Dict[str, int] = None,
    instance_cmap: np.ndarray = None,
    classname_map: Dict[str, int] = None,
    class_cmap: np.ndarray = None, 
    ):
    H, W, *_ = img.shape
    if not inplace:
        img = img.copy()

    oid = obj.id
    clsname = obj.class_name
    
    ocolor = (255, 0, 0)
    if instance_cmap is not None:
        oind = oind if oind is not None else instance_id_map[oid]
        ocolor = instance_cmap[oind].tolist()

    clscolor = (255, 0, 0)
    if class_cmap is not None:
        clsind = cind if cind is not None else classname_map[clsname]
        clscolor = class_cmap[clsind].tolist()

    oid = oid[:4]

    default_cube = DEFAULT_CUBE.clone().to(cam.device)
    if hasattr(obj, "extent"):
        cube_pts_in_cam = cam.world_transform(
            obj.world_transform(obj.extent.tensor * default_cube), inv=True) # (8, 3)
    else:
        cube_pts_in_cam = cam.world_transform(
            obj.world_transform(obj.scale.vec_3() * default_cube), inv=True) # (8, 3)
    cube_edge_indices = cube_pts_in_cam.new_tensor(BBOX_LINES, dtype=torch.long) - 1 # (12, 2)
    cube_edge_pts_in_cam: torch.Tensor = cube_pts_in_cam[cube_edge_indices].cpu() # (12, 2, 3)
    for i in range(12):
        p1 = cube_edge_pts_in_cam[i, 0].clone()
        p2 = cube_edge_pts_in_cam[i, 1].clone()
        if p1[2] > p2[2]:
            p1 = cube_edge_pts_in_cam[i, 1].clone()
            p2 = cube_edge_pts_in_cam[i, 0].clone()
        if p1[2] < 0.01 and p2[2] >= 0.01:
            k = (0.01 - p1[2]) / (p2[2] - p1[2])
            p1 += k * (p2 - p1)
        cube_edge_pts_in_cam[i, 0] = p1
        cube_edge_pts_in_cam[i, 1] = p2
    
    #print(cube_pts_in_cam, cube_edge_pts_in_cam)
    u, v, d = proj(cam.intr.mat_3x3(), cube_pts_in_cam)
    u1, v1, _ = proj(cam.intr.mat_3x3(), cube_edge_pts_in_cam.to(cam.device))
    d1 = cube_edge_pts_in_cam[..., 2]
    #print(torch.stack([u, v, d], -1))
    if (d1 < 0.01).all() or (((u1 < 0) | (u1 >= W)) | ((v1 < 0) | (v1 >= H))).all():
        return None
    u = u.long().cpu().numpy()
    v = v.long().cpu().numpy()
    u1 = u1.long().cpu().numpy()
    v1 = v1.long().cpu().numpy()

    olabel = f"id: {oid}"
    olabelsize = cv2.getTextSize(olabel, cv2.FONT_HERSHEY_SIMPLEX, fontscale, thickness)[0]
    clabel = f"cls: {clsname}"
    clabelsize = cv2.getTextSize(clabel, cv2.FONT_HERSHEY_SIMPLEX, fontscale, thickness)[0]
    label_w, label_h = max(olabelsize[0], clabelsize[0]) + 8, olabelsize[1] + clabelsize[1]

    # label_color = ocolor
    label_color = clscolor if use_class_cmap else ocolor
    # font_color = max(label_color) + min(label_color) - label_color  # Complementary color
    font_color = choose_opposite_color(label_color)

    for i in range(12):
        if d1[i, 0] < 0.01 and d1[i, 1] < 0.01:
            continue
        p1 = u1[i, 0], v1[i, 0]
        p2 = u1[i, 1], v1[i, 1]
        cv2.line(img, p1, p2, label_color, thickness)

    vert_ind = 7  # Label on which box vertice
    if not nolabel:
        cv2.rectangle(img, (u[vert_ind], v[vert_ind] - label_h - thickness - 15),
                      (u[vert_ind] + label_w, v[vert_ind]), label_color, thickness=-1)
        cv2.putText(img, olabel, (u[vert_ind], v[vert_ind] - clabelsize[1] - thickness - 12),
                    cv2.FONT_HERSHEY_DUPLEX, fontscale, font_color, thickness, cv2.LINE_AA)
        cv2.putText(img, clabel, (u[vert_ind], v[vert_ind] - thickness - 1),
                    cv2.FONT_HERSHEY_DUPLEX, fontscale, font_color, thickness, cv2.LINE_AA)

    return img


def plot_box3d(
    obj: SceneNode, 
    cam: Camera, 
    oind: int = None, 
    cind: int = None,
    thickness: int = 2,
    use_class_cmap: bool = True,
    instance_id_map: Dict[str, int] = ...,
    instance_cmap: np.ndarray = ...,
    classname_map: Dict[str, int] = ...,
    class_cmap: np.ndarray = ...):
    from mayavi import mlab

    oid = obj.id
    oind = oind if oind is not None else instance_id_map[oid]
    ocolor = instance_cmap[oind]

    clsname = obj.class_name
    clsind = cind if cind is not None else classname_map[clsname]
    clscolor = class_cmap[clsind]

    default_cube = DEFAULT_CUBE.clone().to(cam.device)
    if hasattr(obj, "extent"):
        cube_pts_in_cam = cam.world_transform(
            obj.world_transform(obj.extent.tensor * default_cube), inv=True) # (8, 3)
    else:
        cube_pts_in_cam = cam.world_transform(
            obj.world_transform(obj.scale.vec_3() * default_cube), inv=True) # (8, 3)
    cube_edge_indices = cube_pts_in_cam.new_tensor(BBOX_LINES, dtype=torch.long) - 1 # (12, 2)
    cube_edge_pts_in_cam: torch.Tensor = cube_pts_in_cam[cube_edge_indices].cpu() # (12, 2, 3)
    
    anno_color = clscolor if use_class_cmap else ocolor
    anno_color = tuple(anno_color.astype(np.float32) / 255.)

    for i in range(12):
        x1, y1, z1 = cube_edge_pts_in_cam[i, 0]
        x2, y2, z2 = cube_edge_pts_in_cam[i, 1]
        mlab.plot3d((x1, x2), (y1, y2), (z1, z2), color=anno_color, line_width=thickness)
