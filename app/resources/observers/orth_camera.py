"""
@file   orth_camera.py
@author Nianchen Deng, Shanghai AI Lab
@brief  Orthogonal camera
"""

__all__ = [
    "OrthogonalCamera"
]

from typing import NamedTuple, List, Tuple

import torch
import torch.nn.functional as F

from nr3d_lib.models.attributes import OrthoCameraIntrinsics
from nr3d_lib.graphics.cameras import pinhole_view_frustum

from app.resources import SceneNode
from app.resources.observers.cameras import Camera, CameraBase, namedtuple_mask_nuvd, namedtuple_mask_niuvd

class OrthogonalCamera(Camera):
    def __init__(self, unique_id: str, scene=..., device=None, dtype=torch.float):
        super().__init__(unique_id=unique_id, scene=scene, device=device, dtype=dtype)
        # Additional attributes
        self.intr = OrthoCameraIntrinsics(device=device)

    def get_all_rays(self, return_ts=False) -> List[torch.Tensor]:
        """
        - support single frame:     ✓
        - support batched frames:   o      should only be used when H or W is the same across differerent batches.
        """
        H, W, device = self.intr.H, self.intr.W, self.device
        i, j = torch.meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device), indexing='ij')
        i, j = i.t().reshape(H*W)+0.5, j.t().reshape(H*W)+0.5 # Pixel centers
        rays_o = self.world_transform(self.intr.lift(i, j, torch.full_like(i, self.near or 0.)))
        rays_d = self.world_transform.mat_3x4()[..., :, 2].expand_as(rays_o)
        
        ret = [rays_o, rays_d]
        return ret

    def build_view_frustum(self):
        pass

    # def check_spheres_inside_frustum(self, sphere_center_radius: torch.Tensor, holistic_body=False):
    #     pass

    @torch.no_grad()
    def get_view_frustum_pts(self, near=None, far=None):
        """
        - support single frame:     ✓
        - support batched frames:   ✓      `...` means arbitary prefix-dims (i.e. `self.i_prefix`)
        """
        near_clip = self.near or near or 0.
        far_clip = self.far or far or 100.
        _W, _H = self.intr.wh().movedim(-1, 0)
        _0, _near, _far = _W.new_zeros(_W.shape), _W.new_full(_W.shape, near_clip), \
            _W.new_full(_W.shape, far_clip)
        # [..., 8]
        u, v, d = torch.stack([
            torch.stack([_0, _0, _near], 0),
            torch.stack([_W, _0, _near], 0),
            torch.stack([_W, _H, _near], 0),
            torch.stack([_0, _H, _near], 0),
            torch.stack([_0, _0, _far], 0),
            torch.stack([_W, _0, _far], 0),
            torch.stack([_W, _H, _far], 0),
            torch.stack([_0, _H, _far], 0),
        ], dim=-1)
        # [..., 8, 3]
        pts = self.world_transform(self.intr.lift(u, v, d)[..., :3])
        return pts

    # @profile
    def filter_drawable_groups(self, drawables: List[SceneNode], draw_self=False):
        """
        Frustum culling to filter drawables by checking whether drawables' bounding spheres has intersection with view frustums.
        - support single frame:     ✓
        - support batched frames:   ✓      `...` means arbitary prefix-dims (i.e. `self.i_prefix`)
        """
        # For now, we don't consider observer's self shading
        if self.model is not None and not draw_self:
            drawables = list(filter(lambda obj: obj.id != self.id, drawables))

        if len(collected := [[obj, obj.model_bounding_sphere, obj.i_valid_flags] for obj in drawables if obj.model_bounding_sphere is not None]) > 0:
            drawables_with_bound, model_bounding_spheres, obj_valids = zip(*collected)
        else:
            drawables_with_bound = []
        drawables_no_bound = [obj for obj in drawables if obj.model_bounding_sphere is None]

        # No frustum Culling

        return list(drawables_with_bound) + drawables_no_bound
