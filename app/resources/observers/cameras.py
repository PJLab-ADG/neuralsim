"""
@file   cameras.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Special kinds of SceneNode: camera observers.
        Support 3 kinds of usage:
        - single camera             `Camera` frozen_at single frame
        - single camera             `Camera` frozen_at batched multi frames
        - bundled multiple camera   `MultiCamBundle` frozen_at single frame
"""

__all__ = [
    "namedtuple_mask_nuvd", 
    "namedtuple_mask_niuvd", 
    "Camera", 
    "MultiCamBundle", 
    "CAMERA_CLASS_NAMES"
]

from typing import NamedTuple, List, Tuple, Any

import torch
import torch.nn.functional as F

from nr3d_lib.utils import is_scalar
from nr3d_lib.models.attributes import *
from nr3d_lib.render.cameras import pinhole_lift
from nr3d_lib.geometry import sphere_inside_planes

from app.resources.nodes import SceneNode

CAMERA_CLASS_NAMES = ['Camera']

class namedtuple_mask_nuvd(NamedTuple):
    mask: torch.Tensor
    n: int
    u: torch.Tensor
    v: torch.Tensor
    d: torch.Tensor

class namedtuple_mask_niuvd(NamedTuple):
    mask: torch.Tensor
    n: int
    i: torch.Tensor
    u: torch.Tensor
    v: torch.Tensor
    d: torch.Tensor

class Camera(SceneNode):
    def __init__(self, unique_id: str, scene=..., device=torch.device('cuda'), dtype=torch.float):
        super().__init__(unique_id=unique_id, class_name='Camera', scene=scene, device=device, dtype=dtype)
        # Additional attributes
        self.intr = CameraBase(device=device)
        self.exposure = Scalar(device=device)
        self.near = None
        self.far = None
        self.frustum = None # [..., num_frustum_planes, 4(n3+d1)]
        # self.min_radial_dist = None
        # self.max_radial_dist = None

    # @profile
    def update(self):
        SceneNode.update(self)
        # Update view frustum
        with torch.no_grad():
            self.build_view_frustum()
    
    @torch.no_grad()
    # @profile
    def get_view_frustum(self, near: float=None, far: float=None) -> torch.Tensor:
        """ Calculate view frustum boundary planes with given near, far clip
            - support single frame:     ✓
            - support batched frames:   ✓      `...` means arbitary prefix-batch-dims (self.frozen_prefix)

        Args:
            near (float, optional): The given near clip value. Defaults to None.
            far (float, optional): The given far clip value. Defaults to None.

        Returns:
            torch.Tensor: [..., num_frustum_planes, 4] The calculated frustum border planes representation.
                The last dim 4 = 3 (normals) + 1 (distances)
        """
        c2w = self.world_transform.mat_4x4()
        planes = self.intr.get_view_frustum(
            c2w, near=self.near if near is None else near, far=self.far if far is None else far)
        return planes

    # @profile
    def build_view_frustum(self):
        """ Build view frustum by calculating frustum planes repr
            - support single frame:     ✓
            - support batched frames:   ✓      `...` means arbitary prefix-batch-dims (self.frozen_prefix)
        
        """
        self.frustum = self.get_view_frustum() # [..., num_frustum_planes, 4(n3+d1)]

    @torch.no_grad()
    def get_view_frustum_pts(self, near: float=None, far: float=None)-> torch.Tensor:
        """ Get view frustum vertices with given near, far clip
            - support single frame:     ✓
            - support batched frames:   ✓      `...` means arbitary prefix-batch-dims (self.frozen_prefix)

        Args:
            near (float, optional): The given near clip value. Defaults to None.
            far (float, optional): The given far clip value. Defaults to None.

        Returns:
            torch.Tensor: [..., 8, 3] The vertices of the view frustum(s)
        """
        near_clip = (0. if self.near is None else self.near) if near is None else near
        far_clip = (100.0 if self.far is None else self.far) if far is None else far
        _W, _H = self.intr.wh().movedim(-1,0)
        _0, _near, _far = _W.new_zeros(_W.shape), _W.new_full(_W.shape, near_clip), _W.new_full(_W.shape, far_clip)
        # [..., 8]
        u, v, d = torch.stack(
            [
                torch.stack([_0, _0, _near], 0),
                torch.stack([_W, _0, _near], 0),
                torch.stack([_W, _H, _near], 0),
                torch.stack([_0, _H, _near], 0),
                torch.stack([_0, _0, _far], 0),
                torch.stack([_W, _0, _far], 0),
                torch.stack([_W, _H, _far], 0),
                torch.stack([_0, _H, _far], 0),   
            ], dim=-1
        )
        
        # [..., 8, 3]
        # pts = self.intr.lift(u, v, d)[...,:3]
        # pts = self.world_transform(pts)
        
        mat = self.intr.mat_3x3()
        if len(mat_prefix:=mat.shape[:-2]) > 0:
            mat = mat.view(*mat_prefix, *[1]*(u.dim()-len(mat_prefix)), 3, 3)
        # [..., 8, 3]
        # NOTE: Force to use pinhole lift instead.
        pts = pinhole_lift(u, v, d, mat)[..., :3]
        pts = self.world_transform(pts)
        return pts

    def check_spheres_inside_frustum(self, sphere_center_radius: torch.Tensor, holistic_body=False) -> torch.BoolTensor:
        """ Check spheres inside camera frustum
            - support single frame:     ✓
            - support batched frames:   ✓      `...` means arbitary prefix-batch-dims (self.frozen_prefix)

        Args:
            sphere_center_radius (torch.Tensor): [..., num_spheres, 4], The given spheres repr, last dim = center (3) + radius (1)
            holistic_body (bool, optional): Whether the holistic sphere body must be in the planes (T), or any part counts (F). Defaults to False.

        Returns:
            torch.BoolTensor: [..., num_spheres] Inside checks of the given spheres
        """
        inside = sphere_inside_planes(sphere_center_radius, self.frustum, holistic=holistic_body)
        return inside
        
    # @profile
    def filter_drawable_groups(self, drawables: List[SceneNode], draw_self=False) -> List[SceneNode]:
        """ Frustum culling to filter drawables by checking whether drawables' bounding spheres has intersection with view frustums.
            - support single frame:     ✓
            - support batched frames:   ✓      `...` means arbitary prefix-batch-dims (self.frozen_prefix)
        
        Args:
            drawables (List[SceneNode]): List of drawables
            draw_self (bool, optional): Whether draw self node. Defaults to False.

        Returns:
            List[SceneNode]: The filtered list of drawables
        """
        # For now, we don't consider observer's self shading
        if self.model is not None and not draw_self:
            drawables = list(filter(lambda obj: obj.id!=self.id, drawables))
        
        if len(collected:=[[obj, obj.model_BSPH, obj.valids] for obj in drawables if obj.model_BSPH is not None]) > 0:
            drawables_with_bound, model_bounding_spheres, valids = zip(*collected)
        else:
            drawables_with_bound = []
        drawables_no_bound = [obj for obj in drawables if obj.model_BSPH is None]
        
        # Frustum Culling
        if len(drawables_with_bound) > 0:
            # Only on nodes that has Bounding
            model_bounding_spheres, valids = torch.stack(model_bounding_spheres, -2), torch.stack(valids, -1)
            # NOTE: model_bounding_spheres: [..., num_drawables, 4]
            #       valids:                 [..., num_drawables]
            #       self.frustum:           [..., num_planes, 4]
            #       inside:                 [..., num_drawables]
            inside = self.check_spheres_inside_frustum(model_bounding_spheres) & valids
            if (dims:=inside.dim()) > 1:
                # Dealing with multiple cameras or batched frames: if any camera suffices, then this drawable node suffices.
                inside = inside.any(dim=0) if dims == 2 else (inside.sum(dim=list(range(0,dims-1))) > 0)
            # List [num_drawables]
            inside = inside.data.nonzero(as_tuple=True)[0].tolist()
            drawables_with_bound = [drawables_with_bound[i] for i in inside]
        
        return drawables_with_bound + drawables_no_bound

    def sample_pixel(self, num_samples: int) -> torch.Tensor:
        """ Uniformly sample some pixels xy locations.
            - support single frame:     ✓
            - support batched frames:   ✓      `...` means arbitary prefix-batch-dims (self.frozen_prefix)

        Args:
            num_samples (int): The given number of samples

        Returns:
            torch.Tensor: [..., num_samples, 2] The sampled xy locations, within range [0,1]
        """
        xy = torch.rand([*self.frozen_prefix, num_samples, 2], dtype=self.dtype, device=self.device).clamp_(1e-6, 1-1e-6)
        return xy

    def _get_selected_rays_xy(self, xy: torch.Tensor, *, snap_to_pixel_centers=True) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Lift [0,1] xy pixel locations to 3D rays origins and direction vectors.
            - support single frame:     ✓
            - support batched frames:   ✓      `...` means arbitary prefix-batch-dims (self.frozen_prefix)
            
            NOTE: Support different camera H,W in different batch

        Args:
            xy (torch.Tensor): [..., 2] The given xy locations within range [0,1]
            snap_to_pixel_centers (bool, optional): Whether to snap pixel locations onto pixel borders. Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: [..., 3], [..., 3] The lifted rays_o and rays_d vectors
        """
        intrs, c2ws, prefix = self.intr, self.world_transform, list(self.frozen_prefix)
        
        # If batched: Assume `xy` has the same prefix dims as self.frozen_prefix;
        assert list(xy.shape[:len(prefix)]) == prefix, f"Expects xy to have shape of {prefix + [Ellipsis]}, but currently xy.shape={xy.shape}"
        
        WH = intrs.wh().long()
        WH = WH.view(*WH.shape[:-1], *[1]*(xy.dim()-WH.dim()), 2)
        if snap_to_pixel_centers:
            wh = (xy*WH).long().clamp_(WH.new_zeros([]), WH-1)+0.5
        else:
            wh = xy*WH
        lifted_directions = intrs.lift(wh[..., 0], wh[..., 1], wh.new_ones(wh.shape[:-1]))
        
        rays_d = F.normalize(c2ws.rotate(lifted_directions[..., :3]), dim=-1)
        trans = c2ws.translation()
        rays_o = trans.view(*trans.shape[:-1], *[1]*(xy.dim()-trans.dim()), 3).expand_as(rays_d)
        return rays_o, rays_d

    def _get_selected_rays_ixy(self, i: torch.Tensor, xy: torch.Tensor, *, snap_to_pixel_centers=True) -> Tuple[torch.Tensor, torch.Tensor]:
        """  Lift [0,1] xy pixel locations at given frame inds to rays origins and direction vectors
            - support single frame:     x      NOTE: This function should only be called when in batched.
            - support batched frames:   ✓
            
            NOTE: Support different camera H,W in different batch

        Args:
            i (torch.Tensor): [..., ] The given frame indices
            xy (torch.Tensor): [..., 2] The given xy locations within range [0,1]
            snap_to_pixel_centers (bool, optional): Whether to snap pixel locations onto pixel borders. Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: [..., 3], [..., 3] The lifted rays_o and rays_d vectors
        """
        
        prefix = list(i.shape)
        assert list(xy.shape[:len(prefix)]) == list(prefix), f"Expects xy to have shape of {prefix + [Ellipsis]}, but currently xy.shape={xy.shape}"
        
        intrs, c2ws = self.intr[i], self.world_transform[i]
        # intrs, c2ws = self.intr, self.world_transform
        
        WH = intrs.wh().long()
        if snap_to_pixel_centers:
            wh = (xy*WH).long().clamp_(WH.new_zeros([]), WH-1)+0.5
        else:
            wh = xy*WH
        lifted_directions = intrs.lift(wh[..., 0], wh[..., 1], wh.new_ones(wh.shape[:-1]))
        
        rays_d = F.normalize(c2ws.rotate(lifted_directions[..., :3]), dim=-1)
        rays_o = c2ws.translation()
        return rays_o, rays_d

    def get_selected_rays(self, *, i: torch.Tensor = None, xy: torch.Tensor = ..., snap_to_pixel_centers=True) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Lift pixel locations and optional frame indices to rays_o and rays_d vectors

        Args:
            i (torch.Tensor, optional): [..., ] The given frame indices. Defaults to None.
            xy (torch.Tensor): [..., 2] The given xy locations within range [0,1]
            snap_to_pixel_centers (bool, optional): Whether to snap pixel locations onto pixel borders. Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: [..., 3], [..., 3] The lifted rays_o and rays_d vectors
        """
        if i is not None:
            return self._get_selected_rays_ixy(i, xy, snap_to_pixel_centers=snap_to_pixel_centers)
        else:
            return self._get_selected_rays_xy(xy, snap_to_pixel_centers=snap_to_pixel_centers)

    def get_all_rays(self, return_rays=True, return_ij=False, return_xy=False) -> List[torch.Tensor]:
        """ Get all camera rays of full image(s)
            - support single frame:     ✓
            - support batched frames:   o      should only be used when H or W is the same across differerent batches.

        Args:
            return_rays (bool, optional): Whether return lifted rays_o and rays_d vectors. Defaults to True.
            return_ij (bool, optional): Whether return ij pixel locations (i \in [0,W], j \in [0,H]). Defaults to False.
            return_xy (bool, optional): Whether return xy pixel locations (xy \in [0,1]). Defaults to False.

        Returns:
            List[torch.Tensor]: Only rays_o, rays_d, or also appends ij or xy.
        """
        H, W, device = self.intr.H, self.intr.W, self.device
        i, j = torch.meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device), indexing='xy')
        i, j = i.reshape(H*W), j.reshape(H*W)        
        w, h = i+0.5, j+0.5 # Pixel centers
        
        ret = []
        if return_rays:
            lifted_directions = self.intr.lift(w, h, torch.ones_like(w, device=device))
            
            # NOTE: !!! bmm/mm would introduce errors of 4e-3 magnitude to rays_d, 
            #           which would be broadcasted to coordinate x and enlarged to an error of ~0.1 @(depth=10) and ~1.0 @(depth=100)
            #           DO NOT USE bmm/mm !!! (also, einsum falls back to bmm/mm)
            # rays_d = torch.mm(self.c2w.rotation(), lifted_directions[...,:3].transpose(-1,-2)).transpose(-1, -2)[..., :3]
            # rays_d = torch.einsum('ij,...j->...i', self.c2w.rotation(), lifted_directions[..., :3])
            rays_d = F.normalize(self.world_transform.rotate(lifted_directions[..., :3]), dim=-1)
            rays_o = self.world_transform.translation()[..., None, :].expand_as(rays_d)
            ret = [rays_o, rays_d]
        if return_ij:
            ret.append(torch.stack([i, j], dim=-1))
        if return_xy:
            ret.append(torch.stack([w / W, h / H], dim=-1))
        
        return ret

    def project_pts_in_image(self, pts: torch.Tensor, *, ignore_mask: torch.Tensor = None, near_clip: float=1e-4):
        """ Project 3D points into image(s)
            - support single frame:     ✓
            - support batched frames:   ✓
        
        Args:
            pts (torch.Tensor): [..., num_points, 3] The given 3D points, `...` has the same prefix dims as self.frozen_prefix;
            ignore_mask (torch.Tensor, optional): Optionally ignores projected pts that falls inside this mask. Defaults to None.
            near_clip (float, optional): Clip away too close points as their u,v is usally in-correct. Defaults to 1e-4.

        Returns:
            Union[namedtuple_mask_nuvd, namedtuple_mask_niuvd]: The projection results
        """
        intrs, c2ws, prefix = self.intr, self.world_transform, self.frozen_prefix
        
        # If batched: Assume `pts` has the same prefix dims as self.frozen_prefix;
        #             pts.shape == [*prefix, num_points, 3] or [*[1]*len(prefix), num_points, 3]
        assert pts.dim() == (len(prefix) + 2), f"Expect pts to be of shape {[*prefix, 'N', 3]} or {[*[1]*len(prefix), 'N', 3]}"
        
        W, H = intrs.wh().long().movedim(-1,0).view(2, *prefix, *[1]*(pts.dim()-len(prefix)-1))
        u, v, d = self.intr.proj(c2ws(pts, inv=True))
        mask = (d > near_clip) & (u < W) & (u >= 0) & (v < H) & (v >= 0)
        
        inds = mask.nonzero(as_tuple=True)
        n, u, v, d = inds[0].numel(), u[inds].long(), v[inds].long(), d[inds]

        if len(prefix) == 0:
            if ignore_mask is not None:
                suffice = (~ignore_mask[v.long(),u.long()])
                suffice_i = suffice.nonzero().long()[..., 0]
                mask[inds] = suffice
                n, u, v, d = suffice_i.numel(), u[suffice_i], v[suffice_i], d[suffice_i]
            return namedtuple_mask_nuvd(mask, n, u, v, d)
        elif len(prefix) == 1:
            i = inds[0]
            if ignore_mask is not None:
                suffice = ~ignore_mask[i,v.long(),u.long()]
                suffice_i = suffice.nonzero().long()[..., 0]
                mask[inds] = suffice
                n, i, u, v, d = suffice_i.numel(), i[suffice_i], u[suffice_i], v[suffice_i], d[suffice_i]
            return namedtuple_mask_niuvd(mask, n, i, u, v, d)
        else:
            raise RuntimeError(f"`project_pts_in_image` does not support current frozen_prefix={self.frozen_prefix}")

class MultiCamBundle(object):
    def __init__(self, cams: List[Camera]):
        self.class_name = 'Camera' # TODO
        
        for cam in cams:
            assert is_scalar(cam.i), "Only support bundling cameras that are frozen at a single frame_ind."
        self.dtype = cam.dtype
        self.device = cam.device
        self.scene = cam.scene
        
        self.cams = cams
        self.key_list = [cam.id for cam in cams]
        
        nears = [cam.near for cam in cams if cam.near is not None]
        fars = [cam.far for cam in cams if cam.far is not None]
        self.near = None if len(nears) == 0 else min(nears)
        self.far = None if len(fars) == 0 else max(fars)
        
        intrs = [cam.intr for cam in cams]
        world_transforms = [cam.world_transform for cam in cams]
        self.intr = type(cams[0].intr).stack(intrs)
        self.world_transform = type(cams[0].world_transform).stack(world_transforms)
        self.frustum = torch.stack([cam.frustum for cam in cams], 0)
        
        self.frozen_prefix = (len(cams),)

    def check_spheres_inside_frustum(self, sphere_center_radius: torch.Tensor, holistic_body=False):
        return Camera.check_spheres_inside_frustum(self, sphere_center_radius, holistic_body=holistic_body)

    def filter_drawable_groups(self, drawables: List[SceneNode], draw_self=False) -> List[SceneNode]:
        if len(collected:=[[obj, obj.model_BSPH, obj.valids] for obj in drawables if obj.model_BSPH is not None]) > 0:
            drawables_with_bound, model_bounding_spheres, valids = zip(*collected)
        else:
            drawables_with_bound = []
        drawables_no_bound = [obj for obj in drawables if obj.model_BSPH is None]
        
        # Frustum Culling
        if len(drawables_with_bound) > 0:
            # Only on nodes that has Bounding
            model_bounding_spheres, valids = torch.stack(model_bounding_spheres, -2).unsqueeze(0), torch.stack(valids, -1).unsqueeze(0)
            # NOTE: model_bounding_spheres: [1, num_drawables, 4]
            #       valids:                 [1, num_drawables, ]
            #       self.frustum:           [num_cameras, num_planes, 4]
            #       inside:                 [num_cameras, num_drawables]
            inside = self.check_spheres_inside_frustum(model_bounding_spheres) & valids
            if (dims:=inside.dim()) > 1:
                # Dealing with multiple cameras or batched frames: if any camera suffices, then this drawable node suffices.
                inside = inside.any(dim=0) if dims == 2 else (inside.sum(dim=list(range(0,dims-1))) > 0)
            # List [num_drawables]
            inside = inside.data.nonzero(as_tuple=True)[0].tolist()
            drawables_with_bound = [drawables_with_bound[i] for i in inside]
        
        return drawables_with_bound + drawables_no_bound

    def sample_pixel(self, num_samples: int):
        return Camera.sample_pixel(self, num_samples)

    def _get_selected_rays_xy(self, xy: torch.Tensor, *, snap_to_pixel_centers=True):
        return Camera._get_selected_rays_xy(self, xy, snap_to_pixel_centers=snap_to_pixel_centers)

    def _get_selected_rays_ixy(self, i: torch.Tensor, xy: torch.Tensor, *, snap_to_pixel_centers=True):
        return Camera._get_selected_rays_ixy(self, i, xy, snap_to_pixel_centers=snap_to_pixel_centers)

    def get_selected_rays(self, *, i: torch.Tensor = None, xy: torch.Tensor = None, snap_to_pixel_centers=True):
        return Camera.get_selected_rays(self, i=i, xy=xy, snap_to_pixel_centers=snap_to_pixel_centers)

    def get_all_rays(self):
        raise RuntimeError(f"You should not use `get_all_rays` on {self.__class__.__name__}. Please use {Camera.__name__} instead.")

    def project_pts_in_image(self, pts: torch.Tensor, *, ignore_mask: torch.Tensor = None, near_clip: float=1e-4):
        return Camera.project_pts_in_image(self, pts, ignore_mask=ignore_mask, near_clip=near_clip)

if __name__ == "__main__":
    def unit_test(
        device=torch.device('cuda'), 
        # device=torch.device('cpu'), 
        ):
        
        from icecream import ic
        from nr3d_lib.utils import import_str
        from nr3d_lib.config import ConfigDict
        from nr3d_lib.models.spatial import AABBSpace
        
        from app.resources import create_scene_bank
        from dataio.dataset_io import DatasetIO

        dataset_cfg = ConfigDict(
            target="dataio.autonomous_driving.WaymoDataset", 
            param=ConfigDict(
                root="/data1/waymo/processed/", 
                rgb_dirname="images", 
                lidar_dirname="lidars", 
                mask_dirname="masks"
            )
        )

        scenebank_cfg = ConfigDict(
            scenarios=['segment-6207195415812436731_805_000_825_000_with_camera_labels, 0, 150'], 
            observer_cfgs=ConfigDict(
                Camera=ConfigDict(
                    list=['camera_FRONT', 'camera_FRONT_LEFT', 'camera_FRONT_RIGHT', 'camera_SIDE_LEFT', 'camera_SIDE_RIGHT']
                )
            ), 
            object_cfgs=ConfigDict(
                Vehicle=ConfigDict(dynamic_only=False)
            ), 
            on_load=ConfigDict(
                consider_distortion=False, 
                no_objects=False,  joint_camlidar=True, joint_camlidar_equivalent_extr=True
            )
        )

        dataset_impl: DatasetIO = import_str(dataset_cfg.target)(dataset_cfg.param)
        
        scene_bank, scenebank_meta = create_scene_bank(
            dataset=dataset_impl, device=device, 
            scenebank_cfg=scenebank_cfg, 
            drawable_class_names=['Sky', 'Street', 'Vehicle'], 
            misc_node_class_names=['node', 'Ego', 'EgoVehicle', 'EgoDrone'], 
        )
        
        scene = scene_bank[0]
        
        # Add some dummy object with model and bound
        class Dummy(object):
            id: str
            space: Any
        dummy_model = Dummy()
        dummy_model.id = 'dummy'
        dummy_model.space = AABBSpace(bounding_size=2.0)
        for o in scene.all_nodes_by_class_name['Vehicle']:
            o.model = dummy_model
            scene.add_node_to_drawable(o)

        #--------------------- Slice with batched frame inds
        scene.frozen_at([20,21,22,23,24])
        # Single camera (but support batched frame inds)
        cam: Camera = scene.get_observer_groups_by_class_name('Camera', True)[0]
        cam.filter_drawable_groups(scene.get_drawables(True))
        xy = cam.sample_pixel(4096)
        rays_o, rays_d = cam._get_selected_rays_xy(xy)
        
        i = torch.randint(5, (4096,), dtype=torch.long, device=device)
        xy = torch.rand([4096, 2], dtype=torch.float, device=device)
        rays_o, rays_d = cam._get_selected_rays_ixy(i, xy)
        
        pts = torch.randn([5, 800, 3], dtype=torch.float, device=device) + cam.world_transform.translation()[:,None,:]
        ret = cam.project_pts_in_image(pts)
        pts = cam.get_view_frustum_pts()

        #--------------------- Slice with single frame ind
        scene.frozen_at(20)
        cam: Camera = scene.get_observer_groups_by_class_name('Camera', True)[0]
        cam.filter_drawable_groups(scene.get_drawables(True))
        xy = cam.sample_pixel(4096)
        rays_o, rays_d = cam._get_selected_rays_xy(xy)
        pts = torch.randn([800, 3], dtype=torch.float, device=device) + cam.world_transform.translation()
        ret = cam.project_pts_in_image(pts)
        pts = cam.get_view_frustum_pts()
        
        #--------------------- Multiple camera bundles + single frame ind
        scene.frozen_at(21)
        cams: List[Camera] = scene.get_observer_groups_by_class_name('Camera', True).to_list()
        cam_bundle = MultiCamBundle(cams)
        cam_bundle.filter_drawable_groups(scene.get_drawables(True))
        xy = cam_bundle.sample_pixel(4096)
        rays_o, rays_d = cam_bundle._get_selected_rays_xy(xy)
        
        i = torch.randint(len(cams), (4096,), dtype=torch.long, device=device)
        xy = torch.rand([4096, 2], dtype=torch.float, device=device)
        rays_o, rays_d = cam_bundle._get_selected_rays_ixy(i, xy)
        
        pts = torch.randn([1, 800, 3], dtype=torch.float, device=device) + cams[0].world_transform.translation()
        ret = cam_bundle.project_pts_in_image(pts)
    
    unit_test()