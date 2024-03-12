"""
@file   lidars.py
@author Xinyu Cai, Shanghai AI Lab & Jianfei Guo, Shanghai AI Lab
@brief  Special kinds of SceneNode: lidar observers.
        - RaysLidar / MultiRaysLidarBundle: Dataset pre-computed lidar model.
        - Lidar: self-defined lidar simulation model.
"""

__all__ = [
    'Lidar',
    'RaysLidar',
    'MultiRaysLidarBundle',
    'LIDAR_CLASS_NAMES'
]

LIDAR_CLASS_NAMES = ['Lidar', 'RaysLidar']

import os
import csv
import zipfile
import numpy as np
from glob import glob
from typing import List, Tuple

import torch
import torch.nn.functional as F

from nr3d_lib.utils import check_to_torch, is_scalar
from nr3d_lib.fmt import log

from app.resources.nodes import SceneNode

class RaysLidar(SceneNode):
    """
    Lidar that directly load rays from dataset
    """
    def __init__(self, unique_id: str, scene=..., device=None, dtype=torch.float):
        super().__init__(unique_id=unique_id, class_name='RaysLidar', scene=scene, device=device, dtype=dtype)
        # Additional attributes
        self.near = None
        self.far = None
        self.rolling_shutter_effect = None
    
    # @profile
    def update(self):
        SceneNode.update(self)

    def _parse_attr_data(self, odict: dict, data: dict, device=None):
        # NOTE: Might parse some special data if needed. Currently just invoke the base class' method
        return super()._parse_attr_data(odict, data, device)

    def filter_drawable_groups(self, drawables: List[SceneNode]):
        return drawables
    
    def _get_selected_rays_ov(self, rays_o: torch.Tensor, rays_d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        - support single frame:     ✓
        - support batched frames:   ✓      `...` means arbitary prefix-dims (i.e. `self.i_prefix`)
        """        
        l2ws = self.world_transform
        # If batched: Assume `rays_o/rays_d` has the same prefix dims as self.i_prefix;
        rays_o, rays_d = l2ws.forward(rays_o), l2ws.rotate(rays_d)
        return rays_o, rays_d
    
    def _get_selected_rays_iov(self, i: torch.Tensor, rays_o: torch.Tensor, rays_d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ The meaning of `i` varies differently in different use cases. \
            There are two possible use cases: \
            - RaysLidar @ batched frames; where `i` indicates the indices of multiple frames. \
            - MultiRaysLidarBundle @ single frame; where `i` indicates the indices of multiple lidars. \
        """
        prefix = tuple(i.shape)
        if len(self.i_prefix) == 0:
            l2ws = self.world_transform.tile(prefix)
        else:
            l2ws = self.world_transform[i]
        
        rays_o, rays_d = l2ws.forward(rays_o), l2ws.rotate(rays_d)
        return rays_o, rays_d
    
    def get_selected_rays(self, *, sel: torch.Tensor = None, rays_o: torch.Tensor = ..., rays_d: torch.Tensor = ...):
        """ Convert rays in lidar coords to world coords. \
            NOTE: In common cases, lidars and worlds have different coordinate systems; \
                In some cases, lidars and worlds can share the same coordinate system. \
                The actual behavior depends on the definition of the lidar's transform to its parent \
                    (and their parents' transform to their ancestors, if any.) \

        Args:
            sel (torch.Tensor, optional): [..., ] The given selector indices to slice the attr's prefix-dim. \
                Defaults to None.
            rays_o (torch.Tensor, optional): Lidar beams' origin, in lidar local coords. Defaults to ....
            rays_d (torch.Tensor, optional): Lidar beams' direction, in lidar local coords. Defaults to ....

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: [..., 3], [..., 3] Lidar beams' origin and direction in world coords.
        """
        if sel is not None:
            return self._get_selected_rays_iov(sel=sel, rays_o=rays_o, rays_d=rays_d)
        else:
            return self._get_selected_rays_ov(rays_o=rays_o, rays_d=rays_d)

    def get_timestamps(self, *, fi: torch.Tensor = None, ts_base: torch.Tensor = None, theta_phi: torch.Tensor = None):
        assert bool(ts_base is not None) != bool(fi is not None), f"You should specify one of [fi, ts_base]"
        if ts_base is None:
            ts_base = self.frame_global_ts[fi]
        
        if not self.rolling_shutter_effect:
            return ts_base
        
        """
        NOTE:
        Currently, the rolling shutter effect of LiDARs are already accounted for by the per-beam ego-car pose correction.
        However, we should still freeze the scene at the correct timestamps to interpolate correct poses for dynamic objects.
        This relies on more detailed modeling of the Dataset's LiDAR, and will be left as a TODO for now.
        """
        raise NotImplementedError
        assert theta_phi is not None, \
            f"Requires lidar beam angles `theta_phi` to calculate timestamp for each ray to account for rolling shutter effect."

    @staticmethod
    def make_bundle(l: List['RaysLidar']):
        return MultiRaysLidarBundle(l)

class MultiRaysLidarBundle(object):
    def __init__(self, lidars: List[RaysLidar], li: torch.LongTensor = None):
        self.class_name = 'RaysLidar' # TODO

        # # Whether the lidars are frozen at multiple frames
        # self.frozen_at_multiple = False
        
        for lidar in lidars:
            if not lidar.i_is_single:
                assert li is not None, \
                    f"Requires `li` to gather multiple lidars when frozen at multiple frames."
                assert [*li.shape] == list(lidar.i_prefix), \
                    f"`li` (shape={[*li.shape]}) should have the same shape with "\
                        f"the current frozen prefix `lidar.i_prefix`={list(lidar.i_prefix)}"
                # self.frozen_at_multiple = True
                
        self.dtype = lidar.dtype
        self.device = lidar.device
        self.scene = lidar.scene
        
        self.lidars = lidars
        self.id = [lidar.id for lidar in lidars]

        nears = [lidar.near for lidar in lidars if lidar.near is not None]
        fars = [lidar.far for lidar in lidars if lidar.far is not None]
        self.near = None if len(nears) == 0 else min(nears)
        self.far = None if len(fars) == 0 else max(fars)
        
        lst_world_transform = [lidar.world_transform for lidar in lidars]
        world_transform = type(lidars[0].world_transform).stack(lst_world_transform)

        # Whether the lidar id selection is already done when grouping multiple lidars.
        self.already_selected = False
        # if self.frozen_at_multiple:
        if li is not None:
            # Use `li` to select the lidars, keeping other dimensions untouched (gather)
            self.already_selected = True
            self.i_prefix = (*lidar.i_prefix, )
            self.world_transform = world_transform.take_along_dim(li.unsqueeze(0), dim=0)[0]
        else:
            self.i_prefix = (len(lidars), *lidar.i_prefix)
            self.world_transform = world_transform

    def filter_drawable_groups(self, drawables: List[SceneNode]):
        return drawables

    def get_selected_rays(self, *, sel: torch.Tensor = None, rays_o: torch.Tensor = ..., rays_d: torch.Tensor = ...):
        if not self.already_selected:
            assert sel is not None, "Only support [sel, rays_o, rays_d] input"
            return RaysLidar._get_selected_rays_iov(self, i=sel, rays_o=rays_o, rays_d=rays_d)
        else:
            assert sel is None, "`li` is already given when instantiating. Do not specify `sel` again."
            return RaysLidar._get_selected_rays_ov(self, rays_o=rays_o, rays_d=rays_d)

class Lidar(SceneNode):
    """
    Lidar that is custom-defined with theoretical models
    """
    def __init__(
        self, unique_id: str, lidar_model: str=None, lidar_name: str=None, 
        scene=..., device=None, dtype=torch.float, **lidar_params):
        super().__init__(unique_id, scene=scene, class_name='Lidar', device=device, dtype=dtype)
        self.near = lidar_params.get('near', 0.3)
        self.far = lidar_params.get('far', 120.0)
        
        self.carla_to_opencv = torch.eye(4, device=device, dtype=dtype)
        self.carla_to_opencv[:3, :3] = torch.tensor(
            [[0, 1, 0],
             [0, 0, -1],
             [1, 0, 0]])
        self.lidar_generator = None
        if lidar_model == 'dummy':
            horizon_vfov = np.arange(52) * 0.48 - 12.55
            self.thetas = np.pi / 2. - horizon_vfov / 180. * np.pi
            self.phis = np.arange(-1000, 1000, 1) * 0.001 * np.pi
        else:
            self.lidar_generator = AbstractLidarGenerator.getGenerator(lidar_model)
            self.lidar_generator.select_lidar_by_name(lidar_name)
            # self.lidar_generator.lidar_init_theta_phi()
            self.thetas, self.phis = self.lidar_generator.thetas, self.lidar_generator.phis
            self.near, self.far = self.lidar_generator.near, self.lidar_generator.far

    def filter_drawable_groups(self, drawables: List[SceneNode]):
        return drawables
    
    def get_all_rays(self, return_theta_phi=False, return_ts=False) -> List[torch.Tensor]:
        assert len(self.i_prefix) == 0
        carla_to_opencv = self.carla_to_opencv.to(self.device)
        c2w = (self.world_transform.mat_4x4().unsqueeze(-1) * carla_to_opencv.unsqueeze(-3)).sum(-2)
        dx = c2w[:3, 0]
        dy = c2w[:3, 1]
        dz = c2w[:3, 2]
        if self.lidar_generator is not None:
            Ts, Ps = self.lidar_generator.get_Ts_Ps()
        else:
            Ts, Ps = torch.meshgrid(check_to_torch(self.thetas, ref=self), check_to_torch(self.phis, ref=self), indexing='xy')
        theta_phi = torch.stack((Ts, Ps), dim=-1)
        
        if self.lidar_generator and self.lidar_generator.lidar_name == 'bpearl':
            # rot = torch.tensor([[0,0,1],[0,1,0],[-1,0,0]],dtype=self.dtype,device=self.device) # y 90
            rays_d = (torch.cos(Ts))[..., None] * dx + (torch.sin(Ts) * torch.sin(Ps))[..., None] * dy + \
                 (torch.sin(Ts) * torch.cos(Ps)* -1)[..., None] * dz
        else:
            rays_d = (torch.sin(Ts) * torch.cos(Ps))[..., None] * dx + (torch.sin(Ts) * torch.sin(Ps))[..., None] * dy + \
                 (torch.cos(Ts))[..., None] * dz
        
        rays_o = torch.tile(c2w[:3, 3], [*Ts.shape,1]).view(-1, 3)
        rays_d = F.normalize(rays_d, dim=-1).view(-1,3)
        
        ret = [rays_o, rays_d]
        
        if return_theta_phi:
            ret.append(theta_phi)
        
        if return_ts:
            if [*self.i_prefix] == list(rays_o.shape[:-1]):
                rays_i = self.i
            else:
                lidar_i = self.i.item() if isinstance(self.i, (torch.Tensor, np.ndarray)) else self.i
                rays_i = torch.full(rays_o.shape[:-1], lidar_i, dtype=torch.long, device=rays_o.device)
            if self.i_is_timestamp: # `self.i` represents timestamps
                rays_ts = self.get_timestamps(ts_base=rays_i, theta_phi=theta_phi)
            else: # `self.i` represents frame indices
                rays_ts = self.get_timestamps(fi=rays_i, theta_phi=theta_phi)
            ret.append(rays_ts)
        
        return ret

    def get_timestamps(
        self, *, fi: torch.Tensor = None, ts_base: torch.Tensor = None, 
        theta_phi: torch.Tensor = None):
        
        assert bool(ts_base is not None) != bool(fi is not None), f"You should specify one of [fi, ts_base]"
        if ts_base is None:
            ts_base = self.frame_global_ts[fi]
        
        if not self.rolling_shutter_effect:
            return ts_base
        
        raise NotImplementedError
        assert theta_phi is not None, \
            f"Requires lidar beam angles `theta_phi` to calculate timestamp for each ray to account for rolling shutter effect."

    @staticmethod
    def make_bundle(l: List['Lidar']):
        raise NotImplementedError

class AbstractLidarGenerator:
    def __init__(self, lidar_model):
        self.lidar_model = lidar_model
        self.dtype = torch.float
        self.device = torch.device('cuda')
        self.near = 0.3
        self.far = 120
        self.thetas = None
        self.phis = None

    def get_Ts_Ps(self):
        raise NotImplementedError

    def set_dtype(self, dtype):
        self.dtype = dtype

    def set_device(self, device):
        self.device = device

    def select_lidar_by_name(self, lidar_name):
        if lidar_name is None:
            return
        self.lidar_name = lidar_name

    @staticmethod
    def getGenerator(lidar_model):
        if lidar_model == 'Surround':
            return SurroundLidarGenerator()
        elif lidar_model == 'Solid_state':
            return SolidStateLidarGenerator()
        elif lidar_model == 'Risley_prism':
            return RisleyPrismLidarGenerator()
        else:
            raise NotImplementedError

class SurroundLidarGenerator(AbstractLidarGenerator):
    def __init__(self):
        super(SurroundLidarGenerator, self).__init__('Surround')
        self.select_lidar_by_name('pandar64')

    def select_lidar_by_name(self, lidar_name):
        super(SurroundLidarGenerator, self).select_lidar_by_name(lidar_name)
        self.lidar_init_theta_phi()

    def lidar_init_theta_phi(self):
        lidar_name = self.lidar_name
        if lidar_name == 'pandar64':
            self.near = 0.3
            self.far = 200
            horizon_vfov = np.array([14.882, 11.032, 8.059, 5.057, 3.04, 2.028, 1.86, 1.688,
                                     1.522, 1.351, 1.184, 1.013, -1.184, -1.351, -1.522, -1.688,
                                     -1.86, -2.028, -2.198, -2.365, -2.536, -2.7, -2.873, 0.846,
                                     0.675, 0.508, 0.337, 0.169, 0, -0.169, -0.337, -0.508,
                                     -0.675, -0.845, -1.013, -3.04, -3.21, -3.375, -3.548, -3.712,
                                     -3.884, -4.05, -4.221, -4.385, -4.558, -4.72, -4.892, -5.057,
                                     -5.229, -5.391, -5.565, -5.726, -5.898, -6.061, -7.063, -8.059,
                                     -9.06, -9.885, -11.032, -12.006, -12.974, -13.93, -18.889, -24.897])
            thetas = np.pi / 2. - horizon_vfov / 180. * np.pi
            phis = np.arange(-900, 900, 1) / 900 * np.pi
        elif lidar_name == 'ruby128':
            self.near = 0.4
            self.far = 200
            horizon_vfov = np.array([-13.565, -1.09, -4.39, 1.91, -6.65, -0.29, -3.59, 2.71, -5.79,
                                     0.51, -2.79, 3.51, -4.99, 1.31, -1.99, 5.06, -4.19, 2.11,
                                     -19.582, -1.29, -3.39, 2.91, -7.15, -0.49, -2.59, 3.71, -5.99,
                                     0.31, -1.79, 5.96, -5.19, 1.11, -0.99, -4.29, 2.01, -25,
                                     -0.19, -3.49, 2.81, -7.65, 0.61, -2.69, 3.61, -6.09, 1.41,
                                     -1.89, 5.46, -5.29, 2.21, -16.042, -1.19, -4.49, 3.01, -6.85,
                                     -0.39, -3.69, 3.81, -5.89, 0.41, -2.89, 6.56, -5.09, 1.21,
                                     -2.09, -8.352, -0.69, -3.99, 2.31, -6.19, 0.11, -3.19, 3.11,
                                     -5.39, 0.91, -2.39, 3.96, -4.59, 1.71, -1.59, 7.41, -3.79,
                                     2.51, -10.346, -0.89, -2.99, 3.31, -6.39, -0.09, -2.19, 4.41,
                                     -5.59, 0.71, -1.39, 11.5, -4.79, 1.51, -0.59, -3.89, 2.41,
                                     -11.742, 0.21, -3.09, 3.21, -6.5, 1.01, -2.29, 4.16, -5.69,
                                     1.81, -1.49, 9, -4.89, 2.61, -9.244, -0.79, -4.09, 3.41,
                                     -6.29, 0.01, -3.29, 4.71, -5.49, 0.81, -2.49, 15, -4.69,
                                     1.61, -1.69])
            thetas = np.pi / 2. - horizon_vfov / 180. * np.pi
            phis = np.arange(-1800, 1800, 1) / 1800 * np.pi
        elif lidar_name == 'pandar128':
            self.near = 0.3
            self.far = 200
            horizon_vfov = [-26.0, -25.0] + [-6.5 - 0.5 * i for i in range(35, -1, -1)] + \
                           [-6 + i * 0.125 for i in range(64)] + \
                           [2 + 0.5 * i for i in range(24)] + [14.0, 15.0]
            horizon_vfov = np.array(horizon_vfov)
            thetas = np.pi / 2. - horizon_vfov / 180. * np.pi
            phis = np.arange(-1800, 1800, 1) / 1800 * np.pi
        elif lidar_name == 'vlp16':
            horizon_vfov = np.arange(-15.0, 16.0, 2.0)
            thetas = np.pi / 2. - horizon_vfov / 180. * np.pi
            phis = np.arange(-900, 900, 1) / 900 * np.pi
        elif lidar_name == 'hdl64':
            self.near = 0.3
            self.far = 120
            horizon_vfov = np.array([-24.9 + 0.427 * i for i in range(64)])
            thetas = np.pi / 2. - horizon_vfov / 180. * np.pi
            phis = np.arange(-1080, 1080, 1) / 1080 * np.pi
        elif lidar_name == 'pandar_qt':
            self.near = 0.3
            self.far = 20
            horizon_vfov = np.array([52.133, 49.795, 47.587, 45.487, 43.475, 41.537, 39.662,
                                     37.84, 36.064, 34.328, 32.627, 30.957, 29.315, 27.697,
                                     26.101, 24.524, 22.959, 21.415, 19.885, 18.368, 16.861,
                                     15.365, 13.877, 12.397, 10.923, 9.456, 7.993, 6.534,
                                     5.079, 3.626, 2.175, 0.725, -0.725, -2.175, -3.626,
                                     -5.079, -6.535, -7.994, -9.457, -10.925, -12.399, -13.88,
                                     -15.368, -16.865, -18.372, -19.889, -21.42, -22.964, -24.517,
                                     -26.094, -27.69, -29.308, -30.95, -32.619, -34.32, -36.055,
                                     -37.831, -39.653, -41.528, -43.465, -45.477, -47.577, -49.785,
                                     -52.121])
            thetas = np.pi / 2. - horizon_vfov / 180. * np.pi
            phis = np.arange(-300, 300, 1) / 300 * np.pi
        elif lidar_name == 'bpearl':
            self.near = 0.1
            self.far = 30
            horizon_vfov = np.array([(90 / 32) * i for i in range(32)])
            thetas = np.pi / 2. - horizon_vfov / 180. * np.pi
            phis = np.arange(-1800, 1800, 1) / 1800 * np.pi
        elif lidar_name == 'pandar_40m':
            self.near = 0.3
            self.far = 120
            horizon_vfov = np.array([15, 11, 8, 5, 3, 2, 1.67, 1.33, 1, 0.67,
                                     0.33, 0, -0.33, -0.67, -1, -1.33, -1.67, -2.00, -2.33, -2.67,
                                     -3.00, -3.33, -3.67, -4.00, -4.33, -4.67, -5.00, -5.33, -5.67, -6.00,
                                     -7, -8, -9, -10, -11, -12, -13, -14, -19, -25])
            thetas = np.pi / 2. - horizon_vfov / 180. * np.pi
            phis = np.arange(-900, 900, 1) / 900 * np.pi
        elif lidar_name == 'pandar_40p':  # same as pandar_40m
            self.near = 0.3
            self.far = 200
            horizon_vfov = np.array([15, 11, 8, 5, 3, 2, 1.67, 1.33, 1, 0.67,
                                     0.33, 0, -0.33, -0.67, -1, -1.33, -1.67, -2.00, -2.33, -2.67,
                                     -3.00, -3.33, -3.67, -4.00, -4.33, -4.67, -5.00, -5.33, -5.67, -6.00,
                                     -7, -8, -9, -10, -11, -12, -13, -14, -19, -25])
            thetas = np.pi / 2. - horizon_vfov / 180. * np.pi
            phis = np.arange(-900, 900, 1) / 900 * np.pi
        elif lidar_name == 'pandar_xt':
            self.near = 0.05
            self.far = 80
            horizon_vfov = np.array([15 - i for i in range(0, 32)])
            thetas = np.pi / 2. - horizon_vfov / 180. * np.pi
            phis = np.arange(-1800, 1800, 1) / 1800 * np.pi
        elif lidar_name == 'vlp32':
            horizon_vfov = np.array([-25 + 40.0 / 32.0 * i for i in range(0, 32)])
            thetas = np.pi / 2. - horizon_vfov / 180. * np.pi
            phis = np.arange(-1800, 1800, 1) / 1800 * np.pi
        elif lidar_name == 'os1_64gen2':
            self.near = 0.3
            self.far = 120
            horizon_vfov = np.array([15 - i for i in range(0, 32)])
            thetas = np.pi / 2. - horizon_vfov / 180. * np.pi
            phis = np.arange(-1800, 1800, 1) / 1800 * np.pi
        else:
            raise NotImplementedError
        self.thetas = thetas
        self.phis = phis
        return self.thetas, self.phis

    def get_Ts_Ps(self) -> Tuple[torch.Tensor, ...]:
        if self.thetas is None or self.phis is None:
            self.lidar_init_theta_phi()
        return torch.meshgrid(check_to_torch(self.thetas, dtype=self.dtype, device=self.device), \
                              check_to_torch(self.phis, dtype=self.dtype, device=self.device), indexing='xy')

class SolidStateLidarGenerator(AbstractLidarGenerator):
    def __init__(self):
        super(SolidStateLidarGenerator, self).__init__('Solid_state')
        self.select_lidar_by_name('rs_m1')

    def select_lidar_by_name(self, lidar_name):
        super(SolidStateLidarGenerator, self).select_lidar_by_name(lidar_name)
        self.lidar_init_theta_phi()

    def lidar_init_theta_phi(self):
        if self.lidar_name == 'rs_m1':
            # self.far =
            fps = 10
            wx, wy = 7200.0, 100.0
            phi = 0.5 * np.pi
            theta1, theta2 = 0.01 * np.pi, -0.01 * np.pi
            theta3, theta4 = 0.02 * np.pi, -0.02 * np.pi
            vfov_mat = np.zeros((10, 11501), dtype=float)
            hfov_mat = np.zeros((10, 11501), dtype=float)
            for idx in range(0, 11501):
                time_tick = 1.0 / 11500.0 / fps * idx
                hfov_mat[0][idx] = 12.5 * np.cos(2 * np.pi * wx * time_tick)
                vfov_mat[0][idx] = 9.25 * np.sin(2 * np.pi * wy * time_tick + phi) + 3.25
                hfov_mat[1][idx] = 12.5 * np.cos(2 * np.pi * wx * time_tick)
                vfov_mat[1][idx] = 7.25 * np.sin(2 * np.pi * wy * time_tick + phi) - 5.25
                x, y = 12.5 * np.cos(2 * np.pi * wx * time_tick) - 24, \
                       9.25 * np.sin(2 * np.pi * wy * time_tick + phi) + 2.25
                hfov_mat[2][idx] = x * np.cos(theta2) + y * np.sin(theta2)
                vfov_mat[2][idx] = -x * np.sin(theta1) + y * np.cos(theta1)
                x, y = 12.5 * np.cos(2 * np.pi * wx * time_tick) - 24, \
                       7.25 * np.sin(2 * np.pi * wy * time_tick + phi) - 6.25
                hfov_mat[3][idx] = x * np.cos(theta2) + y * np.sin(theta2)
                vfov_mat[3][idx] = -x * np.sin(theta1) + y * np.cos(theta1)
                x, y = 12.5 * np.cos(2 * np.pi * wx * time_tick) + 24, \
                       9.25 * np.sin(2 * np.pi * wy * time_tick + phi) + 2.25
                hfov_mat[4][idx] = x * np.cos(theta1) + y * np.sin(theta1)
                vfov_mat[4][idx] = -x * np.sin(theta2) + y * np.cos(theta2)
                x, y = 12.5 * np.cos(2 * np.pi * wx * time_tick) + 24, \
                       7.25 * np.sin(2 * np.pi * wy * time_tick + phi) - 6.25
                hfov_mat[5][idx] = x * np.cos(theta2) + y * np.sin(theta2)
                vfov_mat[5][idx] = -x * np.sin(theta2) + y * np.cos(theta2)
                x, y = 12.5 * np.cos(2 * np.pi * wx * time_tick) - 48, \
                       9.25 * np.sin(2 * np.pi * wy * time_tick + phi) + 0.25
                hfov_mat[6][idx] = x * np.cos(theta4) + y * np.sin(theta4)
                vfov_mat[6][idx] = -x * np.sin(theta3) + y * np.cos(theta3)
                x, y = 12.5 * np.cos(2 * np.pi * wx * time_tick) - 48, \
                       7.25 * np.sin(2 * np.pi * wy * time_tick + phi) - 8.25
                hfov_mat[7][idx] = x * np.cos(theta4) + y * np.sin(theta4)
                vfov_mat[7][idx] = -x * np.sin(theta3) + y * np.cos(theta3)
                x, y = 12.5 * np.cos(2 * np.pi * wx * time_tick) + 48, \
                       9.25 * np.sin(2 * np.pi * wy * time_tick + phi) + 0.25
                hfov_mat[8][idx] = x * np.cos(theta3) + y * np.sin(theta3)
                vfov_mat[8][idx] = -x * np.sin(theta4) + y * np.cos(theta4)
                x, y = 12.5 * np.cos(2 * np.pi * wx * time_tick) + 48, \
                       7.25 * np.sin(2 * np.pi * wy * time_tick + phi) - 8.25
                hfov_mat[9][idx] = x * np.cos(theta3) + y * np.sin(theta3)
                vfov_mat[9][idx] = -x * np.sin(theta4) + y * np.cos(theta4)
            hfov = hfov_mat.reshape((1, -1))[0]
            vfov = vfov_mat.reshape((1, -1))[0]
            self.thetas = np.pi / 2. - vfov / 180. * np.pi
            self.phis = hfov / 180. * np.pi
            self.thetas, self.phis = check_to_torch(self.thetas, dtype=self.dtype, device=self.device), \
                                     check_to_torch(self.phis, dtype=self.dtype, device=self.device)
        else:
            raise NotImplementedError

    def get_Ts_Ps(self):
        if self.thetas is None or self.phis is None:
            self.lidar_init_theta_phi()
        return self.thetas, self.phis


class RisleyPrismLidarGenerator(AbstractLidarGenerator):
    def __init__(self, csv_data_dir: str = None):
        super(RisleyPrismLidarGenerator, self).__init__('Risley_prism')
        if csv_data_dir is None:
            csv_data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'RisleyPrismCsvData')
        
        self.url = "https://github.com/PJLab-ADG/neuralsim/releases/download/pre-release/RisleyPrismCsvData.zip"
        self.csv_data_dir = csv_data_dir
        
        os.makedirs(self.csv_data_dir, exist_ok=True)
        if len(list(glob(os.path.join(csv_data_dir, "*.csv")))) == 0:
            log.warning(f"Data directory for `RisleyPrismLidarGenerator` is empty: \n{csv_data_dir}")
            log.warning("Will start downloading into it now...")
            log.warning("You can also manually download `*.csv` files into it via this link:\n"
                        "https://drive.google.com/file/d/1-EKhYQTaf3LHa4cCL_vKSVg25torT6ij/view?usp=sharing")
            
            #---- Download file
            filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'RisleyPrismCsvData.zip')
            torch.hub.download_url_to_file(self.url, filepath, progress=True)
            log.warning(f"=> File downloaded to {filepath}")
            
            #---- Unzip file
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(self.csv_data_dir)
            log.warning(f"=> Files extracted to {self.csv_data_dir}")
            
        self.csv_cycle_times = 0
        self.csv_max_sec = 4
        self.csv_cache = []
        self.select_lidar_by_name('horizon')

    def select_lidar_by_name(self, lidar_name):
        super(RisleyPrismLidarGenerator, self).select_lidar_by_name(lidar_name)
        self.lidar_init_theta_phi()

    def read_theta_phi_from_csv(self):
        filename = os.path.join(self.csv_data_dir, str(self.lidar_name) + '.csv')
        if not os.path.exists(filename):
            raise FileNotFoundError
        times = []
        thetas = []
        phis = []
        max_sec = 0
        import bisect
        with open(filename, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                t = float(row[0])
                max_sec = max(max_sec, int(t))
                times.append(t)
                phis.append(float(row[1]))
                thetas.append(float(row[2]))
        self.csv_max_sec = int(max_sec)
        self.thetas = np.array(thetas) / 180.0 * np.pi
        self.phis = np.array(phis) / 180.0 * np.pi
        self.csv_cache = []
        pre_idx = 0
        for sec in range(1, self.csv_max_sec + 1):
            idx = bisect.bisect_right(times, sec)
            self.csv_cache.append((check_to_torch(self.thetas[pre_idx:idx], dtype=self.dtype, device=self.device),
                                   check_to_torch(self.phis[pre_idx:idx], dtype=self.dtype, device=self.device)))
            pre_idx = idx
    def lidar_init_theta_phi(self):
        if self.lidar_name == 'horizon':
            self.near = 0.3
            self.far = 90
        elif self.lidar_name == 'mid70':
            self.near = 0.3
            self.far = 90
        elif self.lidar_name == 'tele':
            self.near = 0.3
            self.far = 320
        else:
            raise NotImplementedError
        self.read_theta_phi_from_csv()

    def get_Ts_Ps(self, ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.thetas is None or self.phis is None:
            self.lidar_init_theta_phi()
        current_idx = self.csv_cycle_times % self.csv_max_sec
        self.csv_cycle_times += 1
        return self.csv_cache[current_idx]


if __name__ == "__main__":
    def unit_test():
        rl = RisleyPrismLidarGenerator()
        ts_rl, ps_rl = rl.get_Ts_Ps()
        # print('Risley Prism')
        # print(ts_rl.shape)

        ss = SurroundLidarGenerator()
        ts_ss, _ = ss.get_Ts_Ps()
        # print('Surround')
        # print(ts_ss.shape)

        sl = SolidStateLidarGenerator()
        sl_rl, _ = sl.get_Ts_Ps()
        # print('Solid state')
        # print(sl_rl.shape)
    unit_test()