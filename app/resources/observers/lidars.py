"""
@file   lidars.py
@author Xinyu Cai, Shanghai AI Lab & Jianfei Guo, Shanghai AI Lab
@brief  Special kinds of SceneNode: lidar observers.
        - RaysLidar / MultiRaysLidarBundle: Dataset pre-calculated lidar model.
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
    def __init__(self, unique_id: str, scene=..., device=torch.device('cuda'), dtype=torch.float):
        super().__init__(unique_id=unique_id, class_name='RaysLidar', scene=scene, device=device, dtype=dtype)
        # Additional attributes
        self.near = None
        self.far = None
    # @profile
    def update(self):
        SceneNode.update(self)

    def filter_drawable_groups(self, drawables: List[SceneNode]):
        return drawables
    
    def _get_selected_rays_ov(self, rays_o: torch.Tensor, rays_d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        - support single frame:     ✓
        - support batched frames:   ✓      `...` means arbitary prefix-batch-dims (self.frozen_prefix)
        """
        l2ws, prefix = self.world_transform, self.frozen_prefix
        # If batched: Assume `rays_o/rays_d` has the same prefix dims as self.frozen_prefix;
        rays_o, rays_d = l2ws.forward(rays_o), l2ws.rotate(rays_d)
        return rays_o, rays_d
    def _get_selected_rays_iov(self, i: torch.Tensor, rays_o: torch.Tensor, rays_d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        - support single frame:     x      NOTE: This function should only be called when in batched.
        - support batched frames:   ✓
        """
        l2ws = self.world_transform[i]
        rays_o, rays_d = l2ws.forward(rays_o), l2ws.rotate(rays_d)
        return rays_o, rays_d
    def get_selected_rays(self, *, i: torch.Tensor = None, rays_o: torch.Tensor = ..., rays_d: torch.Tensor = ...):
        if i is not None:
            return self._get_selected_rays_iov(i, rays_o, rays_d)
        else:
            return self._get_selected_rays_ov(rays_o, rays_d)

class MultiRaysLidarBundle(object):
    def __init__(self, lidars: List[RaysLidar]):
        self.class_name = 'RaysLidar' # TODO
        
        for lidar in lidars:
            assert is_scalar(lidar.i), "Only support bundling cameras that are frozen at a single frame_ind."
        self.dtype = lidar.dtype
        self.device = lidar.device
        self.scene = lidar.scene
        
        self.lidars = lidars

        nears = [lidar.near for lidar in lidars if lidar.near is not None]
        fars = [lidar.far for lidar in lidars if lidar.far is not None]
        self.near = None if len(nears) == 0 else min(nears)
        self.far = None if len(fars) == 0 else max(fars)
        
        world_transforms = [lidar.world_transform for lidar in lidars]
        self.world_transform = type(lidars[0].world_transform).stack(world_transforms)
        
        self.frozen_prefix = (len(lidars),)

    def filter_drawable_groups(self, drawables: List[SceneNode]):
        return drawables

    def get_selected_rays(self, *, i: torch.Tensor = None, rays_o: torch.Tensor = ..., rays_d: torch.Tensor = ...):
        assert i is not None, "Only support i-ov selecting"
        return RaysLidar._get_selected_rays_iov(self, i, rays_o, rays_d)

class Lidar(SceneNode):
    """
    Lidar that is custom-defined with theoretical models
    """
    def __init__(
        self, unique_id: str, lidar_model: str=None, lidar_name: str=None, 
        scene=..., device=torch.device('cuda'), dtype=torch.float, **lidar_params):
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
    def get_all_rays(self) -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(self.frozen_prefix) == 0
        carla_to_opencv = self.carla_to_opencv.to(self.device)
        c2w = (self.world_transform.mat_4x4().unsqueeze(-1) * carla_to_opencv.unsqueeze(-3)).sum(-2)
        dx = c2w[:3, 0]
        dy = c2w[:3, 1]
        dz = c2w[:3, 2]
        if self.lidar_generator is not None:
            Ts, Ps = self.lidar_generator.get_Ts_Ps()
        else:
            Ts, Ps = torch.meshgrid(check_to_torch(self.thetas, ref=self), check_to_torch(self.phis, ref=self), indexing='xy')
        if self.lidar_generator and self.lidar_generator.lidar_name == 'bpearl':
            # rot = torch.tensor([[0,0,1],[0,1,0],[-1,0,0]],dtype=self.dtype,device=self.device) # y 90
            rays_d = (torch.cos(Ts))[..., None] * dx + (torch.sin(Ts) * torch.sin(Ps))[..., None] * dy + \
                 (torch.sin(Ts) * torch.cos(Ps)* -1)[..., None] * dz
        else:
            rays_d = (torch.sin(Ts) * torch.cos(Ps))[..., None] * dx + (torch.sin(Ts) * torch.sin(Ps))[..., None] * dy + \
                 (torch.cos(Ts))[..., None] * dz
        
        rays_o = torch.tile(c2w[:3, 3], [*Ts.shape,1]).view(-1, 3)
        rays_d = F.normalize(rays_d, dim=-1).view(-1,3)
        return rays_o, rays_d

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