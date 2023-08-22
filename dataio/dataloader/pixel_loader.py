"""
@file   pixel_loader.py
@author Jianfei Guo, Shanghai AI Lab
@brief  
Sampling returns individual rays (pixels); supports importance sampling using error_map;
- `PixelDataset`: in one sampled batch, rays originate from the same frame of image.
- `JointFramePixelDataset`: in one sampled batch, rays could originate from different frames of image, \
    since frame index is jointly sampled along with ray index in importance sampling
"""

__all__ = [
    'PixelDataset', 
    'JointFramePixelDataset', 
]

import random
import numpy as np
from typing import Dict, Iterator, List, Literal, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.data as torch_data
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from nr3d_lib.utils import collate_tuple_of_nested_dict
from nr3d_lib.models.importance import ImpSampler

from .base import SceneDataLoader, FrameRandomSampler

class PixelDatasetBase(torch_data.Dataset):
    def __init__(
        self, dataset: SceneDataLoader,
        num_rays: int = 4096, num_points: int = None,
        equal_mode: Literal['ray_batch', 'point_batch'] = 'ray_batch', 
        error_map = False, error_map_res: Tuple[int,int] = (128,128), 
        uniform_sampling_fraction: float = 0.5, 
        respect_errormap_after: int = 0 # Default to 0: always respect errormap
        ) -> None:
        """
        The common base pixel sampler.

        Args:
            dataset (SceneDataLoader): The base SceneDataLoader.
            equal_mode (Literal['ray_batch', 'point_batch'], optional): 
                Determines the mode of the number of rays sampled each time. 
                - `ray_batch`: A fixed number of rays are sampled each time.
                - `point_batch`: [Not supported yet] A fixed number of sampling points are sampled each time; \
                    the number of rays can vary. Similar to the instant-ngp mode.
                Defaults to 'ray_batch'.
            num_rays (int, optional): The number of rays/pixels sampled each time when the `equal_mode` is set to 'ray_batch'. Defaults to 4096.
            num_points (int, optional): The number of points sampled each time when the `equal_mode` is set to 'point_batch'. Defaults to None.
            error_map (bool, optional): If true, will construct an updatable errormap. Defaults to False.
            error_map_res (Tuple[int,int], optional): The resolution of the error map for each frame of each camera. Defaults to (128,128).
            uniform_sampling_fraction (float, optional): The proportion of uniform sampling when using error maps for importance sampling. Defaults to 0.5.
            respect_errormap_after (int, optional): The error map will be respected after this iteration. Defaults to 0.
        """
        
        super().__init__()

        self.dataset = dataset
        self.scene_id_list = list(self.dataset.scene_bank.keys())
        self.equal_mode = equal_mode

        self.set_n_rays = num_rays
        self.set_n_pts = num_points
        
        # Init num_rays
        self.num_rays = num_rays
        
        if error_map:
            self.imp_samplers: Dict[str, Dict[str,ImpSampler]] = nn.ModuleDict()
            for sid, scene in self.dataset.scene_bank.items():
                self.imp_samplers[sid] = nn.ModuleDict()
                for cid, cam in scene.observer_groups_by_class_name['Camera'].items():
                    imp_sampler = ImpSampler(len(cam), error_map_res, uniform_sampling_fraction=uniform_sampling_fraction, device=self.device)
                    imp_sampler.construct_cdf()
                    self.imp_samplers[sid][cid] = imp_sampler
        else:
            self.imp_samplers: Dict[str, Dict[str,ImpSampler]] = None

        self.cur_it = np.inf
        self.respect_errormap_after: int = respect_errormap_after

    @property
    def device(self):
        return self.dataset.device

    def record_prev(self, totol_n_pts: int):
        raise NotImplementedError("TODO in v0.5.2")
        if self.equal_mode == 'point_batch':
            self.num_rays = int(self.set_n_pts / (totol_n_pts/self.num_rays))

class PixelDataset(PixelDatasetBase):
    def __init__(
        self, dataset: SceneDataLoader, *, 
        # Pixels
        pixel_sample_mode: Literal['uniform', 'error_map']='error_map',
        # Cameras and frames
        camera_sample_mode: Literal['uniform', 'weighted', 'all_list', 'all_stack']='uniform', 
        multi_cam_weight: List[float]=None, 
        frame_sample_mode: Literal['uniform', 'weighted_by_speed', 'error_map']='uniform', 
        sample_frame_multi_scene_balance=True, sample_frame_replacement=True, 
        # Basic
        error_map_res: Tuple[int,int] = (128,128), 
        uniform_sampling_fraction: float = 0.5, 
        respect_errormap_after: int = 0, # Default to 0: always respect errormap
        equal_mode: Literal['ray_batch', 'point_batch'] = 'ray_batch', 
        num_rays: int = 4096, num_points: int = None, 
        ) -> None:
        """ 
        Pixel sampler: in one sampled batch, rays originate from the same frame of the same observer.

        Args:
            dataset (SceneDataLoader): The base SceneDataLoader.
            pixel_sample_mode (Literal['uniform', 'error_map'], optional): 
                Determines the method for sampling pixels. 
                - `uniform`: Samples pixels uniformly.
                - `error_map`: Samples pixels weighted by error map via 2D importance sampling
                Defaults to 'error_map'.
            camera_sample_mode (Literal['uniform', 'weighted', 'all_list', 'all_stack'], optional): 
                Determines the method for sampling cameras from multiple options. 
                - `uniform`: Samples one camera uniformly.
                - `weighted`: Samples one camera based on `multi_cam_weight`.
                - `all_list`: Samples all cameras and returns them as a list.
                - `all_stack`: Samples all cameras and returns them as a stacked tensor (requires all cameras to be the same size).
                Defaults to 'uniform'.
            multi_cam_weight (List[float], optional): Weight applied to different cameras in `weighted` mode. Default is None.
            frame_sample_mode (Literal['uniform', 'weighted_by_speed'], optional): 
                Determines the method for sampling a frame from multiple options. 
                - `uniform`: Each frame has an equal probability of being chosen.
                - `weighted_by_speed`: Probability of a frame being chosen is based on motion speed.
                Defaults to 'uniform'.
            sample_frame_multi_scene_balance (bool, optional): If true, the weights of different scenes will be balanced \
                according to the number of frames in each scene when sampling frames from different scenes. Defaults to True.
                Only meaningful when sampling from multiple scenes.
            sample_frame_replacement (bool, optional): If true, frames will be replaced after sampling. Defaults to True.
            error_map_res (Tuple[int,int], optional): The resolution of the error map for each frame of each camera. Defaults to (128,128).
            uniform_sampling_fraction (float, optional): The proportion of uniform sampling when using error maps for importance sampling. Defaults to 0.5.
            respect_errormap_after (int, optional): The error map will be respected after this iteration. Defaults to 0.
            equal_mode (Literal['ray_batch', 'point_batch'], optional): 
                Determines the mode of the number of rays sampled each time. 
                - `ray_batch`: A fixed number of rays are sampled each time.
                - `point_batch`: [Not supported yet] A fixed number of sampling points are sampled each time; \
                    the number of rays can vary. Similar to the instant-ngp mode.
                Defaults to 'ray_batch'.
            num_rays (int, optional): The number of rays/pixels sampled each time when the `equal_mode` is set to 'ray_batch'. Defaults to 4096.
            num_points (int, optional): The number of points sampled each time when the `equal_mode` is set to 'point_batch'. Defaults to None.
        """
        
        error_map = 'error_map' in (frame_sample_mode, pixel_sample_mode)
        super().__init__(dataset, 
            equal_mode=equal_mode, num_rays=num_rays, num_points=num_points, 
            error_map=error_map, error_map_res=error_map_res, 
            uniform_sampling_fraction=uniform_sampling_fraction, 
            respect_errormap_after=respect_errormap_after)
        
        self.camera_sample_mode = camera_sample_mode
        self.multi_cam_weight = None
        if self.camera_sample_mode == 'weighted':
            assert multi_cam_weight is not None, f"Please specify `multi_cam_weight`"
            multi_cam_weight = np.array(multi_cam_weight)
            self.multi_cam_weight = multi_cam_weight / multi_cam_weight.sum()
        
        self.frame_sample_mode = frame_sample_mode
        self.pixel_sample_mode = pixel_sample_mode
        
        self.sampler = FrameRandomSampler(
            self.dataset, 
            multi_scene_balance=sample_frame_multi_scene_balance, replacement=sample_frame_replacement, 
            frame_sample_mode=self.frame_sample_mode, imp_samplers=self.imp_samplers)
        
    def sample(self, scene_id: str, cam_id: str, frame_id: int, *, num_rays: int = None):
        if num_rays is None: num_rays = self.num_rays
        
        if self.pixel_sample_mode == 'uniform':
            xy = torch.rand([num_rays, 2], device=self.device).clamp_(1e-6, 1-1e-6)
        elif self.pixel_sample_mode == 'error_map':
            imp_sampler = self.imp_samplers[scene_id][cam_id]
            if self.cur_it < self.respect_errormap_after:
                xy = torch.rand([num_rays, 2], device=self.device).clamp_(1e-6, 1-1e-6)
            else:
                xy = imp_sampler.sample_pixel(num_rays, frame_id)
        else:
            raise RuntimeError(f"Invalid pixel_sample_mode={self.pixel_sample_mode}")

        ground_truth = self.dataset.get_rgb_gts(scene_id, cam_id, frame_id)
        WH = ground_truth['rgb_wh'].view(1,2)
        w, h = (WH * xy).long().clamp_(WH.new_zeros([]), WH-1).movedim(-1,0)
        inds = (h, w)
        
        # [num_rays, 3]
        ground_truth.update({k: ground_truth[k][inds] for k in self.dataset.gt_im_fields if k in ground_truth})

        sample = dict(
            scene_id=scene_id, cam_id=cam_id, frame_id=frame_id, selects=dict(xy=xy), 
            rays_xy=xy, rays_fi=torch.full(h.shape, frame_id, dtype=torch.long, device=h.device))
        return sample, ground_truth

    def sample_cam_id(self) -> str:
        if self.multi_cam_weight is not None:
            return np.random.choice(self.dataset.cam_id_list, p=self.multi_cam_weight)
        else:
            return random.choice(self.dataset.cam_id_list)

    def get_index(self, index: int):
        # From holistic index to scene_idx and frame_idx
        scene_idx = 0
        while index >= 0:
            index -= len(self.dataset.scene_bank[scene_idx])
            scene_idx += 1
        return (scene_idx - 1), int(index + len(self.dataset.scene_bank[scene_idx - 1]))

    def __len__(self):
        return sum([len(scene) for scene in self.dataset.scene_bank]) # Total number of frames of all scenes

    def __getitem__(self, index: int) -> Tuple[dict, dict]:
        scene_idx, frame_id = self.get_index(index)
        scene_id = self.scene_id_list[scene_idx]
        if 'all' in self.camera_sample_mode:
            cam_id_list = self.dataset.cam_id_list
            stack = 'stack' in self.camera_sample_mode
            ret = [self.sample(scene_id, cam_id, frame_id) for cam_id in cam_id_list]
            ret = collate_tuple_of_nested_dict(ret, stack=stack)
        else:
            cam_id = self.sample_cam_id()
            ret = self.sample(scene_id, cam_id, frame_id)
        return ret

    def get_random_sampler(self, ddp=False):
        if ddp:
            raise NotImplementedError
        return self.sampler

    def get_dataloader(self, ddp=False):
        return DataLoader(self, sampler=self.get_random_sampler(ddp=ddp), collate_fn=collate_tuple_of_nested_dict, num_workers=0)

class JointFramePixelDataset(PixelDatasetBase):
    def __init__(
        self, dataset: SceneDataLoader, *, 
        # Cameras
        camera_sample_mode: Literal['uniform', 'weighted']='uniform', 
        multi_cam_weight: List[float]=None, 
        # Basic
        error_map_res: Tuple[int,int] = (128,128), 
        uniform_sampling_fraction: float = 0.5, 
        respect_errormap_after: int = 0,  # Default to 0: always respect errormap
        equal_mode: Literal['ray_batch', 'point_batch'] = 'ray_batch', 
        num_rays: int = 4096, num_points: int = None,
        ) -> None:
        """
        Joint frame & pixel sampler: in one sampled batch, rays could originate from different frames of image, \
            since frame index is jointly sampled along with ray index in importance sampling

        Args:
            dataset (SceneDataLoader): The base SceneDataLoader.
            camera_sample_mode (Literal['uniform', 'weighted'], optional): 
                Determines the method for sampling cameras from multiple options. 
                - `uniform`: Samples one camera uniformly.
                - `weighted`: Samples one camera based on `multi_cam_weight`.
                Defaults to 'uniform'.
            multi_cam_weight (List[float], optional): Weight applied to different cameras in `weighted` mode. Default is None.
            error_map_res (Tuple[int,int], optional): The resolution of the error map for each frame of each camera. Defaults to (128,128).
            uniform_sampling_fraction (float, optional): The proportion of uniform sampling when using error maps for importance sampling. Defaults to 0.5.
            respect_errormap_after (int, optional): The error map will be respected after this iteration. Defaults to 0.
            equal_mode (Literal['ray_batch', 'point_batch'], optional): 
                Determines the mode of the number of rays sampled each time. 
                - `ray_batch`: A fixed number of rays are sampled each time.
                - `point_batch`: [Not supported yet] A fixed number of sampling points are sampled each time; \
                    the number of rays can vary. Similar to the instant-ngp mode.
                Defaults to 'ray_batch'.
            num_rays (int, optional): The number of rays/pixels sampled each time when the `equal_mode` is set to 'ray_batch'. Defaults to 4096.
            num_points (int, optional): The number of points sampled each time when the `equal_mode` is set to 'point_batch'. Defaults to None.
        """
        super().__init__(dataset, 
            equal_mode=equal_mode, num_rays=num_rays, num_points=num_points, 
            error_map=True, error_map_res=error_map_res, 
            uniform_sampling_fraction=uniform_sampling_fraction, 
            respect_errormap_after=respect_errormap_after)

        self.camera_sample_mode = camera_sample_mode
        self.multi_cam_weight = None
        if self.camera_sample_mode == 'weighted':
            assert multi_cam_weight is not None, f"Please specify `multi_cam_weight`"
            multi_cam_weight = np.array(multi_cam_weight)
            self.multi_cam_weight = multi_cam_weight / multi_cam_weight.sum()
        
    def sample(self, scene_id: str, cam_id: str, *, num_rays: int = None):
        if num_rays is None: num_rays = self.num_rays
        
        imp_sampler = self.imp_samplers[scene_id][cam_id]
        
        if self.cur_it < self.respect_errormap_after:
            # (Optional) Pure uniform sampling, do not repsect errormap
            frame_id: int = np.random.randint(len(imp_sampler.cdf_img))
            i = torch.full([self.num_rays], frame_id, dtype=torch.long, device=self.device)
            xy = torch.rand([self.num_rays, 2], device=self.device).clamp_(1e-6, 1-1e-6)
            
            ground_truth = self.dataset.load_rgb_gts(scene_id, cam_id, frame_id)
            WH = ground_truth['rgb_wh'].view(1,2)
            w, h = (WH * xy).long().clamp_(WH.new_zeros([]), WH-1).movedim(-1,0)
            inds = (h, w)
        else:
            # (Main) Joint frame-pixel importance sampling based on errormap.
            i, xy = imp_sampler.sample_img_pixel(self.num_rays)
            frame_id, inv = torch.unique(i, return_inverse=True)
            frame_id: List[int] = frame_id.tolist()

            ground_truth = self.dataset.load_rgb_gts(scene_id, cam_id, frame_id)
            ground_truth['rgb_wh'] = WH = ground_truth['rgb_wh'].view(-1,2)[inv]
            ground_truth['rgb_downscale'] = ground_truth['rgb_downscale'].view(-1,2)[inv]
            w, h = (WH * xy).long().clamp_(WH.new_zeros([]), WH-1).movedim(-1,0)
            inds = (inv, h, w)
        
        # [num_rays, 3]
        ground_truth.update({k: ground_truth[k][inds] for k in self.dataset.gt_im_fields if k in ground_truth})

        #---- Option 1: frozen at i; direct use
        sample = dict(scene_id=scene_id, cam_id=cam_id, frame_id=i, rays_fi=i, rays_xy=xy, selects=dict(xy=xy))
        #---- Option 2: frozon at frame_id; use inv to index (selects['i'] != rays_fi)
        # sample = dict(scene_id=scene_id, cam_id=cam_id, frame_id=frame_id, rays_fi=i, rays_xy=xy, selects=dict(i=inv, xy=xy))
        return sample, ground_truth

    def sample_cam_id(self) -> str:
        if self.multi_cam_weight is not None:
            return random.choice(self.dataset.cam_id_list)
        else:
            return np.random.choice(self.dataset.cam_id_list, p=self.multi_cam_weight)

    def __len__(self):
        return len(self.dataset.scene_bank) # Total number of scenes.
    
    def __getitem__(self, index: int) -> Tuple[dict, dict]:
        scene_id = self.scene_id_list[index]
        cam_id = self.sample_cam_id()
        return self.sample(scene_id, cam_id)

    def get_random_sampler(self, multi_scene_balance=True, replacement=True):
        scene_bank = self.dataset.scene_bank
        scene_lengths = [len(scene) for scene in scene_bank]
        num_scenes = len(scene_bank)
        if multi_scene_balance:
            scene_weights = torch.tensor(scene_lengths, device=self.device, dtype=torch.float)
            scene_weights = scene_weights / scene_weights.sum()
        else:
            scene_weights = torch.full([num_scenes, ],  1./num_scenes, device=self.device, dtype=torch.float)
        return WeightedRandomSampler(scene_weights, num_scenes, replacement=replacement)

    def get_dataloader(self, ddp=False):
        return DataLoader(self, sampler=self.get_random_sampler(), collate_fn=collate_tuple_of_nested_dict, num_workers=0)
