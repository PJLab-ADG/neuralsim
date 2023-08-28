"""
@file   image_loader.py
@author Jianfei Guo, Shanghai AI Lab
@brief  
- `ImageDataset`: Sampling returns the full (optionally downsampled) image;
- `ImagePatchDataset`: Sampling extracts a patch from the image according to certain scaling and shifting rules;
"""

__all__ = [
    'ImageDataset', 
    'ImagePatchDataset', 
]

import random
import numpy as np
from math import sqrt
from typing import Dict, Iterator, List, Literal, Tuple, Union

import torch
import torch.utils.data as torch_data
from torch.utils.data.dataloader import DataLoader

from nr3d_lib.fmt import log
from nr3d_lib.config import ConfigDict
from nr3d_lib.utils import collate_nested_dict, collate_tuple_of_nested_dict

from .base import SceneDataLoader
from .sampler import get_frame_sampler

class ImageDataset(torch_data.Dataset):
    def __init__(
        self, dataset: SceneDataLoader, *, 
        camera_sample_mode: Literal['uniform', 'weighted', 'all_list', 'all_stack']='uniform', 
        multi_cam_weight: List[float]=None, 
        frame_sample_mode: Literal['uniform', 'weighted_by_speed']='uniform', 
        ddp=False, **sampler_kwargs
        ) -> None:
        """ Sampler and data loader for full (downscaled) images.

        Args:
            dataset (SceneDataLoader): The base SceneDataLoader.
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
        """
        
        super().__init__()
        
        self.dataset = dataset
        self.scene_id_list = list(self.dataset.scene_bank.keys())
        
        assert camera_sample_mode in ['uniform', 'weighted', 'all_list', 'all_stack'], f"Invalid camera_sample_mode={camera_sample_mode}"
        self.camera_sample_mode = camera_sample_mode
        self.multi_cam_weight = None
        if self.camera_sample_mode == 'weighted':
            assert multi_cam_weight is not None, f"Please specify `multi_cam_weight`"
            multi_cam_weight = np.array(multi_cam_weight)
            self.multi_cam_weight = multi_cam_weight / multi_cam_weight.sum()
        
        self.frame_sample_mode = frame_sample_mode
        self.cur_it: int = np.inf

        self.ddp = ddp
        self.sampler = get_frame_sampler(self.dataset, frame_sample_mode=self.frame_sample_mode, ddp=ddp, **sampler_kwargs)

    @property
    def device(self):
        return self.dataset.device
    
    def sample(self, scene_id: str, cam_id: Union[str, List[str]], frame_id: Union[int, List[int]], stack=True):
        sample = dict(scene_id=scene_id, cam_id=cam_id, frame_id=frame_id)
        if isinstance(cam_id, str):
            ground_truth = self.dataset.load_rgb_gts(scene_id, cam_id, frame_id, stack=stack)
        elif isinstance(cam_id, list):
            if isinstance(frame_id, list):
                assert len(cam_id) == len(frame_id), "If both cam_id and frame_id are lists, they should form one-to-one pairs"
                ground_truth = [self.dataset.load_rgb_gts(scene_id, ci, fi) for ci, fi in zip(cam_id, frame_id)]
            else:
                ground_truth = [self.dataset.load_rgb_gts(scene_id, ci, frame_id) for ci in cam_id]
            ground_truth = collate_nested_dict(ground_truth, stack=stack)
        return sample, ground_truth

    def sample_cam_id(self) -> str:
        if self.multi_cam_weight is not None:
            return random.choice(self.dataset.cam_id_list)
        else:
            return np.random.choice(self.dataset.cam_id_list, p=self.multi_cam_weight)

    def __len__(self):
        # Total number of frames of all scenes
        return sum([len(scene) for scene in self.dataset.scene_bank])

    def __getitem__(self, index: int) -> Tuple[dict, dict]:
        scene_idx, frame_id = self.dataset.get_scene_frame_idx(index)
        scene_id = self.scene_id_list[scene_idx]
        if 'all' in self.camera_sample_mode:
            cam_id = self.dataset.cam_id_list
            stack = 'stack' in self.camera_sample_mode
        else:
            cam_id = self.sample_cam_id()
            stack = False
        return self.sample(scene_id, cam_id, frame_id, stack=stack)

    def get_dataloader(self, num_workers: int=0):
        return DataLoader(
            self, sampler=self.sampler, collate_fn=collate_tuple_of_nested_dict, 
            num_workers=0 if (self.dataset.preload or not self.ddp) else num_workers)

class ImagePatchDataset(torch_data.Dataset):
    def __init__(
        self, dataset: SceneDataLoader, *, 
        num_rays: int = None, HW: Tuple[int, int] = None, 
        random_shift=True, random_scale=True, 
        scale: float=1.0, max_scale: float=1.0, scale_anneal: ConfigDict=None, 
        camera_sample_mode: Literal['uniform', 'weighted', 'all_list', 'all_stack']='uniform', 
        multi_cam_weight: List[float]=None, 
        frame_sample_mode: Literal['uniform', 'weighted_by_speed']='uniform', 
        ddp=False, **sampler_kwargs
        ) -> None:
        """ Sampler and data loader for scaled and shifted patches from full images
        You can use either `num_rays` or `HW` to set the size of the image patch.
        If `num_rays` is provided, a square-shaped image patch with an automatically computed size will be created.
        If not, the image patch will take the size specified by `HW`.

        Args:
            dataset (SceneDataLoader): The base SceneDataLoader
            num_rays (int, optional): The number of rays(pixels) contained in each image patch. Defaults to None.
            HW (Tuple[int, int], optional): The size of each image patch. Defaults to None.
            random_shift (bool, optional): If true, will apply random shift to the whole patch. Defaults to True.
            random_scale (bool, optional): If true, will apply random scale to the whole patch. Defaults to True.
            scale (float, optional): The scale of the patch. Defaults to 1.0.
            max_scale (float, optional): The maximum scale of the patch. Defaults to 1.0.
            scale_anneal (ConfigDict, optional): 随着不同 iteration 变化不同的 scale 控制函数配置. Defaults to None.
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
        """
        
        super().__init__()
        
        assert bool(num_rays is not None) != bool(HW is not None), "You should specify one of `num_rays` and `HW`"

        self.dataset = dataset
        self.scene_id_list = list(self.dataset.scene_bank.keys())
        
        self.camera_sample_mode = camera_sample_mode
        self.multi_cam_weight = None
        if self.camera_sample_mode == 'weighted':
            assert multi_cam_weight is not None, f"Please specify `multi_cam_weight`"
            multi_cam_weight = np.array(multi_cam_weight)
            self.multi_cam_weight = multi_cam_weight / multi_cam_weight.sum()
        
        self.frame_sample_mode = frame_sample_mode
        
        self.random_shift = random_shift
        self.random_scale = random_scale
        self.scale = scale
        # self.scale_fn = get_annealer(scale_anneal)
        self.max_scale = max_scale
        
        if num_rays is not None:
            num_rays_sqrt = int(sqrt(num_rays))
            self.H, self.W = num_rays_sqrt, num_rays_sqrt
        else:
            self.H, self.W = HW
        
        self.max_offset_y = 1./(self.H+1)
        self.max_offset_x = 1./(self.W+1)
        self.max_offset_xy = torch.tensor([self.max_offset_x, self.max_offset_y], dtype=torch.float, device=self.device)
        
        self.y, self.x = torch.meshgrid([
            torch.linspace(0, 1, self.H+2, device=self.device)[1:-1], 
            torch.linspace(0, 1, self.W+2, device=self.device)[1:-1]], indexing='ij')

        self.cur_it: int = np.inf

        self.ddp = ddp
        self.sampler = get_frame_sampler(self.dataset, frame_sample_mode=self.frame_sample_mode, ddp=ddp, **sampler_kwargs)

    @property
    def device(self):
        return self.dataset.device
    
    def sample(self, scene_id: str, cam_id: str, frame_id: int):
        scale = self.scale if not self.random_scale else torch.empty([1], device=self.device, dtype=torch.float).uniform_(self.scale, self.max_scale)
        xy = torch.stack([self.x * scale, self.y * scale], dim=-1)
        if self.random_shift:
            offset = (torch.rand([2], device=self.device, dtype=torch.float)*2-1) * self.max_offset_xy
            xy = (xy + offset).clamp_(0,1)

        ground_truth = self.dataset.get_rgb_gts(scene_id, cam_id, frame_id)
        WH = ground_truth['rgb_wh'].view(1,2)
        w, h = (WH * xy).long().clamp_(WH.new_zeros([]), WH-1).movedim(-1,0)
        inds = (h, w)

        # [self.H, self.W, 3]
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

    def __len__(self):
        return sum([len(scene) for scene in self.dataset.scene_bank]) # Total number of frames of all scenes
    
    def __getitem__(self, index: int) -> Tuple[dict, dict]:
        scene_idx, frame_id = self.dataset.get_scene_frame_idx(index)
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

    def get_dataloader(self, num_workers: int=0):
        return DataLoader(
            self, sampler=self.sampler, collate_fn=collate_tuple_of_nested_dict, 
            num_workers=0 if (self.dataset.preload or not self.ddp) else num_workers)
