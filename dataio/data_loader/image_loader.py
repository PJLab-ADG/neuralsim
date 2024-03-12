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

from .base_loader import SceneDataLoader
from .sampler import get_frame_sampler
from .patch_sampler import get_patch_sampler

class ImageDataset(torch_data.Dataset):
    def __init__(
        self, scene_loader: SceneDataLoader, *, 
        camera_sample_mode: Literal['uniform', 'weighted', 'all_list', 'all_stack']='uniform', 
        multi_cam_weight: List[float]=None, 
        frame_sample_mode: Literal['uniform', 'weighted_by_speed', 'fixed']='uniform', 
        ddp=False, **sampler_kwargs
        ) -> None:
        """ Sampler and data loader for full (downscaled) images.

        Args:
            scene_loader (SceneDataLoader): The base SceneDataLoader.
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
                - `fixed`: Fixed at one single frame
                Defaults to 'uniform'.
        """
        
        super().__init__()
        
        self.scene_loader = scene_loader
        self.scene_id_list = list(self.scene_loader.scene_bank.keys())
        
        assert camera_sample_mode in ['uniform', 'weighted', 'all_list', 'all_stack'], f"Invalid camera_sample_mode={camera_sample_mode}"
        self.camera_sample_mode = camera_sample_mode
        self.multi_cam_weight = None
        if self.camera_sample_mode == 'weighted':
            assert multi_cam_weight is not None, f"Please specify `multi_cam_weight`"
            multi_cam_weight = np.array(multi_cam_weight)
            self.multi_cam_weight = multi_cam_weight / multi_cam_weight.sum()
        
        self.frame_sample_mode = frame_sample_mode
        self.cur_it: int = None

        self.ddp = ddp
        self.sampler, self.scene_weights = get_frame_sampler(self.scene_loader, frame_sample_mode=self.frame_sample_mode, ddp=ddp, **sampler_kwargs)

    @property
    def device(self) -> torch.device:
        return self.scene_loader.device
    
    @torch.no_grad()
    def sample(self, scene_id: str, cam_id: Union[str, List[str]], cam_fi: Union[int, List[int]], stack=True):
        scene = self.scene_loader.scene_bank[scene_id]
        cam = scene.observers[cam_id]
        
        if isinstance(cam_id, str):
            ground_truth = self.scene_loader.get_batched_image_and_gts(scene_id, cam_id, cam_fi, stack=stack)
        elif isinstance(cam_id, list):
            if isinstance(cam_fi, list):
                assert len(cam_id) == len(cam_fi), "If both cam_id and cam_fi are lists, they should form one-to-one pairs"
                ground_truth = [self.scene_loader.get_batched_image_and_gts(scene_id, ci, fi) for ci, fi in zip(cam_id, cam_fi)]
            else:
                ground_truth = [self.scene_loader.get_batched_image_and_gts(scene_id, ci, cam_fi) for ci in cam_id]
            ground_truth = collate_nested_dict(ground_truth, stack=stack)
        
        # sample = dict(scene_id=scene_id, cam_id=cam_id, cam_fi=cam_fi, rays_sel=None)
        # sample = dict(scene_id=scene_id, cam_id=cam_id, cam_fi=cam_fi, cam_sel=None)
        sample = dict(scene_id=scene_id, cam_id=cam_id, cam_fi=cam_fi)
        
        # NOTE: Moved to trainer to allow for differentiable timestamps
        # if isinstance(cam_id, (List, Tuple)):
        #     if cam[0].frame_global_ts is not None:
        #         cam_ts = [c.frame_global_ts[cam_fi] for c in cam]
        #         sample.update(cam_ts=cam_ts)
        # else:
        #     if cam.frame_global_ts is not None:
        #         cam_ts = cam.frame_global_ts[cam_fi]
        #         sample.update(cam_ts=cam_ts)
        
        return sample, ground_truth

    def sample_cam_id(self) -> str:
        if self.multi_cam_weight is not None:
            return random.choice(self.scene_loader.cam_id_list)
        else:
            return np.random.choice(self.scene_loader.cam_id_list, p=self.multi_cam_weight)

    def __len__(self):
        # Total number of frames of all scenes
        return sum([len(scene) for scene in self.scene_loader.scene_bank])

    def __getitem__(self, index: int) -> Tuple[dict, dict]:
        # TODO: Allow for custom and different frame lengths across different sensors
        scene_idx, cam_fi = self.scene_loader.get_scene_frame_idx(index)
        scene_id = self.scene_id_list[scene_idx]
        if 'all' in self.camera_sample_mode:
            cam_id = self.scene_loader.cam_id_list
            stack = 'stack' in self.camera_sample_mode
        else:
            cam_id = self.sample_cam_id()
            stack = False
        return self.sample(scene_id, cam_id, cam_fi, stack=stack)

    def get_dataloader(self, num_workers: int=0):
        return DataLoader(
            self, sampler=self.sampler, collate_fn=collate_tuple_of_nested_dict, 
            num_workers=0 if (self.scene_loader.preload or not self.ddp) else num_workers)

class ImagePatchDataset(torch_data.Dataset):
    def __init__(
        self, scene_loader: SceneDataLoader, *, 
        patch_sampler_cfg: dict = dict(), 
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
            scene_loader (SceneDataLoader): The base SceneDataLoader
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
        
        self.scene_loader = scene_loader
        self.scene_id_list = list(self.scene_loader.scene_bank.keys())
        
        self.camera_sample_mode = camera_sample_mode
        self.multi_cam_weight = None
        if self.camera_sample_mode == 'weighted':
            assert multi_cam_weight is not None, f"Please specify `multi_cam_weight`"
            multi_cam_weight = np.array(multi_cam_weight)
            self.multi_cam_weight = multi_cam_weight / multi_cam_weight.sum()
        
        self.frame_sample_mode = frame_sample_mode
        
        self.cur_it: int = None
        self.patch_sampler = get_patch_sampler(**patch_sampler_cfg)

        self.ddp = ddp
        self.sampler, self.scene_weights = get_frame_sampler(self.scene_loader, frame_sample_mode=self.frame_sample_mode, ddp=ddp, **sampler_kwargs)

    @property
    def device(self) -> torch.device:
        return self.scene_loader.device
    
    def sample(self, scene_id: str, cam_id: str, cam_fi: int):
        assert self.cur_it is not None, f"Requires settings `cur_it` for {self.__class__.__name__}"
        
        scene = self.scene_loader.scene_bank[scene_id]
        cam = scene.observers[cam_id]
        
        # Get image width and height
        im_metas = self.scene_loader.get_image_metas(scene_id, cam_id, cam_fi)
        W, H = im_metas['image_wh'].tolist()
        
        # Sample a patch
        xy, hw, dbg_infos = self.patch_sampler.sample_pixels(self.cur_it, (H,W), device=self.device)
        
        # Get ground truth
        ground_truth = self.scene_loader.get_image_and_gts(scene_id, cam_id, cam_fi, hw[..., 0], hw[..., 1], device=self.device)
        
        # Prepare output
        rays_fidx = torch.full(xy.shape[:-1], cam_fi, dtype=torch.long, device=self.device)
        sample = dict(
            scene_id=scene_id, 
            cam_id=cam_id, cam_fi=cam_fi, # cam_sel=None, 
            rays_pix=xy, rays_fidx=rays_fidx, rays_sel=None, 
            dbg_infos=dbg_infos
        )
        
        # NOTE: Moved to trainer to allow for differentiable timestamps
        # # Get timestamps
        # if cam.frame_global_ts is not None:
        #     # NOTE: `cam_ts` is for freezing the scene at a certain timestamp;
        #     #       `rays_ts` is for network input.
        #     #---- Opt1: Respect potential rolling shutter effect.
        #     sample['cam_ts'] = sample['rays_ts'] = cam.get_timestamps(fi=rays_fidx, pix=xy)
        #     #---- Opt2: Ignore rolling shutter effect on time slice for now; \
        #     #           Still interp at single timestamp, while the input `rays_ts` can be different.
        #     # sample['cam_ts'] = cam.frame_global_ts[cam_fi]
        #     # sample['rays_ts'] = cam.get_timestamps(fi=rays_fidx, pix=xy)
        
        return sample, ground_truth
    
    def sample_cam_id(self) -> str:
        if self.multi_cam_weight is not None:
            return np.random.choice(self.scene_loader.cam_id_list, p=self.multi_cam_weight)
        else:
            return random.choice(self.scene_loader.cam_id_list)

    def __len__(self):
        return sum([len(scene) for scene in self.scene_loader.scene_bank]) # Total number of frames of all scenes
    
    def __getitem__(self, index: int) -> Tuple[dict, dict]:
        # TODO: Allow for custom and different frame lengths across different sensors
        scene_idx, cam_fi = self.scene_loader.get_scene_frame_idx(index)
        scene_id = self.scene_id_list[scene_idx]
        if 'all' in self.camera_sample_mode:
            cam_id_list = self.scene_loader.cam_id_list
            stack = 'stack' in self.camera_sample_mode
            ret = [self.sample(scene_id, cam_id, cam_fi) for cam_id in cam_id_list]
            ret = collate_tuple_of_nested_dict(ret, stack=stack)
        else:
            cam_id = self.sample_cam_id()
            ret = self.sample(scene_id, cam_id, cam_fi)
        return ret

    def get_dataloader(self, num_workers: int=0):
        return DataLoader(
            self, sampler=self.sampler, collate_fn=collate_tuple_of_nested_dict, 
            num_workers=0 if (self.scene_loader.preload or not self.ddp) else num_workers)
