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

from nr3d_lib.utils import collate_tuple_of_nested_dict
from nr3d_lib.models.importance import ImpSampler

from .base_loader import SceneDataLoader
from .sampler import get_scene_sampler, get_frame_sampler, get_frame_weights_from_error_map

class PixelDatasetBase(torch_data.Dataset):
    def __init__(
        self, scene_loader: SceneDataLoader,
        num_rays: int = 4096, num_points: int = None,
        equal_mode: Literal['ray_batch', 'point_batch'] = 'ray_batch', 
        ) -> None:
        """
        The common base pixel sampler.

        Args:
            scene_loader (SceneDataLoader): The base SceneDataLoader.
            equal_mode (Literal['ray_batch', 'point_batch'], optional): 
                Determines the mode of the number of rays sampled each time. 
                - `ray_batch`: A fixed number of rays are sampled each time.
                - `point_batch`: [Not supported yet] A fixed number of sampling points are sampled each time; \
                    the number of rays can vary. Similar to the instant-ngp mode.
                Defaults to 'ray_batch'.
            num_rays (int, optional): The number of rays/pixels sampled each time when the `equal_mode` is set to 'ray_batch'. Defaults to 4096.
            num_points (int, optional): The number of points sampled each time when the `equal_mode` is set to 'point_batch'. Defaults to None.
        """
        
        super().__init__()

        self.scene_loader = scene_loader
        self.scene_id_list = list(self.scene_loader.scene_bank.keys())
        self.equal_mode = equal_mode

        self.set_n_rays = num_rays
        self.set_n_pts = num_points
        
        # Init num_rays
        self.num_rays = num_rays
        self.cur_it = np.inf
        
    @property
    def device(self) -> torch.device:
        return self.scene_loader.device

    def set_imp_samplers(self, imp_samplers, enable_after: int = 0):
        self.imp_samplers_enable_after: int = enable_after
        self.imp_samplers: Dict[str, Dict[str,ImpSampler]] = imp_samplers

    def record_prev(self, totol_n_pts: int):
        raise NotImplementedError("TODO in v0.5.2")
        if self.equal_mode == 'point_batch':
            self.num_rays = int(self.set_n_pts / (totol_n_pts/self.num_rays))

class PixelDataset(PixelDatasetBase):
    def __init__(
        self, scene_loader: SceneDataLoader, *, 
        # Pixels
        pixel_sample_mode: Literal['uniform', 'error_map']='error_map',
        # Cameras and frames
        camera_sample_mode: Literal['uniform', 'weighted', 'all_list', 'all_stack']='uniform', 
        multi_cam_weight: List[float]=None, 
        frame_sample_mode: Literal['uniform', 'weighted_by_speed', 'error_map']='uniform', 
        # Basic
        equal_mode: Literal['ray_batch', 'point_batch'] = 'ray_batch', 
        num_rays: int = 4096, num_points: int = None, 
        ddp=False, **sampler_kwargs
        ) -> None:
        """ 
        Pixel sampler: in one sampled batch, rays originate from the same frame of the same observer.

        Args:
            scene_loader (SceneDataLoader): The base SceneDataLoader.
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
            equal_mode (Literal['ray_batch', 'point_batch'], optional): 
                Determines the mode of the number of rays sampled each time. 
                - `ray_batch`: A fixed number of rays are sampled each time.
                - `point_batch`: [Not supported yet] A fixed number of sampling points are sampled each time; \
                    the number of rays can vary. Similar to the instant-ngp mode.
                Defaults to 'ray_batch'.
            num_rays (int, optional): The number of rays/pixels sampled each time when the `equal_mode` is set to 'ray_batch'. Defaults to 4096.
            num_points (int, optional): The number of points sampled each time when the `equal_mode` is set to 'point_batch'. Defaults to None.
        """
        
        requires_error_map = 'error_map' in (frame_sample_mode, pixel_sample_mode)
        super().__init__(scene_loader, 
            equal_mode=equal_mode, num_rays=num_rays, num_points=num_points)
        
        self.camera_sample_mode = camera_sample_mode
        self.multi_cam_weight = None
        if self.camera_sample_mode == 'weighted':
            assert multi_cam_weight is not None, f"Please specify `multi_cam_weight`"
            multi_cam_weight = np.array(multi_cam_weight)
            self.multi_cam_weight = multi_cam_weight / multi_cam_weight.sum()
        
        self.frame_sample_mode = frame_sample_mode
        self.pixel_sample_mode = pixel_sample_mode
        
        self.ddp = ddp
        self.sampler, self.scene_weights = get_frame_sampler(
            self.scene_loader, frame_sample_mode=self.frame_sample_mode,
            imp_samplers=None, ddp=ddp, **sampler_kwargs)
    
    def set_imp_samplers(self, imp_samplers, enable_after: int = 0):
        super().set_imp_samplers(imp_samplers, enable_after)
        self.update_weights()
    
    def update_weights(self):
        if self.frame_sample_mode == 'error_map':
            assert self.imp_samplers is not None, \
                f"Please call set_imp_samplers() in after initializing {type(self)}"
            frame_weights_new = get_frame_weights_from_error_map(
                self.scene_loader, self.scene_weights, self.imp_samplers)
            self.sampler.set_weights(frame_weights_new)
    
    @torch.no_grad()
    def sample(self, scene_id: str, cam_id: str, cam_fi: int, *, num_rays: int = None):
        if num_rays is None: num_rays = self.num_rays
        
        if self.pixel_sample_mode == 'uniform':
            xy = torch.rand([num_rays, 2], device=self.device).clamp_(1e-6, 1-1e-6)
        elif self.pixel_sample_mode == 'error_map':
            assert self.imp_samplers is not None, \
                f"Please call set_imp_samplers() in after initializing {type(self)}"
            imp_sampler = self.imp_samplers[scene_id][cam_id]
            if self.cur_it < self.imp_samplers_enable_after:
                xy = torch.rand([num_rays, 2], device=self.device).clamp_(1e-6, 1-1e-6)
            else:
                xy = imp_sampler.sample_pixel(num_rays, cam_fi)
        else:
            raise RuntimeError(f"Invalid pixel_sample_mode={self.pixel_sample_mode}")

        ground_truth = self.scene_loader.get_image_and_gts(scene_id, cam_id, cam_fi)
        WH = ground_truth['image_wh'].view(1,2)
        w, h = (WH * xy).long().clamp_(WH.new_zeros([]), WH-1).movedim(-1,0)
        inds = (h, w)
        
        scene = self.scene_loader.scene_bank[scene_id]
        cam = scene.observers[cam_id]
        
        rays_fidx = torch.full(h.shape, cam_fi, dtype=torch.long, device=h.device)
        
        # [num_rays, 3]
        ground_truth.update({k: ground_truth[k][inds] for k in self.scene_loader.gt_im_keys if k in ground_truth})

        sample = dict(
            scene_id=scene_id, 
            cam_id=cam_id, cam_fi=cam_fi, # cam_sel=None,
            rays_pix=xy, rays_fidx=rays_fidx, rays_sel=None, 
        )
        
        # NOTE: Moved to trainer to allow for differentiable timestamps
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
        # Total number of frames of all scenes
        return sum([len(scene) for scene in self.scene_loader.scene_bank])

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

class JointFramePixelDataset(PixelDatasetBase):
    def __init__(
        self, scene_loader: SceneDataLoader, *, 
        # Cameras
        camera_sample_mode: Literal['uniform', 'weighted']='uniform', 
        multi_cam_weight: List[float]=None, 
        # Basic
        equal_mode: Literal['ray_batch', 'point_batch'] = 'ray_batch', 
        num_rays: int = 4096, num_points: int = None,
        ddp=False, **sampler_kwargs
        ) -> None:
        """
        Joint frame & pixel sampler: in one sampled batch, rays could originate from different frames of image, \
            since frame index is jointly sampled along with ray index in importance sampling

        Args:
            scene_loader (SceneDataLoader): The base SceneDataLoader.
            camera_sample_mode (Literal['uniform', 'weighted'], optional): 
                Determines the method for sampling cameras from multiple options. 
                - `uniform`: Samples one camera uniformly.
                - `weighted`: Samples one camera based on `multi_cam_weight`.
                Defaults to 'uniform'.
            multi_cam_weight (List[float], optional): Weight applied to different cameras in `weighted` mode. Default is None.
            equal_mode (Literal['ray_batch', 'point_batch'], optional): 
                Determines the mode of the number of rays sampled each time. 
                - `ray_batch`: A fixed number of rays are sampled each time.
                - `point_batch`: [Not supported yet] A fixed number of sampling points are sampled each time; \
                    the number of rays can vary. Similar to the instant-ngp mode.
                Defaults to 'ray_batch'.
            num_rays (int, optional): The number of rays/pixels sampled each time when the `equal_mode` is set to 'ray_batch'. Defaults to 4096.
            num_points (int, optional): The number of points sampled each time when the `equal_mode` is set to 'point_batch'. Defaults to None.
        """
        super().__init__(
            scene_loader, 
            equal_mode=equal_mode, num_rays=num_rays, num_points=num_points)

        self.camera_sample_mode = camera_sample_mode
        self.multi_cam_weight = None
        if self.camera_sample_mode == 'weighted':
            assert multi_cam_weight is not None, f"Please specify `multi_cam_weight`"
            multi_cam_weight = np.array(multi_cam_weight)
            self.multi_cam_weight = multi_cam_weight / multi_cam_weight.sum()
        
        self.ddp = ddp
        self.sampler = get_scene_sampler(self.scene_loader, ddp=ddp, **sampler_kwargs)
    
    @torch.no_grad()
    def sample(self, scene_id: str, cam_id: str, *, num_rays: int = None):
        assert self.imp_samplers is not None, \
            f"Please call set_imp_samplers() in after initializing {type(self)}"
        
        if num_rays is None: num_rays = self.num_rays
        imp_sampler = self.imp_samplers[scene_id][cam_id]
        if self.cur_it < self.imp_samplers_enable_after:
            # (Optional) Pure single-frame uniform sampling, do not repsect errormap
            cam_fi: int = np.random.randint(imp_sampler.n_images)
            fi = torch.full([self.num_rays], cam_fi, dtype=torch.long, device=self.device)
            xy = torch.rand([self.num_rays, 2], device=self.device).clamp_(1e-6, 1-1e-6)
            
            ground_truth = self.scene_loader.get_batched_image_and_gts(scene_id, cam_id, cam_fi)
            WH = ground_truth['image_wh'].view(1,2)
            w, h = (WH * xy).long().clamp_(WH.new_zeros([]), WH-1).movedim(-1,0)
            
            # [num_rays, 3]
            ground_truth.update({k: ground_truth[k][h, w] for k in self.scene_loader.gt_im_keys if k in ground_truth})
        
        else:
            # (Main) Joint frame-pixel importance sampling based on errormap.
            fi, xy = imp_sampler.sample_img_pixel(self.num_rays)
            # cam_fi, inv = torch.unique(fi, return_inverse=True)
            # cam_fi: np.ndarray = cam_fi.cpu().numpy()
            
            # [num_rays, 2]
            #-- Opt1: Get from scene_loader
            # WH = self.scene_loader.get_image_metas(scene_id, cam_id, cam_fi, device=self.device)['image_wh'][inv]
            WH = self.scene_loader.get_image_metas(scene_id, cam_id, fi.cpu(), device=self.device)['image_wh']
            #-- Opt2: Get from camera
            # scene = self.scene_loader.scene_bank[scene_id]
            # cam = scene.observers[cam_id]
            # cam.frame_data.intr.set_downscale(self.config.tags.camera.downscale)
            # WH = cam.frame_data.intr.wh()[fi] # Directly get from frame_data (raw data)
            
            w, h = (WH * xy).long().clamp_(WH.new_zeros([]), WH-1).movedim(-1,0).cpu()
            # [num_rays, 3]
            ground_truth = self.scene_loader.get_image_and_gts(scene_id, cam_id, fi.cpu(), h, w)

        scene = self.scene_loader.scene_bank[scene_id]
        cam = scene.observers[cam_id]

        sample = dict(
            scene_id=scene_id, 
            cam_id=cam_id, cam_fi=fi, # cam_sel=None, 
            rays_fidx=fi, rays_pix=xy, rays_sel=None, 
        )
        
        # NOTE: Moved to trainer to allow for differentiable timestamps
        # if cam.frame_global_ts is not None:
        #     sample['cam_ts'] = sample['rays_ts'] = cam.get_timestamps(fi=fi, pix=xy)
        
        return sample, ground_truth

    def sample_cam_id(self) -> str:
        if self.multi_cam_weight is not None:
            return random.choice(self.scene_loader.cam_id_list)
        else:
            return np.random.choice(self.scene_loader.cam_id_list, p=self.multi_cam_weight)

    def __len__(self):
        # Total number of scenes. (Since this is a scene-wise scene_loader)
        return len(self.scene_loader.scene_bank)
    
    def __getitem__(self, index: int) -> Tuple[dict, dict]:
        scene_id = self.scene_id_list[index]
        cam_id = self.sample_cam_id()
        return self.sample(scene_id, cam_id)

    def get_dataloader(self, num_workers: int=0):
        return DataLoader(
            self, sampler=self.sampler, collate_fn=collate_tuple_of_nested_dict, 
            num_workers=0 if (self.scene_loader.preload or not self.ddp) else num_workers)
