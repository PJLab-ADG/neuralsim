"""
@file   base.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Defines basic common IO logics for dataset loading and caching.
"""

__all__ = [
    'SceneDataLoader'
]

import numpy as np
from tqdm import tqdm
from typing import Dict, Iterator, List, Literal, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.data as torch_data

from nr3d_lib.fmt import log
from nr3d_lib.geometry import gmean
from nr3d_lib.config import ConfigDict
from nr3d_lib.models.attributes import TransformMat3x4, Scale
from nr3d_lib.utils import IDListedDict, collate_nested_dict, img_to_torch_and_downscale, pad_images_to_same_size

from nr3d_lib.models.importance import ImpSampler

from app.resources import Scene
from app.resources.observers import MultiCamBundle, MultiRaysLidarBundle

from dataio.dataset_io import DatasetIO

class SceneDataLoader(object):
    """
    - Defines the basic and common IO logic for the dataset.
    - Loads and caches common APIs used in dataio/dataset_impl.py, including scenarios, images, LiDAR data, and annotations.
    - Performs image downsampling.
    
    NOTE: The difference between `_get_xxx` and `get_xxx`
    - `self._get_xxx` directly reads data from disks.
    - `self.get_xxx` first checks whether it's in cache mode and if cache data already exists. \
        If so, it retrieves data directly from the cache; otherwise, it reads data from the disk (by invoking `self._get_xxx`).
    """
    def __init__(
        self, 
        scene_or_scene_bank: Union[IDListedDict[Scene], Scene], 
        dataset_impl: DatasetIO, *, 
        config: ConfigDict, 
        device=torch.device('cuda:0')) -> None:
        super().__init__()

        self.device = device
        self.config: ConfigDict = config.deepcopy()
        
        if isinstance(scene_or_scene_bank, Scene):
            self.scene_bank: IDListedDict[Scene] = IDListedDict([scene_or_scene_bank])
        elif isinstance(scene_or_scene_bank, IDListedDict):
            self.scene_bank: IDListedDict[Scene] = scene_or_scene_bank
        else:
            raise RuntimeError(f"Invalid input type: {type(scene_or_scene_bank)}")
        
        # Dataset-specific operations
        self.dataset_impl: DatasetIO = dataset_impl

        # Ground-truth image-like data fields (not all necessary)
        self.gt_im_fields = ['rgb', 'rgb_mask', 'rgb_dynamic_mask', 'rgb_human_mask', 'rgb_road_mask', 'rgb_ignore_mask', 'rgb_mono_depth', 'rgb_mono_normals']

        # (Optional) Data caching
        self._cache_rgbs: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {}
        self._cache_rgb_masks: Dict[str, Dict[str, torch.Tensor]] = {}
        self._cache_rgb_dynamic_masks: Dict[str, Dict[str, torch.Tensor]] = {}
        self._cache_rgb_human_masks: Dict[str, Dict[str, torch.Tensor]] = {}
        self._cache_rgb_road_masks: Dict[str, Dict[str, torch.Tensor]] = {}
        self._cache_rgb_dataset_ignore_masks: Dict[str, Dict[str, torch.Tensor]] = {}
        self._cache_rgb_depths: Dict[str, Dict[str, torch.Tensor]] = {}
        self._cache_rgb_normals: Dict[str, Dict[str, torch.Tensor]] = {}
        self._cache_lidars: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {}
        self._cache_merged_lidars: Dict[str, List[Dict[str, torch.Tensor]]] = {}
        self._cache_rays: Dict[str, Dict[str, torch.Tensor]] = {}

        self.preload = self.config.get('preload', False)
        cache_device = self.device if self.config.get('preload_on_gpu', False) else torch.device('cpu')
        if 'lidar' in self.config.tags:
            self.config.tags.lidar.setdefault('filter_when_preload', False)
        if self.preload:
            self._build_cache(cache_device)

    def _build_cache(self, cache_device):
        log.info(f"=> Caching data to device={cache_device}...")
        
        for scene in self.scene_bank:
            scene_id = scene.id
            
            if 'camera' in self.config.tags:
                self._cache_rgbs[scene_id] = {}
                log.info("=> Caching camera data...")
                for cam_id in tqdm(self.cam_id_list, "Caching cameras..."):
                    obs = scene.observers[cam_id]
                    if self.config.tags.camera.get('load_images', True):
                        rgb_gts = {}
                        for frame_ind in range(len(scene)):
                            _ret = self._get_rgb(scene_id=scene_id, cam_id=cam_id, frame_ind=frame_ind, device=cache_device)
                            for k, v in _ret.items():
                                rgb_gts.setdefault(k, []).append(v)
                        for k, v in rgb_gts.items():
                            rgb_gts[k] = torch.stack(v, dim=0)
                        self._cache_rgbs[scene_id][cam_id] = rgb_gts

                    # Cache occupancy masks
                    if 'rgb_mask' in self.config.tags:
                        if scene_id not in self._cache_rgb_masks:
                            self._cache_rgb_masks[scene_id] = {}
                        masks = []
                        for frame_ind in range(len(scene)):
                            masks.append(self._get_occupancy_mask(scene_id=scene_id, cam_id=cam_id, frame_ind=frame_ind, device=cache_device).to(torch.bool))
                        self._cache_rgb_masks[scene_id][cam_id] = torch.stack(masks, dim=0)

                    # Cache dynamic masks
                    if 'rgb_dynamic_mask' in self.config.tags:
                        if scene_id not in self._cache_rgb_dynamic_masks:
                            self._cache_rgb_dynamic_masks[scene_id] = {}
                        dynamic_masks = []
                        for frame_ind in range(len(scene)):
                            dynamic_masks.append(self._get_rgb_dynamic_mask(scene_id=scene_id, cam_id=cam_id, frame_ind=frame_ind, device=cache_device).to(torch.bool))
                        self._cache_rgb_dynamic_masks[scene_id][cam_id] = torch.stack(dynamic_masks, dim=0)

                    # Cache human masks
                    if 'rgb_human_mask' in self.config.tags:
                        if scene_id not in self._cache_rgb_human_masks:
                            self._cache_rgb_human_masks[scene_id] = {}
                        human_masks = []
                        for frame_ind in range(len(scene)):
                            human_masks.append(self._get_rgb_human_mask(scene_id=scene_id, cam_id=cam_id, frame_ind=frame_ind, device=cache_device).to(torch.bool))
                        self._cache_rgb_human_masks[scene_id][cam_id] = torch.stack(human_masks, dim=0)

                    # Cache road masks
                    if 'rgb_road_mask' in self.config.tags:
                        if scene_id not in self._cache_rgb_road_masks:
                            self._cache_rgb_road_masks[scene_id] = {}
                        road_masks = []
                        for frame_ind in range(len(scene)):
                            road_masks.append(self._get_rgb_road_mask(scene_id=scene_id, cam_id=cam_id, frame_ind=frame_ind, device=cache_device).to(torch.bool))
                        self._cache_rgb_road_masks[scene_id][cam_id] = torch.stack(road_masks, dim=0)

                    # Cache dataset ignore mask
                    if 'rgb_ignore_mask' in self.config.tags and self.config.tags.rgb_ignore_mask.get('from_dataset', False):
                        if scene_id not in self._cache_rgb_dataset_ignore_masks:
                            self._cache_rgb_dataset_ignore_masks[scene_id] = {}
                        data_ignore_masks = []
                        for frame_ind in range(len(scene)):
                            data_ignore_masks.append(self._get_rgb_dataset_ignore_mask(scene_id=scene_id, cam_id=cam_id, frame_ind=frame_ind, device=cache_device).to(torch.bool))
                        self._cache_rgb_dataset_ignore_masks[scene_id][cam_id] = torch.stack(data_ignore_masks, dim=0)

                    # Cache depth gts
                    if 'rgb_mono_depth' in self.config.tags:
                        if scene_id not in self._cache_rgb_depths:
                            self._cache_rgb_depths[scene_id] = {}
                        depths = []
                        for frame_ind in range(len(scene)):
                            depths.append(self._get_mono_depth(scene_id=scene_id, cam_id=cam_id, frame_ind=frame_ind, device=cache_device).to(torch.float))
                        self._cache_rgb_depths[scene_id][cam_id] = torch.stack(depths, dim=0)

                    # Cache normal gts
                    if 'rgb_mono_normals' in self.config.tags:
                        if scene_id not in self._cache_rgb_normals:
                            self._cache_rgb_normals[scene_id] = {}
                        normals = []
                        for frame_ind in range(len(scene)):
                            normals.append(self._get_mono_normals(scene_id=scene_id, cam_id=cam_id, frame_ind=frame_ind, device=cache_device).to(torch.float))
                        self._cache_rgb_normals[scene_id][cam_id] = torch.stack(normals, dim=0)

                    # Cache rays directly if camera intr is not learnable
                    # NOTE: OOM, cost very high
                    # if len(list(obs.attr_segments.parameters())) == 0:
                    #     if scene_id not in self._cache_rays:
                    #         self._cache_rays[scene_id] = {}
                    #     rays = []
                    #     for frame_ind in range(len(scene)):
                    #         obs.to(self.device)
                    #         obs.frozen_at(frame_ind)
                    #         if (ds:=self.config.tags.camera.downscale) != 1:
                    #             obs.set_camera_downscale(ds)
                    #         rays_o, rays_d = obs.get_all_rays()
                    #         rays.append(torch.cat([rays_o, rays_d], dim=-1))
                    #     self._cache_rays[scene_id][cam_id] = torch.stack(rays, dim=0)
            
            if 'lidar' in self.config.tags:
                log.info("=> Caching lidar data...")
                multi_lidar_merge = self.config.tags.lidar.multi_lidar_merge
                filter_when_preload = self.config.tags.lidar.filter_when_preload
                if filter_when_preload:
                    assert 'filter_kwargs' in self.config.tags.lidar, "You need to specify filter_kwargs when filter_when_preload"
                    filter_kwargs = self.config.tags.lidar.filter_kwargs
                
                if multi_lidar_merge:
                    self._cache_merged_lidars[scene_id] = []
                    for frame_ind in tqdm(range(len(scene)), "Caching merged LiDAR frames..."):
                        _ret = self._get_merged_lidar_gts(scene_id=scene_id, frame_ind=frame_ind, device=cache_device)
                        if filter_when_preload: # Calculates on cuda device 
                            _ret = {k: v.to(scene.device) for k, v in _ret.items()}
                            _ret = self.filter_lidar_gts(scene_id=scene_id, lidar_id=self.lidar_id_list, frame_ind=frame_ind, 
                                                         lidar_data=_ret, inplace=True, **filter_kwargs)
                            _ret = {k: v.to(cache_device) for k, v in _ret.items()}
                        self._cache_merged_lidars[scene_id].append(_ret)
                else: # not multi_lidar_merge
                    self._cache_lidars[scene_id] = {}
                    for lidar_id in tqdm(self.lidar_id_list, "Caching LiDARs..."):
                        lidar_gts = {}
                        for frame_ind in range(len(scene)):
                            _ret = self._get_lidar_gts(scene_id=scene_id, lidar_id=lidar_id, frame_ind=frame_ind, device=cache_device)
                            if filter_when_preload: # Calculates on cuda device 
                                _ret = {k: v.to(scene.device) for k, v in _ret.items()}
                                _ret = self.filter_lidar_gts(scene_id=scene_id, lidar_id=lidar_id, frame_ind=frame_ind, 
                                                             lidar_data=_ret, inplace=True, **filter_kwargs)
                                _ret = {k: v.to(cache_device) for k, v in _ret.items()}
                            for k, v in _ret.items():
                                lidar_gts.setdefault(k, []).append(v)
                        self._cache_lidars[scene_id][lidar_id] = lidar_gts
        
        log.info("=> Done caching data.")

    def set_camera_downscale(self, downscale: float):
        self.config.tags.camera.downscale = downscale

    @property
    def cam_id_list(self) -> List[str]:
        if 'camera' in self.config.tags:
            return self.config.tags.camera.list
        else:
            return []
    @cam_id_list.setter
    def cam_id_list(self, value: List[str]):
        self.config.tags.camera.list = value
    
    @property
    def lidar_id_list(self) -> List[str]:
        return self.config.tags.lidar.list
    @lidar_id_list.setter
    def lidar_id_list(self, value: List[str]):
        self.config.tags.lidar.list = value

    def _get_rgb(self, scene_id: str, cam_id: str, frame_ind: int, device=None, bypass_downscale:float = None):
        data_frame_offset = self.scene_bank[scene_id].data_frame_offset
        rgb_np = self.dataset_impl.get_image(scene_id, cam_id, frame_ind + data_frame_offset)
        H, W, *_ = rgb_np.shape
        rgb = img_to_torch_and_downscale(rgb_np, dtype=torch.float, device=device,
            downscale=bypass_downscale or self.config.tags.camera.downscale, 
            use_cpu_downscale=self.config.get('use_cpu_downscale', False))
        H_, W_, *_ = rgb.shape
        return dict(
            rgb=rgb.to(dtype=torch.float, device=device),
            rgb_downscale=torch.tensor([W/W_, H/H_], dtype=torch.float, device=device),
            rgb_wh=torch.tensor([W_, H_], dtype=torch.long, device=device)
        )
    # @profile
    def get_rgb(self, scene_id: str, cam_id: str, frame_ind: Union[int, List[int]], device=None, bypass_downscale: float = None) -> Dict[str, torch.Tensor]:
        ds_need_reload = (bypass_downscale is not None) and (abs(bypass_downscale-self.config.tags.camera.downscale)>1e-5)
        if (not ds_need_reload) and (scene_id in self._cache_rgbs) and (cam_id in self._cache_rgbs[scene_id]):
            _ret = self._cache_rgbs[scene_id][cam_id]
            return {k: v[frame_ind].to(device) for k,v in _ret.items()}
        else:
            assert isinstance(frame_ind, int), "Only support single frame_ind when not caching."
            return self._get_rgb(scene_id, cam_id, frame_ind, device=device, bypass_downscale=bypass_downscale)

    def _get_occupancy_mask(self, scene_id: str, cam_id: str, frame_ind: int, device=None) -> torch.BoolTensor:
        data_frame_offset = self.scene_bank[scene_id].data_frame_offset
        mask_np = self.dataset_impl.get_occupancy_mask(scene_id, cam_id, frame_ind + data_frame_offset)
        mask = img_to_torch_and_downscale(mask_np, dtype=torch.float, device=device, 
            downscale=self.config.tags.camera.downscale, 
            use_cpu_downscale=self.config.get('use_cpu_downscale', False))
        return mask.to(dtype=torch.bool, device=device)
    def get_occupancy_mask(self, scene_id: str, cam_id: str, frame_ind: Union[int, List[int]], device=None) -> torch.BoolTensor:
        if scene_id in self._cache_rgb_masks and cam_id in self._cache_rgb_masks[scene_id]:
            return self._cache_rgb_masks[scene_id][cam_id][frame_ind].to(device=device, dtype=torch.bool)
        else:
            assert isinstance(frame_ind, int), "Only support single frame_ind when not caching."
            return self._get_occupancy_mask(scene_id, cam_id, frame_ind, device=device).to(dtype=torch.bool)

    def _get_rgb_dynamic_mask(self, scene_id: str, cam_id: str, frame_ind: int, device=None) -> torch.BoolTensor:
        data_frame_offset = self.scene_bank[scene_id].data_frame_offset
        mask_np = self.dataset_impl.get_dynamic_mask(scene_id, cam_id, frame_ind + data_frame_offset)
        mask = img_to_torch_and_downscale(mask_np, dtype=torch.float, device=device, 
            downscale=self.config.tags.camera.downscale, 
            use_cpu_downscale=self.config.get('use_cpu_downscale', False))
        return mask.to(dtype=torch.bool, device=device)
    def get_rgb_dynamic_mask(self, scene_id: str, cam_id: str, frame_ind: Union[int, List[int]], device=None) -> torch.BoolTensor:
        if scene_id in self._cache_rgb_dynamic_masks and cam_id in self._cache_rgb_dynamic_masks[scene_id]:
            return self._cache_rgb_dynamic_masks[scene_id][cam_id][frame_ind].to(device=device, dtype=torch.bool)
        else:
            assert isinstance(frame_ind, int), "Only support single frame_ind when not caching."
            return self._get_rgb_dynamic_mask(scene_id, cam_id, frame_ind, device=device).to(dtype=torch.bool)

    def _get_rgb_human_mask(self, scene_id: str, cam_id: str, frame_ind: int, device=None) -> torch.BoolTensor:
        data_frame_offset = self.scene_bank[scene_id].data_frame_offset
        mask_np = self.dataset_impl.get_human_mask(scene_id, cam_id, frame_ind + data_frame_offset)
        mask = img_to_torch_and_downscale(mask_np, dtype=torch.float, device=device, 
            downscale=self.config.tags.camera.downscale, 
            use_cpu_downscale=self.config.get('use_cpu_downscale', False))
        return mask.to(dtype=torch.bool, device=device)
    def get_rgb_human_mask(self, scene_id: str, cam_id: str, frame_ind: Union[int, List[int]], device=None) -> torch.BoolTensor:
        if scene_id in self._cache_rgb_human_masks and cam_id in self._cache_rgb_human_masks[scene_id]:
            return self._cache_rgb_human_masks[scene_id][cam_id][frame_ind].to(device=device, dtype=torch.bool)
        else:
            assert isinstance(frame_ind, int), "Only support single frame_ind when not caching."
            return self._get_rgb_human_mask(scene_id, cam_id, frame_ind, device=device).to(dtype=torch.bool)

    def _get_rgb_road_mask(self, scene_id: str, cam_id: str, frame_ind: int, device=None) -> torch.BoolTensor:
        data_frame_offset = self.scene_bank[scene_id].data_frame_offset
        mask_np = self.dataset_impl.get_road_mask(scene_id, cam_id, frame_ind + data_frame_offset)
        mask = img_to_torch_and_downscale(mask_np, dtype=torch.float, device=device, 
            downscale=self.config.tags.camera.downscale, 
            use_cpu_downscale=self.config.get('use_cpu_downscale', False))
        return mask.to(dtype=torch.bool, device=device)
    def get_rgb_road_mask(self, scene_id: str, cam_id: str, frame_ind: Union[int, List[int]], device=None) -> torch.BoolTensor:
        if scene_id in self._cache_rgb_road_masks and cam_id in self._cache_rgb_road_masks[scene_id]:
            return self._cache_rgb_road_masks[scene_id][cam_id][frame_ind].to(device=device, dtype=torch.bool)
        else:
            assert isinstance(frame_ind, int), "Only support single frame_ind when not caching."
            return self._get_rgb_road_mask(scene_id, cam_id, frame_ind, device=device).to(dtype=torch.bool)

    def _get_rgb_dataset_ignore_mask(self, scene_id: str, cam_id: str, frame_ind: int, device=None) -> torch.BoolTensor:
        data_frame_offset = self.scene_bank[scene_id].data_frame_offset
        mask_np = self.dataset_impl.get_ignore_mask(scene_id, cam_id, frame_ind + data_frame_offset)
        mask = img_to_torch_and_downscale(mask_np, dtype=torch.float, device=device, 
            downscale=self.config.tags.camera.downscale, 
            use_cpu_downscale=self.config.get('use_cpu_downscale', False))
        return mask.to(dtype=torch.bool, device=device)
    def get_rgb_dataset_ignore_mask(self, scene_id: str, cam_id: str, frame_ind: Union[int, List[int]], device=None) -> torch.BoolTensor:
        if scene_id in self._cache_rgb_dataset_ignore_masks and cam_id in self._cache_rgb_dataset_ignore_masks[scene_id]:
            return self._cache_rgb_dataset_ignore_masks[scene_id][cam_id][frame_ind].to(device=device, dtype=torch.bool)
        else:
            assert isinstance(frame_ind, int), "Only support single frame_ind when not caching."
            return self._get_rgb_dataset_ignore_mask(scene_id, cam_id, frame_ind, device=device).to(dtype=torch.bool)

    def get_rgb_ignore_mask(self, scene_id: str, cam_id: str, frame_ind: Union[int, List[int]], device=None) -> torch.BoolTensor:
        cfg = self.config.tags.rgb_ignore_mask
        ignore_mask = None
        if cfg.get('from_dataset', False):
            dataset_ignore_mask = self.get_rgb_dataset_ignore_mask(scene_id, cam_id, frame_ind)
            ignore_mask = (dataset_ignore_mask | ignore_mask) if ignore_mask is not None else dataset_ignore_mask
        if cfg.get('ignore_not_occupied', False):
            not_occ_mask = self.get_occupancy_mask(scene_id, cam_id, frame_ind, device=device).logical_not()
            ignore_mask = (not_occ_mask | ignore_mask) if ignore_mask is not None else not_occ_mask
        if cfg.get('ignore_dynamic', False):
            dynamic_mask = self.get_rgb_dynamic_mask(scene_id, cam_id, frame_ind, device=device)
            ignore_mask = (dynamic_mask | ignore_mask) if ignore_mask is not None else dynamic_mask
        if cfg.get('ignore_human', False):
            human_mask = self.get_rgb_human_mask(scene_id, cam_id, frame_ind, device=device)
            ignore_mask = (human_mask | ignore_mask) if ignore_mask is not None else human_mask
        return ignore_mask

    def _get_mono_depth(self, scene_id: str, cam_id: str, frame_ind: int, device=None) -> torch.Tensor:
        data_frame_offset = self.scene_bank[scene_id].data_frame_offset
        rgb_mono_depth_np = self.dataset_impl.get_mono_depth(scene_id, cam_id, frame_ind + data_frame_offset)
        rgb_mono_depth = img_to_torch_and_downscale(rgb_mono_depth_np, dtype=torch.float, device=device, 
            downscale=self.config.tags.camera.downscale, 
            use_cpu_downscale=self.config.get('use_cpu_downscale', False))
        return rgb_mono_depth.to(dtype=torch.float, device=device)
    def get_mono_depth(self, scene_id: str, cam_id: str, frame_ind: Union[int, List[int]], device=None) -> torch.Tensor:
        if scene_id in self._cache_rgb_depths and cam_id in self._cache_rgb_depths[scene_id]:
            return self._cache_rgb_depths[scene_id][cam_id][frame_ind].to(device=device, dtype=torch.float)
        else:
            assert isinstance(frame_ind, int), "Only support single frame_ind when not caching."
            return self._get_mono_depth(scene_id, cam_id, frame_ind, device=device).to(dtype=torch.float)

    def _get_mono_normals(self, scene_id: str, cam_id: str, frame_ind: int, device=None) -> torch.Tensor:
        data_frame_offset = self.scene_bank[scene_id].data_frame_offset
        rgb_mono_normals_np = self.dataset_impl.get_mono_normals(scene_id, cam_id, frame_ind + data_frame_offset)
        rgb_mono_normals = img_to_torch_and_downscale(rgb_mono_normals_np, dtype=torch.float, device=device, 
            downscale=self.config.tags.camera.downscale, 
            use_cpu_downscale=self.config.get('use_cpu_downscale', False))
        return rgb_mono_normals.to(dtype=torch.float, device=device)
    def get_mono_normals(self, scene_id: str, cam_id: str, frame_ind: Union[int, List[int]], device=None) -> torch.Tensor:
        if scene_id in self._cache_rgb_normals and cam_id in self._cache_rgb_normals[scene_id]:
            return self._cache_rgb_normals[scene_id][cam_id][frame_ind].to(device=device, dtype=torch.float)
        else:
            assert isinstance(frame_ind, int), "Only support single frame_ind when not caching."
            return self._get_mono_normals(scene_id, cam_id, frame_ind, device=device).to(dtype=torch.float)
    # @profile
    def get_rgb_gts(self, scene_id: str, cam_id: str, frame_ind: Union[int, List[int]], device=None) -> Dict[str, torch.Tensor]:
        device = device or self.device
        # Load RGB observation
        if self.config.tags.camera.get('load_images', True):
            gt = self.get_rgb(scene_id=scene_id, cam_id=cam_id, frame_ind=frame_ind, device=device)
        # Load mask annotation on image (if any)
        if 'rgb_mask' in self.config.tags:
            gt['rgb_mask'] = self.get_occupancy_mask(scene_id=scene_id, cam_id=cam_id, frame_ind=frame_ind, device=device)
        if 'rgb_dynamic_mask' in self.config.tags:
            gt['rgb_dynamic_mask'] = self.get_rgb_dynamic_mask(scene_id=scene_id, cam_id=cam_id, frame_ind=frame_ind, device=device)
        if 'rgb_human_mask' in self.config.tags:
            gt['rgb_human_mask'] = self.get_rgb_human_mask(scene_id=scene_id, cam_id=cam_id, frame_ind=frame_ind, device=device)
        if 'rgb_road_mask' in self.config.tags:
            gt['rgb_road_mask'] = self.get_rgb_road_mask(scene_id=scene_id, cam_id=cam_id, frame_ind=frame_ind, device=device)
        if 'rgb_ignore_mask' in self.config.tags:
            cfg = self.config.tags.rgb_ignore_mask
            ignore_mask = None
            if cfg.get('from_dataset', False):
                dataset_ignore_mask = self.get_rgb_dataset_ignore_mask(scene_id, cam_id, frame_ind)
                ignore_mask = (dataset_ignore_mask | ignore_mask) if ignore_mask is not None else dataset_ignore_mask
            if cfg.get('ignore_not_occupied', False):
                not_occ_mask = gt['rgb_mask'].logical_not()
                ignore_mask = (not_occ_mask | ignore_mask) if ignore_mask is not None else not_occ_mask
            if cfg.get('ignore_dynamic', False):
                dynamic_mask = gt['rgb_dynamic_mask']
                ignore_mask = (dynamic_mask | ignore_mask) if ignore_mask is not None else dynamic_mask
            if cfg.get('ignore_human', False):
                human_mask = gt['rgb_human_mask']
                ignore_mask = (human_mask | ignore_mask) if ignore_mask is not None else human_mask
            gt['rgb_ignore_mask'] = ignore_mask
        # Load depth on image (if any)
        if 'rgb_mono_depth' in self.config.tags:
            gt['rgb_mono_depth'] = self.get_mono_depth(scene_id=scene_id, cam_id=cam_id, frame_ind=frame_ind, device=device)
        # Load normals on image (if any)
        if 'rgb_mono_normals' in self.config.tags:
            gt['rgb_mono_normals'] = self.get_mono_normals(scene_id=scene_id, cam_id=cam_id, frame_ind=frame_ind, device=device)
        # Load bbox annotation on image (if any)
        # if 'box2d' in self.config.tags:
        #     gt['rgb_box'] = self.get_rgb_2dbox()
        return gt
    
    def load_rgb_gts(self, scene_id: str, cam_id: str, frame_ind: Union[int, List[int]], device=None, stack=True) -> Dict[str, torch.Tensor]:
        device = device or self.device
        if isinstance(frame_ind, int):
            return self.get_rgb_gts(scene_id, cam_id, frame_ind, device=device)
        else:
            # NOTE: Batched get when preload. Otherwise get each item iteratively
            return self.get_rgb_gts(scene_id, cam_id, frame_ind, device=device) if self.preload else \
                collate_nested_dict([self.get_rgb_gts(scene_id, cam_id, i) for i in frame_ind], stack=stack)


    @torch.no_grad()
    def filter_lidar_gts(
        self, 
        scene_id: str, lidar_id: Union[str, List[str]], 
        frame_ind: int, lidar_data: Dict[str, torch.Tensor], *, 
        # Default: only filter valid !
        filter_valid=True, filter_in_cams=False, 
        filter_in_aabb: torch.Tensor = None, 
        filter_out_objs=False, filter_out_obj_dynamic_only=False, 
        filter_out_obj_classnames: List[str] = None, 
        inplace=False) -> dict:
        """ 
        Filter LiDAR Ground Truths. By default, filtering is based on validity (GT range > 0).
        Additional optional filter rules include:
        - `filter_in_cams`: If true, only beam points within the camera(s)' viewport are retained.
        - `filter_out_objs`: If true, beam points falling within the 3D bounding boxes of objects are excluded. 
            This can be applied to dynamic objects only, or to all objects; see the next two options.
        - `filter_out_obj_dynamic_only`: If true, only dynamic objects are considered when `filter_out_objs` is enabled. 
            If false, all annotated objects are considered.
        - `filter_out_obj_classnames`: If specified, only objects of these class_name(s) are considered when `filter_out_objs` is enabled.
        - `filter_in_aabb`: If true, only beam points within the 3D Axis-Aligned Bounding Box (AABB) of the scene's main object are retained.

        Args:
            scene_id (str): The correponding scene_id
            lidar_id (Union[str, List[str]]): The correponding lidar_id
            frame_ind (int): The correponding frame_ind
            lidar_data (Dict[str, torch.Tensor]): The unfiltered raw LiDAR GT
            filter_valid (bool, optional): If true, will only retain valid LiDAR data with range>0. Defaults to True.
            filter_in_cams (bool, optional): If true, only beam points within the camera(s)' viewport are retained. Defaults to False.
            filter_in_aabb (torch.Tensor, optional): If true, only beam points within the 3D Axis-Aligned Bounding Box (AABB) \
                of the scene's main object are retained. Defaults to None.
            filter_out_objs (bool, optional): If true, beam points falling within the 3D bounding boxes of objects are excluded. \
                This can be applied to dynamic objects only, or to all objects; see the next two options. Defaults to False.
            filter_out_obj_dynamic_only (bool, optional): If true, only dynamic objects are considered when `filter_out_objs` is enabled. \
                If false, all annotated objects are considered. Defaults to False.
            filter_out_obj_classnames (List[str], optional): If specified, only objects of these class_name(s) are considered \
                when `filter_out_objs` is enabled. Defaults to None.
            inplace (bool, optional): If true, filtering is done in-place that will modify `lidar_data` contents. Defaults to False.

        Returns:
            lidar_data: dict: The filtered LiDAR GT data.
        """
        
        # Freeze scene at current frame
        scene = self.scene_bank[scene_id]
        scene.frozen_at(frame_ind)
        
        if isinstance(lidar_id, str):
            # Get current single lidar node
            lidar = scene.observers[lidar_id]
            # Lidar points in local coordinates
            pts = torch.addcmul(lidar_data['rays_o'], lidar_data['rays_d'], lidar_data['ranges'].unsqueeze(-1))
            # Lidar points in world coordinates
            pts = lidar.world_transform(pts)
        else:
            assert 'i' in lidar_data.keys(), "Missing lidar indices when filtering gt from merged multi lidar"
            # Assemble multiLidarBundle node
            lidars = [scene.observers[lid] for lid in lidar_id]
            lidar = MultiRaysLidarBundle(lidars)
            # Lidar points in local coordinates
            pts = torch.addcmul(lidar_data['rays_o'], lidar_data['rays_d'], lidar_data['ranges'].unsqueeze(-1))
            # Local to world transform of each point
            l2w = lidar.world_transform[lidar_data['i']]
            # Lidar points in world coordinates
            pts = l2w.forward(pts)
        
        # Filter lidar `data` inplace
        lidar_data = self._filter_lidar_gts(
            lidar_data, pts, scene, 
            filter_valid=filter_valid, filter_in_cams=filter_in_cams, filter_in_aabb=filter_in_aabb, 
            filter_out_objs=filter_out_objs, filter_out_obj_dynamic_only=filter_out_obj_dynamic_only, 
            filter_out_obj_classnames=filter_out_obj_classnames, 
            inplace=inplace)
        # NOTE: Unfrozen to restore default
        scene.unfrozen()
        return lidar_data

    @torch.no_grad()
    def _filter_lidar_gts(
        self, 
        data: Dict[str, torch.Tensor], pts: torch.Tensor, frozen_scene: Scene, *, 
        # Default: only filter valid !
        filter_valid=True, filter_in_cams=False, 
        filter_out_objs=False, filter_out_obj_dynamic_only=False, 
        filter_out_obj_classnames: List[str] = None, 
        filter_in_aabb: torch.Tensor = None,
        inplace=False):
        
        if not inplace:
            data = data.copy()

        assert frozen_scene.i is not None,  "scene needs to be frozen in advance."
        frame_ind = int(frozen_scene.i)

        if filter_valid:
            lidar_filter_inds = (data['ranges'] > 0).nonzero().long()[..., 0]
            pts = pts[lidar_filter_inds]
            data = {k: v[lidar_filter_inds] for k,v in data.items()}

        if filter_in_cams:
            cam_id_list = self.cam_id_list
            cams = [frozen_scene.observers[cid] for cid in cam_id_list]
            cam = MultiCamBundle(cams)
            cam.intr.set_downscale(self.config.tags.camera.downscale)
            
            ignore_masks = None
            if 'rgb_ignore_mask' in self.config.tags:
                ignore_masks = [
                    self.get_rgb_ignore_mask(
                        scene_id=frozen_scene.id, cam_id=cid, frame_ind=frame_ind, device=self.device)
                    for cid in cam_id_list]
                # NOTE: Due with possibly different mask image shapes before stack
                #       Must be `bottom_left`, so that the image is placed at the top-left corner and the pixel indices are reserved.
                ignore_masks = pad_images_to_same_size(ignore_masks, value=False, padding='bottom_right')
                ignore_masks = torch.stack(ignore_masks, dim=0)
            proj_ret = cam.project_pts_in_image(pts.view(1, -1, 3), ignore_mask=ignore_masks)
            lidar_filter_mask = proj_ret.mask.any(dim=0)
            lidar_filter_inds = lidar_filter_mask.nonzero().long()[..., 0]
            
            # DEBUG: masked & projected lidar points considering distortion
            # from nr3d_lib.plot import color_depth
            # from nr3d_lib.utils import check_to_torch
            # use_cam_ind = 0
            # rgb_gts = self.get_rgb_gts(frozen_scene.id, cam_id_list[use_cam_ind], frame_ind)
            # rgb = rgb_gts['rgb']
            # used = (proj_ret.i == use_cam_ind)
            # rgb[proj_ret.v[used], proj_ret.u[used]] = check_to_torch(
            #     color_depth(proj_ret.d[used].data.cpu().numpy(), cmap='turbo').astype(np.float32)/255., 
            #     dtype=torch.float, device=self.device) 
            
            # Filtered pts (might be used later)
            pts = pts[lidar_filter_inds]
            # Filter data inplace
            data = {k: v[lidar_filter_inds] for k,v in data.items()}

        if filter_in_aabb is not None:
            assert isinstance(filter_in_aabb, torch.Tensor) and list(filter_in_aabb.shape) == [2,3], \
                f"Invalid filter_in_aabb={filter_in_aabb}"
            aabb_min = filter_in_aabb[0]
            aabb_max = filter_in_aabb[1]
            lidar_filter_mask = ((pts >= aabb_min) & (pts <= aabb_max)).all(dim=-1)
            lidar_filter_inds = lidar_filter_mask.nonzero().long()[..., 0]

            # Filtered pts (might be used later)
            pts = pts[lidar_filter_inds]
            # Filter data inplace
            data = {k: v[lidar_filter_inds] for k,v in data.items()}

        if filter_out_objs:
            assert 'obj_box_list_per_frame' in frozen_scene.metas.keys()
            if filter_out_obj_dynamic_only:
                obj_box_list = frozen_scene.metas['obj_box_list_per_frame_dynamic_only']
            else:
                obj_box_list = frozen_scene.metas['obj_box_list_per_frame']
            class_names = filter_out_obj_classnames or list(obj_box_list.keys())
            # NOTE: meta data is not clipped by start & stop settings; hence `data_frame_offset` is needed.
            data_frame_ind = frame_ind + frozen_scene.data_frame_offset
            all_box_list = [(obj_box_list[c][data_frame_ind] 
                             if (c in obj_box_list ) and (len(obj_box_list[c][data_frame_ind]) > 0)
                             else np.empty([0,15])) for c in class_names]
            all_box_list = np.concatenate(all_box_list, axis=0)
            # Only filter when all_box_list is not empty.
            if len(all_box_list) > 0:
                # [num_obj, 15], where 15 = 12 (transform 3x4) + 3 (size)
                all_box_list = torch.tensor(all_box_list, dtype=torch.float, device=self.device)
                all_box_transform = TransformMat3x4(all_box_list[..., :12].unflatten(-1, (3,4)), device=self.device)
                all_box_scale = Scale(all_box_list[..., 12:], device=self.device)
                
                # [num_obj, num_pts, 3]
                pts_in_objs = all_box_transform.forward(pts.unsqueeze(0), inv=True)
                # [num_obj, 3]
                aabb_min = - all_box_scale.ratio() / 2.
                aabb_max = all_box_scale.ratio() / 2.
                
                # [num_obj, num_pts, 3] --all--> [num_obj, num_pts] --any--> [num_pts]
                lidar_filter_mask = (
                    (pts_in_objs >= aabb_min.unsqueeze(1)).all(dim=-1) & \
                    (pts_in_objs <= aabb_max.unsqueeze(1)).all(dim=-1)
                ).any(dim=0).logical_not()
                lidar_filter_inds = lidar_filter_mask.nonzero().long()[..., 0]
                
                # DEBUG
                # from nr3d_lib.plot import vis_lidar_and_boxes_o3d
                # vis_lidar_and_boxes_o3d(pts.data.cpu().numpy(), all_box_list.data.cpu().numpy())
                
                # Filtered pts (might be used later)
                pts = pts[lidar_filter_inds]
                # Filter data inplace
                data = {k: v[lidar_filter_inds] for k,v in data.items()}

        return data
    
    @torch.no_grad()
    def _check_and_filter_lidar_gts(
        self, 
        scene_id: str, lidar_id: Union[str, List[str]], 
        frame_ind: int, lidar_data: Dict[str, torch.Tensor], *, 
        inplace=False):
        
        # If not configured with lidar filtering, directly return.
        if 'filter_kwargs' not in self.config.tags.lidar:
            return lidar_data
        
        # If already filtered when caching data, directly return.
        if self.preload and self.config.tags.lidar.filter_when_preload:
            return lidar_data
        
        return self.filter_lidar_gts(
            scene_id=scene_id, lidar_id=lidar_id, frame_ind=frame_ind, 
            lidar_data=lidar_data, inplace=inplace, **self.config.tags.lidar.filter_kwargs)
    
    # @profile
    def _get_lidar_gts(self, scene_id: str, lidar_id: str, frame_ind: int, device=None):
        data_frame_offset = self.scene_bank[scene_id].data_frame_offset
        ret = self.dataset_impl.get_lidar(scene_id, lidar_id, frame_ind + data_frame_offset)
        if (downsample := self.config.tags.lidar.get('downsample_cfg', {}).get(lidar_id, 1)) != 1:
            ret = {'rays_o': ret['rays_o'].reshape(-1, 3)[::downsample], 
                   'rays_d': ret['rays_d'].reshape(-1, 3)[::downsample], 
                   'ranges': ret['ranges'].reshape(-1)[::downsample]}
        lidar_data = dict(
            rays_o = torch.tensor(ret['rays_o'].reshape(-1, 3), dtype=torch.float, device=device),
            rays_d = torch.tensor(ret['rays_d'].reshape(-1, 3), dtype=torch.float, device=device),
            ranges = torch.tensor(ret['ranges'].reshape(-1), dtype=torch.float, device=device),
        )
        return lidar_data
    # @profile
    def get_lidar_gts(self, scene_id: str, lidar_id: str, frame_ind: Union[int, List[int]], device=None, 
                      filter_if_configured=False) -> Dict[str, torch.Tensor]:
        if scene_id in self._cache_lidars and lidar_id in self._cache_lidars[scene_id]:
            lidar_data = self._cache_lidars[scene_id][lidar_id]
            lidar_data =  {k: v[frame_ind].to(device) for k,v in lidar_data.items()}
        else:
            assert isinstance(frame_ind, int), "Only support single frame_ind when not caching."
            lidar_data =  self._get_lidar_gts(scene_id, lidar_id, frame_ind, device=device)
        if filter_if_configured:
            lidar_data = self._check_and_filter_lidar_gts(scene_id, lidar_id, frame_ind, lidar_data, inplace=False)
        return lidar_data
    
    # @profile
    def _get_merged_lidar_gts(self, scene_id: str, frame_ind: int, device=None):
        lidar_data = {'rays_o': [], 'rays_d': [], 'ranges': [], 'i': []}
        for i, lidar_id in enumerate(self.lidar_id_list):
            cur_gt = self._get_lidar_gts(
                scene_id=scene_id, lidar_id=lidar_id, frame_ind=frame_ind, device=device)
            for k, v in cur_gt.items():
                lidar_data[k].append(v)
            lidar_data['i'].append(torch.full_like(cur_gt['ranges'], i, device=device, dtype=torch.long))
        lidar_data = {k: torch.cat(v, dim=0) for k, v in lidar_data.items()}
        return lidar_data
    # @profile
    def get_merged_lidar_gts(self, scene_id: str, frame_ind: Union[int, List[int]], device=None, 
                             filter_if_configured=False) -> Dict[str, torch.Tensor]:
        if scene_id in self._cache_merged_lidars:
            lidar_data = self._cache_merged_lidars[scene_id][frame_ind]
            lidar_data = {k:v.to(device) for k,v in lidar_data.items()}
        else:
            assert isinstance(frame_ind, int), "Only support single frame_ind when not caching."
            lidar_data = self._get_merged_lidar_gts(scene_id, frame_ind, device=device)
        if filter_if_configured:
            lidar_data = self._check_and_filter_lidar_gts(scene_id, self.lidar_id_list, frame_ind, lidar_data, inplace=False)
        return lidar_data

class FrameRandomSampler(torch_data.Sampler):
    def __init__(
        self, dataset: SceneDataLoader, *, 
        multi_scene_balance = True, replacement: bool = True, 
        frame_sample_mode: Literal['uniform', 'weighted_by_speed', 'error_map'] = 'uniform', 
        imp_samplers: Dict[str, Dict[str,ImpSampler]]=None) -> None:
        
        self.dataset = dataset
        self.frame_sample_mode = frame_sample_mode
        self.replacement = replacement
        self.multi_scene_balance = multi_scene_balance

        scene_bank = self.dataset.scene_bank
        scene_lengths = [len(scene) for scene in scene_bank]
        self.num_samples = sum(scene_lengths)
        
        if self.multi_scene_balance:
            scene_weights = torch.tensor(scene_lengths, device=self.device, dtype=torch.float)
            self.scene_weights = scene_weights / scene_weights.sum()
        else:
            self.scene_weights = torch.full([len(scene_bank), ],  1./len(scene_bank), device=self.device, dtype=torch.float)

        if self.frame_sample_mode == 'uniform':
            self.weights = self._get_weights_uniform()
        elif self.frame_sample_mode == 'weighted_by_speed':
            self.weights = self._get_weights_by_speed()
        elif self.frame_sample_mode == 'error_map':
            assert imp_samplers is not None, f"Please provide `imp_samplers` for frame_sample_mode={self.frame_sample_mode}"
            self.imp_samplers = imp_samplers
            self.weights: torch.Tensor = self._get_weights_from_error_map()
        else:
            raise RuntimeError(f"Invalid frame_sample_mode={self.frame_sample_mode}")

    def __len__(self):
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        rand_tensor = torch.multinomial(self.weights, self.num_samples, self.replacement, generator=None)
        yield from iter(rand_tensor.tolist())

    @property
    def device(self):
        return self.dataset.device

    @torch.no_grad()
    def _get_weights_uniform(self):
        total_weights = []
        for i, scene in enumerate(iter(self.dataset.scene_bank)):
            weights = torch.full([len(scene),], 1./len(scene), device=self.device, dtype=torch.float)
            total_weights.append( weights * self.scene_weights[i] )
        total_weights = torch.cat(total_weights)
        return total_weights

    @torch.no_grad()
    def _get_weights_by_speed(self, mode: Literal['linear', 'trunc_linear'] = 'linear', multiplier: float=4.0):
        total_weights = []
        for i, scene in enumerate(iter(self.dataset.scene_bank)):
            tracks = scene.process_observer_infos().tracks
            dtrans = tracks.new_zeros([*tracks.shape[:-1]])
            dtrans[...,:-1] = (tracks[...,1:,:] - tracks[...,:-1,:]).norm(dim=-1)
            dtrans[...,-1] = dtrans[...,-2]
            
            if mode == 'linear':
                weights = dtrans.clamp(1e-5)
                weights /= weights.sum()
            elif mode == 'trunc_linear':
                w_mean = gmean(dtrans, dim=-1)
                weights = dtrans.clip(w_mean/np.sqrt(multiplier), w_mean*np.sqrt(multiplier))
                weights /= weights.sum()
            else:
                raise RuntimeError(f"Invalid mode={mode}")
            
            total_weights.append(weights * self.scene_weights[i])
        total_weights = torch.cat(total_weights)
        return total_weights
    
    @torch.no_grad()
    def _get_weights_from_error_map(self):
        total_weights = []
        for i, (scene_id, scene) in enumerate(self.dataset.scene_bank.items()):
            weights = torch.zeros([len(scene)], device=self.device, dtype=torch.float)
            for cam_id, imp_sampler in self.imp_samplers[scene_id].items():
                weights += imp_sampler.get_pdf_image()
            weights /= weights.sum()
            total_weights.append(weights * self.scene_weights[i])
        total_weights = torch.cat(total_weights)
        return total_weights

    @torch.no_grad()
    def update_weights(self):
        if self.frame_sample_mode == 'error_map':
            self.weights = self._get_weights_from_error_map()
