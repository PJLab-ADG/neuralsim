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
from nr3d_lib.maths import geometric_mean
from nr3d_lib.config import ConfigDict
from nr3d_lib.models.attributes import TransformMat3x4, Scale
from nr3d_lib.utils import IDListedDict, collate_nested_dict, img_to_torch_and_downscale, pad_images_to_same_size

from app.resources import Scene
from app.resources.observers import MultiCamBundle, MultiRaysLidarBundle

from dataio.scene_dataset import SceneDataset

class SceneDataLoader(object):
    """
    - Defines the basic and common IO logic for the dataset.
    - Loads and caches common APIs used in dataio/dataset_impl.py, including scenarios, images, LiDAR data, and annotations.
    - Performs image downsampling.
    
    NOTE: The difference between `_get_xxx` and `get_xxx`
    - `self._get_xxx` directly reads data from disks.
    - `self.get_xxx` first checks whether it's in cache mode and if cache data already exists. \
        If so, it retrieves data directly from the cache; otherwise, it reads data from the disk (by invoking `self._get_xxx`).
    
    Methods:
    ------------------------------------------------------------------------
    Get 'image_rgb', 'image_wh', 'image_downscale'
    - _get_image_and_metas(): Direct load full images from disk.
    - get_image_and_metas(): Load from cache or disk accordingly; support loading individual pixels.
    ------------------------------------------------------------------------
    Get 'image_occupancy_mask'
    - _get_image_occupancy_mask(): Direct load full images from disk.
    - get_image_occupancy_mask(): Load from cache or disk accordingly; support loading individual pixels.
    ------------------------------------------------------------------------
    Group 'image_ignore_mask' (consists of combination of configured ignore keys in `config.tags.image_ignore_mask`)
    - get_image_ignore_mask(): Load from cache or disk accordingly; support loading individual pixels.
    ------------------------------------------------------------------------
    Get 'image_mono_depth'
    - _get_image_mono_depth(): Direct load full images from disk.
    - get_image_mono_depth(): Load from cache or disk accordingly; support loading individual pixels.
    ------------------------------------------------------------------------
    Get 'image_mono_normals'
    - _get_image_mono_normals(): Direct load full images from disk.
    - get_image_mono_normals(): Load from cache or disk accordingly; support loading individual pixels.
    ------------------------------------------------------------------------
    Group 'image_rgb' and 'image_xxx' ground truths
    - get_image_and_gts(): Load from cache or disk accordingly; support loading individual pixels.
    - get_batched_image_and_gts(): Load batched (multi-frame) images / individual pixels from cache or disk accordingly.
    ------------------------------------------------------------------------
    - filter_lidar_gts(): Filter LiDAR Ground Truths. By default, filtering is based on validity (GT range > 0).
    - _get_lidar_gts()
    - get_lidar_gts()
    - _get_merged_lidar_gts()
    - get_merged_lidar_gts()

    """
    def __init__(
        self, 
        scene_or_scene_bank: Union[IDListedDict[Scene], Scene], 
        dataset_impl: SceneDataset, *, 
        config: dict, 
        device=None, 
        is_master=True # Mainly for logging controlling
        ) -> None:
        super().__init__()

        self.device = device
        self.is_master = is_master
        self.config: dict = config.deepcopy()
        
        if isinstance(scene_or_scene_bank, Scene):
            self.scene_bank: IDListedDict[Scene] = IDListedDict([scene_or_scene_bank])
        elif isinstance(scene_or_scene_bank, IDListedDict):
            self.scene_bank: IDListedDict[Scene] = scene_or_scene_bank
        else:
            raise RuntimeError(f"Invalid input type: {type(scene_or_scene_bank)}")
        
        #---- Dataset-specific operations
        self.dataset_impl: SceneDataset = dataset_impl

        #---- Ground-truth image-like data fields (not all necessary)
        self.gt_im_keys = [
            'image_rgb', 
            'image_mono_depth', 
            'image_mono_normals', 
            'image_occupancy_mask', 
            
            # Other semantic masks
            'image_dynamic_mask', 
            'image_human_mask', 
            'image_road_mask', 
            'image_ignore_mask'
        ]

        self.gt_im_semantic_mask_mappings = {
            'image_dynamic_mask': 'dynamic', 
            'image_human_mask': 'human', 
            'image_road_mask': 'road', 
            'image_anno_dontcare_mask': 'anno_dontcare',
        }

        #---- (Optional) Data caching
        # (Always available) [image_wh, image_downscale]
        self._cache_image_metas: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {} 
        
        # [image_rgb]
        self._cache_image_raw: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {} 
        
        # [image_occupancy_mask]
        self._cache_image_occupancy_mask: Dict[str, Dict[str, torch.Tensor]] = {}
        
        # [image_dynamic_mask, image_human_mask, image_road_mask, image_anno_dontcare_mask, ]
        self._cache_image_semantic_masks: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {} 
        
        self._cache_image_mono_depth: Dict[str, Dict[str, torch.Tensor]] = {}
        self._cache_image_mono_normals: Dict[str, Dict[str, torch.Tensor]] = {}
        self._cache_lidar: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {}
        self._cache_merged_lidar: Dict[str, List[Dict[str, torch.Tensor]]] = {}
        self._cache_rays: Dict[str, Dict[str, torch.Tensor]] = {}

        self.preload = self.config.get('preload', False)
        cache_device = self.device if self.config.get('preload_on_gpu', False) else torch.device('cpu')
        if 'lidar' in self.config.tags:
            self.config.tags.lidar.setdefault('filter_when_preload', False)
        
        if self.preload:
            self._build_full_cache(cache_device)
        else:
            self._build_meta_cache()

    def get_scene_frame_idx(self, index: int):
        # From overall index to scene_idx and frame_idx
        scene_idx = 0
        while index >= 0:
            index -= len(self.scene_bank[scene_idx])
            scene_idx += 1
        return (scene_idx - 1), int(index + len(self.scene_bank[scene_idx - 1]))

    def _build_meta_cache(self):
        for scene in self.scene_bank:
            scene_id = scene.id
            if 'camera' in self.config.tags:
                self._cache_image_metas[scene_id] = {}
                for cam_id in tqdm(self.cam_id_list, "=> Caching metas...", disable=not self.is_master):
                    obs = scene.observers[cam_id]
                    if self.config.tags.camera.get('load_images', True):
                        image_metas = {}
                        for frame_ind in range(len(scene)):
                            _ret = self._get_image_metas(scene_id=scene_id, cam_id=cam_id, frame_ind=frame_ind)
                            for k, v in _ret.items():
                                image_metas.setdefault(k, []).append(v)
                        for k, v in image_metas.items():
                            image_metas[k] = torch.stack(v, dim=0).to(self.device)
                        self._cache_image_metas[scene_id][cam_id] = image_metas

    def _build_full_cache(self, cache_device):
        log.info(f"=> Caching data to device={cache_device}...")
        
        for scene in self.scene_bank:
            scene_id = scene.id
            
            if 'camera' in self.config.tags:
                self._cache_image_metas[scene_id] = {}
                self._cache_image_raw[scene_id] = {}
                log.info("=> Caching camera data...")
                for cam_id in tqdm(self.cam_id_list, "Caching cameras...", disable=not self.is_master):
                    obs = scene.observers[cam_id]
                    if self.config.tags.camera.get('load_images', True):
                        self._cache_image_metas[scene_id][cam_id] = {}
                        self._cache_image_raw[scene_id][cam_id] = {}
                        image_and_metas = {}
                        for frame_ind in range(len(scene)):
                            _ret = self._get_image_and_metas(scene_id, cam_id, frame_ind, device=cache_device)
                            for k, v in _ret.items():
                                image_and_metas.setdefault(k, []).append(v)
                        for k, v in image_and_metas.items():
                            v = torch.stack(v, dim=0)
                            if k in ('image_wh', 'image_downscale'):
                                self._cache_image_metas[scene_id][cam_id][k] = v
                            else:
                                self._cache_image_raw[scene_id][cam_id][k] = v

                    # Cache occupancy masks
                    if 'image_occupancy_mask' in self.config.tags:
                        if scene_id not in self._cache_image_occupancy_mask:
                            self._cache_image_occupancy_mask[scene_id] = {}
                        masks = []
                        for frame_ind in range(len(scene)):
                            masks.append(self._get_image_occupancy_mask(scene_id, cam_id, frame_ind, device=cache_device).to(torch.bool))
                        self._cache_image_occupancy_mask[scene_id][cam_id] = torch.stack(masks, dim=0)

                    # Cache other semantic masks
                    for gt_k, sem_type in self.gt_im_semantic_mask_mappings.items():
                        if gt_k in self.config.tags:
                            if scene_id not in self._cache_image_semantic_masks:
                                self._cache_image_semantic_masks[scene_id] = {}
                            if cam_id not in self._cache_image_semantic_masks[scene_id]:
                                self._cache_image_semantic_masks[scene_id][cam_id] = {}
                            masks = []
                            for frame_ind in range(len(scene)):
                                m = self._get_image_semantic_mask_by_type(scene_id, cam_id, sem_type, frame_ind, device=cache_device).to(torch.bool)
                                masks.append(m)
                            self._cache_image_semantic_masks[scene_id][cam_id][sem_type] = torch.stack(masks, dim=0)

                    # Cache depth gts
                    if 'image_mono_depth' in self.config.tags:
                        if scene_id not in self._cache_image_mono_depth:
                            self._cache_image_mono_depth[scene_id] = {}
                        depths = []
                        for frame_ind in range(len(scene)):
                            depths.append(self._get_image_mono_depth(scene_id, cam_id, frame_ind, device=cache_device).to(torch.float))
                        self._cache_image_mono_depth[scene_id][cam_id] = torch.stack(depths, dim=0)

                    # Cache normal gts
                    if 'image_mono_normals' in self.config.tags:
                        if scene_id not in self._cache_image_mono_normals:
                            self._cache_image_mono_normals[scene_id] = {}
                        normals = []
                        for frame_ind in range(len(scene)):
                            normals.append(self._get_image_mono_normals(scene_id, cam_id, frame_ind, device=cache_device).to(torch.float))
                        self._cache_image_mono_normals[scene_id][cam_id] = torch.stack(normals, dim=0)

                    # Cache rays directly if camera intr is not learnable
                    # NOTE: OOM, cost very high
                    # if len(list(obs.attr_segments.parameters())) == 0:
                    #     if scene_id not in self._cache_rays:
                    #         self._cache_rays[scene_id] = {}
                    #     rays = []
                    #     for frame_ind in range(len(scene)):
                    #         obs.to(self.device)
                    #         obs.slice_at(frame_ind)
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
                    self._cache_merged_lidar[scene_id] = []
                    for frame_ind in tqdm(range(len(scene)), "=> Caching merged LiDAR frames...", disable=not self.is_master):
                        _ret = self._get_merged_lidar_gts(scene_id=scene_id, frame_ind=frame_ind, device=cache_device)
                        if filter_when_preload: # Calculates on cuda device 
                            _ret = {k: v.to(scene.device) for k, v in _ret.items()}
                            _ret = self.filter_lidar_gts(scene_id=scene_id, lidar_id=self.lidar_id_list, frame_ind=frame_ind, 
                                                         lidar_data=_ret, inplace=True, **filter_kwargs)
                            _ret = {k: v.to(cache_device) for k, v in _ret.items()}
                        self._cache_merged_lidar[scene_id].append(_ret)
                else: # not multi_lidar_merge
                    self._cache_lidar[scene_id] = {}
                    for lidar_id in tqdm(self.lidar_id_list, "=> Caching LiDARs...", disable=not self.is_master):
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
                        self._cache_lidar[scene_id][lidar_id] = lidar_gts
        
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


    """------------------------------------------------------------------------
    Get 'image_rgb', 'image_wh', 'image_downscale'
    - _get_image_and_metas(): Direct load full images from disk.
    - get_image_and_metas(): Load from cache or disk accordingly; support loading individual pixels.
    
    NOTE: Debug the performance of loading full images vs. individual pixels
    >>> from math import prod
    >>> from torch.utils.benchmark import Timer
    
    # [!!!] First index the list frame id, then transfer to device: 1.72 s
    >>> print(Timer(stmt="{k: v[frame_ind].to(device) for k,v in _ret.items()}", globals={'_ret':_ret, 'frame_ind':frame_ind, 'device': device}).blocked_autorange())
    
    # First transfer to device, then index the list frame id: 480 ms
    >>> print(Timer(stmt="{k: v.to(device)[frame_ind] for k,v in _ret.items()}", globals={'_ret':_ret, 'frame_ind':frame_ind, 'device': device}).blocked_autorange())
    
    # First transfer to device, then index the single frame id: 480 ms
    >>> print(Timer(stmt="{k: v.to(device)[0] for k,v in _ret.items()}", globals={'_ret':_ret, 'frame_ind':frame_ind, 'device': device}).blocked_autorange())
    
    # First index the single frame id, then transfer to device: 2.83 ms
    >>> print(Timer(stmt="{k: v[0].to(device) for k,v in _ret.items()}", globals={'_ret':_ret, 'frame_ind':frame_ind, 'device': device}).blocked_autorange())
    
    >>> num_pixels = 8192
    >>> num_frames, H, W, _ = _ret['image_rgb'].shape # [num_frames, H, W, 3] e.g. [163, 886, 1920, 3]
    >>> h = torch.randint(H, size=[num_pixels,]).tolist()
    >>> w = torch.randint(W, size=[num_pixels,]).tolist()
    >>> fi = torch.randint(num_frames, size=[num_pixels,]).tolist()
    
    # e.g. 3173 MiB Mem, can not be placed on GPU especially when there are multiple cameras
    >>> print(prod(_ret['image_rgb'].shape) * 4) // (1024**2) 
    
    # Index individual pixel locations
    >>> tt = _ret['image_rgb'][fi, h, w] # [num_pixels, 3]
    
    # [!!!] First directly index individual pixels, then transfer to device: 2.38 ms
    >>> print(Timer(stmt="{k: v[fi, h, w].to(device) if k == 'image_rgb' else v[fi].to(device) for k,v in _ret.items()}", globals={'_ret':_ret, 'fi':fi, 'h':h, 'w':w, 'device': device}).blocked_autorange())
    
    # First transfer to device, then directly index individual pixels: 480 ms
    >>> print(Timer(stmt="{k: v.to(device)[fi, h, w] if k == 'image_rgb' else v.to(device)[fi] for k,v in _ret.items()}", globals={'_ret':_ret, 'fi':fi, 'h':h, 'w':w, 'device': device}).blocked_autorange())
    ------------------------------------------------------------------------"""
    def _get_image_metas(self, scene_id: str, cam_id: str, frame_ind: int, device=None) -> Dict[str, torch.Tensor]:
        data_frame_offset = self.scene_bank[scene_id].data_frame_offset
        WH0 = torch.tensor(self.dataset_impl.get_image_wh(scene_id, cam_id, frame_ind + data_frame_offset)).long()
        ds = self.config.tags.camera.downscale
        WH = torch.floor(WH0 / ds).long() # Rounding to floor
        return dict(
            image_downscale=(WH0/WH).to(dtype=torch.float, device=device), 
            image_wh=WH.to(device)
        )
    
    def get_image_metas(
        self, scene_id: str, cam_id: str, frame_ind: Union[torch.LongTensor, List[int], int], 
        device=None) -> Dict[str, torch.Tensor]:
        """
        {
            'image_downscale': [downscale_w, downscale_h], 
            'image_wh': [width, height]
        }
        """
        _ret = self._cache_image_metas[scene_id][cam_id] # Always available
        return {k: v[frame_ind].to(device) for k, v in _ret.items()}
    
    def _get_image_and_metas(
        self, scene_id: str, cam_id: str, frame_ind: int, 
        device=None, bypass_downscale:float = None) -> Dict[str, torch.Tensor]:
        
        data_frame_offset = self.scene_bank[scene_id].data_frame_offset
        rgb_np = self.dataset_impl.get_image(scene_id, cam_id, frame_ind + data_frame_offset)
        H, W, *_ = rgb_np.shape
        rgb = img_to_torch_and_downscale(rgb_np, dtype=torch.float, device=device,
            downscale=bypass_downscale or self.config.tags.camera.downscale, 
            use_cpu_downscale=self.config.get('use_cpu_downscale', False))
        H_, W_, *_ = rgb.shape
        return dict(
            image_rgb=rgb.to(dtype=torch.float, device=device),
            image_downscale=torch.tensor([W/W_, H/H_], dtype=torch.float, device=device), # old / new
            image_wh=torch.tensor([W_, H_], dtype=torch.long, device=device)
        )
    
    def get_image_and_metas(
        self, scene_id: str, cam_id: str, frame_ind: Union[torch.LongTensor, List[int], int], 
        h: Union[torch.LongTensor, List[int]] = None, w: Union[torch.LongTensor, List[int]] = None, 
        device=None, bypass_downscale: float = None) -> Dict[str, torch.Tensor]:
        
        ds_need_reload = (bypass_downscale is not None) and (abs(bypass_downscale-self.config.tags.camera.downscale)>1e-5)
        if (not ds_need_reload) and (scene_id in self._cache_image_raw) and (cam_id in self._cache_image_raw[scene_id]):
            _ret_image_raw = self._cache_image_raw[scene_id][cam_id]
            _ret_image_meta = self._cache_image_metas[scene_id][cam_id] # Always available
            
            _ret = {k: v[frame_ind].to(device) for k,v in _ret_image_meta.items()}
            if h is not None or w is not None:
                _ret.update({k: v[frame_ind, h, w].to(device) for k,v in _ret_image_raw.items()})
            else:
                _ret.update({k: v[frame_ind].to(device) for k,v in _ret_image_raw.items()})
            return _ret
        else:
            assert isinstance(frame_ind, int), \
                "When no caching/preload, only single frame_ind data loading is supported."\
                "If you are using JointFramePixel mode that will load data at a list of frames, "\
                "Please turn on caching(training:dataloader:preload...) instead."
            _ret = self._get_image_and_metas(scene_id, cam_id, frame_ind, device=device, bypass_downscale=bypass_downscale)
            if h is not None or w is not None:
                _ret['image_rgb'] = _ret['image_rgb'][h, w]
            return _ret

    """------------------------------------------------------------------------
    Get 'image_occupancy_mask'
    - _get_image_occupancy_mask(): Direct load full images from disk.
    - get_image_occupancy_mask(): Load from cache or disk accordingly; support loading individual pixels.
    ------------------------------------------------------------------------"""
    def _get_image_occupancy_mask(self, scene_id: str, cam_id: str, frame_ind: int, device=None) -> torch.BoolTensor:
        
        data_frame_offset = self.scene_bank[scene_id].data_frame_offset
        mask_np = self.dataset_impl.get_image_occupancy_mask(scene_id, cam_id, frame_ind + data_frame_offset)
        mask = img_to_torch_and_downscale(mask_np, dtype=torch.float, device=device, 
            downscale=self.config.tags.camera.downscale, 
            use_cpu_downscale=self.config.get('use_cpu_downscale', False))
        return mask.to(dtype=torch.bool, device=device)
    
    def get_image_occupancy_mask(
        self, scene_id: str, cam_id: str, frame_ind: Union[torch.LongTensor, List[int], int], 
        h: Union[torch.LongTensor, List[int]] = None, w: Union[torch.LongTensor, List[int]] = None, 
        device=None) -> torch.BoolTensor:
        
        if scene_id in self._cache_image_occupancy_mask and cam_id in self._cache_image_occupancy_mask[scene_id]:
            _ret = self._cache_image_occupancy_mask[scene_id][cam_id]
            if h is not None or w is not None:
                return _ret[frame_ind, h, w].to(device=device, dtype=torch.bool)
            else:
                return _ret[frame_ind].to(device=device, dtype=torch.bool)
        else:
            assert isinstance(frame_ind, int), \
                "When no caching/preload, only single frame_ind data loading is supported."\
                "If you are using JointFramePixel mode that will load data at a list of frames, "\
                "Please turn on caching(training:dataloader:preload...) instead."
            _ret = self._get_image_occupancy_mask(scene_id, cam_id, frame_ind, device=device).to(dtype=torch.bool)
            if h is not None or w is not None:
                _ret = _ret[h, w]
            return _ret

    def _get_image_semantic_mask_by_type(
        self, scene_id: str, cam_id: str, sem_type: Literal['dynamic', 'human', 'road', 'anno_dontcare'], 
        frame_ind: int, device=None)-> torch.BoolTensor:
        
        data_frame_offset = self.scene_bank[scene_id].data_frame_offset
        mask_np = self.dataset_impl.get_image_semantic_mask_by_type(scene_id, cam_id, sem_type, frame_ind + data_frame_offset)
        mask = img_to_torch_and_downscale(mask_np, dtype=torch.float, device=device, 
            downscale=self.config.tags.camera.downscale, 
            use_cpu_downscale=self.config.get('use_cpu_downscale', False))
        return mask.to(dtype=torch.bool, device=device)

    def get_image_semantic_mask_by_type(
        self, scene_id: str, cam_id: str, 
        sem_type: Literal['dynamic', 'human', 'road', 'anno_dontcare'], 
        frame_ind: Union[torch.LongTensor, List[int], int], 
        h: Union[torch.LongTensor, List[int]] = None, w: Union[torch.LongTensor, List[int]] = None, 
        device=None) -> torch.BoolTensor:
        
        if scene_id in self._cache_image_semantic_masks \
            and cam_id in self._cache_image_semantic_masks[scene_id] \
            and sem_type in self._cache_image_semantic_masks[scene_id][cam_id]:
            _ret = self._cache_image_semantic_masks[scene_id][cam_id][sem_type]
            if h is not None or w is not None:
                return _ret[frame_ind, h, w].to(device=device, dtype=torch.bool)
            else:
                return _ret[frame_ind].to(device=device, dtype=torch.bool)
        else:
            assert isinstance(frame_ind, int), \
                "When no caching/preload, only single frame_ind data loading is supported."\
                "If you are using JointFramePixel mode that will load data at a list of frames, "\
                "Please turn on caching(training:dataloader:preload...) instead."
            _ret = self._get_image_semantic_mask_by_type(scene_id, cam_id, sem_type, frame_ind, device=device).to(dtype=torch.bool)
            if h is not None or w is not None:
                _ret = _ret[h, w]
            return _ret

    """------------------------------------------------------------------------
    Group 'image_ignore_mask' (consists of combination of configured ignore keys in `config.tags.image_ignore_mask`)
    - get_image_ignore_mask(): Load from cache or disk accordingly; support loading individual pixels.
    ------------------------------------------------------------------------"""
    def get_image_ignore_mask(
        self, scene_id: str, cam_id: str, frame_ind: Union[torch.LongTensor, List[int], int], 
        h: Union[torch.LongTensor, List[int]] = None, w: Union[torch.LongTensor, List[int]] = None, 
        device=None) -> torch.BoolTensor:
        
        cfg = self.config.tags.image_ignore_mask
        ignore_mask = None
        if cfg.get('ignore_not_occupied', False):
            not_occ_mask = self.get_image_occupancy_mask(scene_id, cam_id, frame_ind, h, w, device=device).logical_not()
            ignore_mask = (not_occ_mask | ignore_mask) if ignore_mask is not None else not_occ_mask
        if cfg.get('ignore_anno_dontcare', False):
            dataset_ignore_mask = self.get_image_semantic_mask_by_type(scene_id, cam_id, 'anno_dontcare', frame_ind, h, w, device=device)
            ignore_mask = (dataset_ignore_mask | ignore_mask) if ignore_mask is not None else dataset_ignore_mask
        if cfg.get('ignore_dynamic', False):
            dynamic_mask = self.get_image_semantic_mask_by_type(scene_id, cam_id, 'dynamic', frame_ind, h, w, device=device)
            ignore_mask = (dynamic_mask | ignore_mask) if ignore_mask is not None else dynamic_mask
        if cfg.get('ignore_human', False):
            human_mask = self.get_image_semantic_mask_by_type(scene_id, cam_id, 'human', frame_ind, h, w, device=device)
            ignore_mask = (human_mask | ignore_mask) if ignore_mask is not None else human_mask
        return ignore_mask

    """------------------------------------------------------------------------
    Get 'image_mono_depth'
    - _get_image_mono_depth(): Direct load full images from disk.
    - get_image_mono_depth(): Load from cache or disk accordingly; support loading individual pixels.
    ------------------------------------------------------------------------"""
    def _get_image_mono_depth(self, scene_id: str, cam_id: str, frame_ind: int, device=None) -> torch.Tensor:
        
        data_frame_offset = self.scene_bank[scene_id].data_frame_offset
        image_mono_depth_np = self.dataset_impl.get_image_mono_depth(scene_id, cam_id, frame_ind + data_frame_offset)
        image_mono_depth = img_to_torch_and_downscale(image_mono_depth_np, dtype=torch.float, device=device, 
            downscale=self.config.tags.camera.downscale, 
            use_cpu_downscale=self.config.get('use_cpu_downscale', False))
        return image_mono_depth.to(dtype=torch.float, device=device)
    
    def get_image_mono_depth(
        self, scene_id: str, cam_id: str, frame_ind: Union[torch.LongTensor, List[int], int], 
        h: Union[torch.LongTensor, List[int]] = None, w: Union[torch.LongTensor, List[int]] = None, 
        device=None) -> torch.Tensor:
        
        if scene_id in self._cache_image_mono_depth and cam_id in self._cache_image_mono_depth[scene_id]:
            _ret = self._cache_image_mono_depth[scene_id][cam_id]
            if h is not None or w is not None:
                return _ret[frame_ind, h, w].to(device=device, dtype=torch.float)
            else:
                return _ret[frame_ind].to(device=device, dtype=torch.float)
        else:
            assert isinstance(frame_ind, int), \
                "When no caching/preload, only single frame_ind data loading is supported."\
                "If you are using JointFramePixel mode that will load data at a list of frames, "\
                "Please turn on caching(training:dataloader:preload...) instead."
            _ret = self._get_image_mono_depth(scene_id, cam_id, frame_ind, device=device).to(dtype=torch.float)
            if h is not None or w is not None:
                _ret = _ret[h, w]
            return _ret

    """------------------------------------------------------------------------
    Get 'image_mono_normals'
    - _get_image_mono_normals(): Direct load full images from disk.
    - get_image_mono_normals(): Load from cache or disk accordingly; support loading individual pixels.
    ------------------------------------------------------------------------"""
    def _get_image_mono_normals(self, scene_id: str, cam_id: str, frame_ind: int, device=None) -> torch.Tensor:
        
        data_frame_offset = self.scene_bank[scene_id].data_frame_offset
        image_mono_normals_np = self.dataset_impl.get_image_mono_normals(scene_id, cam_id, frame_ind + data_frame_offset)
        image_mono_normals = img_to_torch_and_downscale(image_mono_normals_np, dtype=torch.float, device=device, 
            downscale=self.config.tags.camera.downscale, 
            use_cpu_downscale=self.config.get('use_cpu_downscale', False))
        return image_mono_normals.to(dtype=torch.float, device=device)
    
    def get_image_mono_normals(
        self, scene_id: str, cam_id: str, frame_ind: Union[torch.LongTensor, List[int], int], 
        h: Union[torch.LongTensor, List[int]] = None, w: Union[torch.LongTensor, List[int]] = None, 
        device=None)  -> torch.Tensor:
        
        if scene_id in self._cache_image_mono_normals and cam_id in self._cache_image_mono_normals[scene_id]:
            _ret = self._cache_image_mono_normals[scene_id][cam_id]
            if h is not None or w is not None:
                return _ret[frame_ind, h, w].to(device=device, dtype=torch.float)
            else:
                return _ret[frame_ind].to(device=device, dtype=torch.float)
        else:
            assert isinstance(frame_ind, int), \
                "When no caching/preload, only single frame_ind data loading is supported."\
                "If you are using JointFramePixel mode that will load data at a list of frames, "\
                "Please turn on caching(training:dataloader:preload...) instead."
            _ret = self._get_image_mono_normals(scene_id, cam_id, frame_ind, device=device).to(dtype=torch.float)
            if h is not None or w is not None:
                _ret = _ret[h, w]
            return _ret

    """------------------------------------------------------------------------
    Group 'image_rgb' and 'image_xxx' ground truths
    - get_image_and_gts(): Load from cache or disk accordingly; support loading individual pixels.
    - get_batched_image_and_gts(): Load batched (multi-frame) images / individual pixels from cache or disk accordingly.
    ------------------------------------------------------------------------"""
    # @profile
    def get_image_and_gts(
        self, scene_id: str, cam_id: str, frame_ind: Union[torch.LongTensor, List[int], int], 
        h: Union[torch.LongTensor, List[int]] = None, w: Union[torch.LongTensor, List[int]] = None, 
        device=None) -> Dict[str, torch.Tensor]:
        
        device = device or self.device
        # Load RGB observation
        if self.config.tags.camera.get('load_images', True):
            gt = self.get_image_and_metas(scene_id, cam_id, frame_ind, h, w, device=device)
        else:
            # gt = self.get_image_metas(scene_id, cam_id, frame_ind, device=device)
            gt = {}
        # Load mask annotation on image (if any)
        if 'image_occupancy_mask' in self.config.tags:
            gt['image_occupancy_mask'] = self.get_image_occupancy_mask(scene_id, cam_id, frame_ind, h, w, device=device)
        if 'image_dynamic_mask' in self.config.tags:
            gt['image_dynamic_mask'] = self.get_image_semantic_mask_by_type(scene_id, cam_id, 'dynamic', frame_ind, h, w, device=device)
        if 'image_human_mask' in self.config.tags:
            gt['image_human_mask'] = self.get_image_semantic_mask_by_type(scene_id, cam_id, 'human', frame_ind, h, w, device=device)
        if 'image_road_mask' in self.config.tags:
            gt['image_road_mask'] = self.get_image_semantic_mask_by_type(scene_id, cam_id, 'road', frame_ind, h, w, device=device)
        if 'image_ignore_mask' in self.config.tags:
            cfg = self.config.tags.image_ignore_mask
            ignore_mask = None
            if cfg.get('ignore_anno_dontcare', False):
                dataset_ignore_mask = self.get_image_anno_dontcare_mask(scene_id, cam_id, frame_ind, h, w, device=device)
                ignore_mask = (dataset_ignore_mask | ignore_mask) if ignore_mask is not None else dataset_ignore_mask
            if cfg.get('ignore_not_occupied', False):
                not_occ_mask = gt['image_occupancy_mask'].logical_not()
                ignore_mask = (not_occ_mask | ignore_mask) if ignore_mask is not None else not_occ_mask
            if cfg.get('ignore_dynamic', False):
                dynamic_mask = gt['image_dynamic_mask']
                ignore_mask = (dynamic_mask | ignore_mask) if ignore_mask is not None else dynamic_mask
            if cfg.get('ignore_human', False):
                human_mask = gt['image_human_mask']
                ignore_mask = (human_mask | ignore_mask) if ignore_mask is not None else human_mask
            gt['image_ignore_mask'] = ignore_mask
        # Load depth on image (if any)
        if 'image_mono_depth' in self.config.tags:
            gt['image_mono_depth'] = self.get_image_mono_depth(scene_id, cam_id, frame_ind, h, w, device=device)
        # Load normals on image (if any)
        if 'image_mono_normals' in self.config.tags:
            gt['image_mono_normals'] = self.get_image_mono_normals(scene_id, cam_id, frame_ind, h, w, device=device)
        # Load bbox annotation on image (if any)
        # if 'box2d' in self.config.tags:
        #     gt['rgb_box'] = self.get_image_2dbox()
        return gt
    
    def get_batched_image_and_gts(
        self, scene_id: str, cam_id: str, frame_ind: Union[List[int], int], 
        h: Union[torch.LongTensor, List[int]] = None, w: Union[torch.LongTensor, List[int]] = None, 
        device=None, stack=True) -> Dict[str, torch.Tensor]:
        
        assert not isinstance(frame_ind, torch.Tensor), \
            "Joint frame-pixel sampling should not be applied to get_batched_image_and_gts()"
        
        device = device or self.device
        if isinstance(frame_ind, int) or self.preload:
            return self.get_image_and_gts(scene_id, cam_id, frame_ind, h, w, device=device)
        else:
            return collate_nested_dict([self.get_image_and_gts(scene_id, cam_id, i) for i in frame_ind], stack=stack)

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
        
        scene = self.scene_bank[scene_id]
        _i = scene.i # If the scene is already frozen, we need to freeze it again after filtering is done.
        # Freeze scene at current frame
        scene.frozen_at_global_frame(frame_ind)
        
        if isinstance(lidar_id, str):
            # Get current single lidar node
            lidar = scene.observers[lidar_id]
            # Lidar points in local coordinates
            pts = torch.addcmul(lidar_data['rays_o'], lidar_data['rays_d'], lidar_data['ranges'].unsqueeze(-1))
            # Lidar points in world coordinates
            pts = lidar.world_transform(pts)
        else:
            assert 'li' in lidar_data.keys(), "Missing lidar indices when filtering gt from merged multi lidar"
            # Assemble multiLidarBundle node
            lidars = [scene.observers[lid] for lid in lidar_id]
            lidar = MultiRaysLidarBundle(lidars)
            # Lidar points in local coordinates
            pts = torch.addcmul(lidar_data['rays_o'], lidar_data['rays_d'], lidar_data['ranges'].unsqueeze(-1))
            # Local to world transform of each point
            l2w = lidar.world_transform[lidar_data['li']]
            # Lidar points in world coordinates
            pts = l2w.forward(pts)
        
        # Filter lidar `data` inplace
        lidar_data = self._filter_lidar_gts(
            lidar_data, pts, scene, 
            filter_valid=filter_valid, filter_in_cams=filter_in_cams, filter_in_aabb=filter_in_aabb, 
            filter_out_objs=filter_out_objs, filter_out_obj_dynamic_only=filter_out_obj_dynamic_only, 
            filter_out_obj_classnames=filter_out_obj_classnames, 
            inplace=inplace)
        # NOTE: Unfrozen and re-freeze again to restore original state
        scene.unfrozen()
        if _i is not None:
            if scene.use_ts_interp:
                scene.interp_at(_i)
            else:
                scene.slice_at(_i) 
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
            if 'image_ignore_mask' in self.config.tags:
                ignore_masks = [
                    self.get_image_ignore_mask(
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
            # rgb_gts = self.get_image_and_gts(frozen_scene.id, cam_id_list[use_cam_ind], frame_ind)
            # rgb = rgb_gts['image_rgb']
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
                aabb_min = - all_box_scale.vec_3() / 2.
                aabb_max = all_box_scale.vec_3() / 2.
                
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
        if scene_id in self._cache_lidar and lidar_id in self._cache_lidar[scene_id]:
            lidar_data = self._cache_lidar[scene_id][lidar_id]
            lidar_data =  {k: v[frame_ind].to(device) for k,v in lidar_data.items()}
        else:
            assert isinstance(frame_ind, int), \
                "When no caching/preload, only single frame_ind data loading is supported."\
                "If you are using JointFramePixel mode that will load data at a list of frames, "\
                "Please turn on caching(training:dataloader:preload...) instead."
            lidar_data =  self._get_lidar_gts(scene_id, lidar_id, frame_ind, device=device)
        if filter_if_configured:
            lidar_data = self._check_and_filter_lidar_gts(scene_id, lidar_id, frame_ind, lidar_data, inplace=False)
        return lidar_data
    
    # @profile
    def _get_merged_lidar_gts(self, scene_id: str, frame_ind: int, device=None):
        lidar_data = {'rays_o': [], 'rays_d': [], 'ranges': [], 'li': []}
        for i, lidar_id in enumerate(self.lidar_id_list):
            cur_gt = self._get_lidar_gts(
                scene_id=scene_id, lidar_id=lidar_id, frame_ind=frame_ind, device=device)
            for k, v in cur_gt.items():
                lidar_data[k].append(v)
            lidar_data['li'].append(torch.full_like(cur_gt['ranges'], i, device=device, dtype=torch.long))
        lidar_data = {k: torch.cat(v, dim=0) for k, v in lidar_data.items()}
        return lidar_data
    # @profile
    def get_merged_lidar_gts(self, scene_id: str, frame_ind: Union[int, List[int]], device=None, 
                             filter_if_configured=False) -> Dict[str, torch.Tensor]:
        if scene_id in self._cache_merged_lidar:
            lidar_data = self._cache_merged_lidar[scene_id][frame_ind]
            lidar_data = {k:v.to(device) for k,v in lidar_data.items()}
        else:
            assert isinstance(frame_ind, int), \
                "When no caching/preload, only single frame_ind data loading is supported."\
                "If you are using JointFramePixel mode that will load data at a list of frames, "\
                "Please turn on caching(training:dataloader:preload...) instead."
            lidar_data = self._get_merged_lidar_gts(scene_id, frame_ind, device=device)
        if filter_if_configured:
            lidar_data = self._check_and_filter_lidar_gts(scene_id, self.lidar_id_list, frame_ind, lidar_data, inplace=False)
        return lidar_data

