"""
@file   mega_nerf_dataset.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Dataset IO for MegaNeRF dataset
"""

import os
import numpy as np
from math import ceil
from zipfile import ZipFile
from typing import Any, Dict, List, Literal, Tuple, Union

import torch

from nr3d_lib.utils import load_rgb
from nr3d_lib.config import ConfigDict
from nr3d_lib.models.fields_forest.utils import prepare_dense_grids

from dataio.scene_dataset import SceneDataset

def load_mask(path: str):
    with ZipFile(path) as zf:
        base = os.path.basename(os.path.normpath(path))
        with zf.open(base) as f:
            keep_mask = torch.load(f, map_location='cpu')
    return keep_mask

class MegaNeRFDataset(SceneDataset):
    def __init__(self, config: dict) -> None:
        self.config = config
        self.populate(**config)
    
    def populate(
        self, 
        dataset_path: str, 
        dataset_name: str, 
        train_every: int = 1, 
        split: Literal['train', 'val'] = 'train', 
        cluster_mask_path: str = None, 
        ray_altitude_range: Tuple[int, int] = None
        ):
        
        self.dataset_name = dataset_name
        self.main_class_name = "Main"
        
        coordinate_info = torch.load(os.path.join(dataset_path, 'coordinates.pt'), map_location='cpu')
        self.origin_drb = coordinate_info['origin_drb']
        self.pose_scale_factor = coordinate_info['pose_scale_factor']
        
        if ray_altitude_range is not None:
            # 0th dim is the altitude dim
            self.x_minmax = [(x - self.origin_drb[0]) / self.pose_scale_factor for x in ray_altitude_range]
        else:
            self.x_minmax = [-1., 1.]
        
        val_meta_dir = os.path.join(dataset_path, 'val', 'metadata')
        val_paths = [os.path.join(val_meta_dir, l) for l in sorted(os.listdir(val_meta_dir))]
        
        if split == 'train':
            train_meta_dir = os.path.join(dataset_path, 'train', 'metadata')
            train_path_candidates = [os.path.join(train_meta_dir, l) for l in sorted(os.listdir(train_meta_dir))]
            train_paths = [train_path_candidates[i] for i in range(0, len(train_path_candidates), train_every)]
            total_paths = list(sorted(train_paths + val_paths))
        else:
            total_paths = val_paths
        
        intrs_all = []
        c2ws_all = []
        hws_all = []
        image_paths_all = []
        mask_paths_all = []
        for metadata_path in total_paths:
            base = os.path.basename(os.path.normpath(metadata_path))
            stem = os.path.splitext(base)[0]
            cur_split = os.path.split(os.path.split(os.path.split(metadata_path)[0])[0])[1]
            
            image_path = None
            for extension in ['.jpg', '.JPG', '.png', '.PNG']:
                candidate = os.path.join(dataset_path, cur_split, 'rgbs', f"{stem}{extension}")
                if os.path.exists(candidate):
                    image_path = candidate
                    break
            
            assert os.path.exists(image_path)
            
            metadata = torch.load(metadata_path, map_location='cpu')
            
            c2ws_all.append(metadata['c2w'].data.cpu().numpy())
            intrs_all.append(metadata['intrinsics'].data.cpu().numpy())
            hws_all.append([metadata['H'], metadata['W']])
            
            image_paths_all.append(image_path)
            if cluster_mask_path is not None:
                mask_path = os.path.join(cluster_mask_path, base)
            elif os.path.exists(dataset_mask:=os.path.join(dataset_path, 'masks', base)):
                mask_path = dataset_mask
            else:
                mask_path = None
            if mask_path is not None:
                mask_paths_all.append(mask_path)
            
        intrs_all = np.array(intrs_all)
        self.intrs_all = np.zeros([len(intrs_all), 3, 3], dtype=float)
        self.intrs_all[:, 0, 0] = intrs_all[:, 0]
        self.intrs_all[:, 1, 1] = intrs_all[:, 1]
        self.intrs_all[:, 0, 2] = intrs_all[:, 2]
        self.intrs_all[:, 1, 2] = intrs_all[:, 3]
        self.intrs_all[:, 2, 2] = 1.

        hws_all = np.array(hws_all)
        self.hws_all = hws_all


        # NOTE: Mega-NeRF use OpenGL camera coordiantes
        """
            < opencv / colmap convention >                 --->>>   < openGL convention >
            facing [+z] direction, x right, y downwards    --->>>  facing [-z] direction, x right, y upwards, 
                        z                                                ↑ y               
                      ↗                                                  |                  
                     /                                                   |               
                    /                                                    |                 
                    o------> x                                           o------> x      
                    |                                                   /                       
                    |                                                  /                    
                    |                                                 ↙              
                    ↓                                               z                
                    y                                                                        
        """
        opencv_to_opengl = np.eye(4)
        opencv_to_opengl[:3, :3] = np.array(
            [[1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]]
        )
        c2ws_all = np.array(c2ws_all)
        
        self.c2ws_all = np.eye(4)[None,...].repeat(c2ws_all.shape[0], axis=0)
        self.c2ws_all[:, :3, :4] = c2ws_all[:, :3, :4]
        self.c2ws_all = self.c2ws_all @ opencv_to_opengl[None,...]
        # NOTE:
        #       c2w tracks are already in [-1,1] normalized range, thanks to `origin_drb` and `pose_scale_factor`
        
        self.image_paths_all = image_paths_all
        self.mask_paths_all = mask_paths_all
        self.n_images = len(self.image_paths_all)
    
    def get_scenario(self, scene_id: str, should_split_block: True, split_block_cfg: dict = None) -> Dict[str, Any]:
        metas = dict(
            n_frames=self.n_images, 
            main_class_name=self.main_class_name
        )
        
        tracks = self.c2ws_all[:, :3, 3]
        # NOTE: Ignore the 0th dim (altitude dim)
        tracks_2d = tracks[:, 1:]
        aabb_2d = np.stack([tracks_2d.min(0), tracks_2d.max(0)], 0)
        aabb = np.zeros([2, 3])
        aabb[:, 1:] = aabb_2d
        aabb[:, 0] = np.array(self.x_minmax)
        metas['aabb'] = aabb
        
        # NOTE: floor_up_sign = -1: 
        #       (x - floor_at < 0) means above ground (+sdf), (x - floor_at > 0) means below ground (-sdf)
        metas['floor_info'] = dict(floor_dim='x', floor_up_sign=-1,  floor_at=(self.x_minmax[0]+self.x_minmax[1])/2.)
        
        if should_split_block:
            assert split_block_cfg is not None, "Need to provide split_block_cfg if split_block"
            
            # resolution, world_origin, world_block_size, level = prepare_dense_grids(aabb, **split_block_cfg)
            # world_origin = world_origin.data.cpu().numpy()
            
            resolution_2d, world_origin_2d, world_block_size, level = prepare_dense_grids(aabb_2d, **split_block_cfg)
            world_origin = np.zeros([3])
            world_origin[1:] = world_origin_2d.data.cpu().numpy()
            world_origin[0] = self.x_minmax[0]
            
            resolution = np.zeros([3])
            resolution[1:] = resolution_2d.data.cpu().numpy()
            resolution[0] = ceil((self.x_minmax[1] - self.x_minmax[0])/world_block_size)
            
            block_ks = np.stack(np.meshgrid(*[np.arange(r) for r in resolution.tolist()], indexing='ij'), -1).reshape(-1, 3)
            
            split_block_info = metas['split_block_info'] = {}
            split_block_info['block_ks'] = block_ks
            split_block_info['world_origin'] = world_origin
            split_block_info['world_block_size'] = world_block_size
            split_block_info['level'] = level
        
        cam = dict(
            id='camera',
            class_name='Camera', 
            n_frames=self.n_images, 
            data=dict(
                hw=self.hws_all, 
                intr=self.intrs_all,
                transform=self.c2ws_all,
                global_frame_inds=np.arange(self.n_images), 
            )
        )
        obj = dict(
            id=self.main_class_name.lower(), 
            class_name=self.main_class_name, 
            # Has no recorded data.
        )
        scenario = dict(
            scene_id=self.dataset_name, 
            metas=metas, 
            objects={obj['id']: obj}, 
            observers={cam['id']: cam}
        )
        return scenario
    
    def get_image_wh(self, scene_id: str, camera_id: str, frame_index: Union[int, List[int]]):
        return self.hws_all[frame_index][::-1]
    
    def get_image(self, scene_id: str, camera_id: str, frame_index: int) -> np.ndarray:
        fpath = self.image_paths_all[frame_index]
        return load_rgb(fpath)
    
    def get_image_occupancy_mask(self, scene_id: str, camera_id: str, frame_index: int) -> np.ndarray:
        fpath = self.mask_paths_all[frame_index]
        return load_mask(fpath)
