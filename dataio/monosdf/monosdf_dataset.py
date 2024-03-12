"""
@file   monosdf_dataset.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Dataset IO for MonoSDF dataset
"""

import os
from numbers import Number
import imageio
import skimage
import numpy as np
from glob import glob
from typing import Any, Dict, List, Literal, Union

from nr3d_lib.config import ConfigDict
from nr3d_lib.utils import load_rgb, get_image_size, cpu_resize
from nr3d_lib.graphics.cameras import decompose_intr_c2w_from_proj_np

from dataio.scene_dataset import SceneDataset

def load_mask(path, downscale: Number=1):
    alpha = np.load(path)
    if downscale != 1:
        H, W, _ = alpha.shape
        alpha = cpu_resize(alpha, (int(H//downscale), int(W//downscale)), anti_aliasing=False)
    object_mask = alpha > 0.5

    return object_mask

class MonoSDFDataset(SceneDataset):
    def __init__(self, config: dict) -> None:
        self.config = config
        self.populate(**config)

    def populate(
        self,
        root: str,
        dataset_id: Literal['replica', 'tnt', 'dtu', 'scannet'], 
        scan_id: str,
        center_crop_type='xxxx',
        # num_views: int = None,
        ):
        
        self.main_class_name = "Main"
        self.dataset_id = dataset_id
        scan_id = str(scan_id)
        self.scan_id = scan_id
        self.instance_dir = os.path.join(root, scan_id)
        
        assert os.path.exists(self.instance_dir), f"Empty directory: {self.instance_dir}"

        self.image_paths = list(sorted(glob(os.path.join(self.instance_dir, "*_rgb.png"))))
        self.mask_paths = list(sorted(glob(os.path.join(self.instance_dir, "*_mask.npy"))))
        self.mono_depth_paths = list(sorted(glob(os.path.join(self.instance_dir, "*_depth.npy"))))
        self.mono_normal_paths = list(sorted(glob(os.path.join(self.instance_dir, "*_normal.npy"))))
        
        self.n_images = len(self.image_paths)
        
        self.cam_file = os.path.join(self.instance_dir, 'cameras.npz')
        camera_dict = np.load(self.cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        
        intrs_all = []
        c2ws_all = []
        
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = decompose_intr_c2w_from_proj_np(P)
            intrinsics, pose = intrinsics.astype(np.float32), pose.astype(np.float32)
            
            if center_crop_type == 'center_crop_for_replica':
                scale = 384 / 680
                offset = (1200 - 680 ) * 0.5
                intrinsics[0, 2] -= offset
                intrinsics[:2, :] *= scale
            elif center_crop_type == 'center_crop_for_tnt':
                scale = 384 / 540
                offset = (960 - 540) * 0.5
                intrinsics[0, 2] -= offset
                intrinsics[:2, :] *= scale
            elif center_crop_type == 'center_crop_for_dtu':
                scale = 384 / 1200
                offset = (1600 - 1200) * 0.5
                intrinsics[0, 2] -= offset
                intrinsics[:2, :] *= scale
            elif center_crop_type == 'padded_for_dtu':
                scale = 384 / 1200
                offset = 0
                intrinsics[0, 2] -= offset
                intrinsics[:2, :] *= scale
            elif center_crop_type == 'no_crop':  # for scannet dataset, we already adjust the camera intrinsic duing preprocessing so nothing to be done here
                pass
            else:
                raise RuntimeError(f"Invalid center_crop_type={center_crop_type}")
            
            intrs_all.append(intrinsics)
            c2ws_all.append(pose)
        
        self.intrs_all = np.array(intrs_all)
        self.c2ws_all = np.array(c2ws_all)
        
        hw = []
        for image_path in self.image_paths:
            W, H = get_image_size(image_path)
            hw.append([H, W])
        hw = np.array(hw)  
        self.hws_all = hw
        self.scale_mat = scale_mats[0]
        
    def get_scenario(self, scene_id: str, **kwargs) -> Dict[str, Any]:
        metas = dict(
            n_frames=self.n_images, 
            main_class_name=self.main_class_name
        )
        cam = dict(
            id='camera',
            class_name='Camera', 
            n_frames=self.n_images, 
            data=dict(
                hw=self.hws_all, 
                intr=self.intrs_all,
                transform=self.c2ws_all,
                global_frame_inds=np.arange(self.n_images)
            )
        )
        obj = dict(
            id=self.scan_id,
            class_name=self.main_class_name, 
            # Has no recorded data.
        )
        scenario = dict(
            scene_id=f"{self.dataset_id}-{self.scan_id}", 
            metas=metas, 
            objects={obj['id']: obj}, 
            observers={cam['id']: cam}
        )
        return scenario

    def get_image_wh(self, scene_id: str, camera_id: str, frame_index: Union[int, List[int]]):
        return self.hws_all[frame_index][::-1]

    def get_image(self, scene_id: str, camera_id: str, frame_index: int) -> np.ndarray:
        fpath = self.image_paths[frame_index]
        return load_rgb(fpath)
    
    def get_image_occupancy_mask(self, scene_id: str, camera_id: str, frame_index: int) -> np.ndarray:
        fpath = self.mask_paths[frame_index]
        return load_mask(fpath)
        
    def get_image_mono_depth(self, scene_id: str, camera_id: str, frame_index: int) -> np.ndarray:
        fpath = self.mono_depth_paths[frame_index]
        depth = np.load(fpath).astype(np.float32)
        return depth # [H, W]
    
    def get_image_mono_normals(self, scene_id: str, camera_id: str, frame_index: int) -> np.ndarray:
        fpath = self.mono_normal_paths[frame_index]
        normal = np.load(fpath).astype(np.float32)* 2.-1. # To [-1,1]
        return np.moveaxis(normal, 0, -1) # [H, W, 3]
