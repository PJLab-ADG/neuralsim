"""
@file   custom_dataset.py
@author Jianfei Guo, Shanghai AI Lab
@brief  (WIP) Custom dataset from COLMAP
"""

import os
import json
from numbers import Number
import imageio
import skimage
import numpy as np
from tqdm import tqdm
from typing import Any, Dict

from nr3d_lib.config import ConfigDict
from nr3d_lib.utils import get_image_size, load_rgb, cpu_resize
from nr3d_lib.graphics.cameras import decompose_intr_c2w_from_proj_np

from dataio.scene_dataset import SceneDataset

def load_mask(path, downscale: Number=1):
    alpha = imageio.imread(path, as_gray=True)
    alpha = skimage.img_as_float32(alpha)
    if downscale != 1:
        H, W, _ = alpha.shape
        alpha = cpu_resize(alpha, (int(H//downscale), int(W//downscale)), anti_aliasing=False)
    object_mask = alpha > 127.5

    return object_mask

class CustomDataset(SceneDataset):
    def __init__(self, config:ConfigDict) -> None:
        self.config = config
        self.populate(**config)
    
    def populate(
        self,
        root: str,
        cam_file: str=None,
        scale_radius: float=-1,
        ):

        self.main_class_name = "Main"
        self.instance_dir = root
        assert os.path.exists(self.instance_dir), "Data directory is empty"

        image_dir = os.path.join(self.instance_dir, 'images')
        mask_dir = os.path.join(self.instance_dir, 'masks')
        mask_ignore_dir = os.path.join(self.instance_dir, 'masks_ignore')
        self.has_mask = os.path.exists(mask_dir) and len(os.listdir(mask_dir)) > 0
        self.has_mask_ignore = os.path.exists(mask_ignore_dir) and len(os.listdir(mask_ignore_dir)) > 0

        cam_file = 'cam.json' if cam_file is None else cam_file
        self.cam_file = os.path.join(self.instance_dir, cam_file)
        camera_dict = json.load(open(self.cam_file))
        self.n_images = len(camera_dict)
        
        cam_center_norms = []
        intrs_all = []
        c2ws_all = []
        hws_all = []
        image_paths_all = []
        mask_paths_all = []
        mask_ignore_paths_all = []
        for imgname, v in tqdm(camera_dict.items(), desc='loading dataset...'):
            world_mat = np.array(v['P'], dtype=np.float32).reshape(4,4)
            if 'SCALE' in v:
                scale_mat = np.array(v['SCALE'], dtype=np.float32).reshape(4,4)
                P = world_mat @ scale_mat
            else:
                P = world_mat
            intrinsics, c2w = decompose_intr_c2w_from_proj_np(P[:3, :4])
            cam_center_norms.append(np.linalg.norm(c2w[:3,3]))

            intrs_all.append(intrinsics.astype(np.float32))
            c2ws_all.append(c2w.astype(np.float32))
            
            image_path = os.path.join(image_dir, imgname)
            W, H = get_image_size(image_path)
            hws_all.append([H, W])
            image_paths_all.append(image_path)

            fname_base = os.path.splitext(imgname)[0]
            
            if self.has_mask:
                mask_path = os.path.join(mask_dir, f"{fname_base}.png")
                W_, H_ = get_image_size(mask_path)
                assert W_ == W and H_ == H
                mask_paths_all.append(mask_path)
            
            if self.has_mask_ignore:
                mask_ignore_path = os.path.join(mask_ignore_dir, f"{fname_base}.png")
                W_, H_ = get_image_size(mask_ignore_path)
                assert W_ == W and H_ == H
                mask_ignore_paths_all.append(mask_ignore_path)

        self.c2ws_all = np.array(c2ws_all)
        self.intrs_all = np.array(intrs_all)
        self.hws_all = np.array(hws_all)
        self.image_paths_all = image_paths_all
        self.mask_paths_all = mask_paths_all
        self.mask_ignore_paths_all = mask_ignore_paths_all
        
        max_cam_norm = max(cam_center_norms)
        if scale_radius > 0:
            for i in range(len(self.c2ws_all)):
                self.c2ws_all[i][:3, 3] *= (scale_radius / max_cam_norm / 1.1)

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
                global_frame_inds=np.arange(self.n_images), 
            )
        )
        obj = dict(
            id=self.main_class_name.lower(), 
            class_name=self.main_class_name, 
            # Has no recorded data.
        )
        scenario = dict(
            scene_id=f"custom", 
            metas=metas, 
            objects={obj['id']: obj}, 
            observers={cam['id']: cam}
        )
        return scenario
    
    def get_image(self, scene_id: str, camera_id: str, frame_index: int) -> np.ndarray:
        image_path = self.image_paths_all[frame_index]
        return load_rgb(image_path)

    def get_image_occupancy_mask(self, scene_id: str, camera_id: str, frame_index: int) -> np.ndarray:
        mask_path = self.mask_paths_all[frame_index]
        return load_mask(mask_path)

    def get_image_ignore_mask(self, scene_id: str, camera_id: str, frame_index: int) -> np.ndarray:
        mask_ignore_path = self.mask_ignore_paths_all[frame_index]
        return ~load_mask(mask_ignore_path)