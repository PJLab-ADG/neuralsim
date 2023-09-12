"""
@file   colmap_dataset.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Dataset IO for COLMAP format datasets (e.g. from mipNeRF360).
"""

import os
import sys
import numpy as np
from typing import Any, Dict, Literal, Tuple

from nr3d_lib.config import ConfigDict
from nr3d_lib.geometry.math import inverse_transform_matrix_np
from nr3d_lib.geometry.normalize_views import normalize_multi_view
from nr3d_lib.utils import load_rgb, glob_imgs, get_image_size

from dataio.dataset_io import DatasetIO
from dataio.colmap.colmap_loader import *

class COLMAPDataset(DatasetIO):
    def __init__(self, config: ConfigDict) -> None:
        self.config = config
        self.populate(**config)

    def populate(
        self,
        data_dir: str, 
        image_dirname: str = "images", 
        image_mono_depth_dirname: str = "depths", 
        image_mono_normals_dirname: str = "normals", 
        downscale: int = 1, 
        estimate_focus_center: Literal['average', 'pixel_center'] = 'solve', 
        normalize_scale: Literal['average', 'max'] = "max", 
        normalize_scale_factor: float = 1.0, 
        normalize_rotation = True, 
        ):
        
        self.data_dir = data_dir
        self.image_dirname = image_dirname
        self.image_mono_depth_dirname = image_mono_depth_dirname
        self.image_mono_normals_dirname = image_mono_normals_dirname
        self.image_dir = os.path.join(self.data_dir, self.image_dirname)
        self.main_class_name = "Main"
        
        assert os.path.exists(self.data_dir), f"Not exist: {self.data_dir}"
        
        try:
            cameras_extrinsic_file = os.path.join(self.data_dir, "sparse", "0", "images.bin")
            cameras_intrinsic_file = os.path.join(self.data_dir, "sparse", "0", "cameras.bin")
            cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        except:
            cameras_extrinsic_file = os.path.join(self.data_dir, "sparse", "0", "images.txt")
            cameras_intrinsic_file = os.path.join(self.data_dir, "sparse", "0", "cameras.txt")
            cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
        
        camera_infos = []
        for idx, key in enumerate(cam_extrinsics.keys()):
            extr = cam_extrinsics[key]
            cam = cam_intrinsics[extr.camera_id]
            image_path = os.path.join(self.image_dir, os.path.basename(extr.name))
            image_name = os.path.basename(image_path).split(".")[0]
            
            height = cam.height // downscale
            width = cam.width // downscale
            W0, H0 = get_image_size(image_path)
            assert max(abs(H0-height), abs(W0-width)) <= 2, \
                f"Image size mismatch between real image size={[H0, W0]} and intrinsics={[height, width]}"
            # NOTE: Always respect the real image size read from the disk.
            hw = (H0, W0)

            # uid = intr.id
            R = qvec2rotmat(extr.qvec)
            T = np.array(extr.tvec)
            w2c = np.eye(4)
            w2c[:3,:3] = R
            w2c[:3, 3] = T
            
            c2w = inverse_transform_matrix_np(w2c)
            intr = np.eye(4)
            intr[0,0], intr[1,1], intr[0,2], intr[1,2] = np.array(list(cam.params)) / downscale
            
            assert cam.model in ["SIMPLE_PINHOLE", "PINHOLE"], \
                f"Invalid COLMAP intr.model={cam.model}: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
            
            camera_infos.append((image_name, intr, c2w, hw, image_path))

        image_names, intrs_all, c2ws_all, hws_all, image_paths = zip(*sorted(camera_infos, key=lambda item: item[0]))

        self.image_names = image_names
        self.image_paths = image_paths
        self.n_images = len(self.image_paths)
        self.intrs_all = np.array(intrs_all).astype(np.float32)
        self.hws_all = np.array(hws_all).astype(np.float32)
        c2ws_all = np.array(c2ws_all).astype(np.float32)

        normalization, new_c2ws = normalize_multi_view(
            c2ws_all, self.intrs_all, self.hws_all[:,0], self.hws_all[:,1],
            estimate_focus_center=estimate_focus_center, 
            normalize_scale=normalize_scale, 
            normalize_scale_factor=normalize_scale_factor, 
            normalize_rotation=normalize_rotation, 
        )
        self.c2ws_all = new_c2ws

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
                global_frame_ind=np.arange(self.n_images)
            )
        )
        obj = dict(
            id='object',
            class_name=self.main_class_name, 
            # Has no recorded data.
        )
        scenario = dict(
            scene_id='COLMAP', 
            metas=metas, 
            objects={obj['id']: obj}, 
            observers={cam['id']: cam}
        )
        return scenario

    def get_image(self, scene_id: str, camera_id: str, frame_index: int) -> np.ndarray:
        fpath = self.image_paths[frame_index]
        return load_rgb(fpath)

    def get_mono_depth(self, scene_id: str, camera_id: str, frame_index: int) -> np.ndarray:
        assert self.image_mono_depth_dirname is not None, "You should specify image_mono_depth_dirname"
        img_name = self.image_names[frame_index]
        fpath = os.path.join(self.data_dir, self.image_mono_depth_dirname, f'{img_name}.npz')
        assert os.path.exists(fpath), f"Not exist: {fpath}"
        depth = np.load(fpath)['arr_0'].astype(np.float32)
        return depth

    def get_mono_normals(self, scene_id: str, camera_id: str, frame_index: int) -> np.ndarray:
        assert self.image_mono_normals_dirname is not None, "You should specify image_mono_normals_dirname"
        img_name = self.image_names[frame_index]
        fpath = os.path.join(self.data_dir, self.image_mono_normals_dirname, f'{img_name}.jpg')
        assert os.path.exists(fpath), f"Not exist: {fpath}"
        # [-1.,1.] np.float32 in OpenCV local coords
        normals = load_rgb(fpath)*2-1
        return normals

if __name__ == "__main__":
    def unit_test():
        dataset = COLMAPDataset(config=ConfigDict(
            data_dir="/data1/360_v2/bicycle"
        ))
        
        import matplotlib.pyplot as plt
        from nr3d_lib.plot import vis_camera_mplot
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        vis_camera_mplot(ax, dataset.intrs_all, dataset.c2ws_all, dataset.hws_all[0, 0], dataset.hws_all[0, 1], 
                         cam_size=0.03, annotation=True, per_cam_axis=False)
        plt.show()
    unit_test()
