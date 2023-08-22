"""
@file   bmvs_dataset.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Dataset IO for IDR/NeuS/DTU format datasets.
"""

import os
import numpy as np
from typing import Any, Dict, Tuple

from nr3d_lib.config import ConfigDict
from nr3d_lib.geometry import decompose_K_Rt_from_P
from nr3d_lib.utils import load_rgb, glob_imgs, get_image_size

from dataio.dataset_io import DatasetIO

class BMVSDataset(DatasetIO):
    def __init__(self, config: ConfigDict) -> None:
        self.config = config
        self.populate(**config)

    def populate(
        self,
        root: str,
        instance_id: str,
        cam_file='cameras_sphere.npz',
        scale_radius=-1):
        
        self.main_class_name = "Main"
        self.instance_id = instance_id
        self.instance_dir = os.path.join(root, instance_id)
        
        assert os.path.exists(self.instance_dir), f"Not exist: {self.instance_dir}"

        self.image_paths = list(sorted(glob_imgs(os.path.join(self.instance_dir, 'blended_images'))))
        self.image_paths = [p for p in self.image_paths if not "masked" in p]
        self.n_images = len(self.image_paths)

        self.cam_file = os.path.join(self.instance_dir, cam_file)

        camera_dict = np.load(self.cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        intrs_all = []
        c2ws_all = []
        cam_center_norms = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = decompose_K_Rt_from_P(P)
            cam_center_norms.append(np.linalg.norm(pose[:3,3]))
            intrs_all.append(intrinsics.astype(np.float32))
            c2ws_all.append(pose.astype(np.float32))
        
        self.intrs_all = np.array(intrs_all)
        self.c2ws_all = np.array(c2ws_all)
        max_cam_norm = max(cam_center_norms)
        if scale_radius > 0:
            for i in range(len(self.c2ws_all)):
                self.c2ws_all[i][:3, 3] *= (scale_radius / max_cam_norm / 1.1)

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
                global_frame_ind=np.arange(self.n_images)
            )
        )
        obj = dict(
            id=self.instance_id,
            class_name=self.main_class_name, 
            # Has no recorded data.
        )
        scenario = dict(
            scene_id=f"BMVS-{self.instance_id}", 
            metas=metas, 
            objects={obj['id']: obj}, 
            observers={cam['id']: cam}
        )
        return scenario

    def get_image(self, scene_id: str, camera_id: str, frame_index: int) -> np.ndarray:
        fpath = self.image_paths[frame_index]
        return load_rgb(fpath)

if __name__ == '__main__':
    def unit_test_cams():
        eg_config = ConfigDict(
            root="/data1/bmvs", 
            # instance_id="5aa515e613d42d091d29d300", 
            # instance_id="5a4a38dad38c8a075495b5d2", 
            instance_id="5a8315f624b8e938486e0bd8", 
            cam_file="cameras_sphere.npz"
        )
        instance = BMVSDataset(eg_config)

        [print(p) for p in instance.image_paths]
        
        # from nr3d_lib.plot import vis_camera_o3d_from_arrays
        # vis_camera_o3d_from_arrays(
        #     instance.intrs_all, instance.c2ws_all, instance.hws_all[..., 0], instance.hws_all[..., 1], 
        #     cam_size=0.03)

        import matplotlib.pyplot as plt
        from nr3d_lib.plot import vis_camera_mplot
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        vis_camera_mplot(ax, instance.intrs_all, instance.c2ws_all, instance.hws_all[0, 0], instance.hws_all[0, 1], 
                         cam_size=0.03, annotation=True, per_cam_axis=False)
        plt.show()
        
    unit_test_cams()
