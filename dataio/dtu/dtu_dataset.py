"""
@file   dtu_dataset.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Dataset IO for IDR/NeuS/DTU format datasets.
"""
import os
import numbers
import imageio
import skimage
import numpy as np
from typing import Any, Dict, Literal

from nr3d_lib.config import ConfigDict
from nr3d_lib.utils import load_rgb, glob_imgs, get_image_size, cpu_resize
from nr3d_lib.geometry import decompose_K_Rt_from_P, inverse_transform_matrix_np

from dataio.dataset_io import DatasetIO

def load_mask(path, downscale: numbers.Number=1):
    alpha = imageio.imread(path, as_gray=True)
    alpha = skimage.img_as_float32(alpha)
    if downscale != 1:
        H, W, _ = alpha.shape
        alpha = cpu_resize(alpha, (int(H//downscale), int(W//downscale)), anti_aliasing=False)
    object_mask = alpha > 127.5

    return object_mask

def write_cams(filepath: str, intrs: np.ndarray, c2ws: np.ndarray, normalization: np.ndarray):
    """ Write intrs, c2ws mat to .npz file following the IDR/NeuS DTU format

    Args:
        filepath (str): Output ".npz" file path
        intrs (np.ndarray): [N, 3/4, 3/4] pinhole intrinsics mat
        c2ws (np.ndarray): [N, 4, 4] camera to world transform mat
        normalization (np.ndarray): [4, 4] normalization mat
    """
    assert filepath.endswith(".npz"), f"Should end with '.npz': {filepath}"
    # Write Ps, normalization following the IDR/NeuS DTU format
    data_dict = {}
    
    w2cs = inverse_transform_matrix_np(c2ws)
    for i, (intr, w2c) in enumerate(zip(intrs, w2cs)):
        intr_ex = np.eye(4, dtype=np.float32)
        intr_ex[:3, :3] = intr[:3, :3]
        P = (intr_ex @ w2c).astype(np.float32)
        data_dict[f"camera_mat_{i:d}"] = intr_ex # [4,4]
        data_dict[f"world_mat_{i:d}"] = P # [4,4]
    
    for i in range(len(w2cs)):
        data_dict[f"scale_mat_{i:d}"] = normalization
    
    np.savez(filepath, **data_dict)

class DTUDataset(DatasetIO):
    def __init__(self, config: ConfigDict) -> None:
        self.config = config
        self.populate(**config)

    def populate(
        self,
        root: str,
        instance_id: str,
        load_mask=True, 
        cam_file='cameras.npz',
        num_few_shot: int = None, # Optional few-shot training
        drop_frame_split: Literal['train', 'test'] = None, # Optional drop frame training / testing
        mono_cues_with_mask: bool = False, 
        scale_radius=-1):
        
        self.main_class_name = "Main"
        self.instance_id = instance_id
        self.instance_dir = os.path.join(root, instance_id)
        
        assert os.path.exists(self.instance_dir), f"Not exist: {self.instance_dir}"

        self.image_dir = os.path.join(self.instance_dir, 'image')
        self.image_paths = list(sorted(glob_imgs(self.image_dir)))
        
        if load_mask:
            self.mask_dir = os.path.join(self.instance_dir, 'mask')
            self.mask_paths = list(sorted(glob_imgs(self.mask_dir)))

        self.cam_file = os.path.join(self.instance_dir, cam_file)
        
        self.mono_cues_with_mask = mono_cues_with_mask

        if num_few_shot is not None and num_few_shot > 0:
            assert num_few_shot in [3, 6, 9]
            # NOTE: The same setting as monoSDF https://github.com/autonomousvision/monosdf/blob/main/code/datasets/scene_dataset.py
            self.sel_frame_idx = [25, 22, 28, 40, 44, 48, 0, 8, 13][:num_few_shot]
        elif drop_frame_split is not None:
            # NOTE: The same setting as NeuS2
            if drop_frame_split == 'train':
                self.sel_frame_idx = [i for i in range(len(self.image_paths)) if i not in [8, 13, 16, 21, 26, 31, 34, 56]]
            elif drop_frame_split == 'test':
                self.sel_frame_idx = [i for i in range(len(self.image_paths)) if i in [8, 13, 16, 21, 26, 31, 34, 56]]
            else:
                raise RuntimeError(f'Invalid drop_frame_split={drop_frame_split}')
        else:
            self.sel_frame_idx = list(range(len(self.image_paths)))

        self.n_images = len(self.sel_frame_idx)
        self.image_paths = [self.image_paths[idx] for idx in self.sel_frame_idx]
        
        if load_mask:
            self.mask_paths = [self.mask_paths[idx] for idx in self.sel_frame_idx]

        camera_dict = np.load(self.cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in self.sel_frame_idx]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in self.sel_frame_idx]

        intrs_all = []
        c2ws_all = []
        cam_center_norms = []
        # dtu_pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = decompose_K_Rt_from_P(P)
            cam_center_norms.append(np.linalg.norm(pose[:3,3]))
            intrs_all.append(intrinsics.astype(np.float32))
            c2ws_all.append(pose.astype(np.float32))
            # dtu_pose = pose.copy()
            # dtu_pose[:, 3:] = scale_mat @ pose[:, 3:]
            # dtu_pose_all.append(dtu_pose)
        
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
            scene_id=f"DTU-{self.instance_id}", 
            metas=metas, 
            objects={obj['id']: obj}, 
            observers={cam['id']: cam}
        )
        return scenario

    def get_image(self, scene_id: str, camera_id: str, frame_index: int) -> np.ndarray:
        fpath = self.image_paths[frame_index]
        return load_rgb(fpath)
    
    def get_occupancy_mask(self, scene_id: str, camera_id: str, frame_index: int) -> np.ndarray:
        fpath = self.mask_paths[frame_index]
        return load_mask(fpath)

    def get_mono_depth(self, scene_id: str, camera_id: str, frame_index: int) -> np.ndarray:
        img_name = self.img_names[frame_index]
        fpath = os.path.join(self.root_dir, self.split, 'depth_wmask' if self.mono_cues_with_mask else 'depth', f'{img_name}.npz')
        assert os.path.exists(fpath), f"Not exist: {fpath}"
        depth = np.load(fpath)['arr_0'].astype(np.float32)
        return depth

    def get_mono_normals(self, scene_id: str, camera_id: str, frame_index: int) -> np.ndarray:
        img_name = self.img_names[frame_index]
        fpath = os.path.join(self.root_dir, self.split, 'normal_wmask' if self.mono_cues_with_mask else 'normal', f'{img_name}.jpg')
        assert os.path.exists(fpath), f"Not exist: {fpath}"
        # [-1.,1.] np.float32
        normals = load_rgb(fpath)*2-1
        # TODO: align coordinate system
        return normals

if __name__ == "__main__":
    def unit_test_cams():
        eg_config = ConfigDict(
            root="/data1/neus", 
            instance_id="bmvs_jade", 
            cam_file="cameras_sphere.npz"
        )
        instance = DTUDataset(eg_config)
        
        [print(p) for p in instance.image_paths]
        
        # from nr3d_lib.plot import vis_camera_o3d_from_arrays
        # vis_camera_o3d_from_arrays(instance.intrs_all, instance.c2ws_all, instance.hws_all[..., 0], instance.hws_all[..., 1])

        import matplotlib.pyplot as plt
        from nr3d_lib.plot import vis_camera_mplot
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        vis_camera_mplot(ax, instance.intrs_all, instance.c2ws_all, instance.hws_all[0, 0], instance.hws_all[0, 1], cam_size=1.0, annotation=True, per_cam_axis=True)
        plt.show()
    
    unit_test_cams()