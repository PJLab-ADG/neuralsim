"""
@file   normalize_bmvs.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Normalize Blended-MVS dataset camera poses
"""
import os
import sys
def set_env(depth: int):
    # Add project root to sys.path
    current_file_path = os.path.abspath(__file__)
    project_root_path = os.path.dirname(current_file_path)
    for _ in range(depth):
        project_root_path = os.path.dirname(project_root_path)
    if project_root_path not in sys.path:
        sys.path.append(project_root_path)
        print(f"Added {project_root_path} to sys.path")
    return project_root_path
project_root_path = set_env(2)

import numpy as np
from glob import glob
from tqdm import tqdm
from typing import Tuple

from dataio.dtu.dtu_dataset import write_cams as write_cams_dtu

from nr3d_lib.utils import get_image_size
from nr3d_lib.geometry.math import inverse_transform_matrix_np
from nr3d_lib.geometry.normalize_views import normalize_multi_view


def load_cam_original(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """ Read camera txt file
    modified from https://github.com/YoYo000/MVSNet/blob/master/mvsnet/preprocess.py

    Args:
        filepath (str): Given `.txt` camera param file path

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            [4,4] w2c, world to camera transform mat
            [4,4] intr, pinhole camera intrinsics mat
    """
    intr = np.eye(4)
    w2c = np.eye(4)
    
    words = open(filepath).read().split()
    # Read extrinsic
    for i in range(0, 4):
        for j in range(0, 4):
            extrinsic_index = 4 * i + j + 1
            w2c[i][j] = words[extrinsic_index]

    # Read intrinsic
    for i in range(0, 3):
        for j in range(0, 3):
            intrinsic_index = 3 * i + j + 18
            intr[i][j] = words[intrinsic_index]

    #---- Original BMVS data on the last row of intr
    # interval_scale = 1.
    # if len(words) == 29:
    #     intr[3][0] = words[27]
    #     intr[3][1] = float(words[28]) * interval_scale
    #     # intr[3][2] = FLAGS.max_d
    #     intr[3][2] = 128 # NOTE: manually fixed here.
    #     intr[3][3] = intr[3][0] + intr[3][1] * intr[3][2]
    # elif len(words) == 30:
    #     intr[3][0] = words[27]
    #     intr[3][1] = float(words[28]) * interval_scale
    #     intr[3][2] = words[29]
    #     intr[3][3] = intr[3][0] + intr[3][1] * intr[3][2]
    # elif len(words) == 31:
    #     intr[3][0] = words[27]
    #     intr[3][1] = float(words[28]) * interval_scale
    #     intr[3][2] = words[29]
    #     intr[3][3] = words[30]
    # else:
    #     intr[3][0] = 0
    #     intr[3][1] = 0
    #     intr[3][2] = 0
    #     intr[3][3] = 0

    return w2c, intr

def write_original(filepath: str, w2c: np.ndarray, intr: np.ndarray):
    """ Write w2c, intr mat to .txt file following the original BMVS format

    Args:
        filepath (str): Output `.txt` camera param file path
        w2c (np.ndarray): [4,4] world to camera transform mat
        intr (np.ndarray): [4,4] pinhole camera intrinsics mat
    """
    f = open(filepath, "w")

    f.write('extrinsic\n')
    for i in range(0, 4):
        for j in range(0, 4):
            f.write(str(w2c[i][j]) + ' ')
        f.write('\n')
    f.write('\n')

    f.write('intrinsic\n')
    for i in range(0, 3):
        for j in range(0, 3):
            f.write(str(intr[i][j]) + ' ')
        f.write('\n')

    f.write('\n' + str(intr[3][0]) + ' ' + str(intr[3][1]) + ' ' + str(intr[3][2]) + ' ' + str(intr[3][3]) + '\n')
    f.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, help="BlendedMVS dataset root", default='/data1/bmvs')
    parser.add_argument("--ins_list", type=str, help="Optional path to given seq list", default=None)
    
    args = parser.parse_args()
    
    assert os.path.isdir(args.root), f"Not exist: {args.root}"
    if args.ins_list is None:
        ins_list = list(sorted(os.listdir(args.root)))
    else:
        with open(args.ins_list, 'r') as f:
            ins_list = f.read().splitlines()
    
    succeeded_list = []
    
    for ins in tqdm(ins_list, "Processing ..."):
        try:
            ins_dir = os.path.join(args.root, ins)
            cam_file_list = list(sorted(glob(os.path.join(ins_dir, "cams", "*_cam.txt"))))
            
            w2cs = []
            intrs = []
            for cam_file in cam_file_list:
                w2c, intr = load_cam_original(cam_file)
                w2cs.append(w2c)
                intrs.append(intr)
            w2cs = np.array(w2cs)
            c2ws = inverse_transform_matrix_np(w2cs)
            intrs = np.array(intrs)
            
            rgb_file_list = glob(os.path.join(ins_dir, "blended_images", "*.jpg"))
            rgb_file_list = [f for f in rgb_file_list if not f.endswith("masked.jpg")]
            rgb_file_list = list(sorted(rgb_file_list))
            
            W,  H = get_image_size(rgb_file_list[0])
            
            normalization, new_c2ws = normalize_multi_view(
                c2ws, intrs, H, W, 
                normalize_scale=True, 
                normalize_rotation=True, 
                estimate_focus_center=True)

            # from nr3d_lib.plot.plot_3d import vis_camera_o3d_from_arrays
            # vis_camera_o3d_from_arrays(intrs, new_c2ws, H, W, cam_size=0.1)

            cam_file_output = os.path.join(ins_dir, "cameras_sphere.npz")
            write_cams_dtu(cam_file_output, intrs, c2ws, normalization)
            
            succeeded_list.append(ins)
        except RuntimeError:
            pass
    
    with open(os.path.join(project_root_path, 'dataio', 'bmvs', 'normalized.lst'), 'w') as f:
        f.writelines([ins + "\n" for ins in succeeded_list])