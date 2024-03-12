
import os
import json
import pickle
import numpy as np
import pandas as pd
from glob import glob
import transforms3d as t3d # pip install transforms3d
from typing import Any, Dict
from operator import itemgetter

from nr3d_lib.config import ConfigDict
from nr3d_lib.utils import get_image_size

from dataio.autonomous_driving.pandaset.pandaset_dataset import idx_to_frame_str


def _heading_position_to_mat(heading, position):
    # From https://github.com/scaleapi/pandaset-devkit
    quat = np.array([heading["w"], heading["x"], heading["y"], heading["z"]])
    pos = np.array([position["x"], position["y"], position["z"]])
    transform_matrix = t3d.affines.compose(np.array(pos),
                                           t3d.quaternions.quat2mat(quat),
                                           [1.0, 1.0, 1.0])
    return transform_matrix

def process_single_sequence(
    sequence_dir: str, 
    rgb_dirname: str = 'camera', 
    lidar_dirname: str = 'lidar', 
    annotations_dirname: str = 'annotations', 
    lidar_out_dirname: str = 'lidar_processed'
    ):

        scene_objects = dict()
        scene_observers = dict()
        
        total_num_frames = None

        #--------------------------------------------
        #---- Process cameras
        #--------------------------------------------
        cam_root = os.path.join(sequence_dir, rgb_dirname)
        camera_directories = list(filter(lambda s: os.path.isdir(os.path.join(cam_root, s)), os.listdir(cam_root)))
        for cd in camera_directories:
            camera_name = cd.split('/')[-1].split('\\')[-1]
            with open(os.path.join(cd, 'intrinsics.json')) as f:
                intr_json = json.load(f)
            with open(os.path.join(cd , 'poses.json')) as f:
                pose_json = json.load(f)
            with open(os.path.join(cd, 'timestamps.json')) as f:
                timestamps_json = json.load(f)
            
            fx, fy, cx, cy = itemgetter('fx', 'fy', 'cx', 'cy')(intr_json)
            
            img_fp_list = list(sorted(glob(os.path.join(cd, '*.jpg'))))
            assert len(img_fp_list) == len(pose_json), \
                f"Camera {camera_name}: Number of frames mismatches for images (len={len(img_fp_list)}) and poses (len={len(pose_json)})"
            if total_num_frames is None:
                total_num_frames = len(len(img_fp_list))
            else:
                assert len(img_fp_list) == total_num_frames, \
                    f"Camera {camera_name} should have {total_num_frames} frames, but got {len(img_fp_list)}"
            
            obs_dict = {
                'id': camera_name, 
                'class_name': 'Camera', 
                'n_frames': total_num_frames, 
                'camera_model': 'pinhole', 
                'data': {
                    # NOTE: Since camera is the direct children of world (there is no ego_car in PandaSet), 
                    #       The transform to parent is directly c2w
                    'transform': [], 
                    'hw': [], 'intr': [], 'global_timestamps': [], 
                    # 'distortion': []
                }
            }
            for idx, (img_fp, pose_entry, timestamp) in enumerate(zip(img_fp_list, pose_json, timestamps_json)):
                img_fn = os.path.splitext(os.path.split(img_fp)[-1])[0]
                assert img_fn == idx_to_frame_str(idx), f"Image name should be consistent, but {img_fn} != {idx_to_frame_str(idx)}"
                W, H = get_image_size(img_fp)
                intr = np.eye(3)
                intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2] = fx, fy, cx, cy
                pose = _heading_position_to_mat(pose_entry['heading'], pose_entry['position'])
                obs_dict['data']['intr'].append(intr)
                obs_dict['data']['transform'].append(pose)
                obs_dict['data']['global_timestamps'].append(timestamp)
                obs_dict['data']['hw'].append((H, W))
            
            obs_dict['data']['intr'] = np.array(obs_dict['data']['intr']).astype(np.float32)
            obs_dict['data']['transform'] = np.array(obs_dict['data']['transform']).astype(np.float32)
            obs_dict['data']['global_timestamps'] = np.array(obs_dict['data']['global_timestamps']).astype(np.float32)
            obs_dict['data']['hw'] = np.array(obs_dict['data']['global_timestamps']).astype(int)
            
            scene_observers[camera_name] = obs_dict
        
        
        #--------------------------------------------
        #---- Process LiDARs
        #--------------------------------------------
        lidar_dir = os.path.join(sequence_dir, lidar_dirname)
        lidar_name = 'lidar'
        with open(os.path.join(lidar_dir , 'poses.json')) as f:
            pose_json = json.load(f)
        with open(os.path.join(lidar_dir, 'timestamps.json')) as f:
            timestamps_json = json.load(f)
        
        lidar_data_fp_list = list(sorted(glob(os.path.join(lidar_dir, '*.pkl.gz'))))
        assert len(lidar_data_fp_list) == len(pose_json), \
            f"Lidar {lidar_name}: Number of frames mismatches for images (len={len(lidar_data_fp_list)}) and poses (len={len(pose_json)})"
        if total_num_frames is None:
            total_num_frames = len(len(lidar_data_fp_list))
        else:
            assert len(lidar_data_fp_list) == total_num_frames, \
                f"Lidar {lidar_name} should have {total_num_frames} frames, but got {len(lidar_data_fp_list)}"
        
        obs_dict = {
            'id': lidar_name, 
            'class_name': 'RaysLidar', 
            'n_frames': total_num_frames, 
            'data': {
                # NOTE: Since lidar is the direct children of world (there is no ego_car in PandaSet), 
                #       The transform to parent is directly l2w
                'transform': [], 
                'global_timestamps': [], 
                # 'distortion': []
            }
        }
        for idx, (lidar_data_fp, pose_entry, timestamp) in enumerate(zip(lidar_data_fp_list, pose_json, timestamps_json)):
            lidar_data_fn = os.path.splitext(os.path.split(lidar_data_fp)[-1])[0]
            assert lidar_data_fn == idx_to_frame_str(idx), f"Lidar data name should be consistent, but {lidar_data_fn} != {idx_to_frame_str(idx)}"
            pose = _heading_position_to_mat(pose_entry['heading'], pose_entry['position'])
            obs_dict['data']['transform'].append(pose)
            obs_dict['data']['global_timestamps'].append(timestamp)
            
            # NOTE: LiDAR points in PandaSet is stored in world. 
            #       We need to get the rays_o, rays_d and ranges in LiDAR local coords
            
            lidar_pts_in_world = pd.read_pickle(lidar_data_fp)
            lidar_pts_in_local = np.einsum('ij,bi->bj', pose[:3,:3].T, lidar_pts_in_world - pose[:3, 3])
            rays_o = np.zeros([len(lidar_pts_in_local), 3], dtype=np.float32)
            ranges = np.linalg.norm(lidar_pts_in_local)
            rays_d = lidar_pts_in_local / ranges.clip(1e-5)
            
            # Save processed lidar data
            np.savez_compressed(
                os.path.join(sequence_dir, lidar_out_dirname), 
                rays_o=rays_o.astype(np.float32), rays_d=rays_d.astype(np.float32), ranges=ranges.astype(np.float32))

        obs_dict['data']['transform'] = np.array(obs_dict['data']['transform']).astype(np.float32)
        obs_dict['data']['global_timestamps'] = np.array(obs_dict['data']['global_timestamps']).astype(np.float32)

        scene_observers[lidar_name] = obs_dict

        #--------------------------------------------
        #---- Process Annotations
        #--------------------------------------------
        
        
        return
