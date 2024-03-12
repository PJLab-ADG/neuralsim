"""
Performance: [2.5 mins @ downscale=2] 
Convert the merged objects to pandas DataFrame at the very beginning, load all into memory, significantly speeding up the process.
"""
import os
import sys

project_root_path = "/home/guojianfei/ai_ws/neuralsim_dev/"
sys.path.append(project_root_path)
print(f"Added {project_root_path} to sys.path")

import io
import os
import cv2
import math
import pickle
import imageio # NOTE: Also needs pip install imageio-ffmpeg
import warnings
import functools
import numpy as np
from tqdm import tqdm
from PIL import Image
from typing import List, Literal
from numbers import Number
import dask.dataframe as dd
import matplotlib.pyplot as plt

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from waymo_open_dataset import v2

from nr3d_lib.utils import pad_images_to_same_size, image_downscale, cpu_resize
from nr3d_lib.plot import draw_2dbox_on_im, draw_bool_mask_on_im, draw_patch_on_im, get_n_ind_colors

# Disable annoying warnings from PyArrow using under the hood.
warnings.simplefilter(action='ignore', category=FutureWarning)

def read(dataset_dir: str, tag: str, context_name: str) -> dd.DataFrame:
    """Creates a Dask DataFrame for the component specified by its tag."""
    paths = tf.io.gfile.glob(f'{dataset_dir}/{tag}/{context_name}.parquet')
    return dd.read_parquet(paths)

def make_video(
    dataset_root: str, context_name: str, vid_base: str, 
    save_perframe=True, downscale: int = 1, 
    mask_type: Literal['semantic', 'object'] = 'object'):
    if save_perframe:
        os.makedirs(vid_base, exist_ok=True)

    # Lazily read DataFrames for all components.
    cam_img_df = read(dataset_root, 'camera_image', context_name)
    lidar_box_df = read(dataset_root, 'lidar_box', context_name)
    lidar_df = read(dataset_root, 'lidar', context_name)

    asset_type = 'ped'
    asset_camera_sensor_df = read(dataset_root, f'{asset_type}_asset_camera_sensor', context_name)
    asset_auto_label_df = read(dataset_root, f'{asset_type}_asset_auto_label', context_name)
    # common: [segment, timestamp, camera_name, laser_object_id]
    ped_asset_df = v2.merge(asset_camera_sensor_df, asset_auto_label_df)
    ped_laser_ids = ped_asset_df['key.laser_object_id'].unique().values.compute()
    ped_ins_cmap = dict(zip(ped_laser_ids, get_n_ind_colors(len(ped_laser_ids))))
    
    asset_type = 'veh'
    asset_camera_sensor_df = read(dataset_root, f'{asset_type}_asset_camera_sensor', context_name)
    asset_auto_label_df = read(dataset_root, f'{asset_type}_asset_auto_label', context_name)
    # common: [segment, timestamp, camera_name, laser_object_id]
    veh_asset_df = v2.merge(asset_camera_sensor_df, asset_auto_label_df)
    veh_laser_ids = veh_asset_df['key.laser_object_id'].unique().values.compute()
    veh_ins_cmap = dict(zip(veh_laser_ids, get_n_ind_colors(len(veh_laser_ids))))

    sem_cmap = np.array(get_n_ind_colors(255))

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # NOTE: Dask.Dataframe is too slow; directly using in-memory pandas.DataFrame
    #       This will store seq data directly into CPU mem; usually several GiB per seq.
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    cam_img_df = cam_img_df.compute()
    ped_asset_df = ped_asset_df.compute()
    veh_asset_df = veh_asset_df.compute()
    
    grouped_cam_img_df = cam_img_df.groupby(['key.segment_context_name', 'key.frame_timestamp_micros', 'key.camera_name'])
    ped_asset_grouped_per_frame_df = ped_asset_df.groupby(['key.segment_context_name', 'key.frame_timestamp_micros', 'key.camera_name'])
    veh_asset_grouped_per_frame_df = veh_asset_df.groupby(['key.segment_context_name', 'key.frame_timestamp_micros', 'key.camera_name'])

    im_per_frame = []
    
    # Unique timestamps (already sorted)
    context_segment_name = cam_img_df.head(1)['key.segment_context_name'].iloc[0]
    unique_ts = list(cam_img_df['key.frame_timestamp_micros'].unique())
    for find, ts in enumerate(tqdm(unique_ts)):
        im_per_cam = []
        
        for camera_name in [v2._camera_image.CameraName.SIDE_LEFT, 
                            v2._camera_image.CameraName.FRONT_LEFT, 
                            v2._camera_image.CameraName.FRONT, 
                            v2._camera_image.CameraName.FRONT_RIGHT, 
                            v2._camera_image.CameraName.SIDE_RIGHT]:
            camera_name = camera_name.value

            
            cam_row = grouped_cam_img_df.get_group((context_segment_name, ts, camera_name)).reset_index()
            cam_row = cam_row.squeeze()
            camera_image = v2.CameraImageComponent.from_dict(cam_row)
            
            # ---- Start processing
            im = tf.io.decode_jpeg(camera_image.image).numpy() # uint8, 0~255, [H, W, 3]
            im = image_downscale(im, downscale=downscale)
            im = (im * 255).clip(0, 255).astype(np.uint8)

            if (context_segment_name, ts, camera_name) in ped_asset_grouped_per_frame_df.groups:
                ped_rows = ped_asset_grouped_per_frame_df.get_group((context_segment_name, ts, camera_name))
            else:
                ped_rows = None
            
            if (context_segment_name, ts, camera_name) in veh_asset_grouped_per_frame_df.groups:
                veh_rows = veh_asset_grouped_per_frame_df.get_group((context_segment_name, ts, camera_name))
            else:
                veh_rows = None
            
            if ped_rows is not None:
                ped_rows = ped_rows.reset_index()
                for _, row in ped_rows.iterrows():
                    asset_auto_label_comp = v2.ObjectAssetAutoLabelComponent.from_dict(row)
                    asset_cam_comp = v2.ObjectAssetCameraSensorComponent.from_dict(row)
                    
                    lid = asset_auto_label_comp.key.laser_object_id
                    lidar_box_label = f"lid: {lid[:6]}"
                    box = asset_cam_comp.camera_region
                    height, width = box.size.y / downscale, box.size.x / downscale
                    center_x, center_y = box.center.x / downscale, box.center.y / downscale
                    
                    im = draw_2dbox_on_im(
                        im, 
                        center_x, center_y, 
                        width=width, height=height, 
                        label=lidar_box_label, fontscale=1./downscale, 
                        color=(128, 0, 0), fillalpha=0.2, 
                    )
                    
                    if mask_type == 'object':
                        object_mask = asset_auto_label_comp.object_mask_numpy
                        object_mask = cpu_resize(object_mask.astype(bool), (int(height), int(width)))
                        im = draw_bool_mask_on_im(
                            im, object_mask, color=ped_ins_cmap[lid], alpha=0.5, 
                            h0=int((center_y - height / 2)), 
                            w0=int((center_x - width / 2)), 
                        )
                    elif mask_type == 'semantic':
                        semantic_mask = asset_auto_label_comp.semantic_mask_numpy
                        semantic_mask = sem_cmap[semantic_mask.squeeze(-1)]
                        semantic_mask = cpu_resize(semantic_mask, (int(height), int(width)), preserve_range=True).astype(semantic_mask.dtype)
                        im = draw_patch_on_im(
                            im, semantic_mask, alpha=1, 
                            h0=int((center_y - height / 2)), 
                            w0=int((center_x - width / 2)), 
                        )
            
            if veh_rows is not None:
                veh_rows = veh_rows.reset_index()
                for _, row in veh_rows.iterrows():
                    asset_auto_label_comp = v2.ObjectAssetAutoLabelComponent.from_dict(row)
                    asset_cam_comp = v2.ObjectAssetCameraSensorComponent.from_dict(row)
                    
                    lid = asset_auto_label_comp.key.laser_object_id
                    lidar_box_label = f"lid: {lid[:6]}"
                    box = asset_cam_comp.camera_region
                    height, width = box.size.y / downscale, box.size.x / downscale
                    center_x, center_y = box.center.x / downscale, box.center.y / downscale
                    
                    """
                    patch = asset_cam_comp.rgb_image_numpy
                    # NOTE: When interacting with the original image, use this size
                    height0, width0 = box.size.y, box.size.x 

                    # NOTE: The size of the patch/mask is always consistent, 
                    # but when the original patch size (camera_region) is too large, the waymo-team will compress the mask/patch
                    height1, width1, *_ = patch.shape
                    height, width, *_ = semantic_mask.shape
                    
                    e.g.:
                    >>> camera_region: (702, 909) 
                    >>> patch: (395, 512) 
                    >>> mask: (395, 512)
                        
                    print((int(height0), int(width0)), (height1, width1), (height, width))
                    if not (int(height0)==height1==height) or not (int(width0)==width1==width):
                        _ = 1
                    """
                    # patch = asset_cam_comp.rgb_image_numpy
                    # height0, width0 = box.size.y, box.size.x
                    # height1, width1, *_ = patch.shape
                    # height, width, *_ = semantic_mask.shape
                    # if not (int(height0)==height1==height) or not (int(width0)==width1==width):
                    #     _ = 1
                    
                    im = draw_2dbox_on_im(
                        im, 
                        center_x, center_y, 
                        width=width, height=height, 
                        label=lidar_box_label, fontscale=1./downscale, 
                        color=(128, 0, 0), fillalpha=0.2, 
                    )
                    
                    if mask_type == 'object':
                        object_mask = asset_auto_label_comp.object_mask_numpy
                        object_mask = cpu_resize(object_mask.astype(bool), (int(height), int(width)))
                        im = draw_bool_mask_on_im(
                            im, object_mask, color=veh_ins_cmap[lid], alpha=0.5, 
                            h0=int((center_y - height / 2)), 
                            w0=int((center_x - width / 2)), 
                        )
                    elif mask_type == 'semantic':
                        semantic_mask = asset_auto_label_comp.semantic_mask_numpy
                        semantic_mask = sem_cmap[semantic_mask.squeeze(-1)]
                        semantic_mask = cpu_resize(semantic_mask, (int(height), int(width)), preserve_range=True).astype(semantic_mask.dtype)
                        im = draw_patch_on_im(
                            im, semantic_mask, alpha=1, 
                            h0=int((center_y - height / 2)), 
                            w0=int((center_x - width / 2)), 
                        )

            im_per_cam.append(im)
            
        im_per_cam = pad_images_to_same_size(im_per_cam, padding='top_left')
        im_per_cam = np.concatenate(im_per_cam, axis=1)
        
        if save_perframe:
            imageio.imwrite(os.path.join(vid_base, f"{find:04d}.jpg"), im_per_cam)
        im_per_frame.append(im_per_cam)
    
    vid_path = vid_base + ".mp4"
    imageio.mimwrite(vid_path, im_per_frame, fps=24)
    print(f"=> Video saved to {vid_path}")
            

if __name__ == "__main__":
    mask_type = 'object' # ['object', 'semantic']
    make_video(
        "/data1/waymo/v2", "7670103006580549715_360_000_380_000", 
        f"./dev_test/767010_cam_assets_{mask_type}_mask", save_perframe=True, downscale=2, 
        mask_type=mask_type)