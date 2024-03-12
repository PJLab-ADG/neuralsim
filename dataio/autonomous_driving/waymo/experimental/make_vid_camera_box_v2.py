"""
Performance: [2.5 mins @ downscale=2] 
Directly convert the merged objects into a pandas DataFrame at the beginning, load them all into memory, significantly increasing the speed.
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
from typing import List
from numbers import Number
import dask.dataframe as dd
import matplotlib.pyplot as plt

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from waymo_open_dataset import v2

from nr3d_lib.utils import pad_images_to_same_size, image_downscale
from nr3d_lib.plot import draw_2dbox_on_im

# Disable annoying warnings from PyArrow using under the hood.
warnings.simplefilter(action='ignore', category=FutureWarning)

def read(dataset_dir: str, tag: str, context_name: str) -> dd.DataFrame:
    """Creates a Dask DataFrame for the component specified by its tag."""
    paths = tf.io.gfile.glob(f'{dataset_dir}/{tag}/{context_name}.parquet')
    return dd.read_parquet(paths)

def make_video(
    dataset_root: str, context_name: str, vid_base: str, 
    save_perframe=True, downscale: int = 1):
    if save_perframe:
        os.makedirs(vid_base, exist_ok=True)

    # Lazily read DataFrames for all components.
    cam_img_df = read(dataset_root, 'camera_image', context_name)
    association_df = read(dataset_root, 'camera_to_lidar_box_association', context_name)
    cam_box_df = read(dataset_root, 'camera_box', context_name)
    lidar_box_df = read(dataset_root, 'lidar_box', context_name)

    # Join all DataFrames using matching columns
    
    # common: [segment, timestamp, camera_name, camera_object_id]
    # NOTE: left_nullable=True to allow for empty frame with no association
    cam_box_w_asso = v2.merge(association_df, cam_box_df, left_nullable=True)
    
    # common: [segment, timestamp, camera_name]
    # NOTE: 1. left_nullable=True to allow for empty association
    #       2. left_group=True to group & agg cam_box_w_asso by common keys in advance
    cam_image_w_box_w_asso = v2.merge(cam_box_w_asso, cam_img_df, left_group=True, left_nullable=True)

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # NOTE: Dask.Dataframe is too slow; directly using in-memory pandas.DataFrame
    #       This will store seq data directly into CPU mem; usually several GiB per seq.
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    cam_image_w_box_w_asso = cam_image_w_box_w_asso.compute()

    im_per_frame = []
    
    # Unique timestamps (already sorted)
    context_segment_name = cam_image_w_box_w_asso.head(1)['key.segment_context_name'].iloc[0]
    unique_ts = list(cam_image_w_box_w_asso['key.frame_timestamp_micros'].unique())
    grouped_df = cam_image_w_box_w_asso.groupby(['key.segment_context_name', 'key.frame_timestamp_micros', 'key.camera_name'])
    for find, ts in enumerate(tqdm(unique_ts)):

        im_per_cam = []
        
        for camera_name in [v2._camera_image.CameraName.SIDE_LEFT, 
                            v2._camera_image.CameraName.FRONT_LEFT, 
                            v2._camera_image.CameraName.FRONT, 
                            v2._camera_image.CameraName.FRONT_RIGHT, 
                            v2._camera_image.CameraName.SIDE_RIGHT]:
            camera_name = camera_name.value

            row = grouped_df.get_group((context_segment_name, ts, camera_name)).reset_index()
            row = row.squeeze()
            
            camera_image = v2.CameraImageComponent.from_dict(row)
            camera_box = v2.CameraBoxComponent.from_dict(row) # Multiple box list (could be empty)
            camera_to_lidar_asso = v2.CameraToLiDARBoxAssociationComponent.from_dict(row)
            
            # ---- Start processing
            im = tf.io.decode_jpeg(camera_image.image).numpy() # uint8, 0~255, [H, W, 3]
            im = image_downscale(im, downscale=downscale)
            im = (im * 255).clip(0, 255).astype(np.uint8)
            
            if not (isinstance(camera_box.key.camera_object_id, Number) and math.isnan(camera_box.key.camera_object_id)):
                for cid, lid, center_x, center_y, width, height in zip(
                        camera_box.key.camera_object_id, 
                        camera_to_lidar_asso.key.laser_object_id, 
                        camera_box.box.center.x, camera_box.box.center.y, 
                        camera_box.box.size.x, camera_box.box.size.y, 
                    ):

                    cam_box_label = f"cid: {cid[:6]}"
                    lidar_box_label = None
                    if isinstance(lid, str): # has association; not math.isnan(lid) and 
                        lidar_box_label = f"lid: {lid[:6]}"
                    
                    im = draw_2dbox_on_im(
                        im, 
                        center_x / downscale, center_y / downscale, 
                        width=width / downscale, height=height / downscale, 
                        label=cam_box_label, label2=lidar_box_label, fontscale=1./downscale, 
                        color=(0, 128, 64), fillalpha=0.2, 
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
    make_video(
        "/data1/waymo/v2", "7670103006580549715_360_000_380_000", 
        "./dev_test/767010_cam_box", save_perframe=True, downscale=2)