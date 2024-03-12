"""
- Annotations of camera box and id for each frame
- Existence of association
"""


import io
import os
import cv2
import pickle
import imageio # NOTE: Also needs pip install imageio-ffmpeg
import functools
import numpy as np
from tqdm import tqdm
from PIL import Image
from typing import List
from numbers import Number
import matplotlib.pyplot as plt

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import frame_utils, transform_utils, range_image_utils
from dataio.autonomous_driving.waymo.waymo_filereader import WaymoDataFileReader
from dataio.autonomous_driving.waymo.filter_dynamic import stat_dynamic_objects
from dataio.autonomous_driving.waymo.waymo_dataset import *

from nr3d_lib.utils import pad_images_to_same_size, image_downscale
from nr3d_lib.plot import draw_2dbox_on_im

def make_video(sequence_file, vid_base: str, save_perframe=True, downscale: int = 1):
    if save_perframe:
        os.makedirs(vid_base, exist_ok=True)

    dataset = WaymoDataFileReader(str(sequence_file))

    im_per_frame = []

    # frame = next(iter(dataset))
    for find, frame in enumerate(tqdm(dataset, "processing...")):
        im_per_cam = []
        
        for camera_name in ['SIDE_LEFT', 'FRONT_LEFT', 'FRONT', 'FRONT_RIGHT', 'SIDE_RIGHT']:
            _j = WAYMO_CAMERAS.index(camera_name)
            camera_image = frame.images[_j]
            for c in frame.context.camera_calibrations:
                if c.name == camera_image.name:
                    break
            for cl in frame.camera_labels:
                if cl.name == camera_image.name:
                    break
            
            im = Image.open(io.BytesIO(camera_image.image))
            im = np.asarray(im) # uint8, 0~255, [H, W, 3]
            im = image_downscale(im, downscale=downscale)
            im = (im * 255).clip(0, 255).astype(np.uint8)
            
            for l in cl.labels:
                box = l.box
                
                cam_box_label = f"cid: {l.id[:6]}"
                lidar_box_label = None
                if l.HasField('association'):
                    lidar_box_label = l.association.laser_object_id
                    lidar_box_label = f"lid: {lidar_box_label[:6]}"
                
                im = draw_2dbox_on_im(
                    im, 
                    box.center_x / downscale, box.center_y / downscale, 
                    width=box.length / downscale, height=box.width / downscale, 
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

if __name__ == "__main__":
    sequence_file = "/data2/neuralsim_open/dataset/waymo_dynamic_81/segment-7670103006580549715_360_000_380_000_with_camera_labels.tfrecord"
    make_video(sequence_file, "./dev_test/767010_cam_box", save_perframe=True, downscale=2)