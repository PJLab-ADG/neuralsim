"""
@file   instance_dataset_cropped.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Dataset IO for NeRS's MVMC dataset; 
        Single instance loader (with cropped image and its corresponding intrinsics).
"""

import os
import json
import numpy as np
from PIL import Image
from typing import Any, Dict, Literal

from nr3d_lib.config import ConfigDict
from nr3d_lib.utils import crop_image, get_image_size

from dataio.dataset_io import DatasetIO

def rle_to_binary_mask(rle):
    """
    rle should be coco format: {"counts": [], "size": []}
    """
    if isinstance(rle, list):
        return np.stack([rle_to_binary_mask(r) for r in rle])
    counts = rle["counts"]
    if isinstance(counts, str):
        counts = list(map(int, counts.split(" ")))
    mask = np.zeros(np.prod(rle["size"]), dtype=bool)
    running_length = 0
    for start, length in zip(counts[::2], counts[1::2]):
        running_length += start
        mask[running_length : running_length + length] = 1
        running_length += length
    return mask.reshape(rle["size"], order="F")

class MVMCNeRSInstanceDatasetCropped(DatasetIO):
    def __init__(self, config: ConfigDict) -> None:
        self.config = config
        self.populate(**config)
    
    def populate(
        self, 
        root: str, 
        instance_id: str, 
        camera_type: Literal['camera_optimized', 'camera_pretrained'] = 'camera_optimized', 
        crop_image_size: int = 256, 
        pad_size: float = 0.1
        ):
        """
        - TODO: (to check) To use these intrinsics and cropped images, the initial aabb should be larger?
        """
        
        instance_id = str(instance_id)
        
        self.main_class_name = "Main"
        self.instance_id = instance_id
        self.instance_dir = os.path.join(root, instance_id)
        
        annotations_json = os.path.join(self.instance_dir, "annotations.json")
        with open(annotations_json) as f:
            annotations = json.load(f)

        self.crop_image_size = crop_image_size
        hws_all = []
        image_paths = []
        Rs = []
        Ts = []
        fovs = []
        masks = []
        square_bboxes = []
        for annotation in annotations["annotations"]:
            image_path = os.path.join(self.instance_dir, "images", annotation["filename"])
            image_paths.append(image_path)
            
            W, H = get_image_size(image_path)
            hws_all.append([H, W])

            Rs.append(annotation[camera_type]["R"])
            Ts.append(annotation[camera_type]["T"])
            fovs.append(annotation[camera_type]["fov"])
            
            # Make a square bbox.
            bbox = np.array(annotation["bbox"])
            center = ((bbox[:2] + bbox[2:]) / 2.0).astype(int)
            s = (max(bbox[2:] - bbox[:2]) / 2.0 * (1 + pad_size)).astype(int)
            square_bbox = np.concatenate([center - s, center + s])
            # Cropped box pixel location in the original image
            square_bboxes.append(square_bbox)
            
            mask = Image.fromarray(rle_to_binary_mask(annotation["mask"]))
            mask = crop_image(mask, square_bbox)
            mask = np.array(mask.resize((crop_image_size, crop_image_size), Image.BILINEAR)) > 0.5
            masks.append(mask)

        Rs, Ts, fovs = np.array(Rs), np.array(Ts), np.array(fovs)
        hws_all = np.array(hws_all)
        self.image_paths = image_paths
        self.n_images = len(self.image_paths)
        masks = np.array(masks)
        self.masks = masks
        square_bboxes = np.array(square_bboxes)
        self.square_bboxes = square_bboxes

        #------ Intrinsics
        fovs = np.deg2rad(fovs)
        focal_length = np.abs(1 / np.tan(fovs / 2))
        focal_length_px = focal_length * crop_image_size/2.
        
        # Original image center
        image_og_center = hws_all[..., [1,0]] / 2 # [x, y]
        # Original image principle point in the cropped images.
        principal_point_px = image_og_center - square_bboxes[..., :2]
        
        intrs_all = np.zeros([self.n_images, 3, 3])
        intrs_all[..., 0, 0] = focal_length_px
        intrs_all[..., 1, 1] = focal_length_px
        intrs_all[..., 0:2, 2] = principal_point_px
        intrs_all[..., 2, 2] = 1.
        self.intrs_all = intrs_all
        self.hws_all = np.full([self.n_images, 2], crop_image_size)
        
        #------ Optimized camera poses from NeRS
        # From pytorch3d's w2c to OpenCV's c2w
        c2ws = np.eye(4)[None, ...].repeat(self.n_images, 0)
        c2ws[..., :3, :3] = Rs
        c2ws[..., :3, 3] = -np.einsum('...ij,...j->...i', Rs, Ts)
        c2ws = c2ws @ np.diagflat([-1,-1,1,1])[None,...]
        self.c2ws_all = c2ws

        intrs_all = np.zeros([self.n_images, 3, 3])
        intrs_all[..., 0, 0] = focal_length_px
        intrs_all[..., 1, 1] = focal_length_px
        intrs_all[..., :2, 2] = principal_point_px
        intrs_all[..., 2, 2] = 1.
        self.intrs_all = intrs_all
        
        #------ Optimized camera poses from NeRS
        # From pytorch3d's w2c to OpenCV's c2w
        c2ws = np.eye(4)[None, ...].repeat(self.n_images, 0)
        c2ws[..., :3, :3] = Rs
        c2ws[..., :3, 3] = -np.einsum('...ij,...j->...i', Rs, Ts)
        c2ws = c2ws @ np.diagflat([-1,-1,1,1])[None,...]
        self.c2ws_all = c2ws
    
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
            scene_id=f"MVMC-{self.instance_id}", 
            metas=metas, 
            objects={obj['id']: obj}, 
            observers={cam['id']: cam}
        )
        return scenario
    
    def get_image(self, scene_id: str, camera_id: str, frame_index: int) -> np.ndarray:
        image = Image.open(self.image_paths[frame_index]).convert("RGB")
        image = crop_image(image, self.square_bboxes[frame_index])
        image = np.array(image.resize((self.crop_image_size, self.crop_image_size), Image.LANCZOS)) / 255.0
        return image
    
    def get_occupancy_mask(self, scene_id: str, camera_id: str, frame_index: int) -> np.ndarray:
        return self.masks[frame_index]