"""
@author Jianfei Guo, Shanghai AI Lab
@brief  This module implements a generic dataloader containing scene graph, images, LiDAR data, and annotations, \
    and defines different specialized datasets for pixel, image, and LiDAR sampling.

FEATURES:
- `SceneDataLoader`
    - Defines the basic and common IO logic for the dataset.
    - Loads and caches common APIs used in dataio/dataset_impl.py, including scenarios, images, LiDAR data, and annotations.
    - Performs image downsampling.

- Defines 4 different specialized dataloader based on `SceneDataLoader`:
    - `PixelDataset` / `JointFramePixelDataset`: Sampling returns individual rays (pixels); supports importance sampling using error_map;
        - `PixelDataset`: in one sampled batch, rays originate from the same frame of image.
        - `JointFramePixelDataset`: in one sampled batch, rays could originate from different frames of image, \
            since frame index is jointly sampled along with ray index in importance sampling
    - `ImageDataset`: Sampling returns the full (optionally downsampled) image;
    - `ImagePatchDataset`: Sampling extracts a patch from the image according to certain scaling and shifting rules;
    - `LidarDataset`: Sampling returns individual rays from the lidar.

NOTE:
(Coding conventions)
- Avoid the use of "scene.slice_at" or any other operations related to the scene graph or nodes, or use them strictly within a "no_grad()" context.
    - The reason for this is that the scene graph may involve propagation of pose gradients across nodes, 
        and we do not expect the dataloader to provide any gradients. 
        These gradients should only be present in the forward process of the trainer.
    - In particular, calculations of camera rays should not be performed here, as they may require pose gradients. 
        Instead, this code outputs `rays_xy` (the sampled pixel location in [0,1]) and `rays_sel` (an optional selector).
        for the "cam.get_selected_rays" function in the trainer's forward method.
    - LiDAR's merging and filter_in_cams require the scene graph, but should be performed within a "no_grad()" context.
"""

from .base_loader import *
from .pixel_loader import *
from .lidar_loader import *
from .image_loader import *