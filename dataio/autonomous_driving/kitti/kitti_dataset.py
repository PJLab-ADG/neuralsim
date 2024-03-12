"""
@file   kitti_dataset.py
@author Jianfei Guo, Shanghai AI Lab.
@brief  Dataset IO for KITTI datasets

To be merged.
"""
import numpy as np
from typing import Dict, Any

from nr3d_lib.config import ConfigDict

from dataio.scene_dataset import SceneDataset

class KITTIDataset(SceneDataset):
    def __init__(self, config: dict) -> None:
        config = config.copy()

    def get_scenario(self, scene_id: str, **kwargs) -> Dict[str, Any]:
        pass

    def get_image(self, scene_id: str, camera_id: str, frame_index: int) -> np.ndarray:
        pass
    
    def get_image_occupancy_mask(self, scene_id: str, camera_id: str, frame_index: int) -> np.ndarray:
        pass
    
    def get_image_semantic_mask_all(self, scene_id: str, camera_id: str, frame_index: int, *, compress=True) -> np.ndarray:
        # Integer semantic mask on RGB image.
        raise NotImplementedError
    
    def get_lidar(self, scene_id: str, lidar_id: str, frame_index: int) -> Dict[str, np.ndarray]:
        pass