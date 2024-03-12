"""
@file   scene_dataset.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Dataset implementation abstract interfaces.
"""

import numpy as np
from typing import Any, Dict, List, Literal, Tuple, Union
from abc import ABC, abstractmethod

from nr3d_lib.config import ConfigDict

class SceneDataset(ABC):
    @abstractmethod # NOTE: This is the only method that must be implemented.
    def get_scenario(self, scene_id: str, **kwargs) -> Dict[str, Any]:
        # NOTE: Must be implemented to enable scene loading and the whole training framework.
        raise NotImplementedError
    
    @property
    def up_vec(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def forward_vec(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def right_vec(self) -> np.ndarray:
        raise NotImplementedError

    def get_all_available_scenarios(self) -> List[str]:
        # All available scene id list of current dataset
        raise NotImplementedError
    
    def get_metadata(self, scene_id: str) -> Dict[str, Any]:
        raise NotImplementedError
    
    def get_image_wh(self, scene_id: str, camera_id: str, frame_index: Union[int, List[int]]) -> np.ndarray:
        raise NotImplementedError
    
    def get_image(self, scene_id: str, camera_id: str, frame_index: int) -> np.ndarray:
        # [H, W, 3], np.float32, range [0,1]
        raise NotImplementedError

    def get_image_mono_depth(self, scene_id: str, camera_id: str, frame_index: int) -> np.ndarray:
        # [H, W], np.float32, range [0, inf]
        raise NotImplementedError

    def get_image_mono_normals(self, scene_id: str, camera_id: str, frame_index: int) -> np.ndarray:
        # [H, W, 3], np.float32, range [-1,1]
        raise NotImplementedError

    def get_image_occupancy_mask(self, scene_id: str, camera_id: str, frame_index: int) -> np.ndarray:
        # [H, W], bool, binary occupancy mask on RGB image. 1 for occpied, 0 for not.
        raise NotImplementedError

    def get_image_semantic_mask_by_type(
        self, scene_id: str, camera_id: str, 
        sem_type: Literal['dynamic', 'human', 'road', 'anno_dontcare'], 
        frame_index: int) -> np.ndarray:
        # Binary semantic mask on RGB image. 1 for matched, 0 for not.
        raise NotImplementedError

    def get_image_semantic_mask_all(self, scene_id: str, camera_id: str, frame_index: int) -> np.ndarray:
        # [H, W], int16, integer semantic mask on RGB image.
        raise NotImplementedError

    def get_lidar(self, scene_id: str, lidar_id: str, frame_index: int) -> Dict[str, np.ndarray]:
        # {
        #   'rays_o': [..., 3], np.float32, lidar beam's starting points
        #   'rays_d': [..., 3], np.float32, lidar beam's direction vectors
        #   'ranges': [...], np.float32, lidar beam's termination depth (usally the first return)
        # }
        raise NotImplementedError

    # def get_aabb(self, scene_id: str, lidar_id: str, frame_index: int) -> Dict[str, np.ndarray]:
    #     raise NotImplementedError

    # def create_dataset(self, source_data_cfg: dict, j=8):
    #     raise NotImplementedError