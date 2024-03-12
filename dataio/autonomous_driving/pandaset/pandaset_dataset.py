"""

"""
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

from dataio.scene_dataset import SceneDataset

def idx_to_frame_str(frame_index):
    return f'{frame_index:02d}'

class PandarsetDataset(SceneDataset):
    def __init__(self, config: dict) -> None:
        self.config = config
        self.populate(**config)
    
    def populate(
        self, root: str, 
        rgb_dirname: str = 'camera', 
        lidar_dirname: str = 'lidar', 
        mask_dirname: str = 'masks', 
        ):
        self.root = root
        self.rgb_dirname = rgb_dirname
        self.lidar_dirname = lidar_dirname

    def get_scenario(self, scene_id: str, **kwargs) -> Dict[str, Any]:
        sequence_dir = os.path.join(self.root, scene_id)
        
        

def group_vis_gps():
    pass

if __name__ == '__main__':
    pass