"""
@file   nerf_dataset.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Dataset IO for NeRF-standard datasets
"""

import os
import json
import numpy as np
from typing import Any, Dict, Literal

from nr3d_lib.config import ConfigDict
from nr3d_lib.utils import get_image_size, load_rgb

from dataio.scene_dataset import SceneDataset

class NeRFDataset(SceneDataset):
    def __init__(self, config: dict) -> None:
        self.config = config
        self.populate(**config)
    
    def populate(
        self, 
        datadir: str, ):
        pass