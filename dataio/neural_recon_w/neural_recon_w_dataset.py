"""
@file   neural_recon_w_dataset.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Dataset IO for Neural Reconstruction in the wild datasets.
"""

import os
import numpy as np
from typing import Any, Dict

from nr3d_lib.config import ConfigDict
from nr3d_lib.utils import load_mask, load_rgb, glob_imgs, get_image_size

from dataio.scene_dataset import SceneDataset

class NeuralReconWDataset(SceneDataset):
    def __init__(self, config: dict) -> None:
        self.config = config
        self.populate(**config)
    
    def populate(
        self, 
        root: str, 
        
        ):
        pass