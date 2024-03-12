"""
@file   clearance.py
@author Jianfei Guo, Shanghai AI Lab
@brief  SDF near field clearance regularization loss, to prevent too small or even minus near field SDFs (minus near SDF = camera inside shape)
"""

from copy import deepcopy
from numbers import Number
from typing import Dict, List, Literal, Union

import torch
import torch.nn as nn

from nr3d_lib.config import ConfigDict
from nr3d_lib.models.annealers import get_anneal_val

from app.resources import Scene, SceneNode

class ClearanceLoss(nn.Module):
    def __init__(
        self, 
        class_name_cfgs: Union[ConfigDict, float], 
        drawable_class_names: List[str]) -> None:
        """
        Near-field SDF clearance regularization.
        This regularization penalizes small or negative near-SDF values to avoid the common local minimum problem of camera-inside-geometry. 
        It also aids in eliminating near-field in-the-air artifacts in street views.

        Args:
            class_name_cfgs (Union[ConfigDict, float]): Regularization configuration for each corresponding model class_name. 
                Each configuration has the following format (for example):
                "Street": 
                {
                    'w': 1.0, # Loss weight, 
                    'anenal': ..., # Optional weight annealing configuration
                    'thresh': 0.01, # Only penalizes near-SDF values below this threshold.
                    'beta': 10.0, # Growth factor for the penalty as the near-SDF value falls further below the threshold.
                }
            drawable_class_names (List[str]): List of all potential class_names. Defaults to [].
        """
        
        super().__init__()
        if isinstance(class_name_cfgs, Number):
            class_name_cfgs = {class_name: {'w': class_name_cfgs} for class_name in drawable_class_names}
        else:
            for k, v in class_name_cfgs.items():
                if isinstance(v, Number):
                    class_name_cfgs[k] = {'w' : v}
        self.class_name_cfgs: Dict[str, ConfigDict] = class_name_cfgs
    
    def fn_penalty_sdf(self, near_sdf: torch.Tensor, beta: float = 1.0, thresh: float = 0.01):
        mask_in = near_sdf < thresh
        num_pen_pts = mask_in.sum().item()
        # penalty = torch.sigmoid(-beta * near_sdf[mask_in]).mean() if num_pen_pts > 0 else near_sdf.new_zeros([1,])
        penalty = (torch.exp(-beta * (near_sdf[mask_in]-thresh)).sum() / mask_in.numel()) if num_pen_pts > 0 else near_sdf.new_zeros([1,])
        return num_pen_pts, penalty
    
    def fn_penalty_density(self, near_density: torch.Tensor, ):
        raise NotImplementedError
    
    def forward(
        self, 
        scene: Scene, ret: dict, uniform_samples: dict, sample: dict, ground_truth: dict, it: int, 
        mode: Literal['pixel', 'lidar', 'image_patch'] = ...) -> Dict[str, torch.Tensor]:
        
        ret_losses = {}
        for _, obj_raw_ret in ret['raw_per_obj_model'].items():
            if obj_raw_ret['volume_buffer']['type'] == 'empty':
                continue # Skip not rendered models
            class_name = obj_raw_ret['class_name']
            model_id = obj_raw_ret['model_id']
            model = scene.asset_bank[model_id]
            if class_name not in self.class_name_cfgs.keys():
                continue
            
            config = deepcopy(self.class_name_cfgs[class_name])
            w = config.pop('w', None)
            if (anneal_cfg:=config.pop('anneal', None)) is not None:
                w = get_anneal_val(it=it, **anneal_cfg)
            assert w is not None, f"Can not get w for {self.__class__.__name__}.{class_name}"
            
            if obj_raw_ret['volume_buffer']['type'] == 'empty':
                continue
            
            if 'near_sdf' in obj_raw_ret['details']:
                near_sdf = obj_raw_ret['details']['near_sdf']
                _, penalty = self.fn_penalty_sdf(near_sdf, **config)
            elif 'near_sigma' in obj_raw_ret['details']:
                near_sigma = obj_raw_ret['details']['near_sdf']
                _, penalty = self.fn_penalty_density(near_sigma, **config)
            else:
                raise RuntimeError(f"Can not find 'near_sdf' or 'near_density' in details for {class_name}")
            ret_losses[f'loss_clearance.{class_name}'] = w * penalty
        
        return ret_losses
