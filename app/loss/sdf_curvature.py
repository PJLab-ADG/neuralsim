"""
@file   sdf_curvature.py
@author Jianfei Guo, Shanghai AI Lab
@brief  SDF curvature smoothness loss. 
        Borrowed from 
            PermutoSDF: Fast Multi-View Reconstruction with Implicit Surfaces using Permutohedral Lattices, 
            Radu Alexandru Rosu and Sven Behnke
"""

from copy import deepcopy
from numbers import Number
from typing import Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from nr3d_lib.config import ConfigDict
from nr3d_lib.models.loss.safe import safe_mse_loss
from nr3d_lib.models.annealers import get_anneal_val

from app.resources import Scene, SceneNode

class SDFCurvatureRegLoss(nn.Module):
    def __init__(
        self, 
        class_name_cfgs: Union[ConfigDict, float], drawable_class_names: List[str], 
        on_uniform_samples=True, eps=1.0e-4, enable_after: int = 0) -> None:
        super().__init__()
        if isinstance(class_name_cfgs, Number):
            class_name_cfgs = {class_name: {'w': class_name_cfgs} for class_name in drawable_class_names}
        else:
            for k, v in class_name_cfgs.items():
                if isinstance(v, Number):
                    class_name_cfgs[k] = {'w' : v}
        self.class_name_cfgs: Dict[str, ConfigDict] = class_name_cfgs
        self.on_uniform_samples = on_uniform_samples
        self.eps = eps
        self.enable_after = enable_after
    
    def fn(self, curvature: torch.Tensor):
        return curvature.clamp_max_(0.5).abs().mean() 
    
    def forward(self, scene: Scene, ret: dict, uniform_samples: dict, sample: dict, ground_truth: dict, it: int) -> Dict[str, torch.Tensor]:
        if it < self.enable_after:
            return {}
        
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
            if (anneal_cfg:=config.get('anneal', None)) is not None:
                w = get_anneal_val(it=it, **anneal_cfg)
            assert w is not None, f"Can not get w for {self.__class__.__name__}.{class_name}"
            if w <= 0:
                continue

            alpha_uniform = 1.0
            if self.on_uniform_samples:
                assert class_name in uniform_samples.keys(), f"uniform_samples should contain {class_name}"
                loss_on_uniform = self.fn(model.get_sdf_curvature_1d(uniform_samples['net_x'], uniform_samples['nablas'], eps=self.eps))
                ret_losses[f"loss_sdf_curvature_reg.{class_name}.uniform"] = w * alpha_uniform * loss_on_uniform

            alpha_render = config.get('alpha_loss_on_render', 0)
            if (alpha_render > 0) and (obj_raw_ret['volume_buffer']['type'] != 'empty'):
                volume_buffer = obj_raw_ret['volume_buffer']
                loss_on_render = self.fn(model.get_sdf_curvature_1d(volume_buffer['net_x'], volume_buffer['nablas'], eps=self.eps))
                ret_losses[f"loss_sdf_curvature_reg.{class_name}.render"] = w * alpha_render * loss_on_render

        return ret_losses
