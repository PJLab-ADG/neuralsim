"""
@file   color_lipshitz.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Color regularization loss
        Borrowed from 
            PermutoSDF: Fast Multi-View Reconstruction with Implicit Surfaces using Permutohedral Lattices, 
            Radu Alexandru Rosu and Sven Behnke
"""

import numbers
from copy import deepcopy
from typing import Dict, List, Union

import torch
import torch.nn as nn

from nr3d_lib.config import ConfigDict
from nr3d_lib.models.loss.recon import *
from nr3d_lib.models.annealers import get_anneal_val

from app.resources import Scene, SceneNode

class ColorLipshitzRegLoss(nn.Module):
    def __init__(
        self, 
        class_name_cfgs: Union[ConfigDict, float], 
        drawable_class_names: List[str], 
        enable_after: int = 0, 
        ) -> None:
        super().__init__()
        
        self.enable_after = enable_after
        if isinstance(class_name_cfgs, numbers.Number):
            class_name_cfgs = {class_name: {'w': class_name_cfgs} for class_name in drawable_class_names}
        else:
            for k, v in class_name_cfgs.items():
                if isinstance(v, numbers.Number):
                    class_name_cfgs[k] = {'w' : v}
        self.class_name_cfgs: Dict[str, ConfigDict] = class_name_cfgs
    
    def forward_code_single(self, obj: SceneNode, ret: dict, sample: dict, ground_truth: dict, it: int):
        if it < self.enable_after:
            return {}
        model = obj.model
        class_name = obj.class_name
        config = deepcopy(self.class_name_cfgs[class_name])
        
        assert hasattr(model, 'get_color_lipshitz_bound'), f"{obj.model.id} has no get_color_lipshitz_bound"
        loss = config.w * model.get_color_lipshitz_bound()
        return {'loss_color_reg': loss}
    
    def forward_code_multi(self, scene: Scene, ret: dict, sample: dict, ground_truth: dict, it: int) -> Dict[str, torch.Tensor]:
        if it < self.enable_after:
            return {}
        ret_losses = {}
        for _, obj_raw_ret in ret['raw_per_obj_model'].items():
            if obj_raw_ret['volume_buffer']['buffer_type'] == 'empty':
                continue # Skip not rendered models to prevent pytorch error (accessing freed tensors)
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
            
            assert hasattr(model, 'get_color_lipshitz_bound'), f"{model.id} has no get_color_lipshitz_bound"
            loss = config.w * model.get_color_lipshitz_bound()
            ret_losses[f"loss_color_reg.{class_name}"] = loss
        return ret_losses
        
