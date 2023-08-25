"""
@file   sparsity.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Geometry sparsity regularization loss
"""

import numbers
from copy import deepcopy
from typing import Dict, List, Literal, Union

import torch
import torch.nn as nn

from nr3d_lib.config import ConfigDict
from nr3d_lib.models.annealers import get_anneal_val
from nr3d_lib.models.utils import normalized_logistic_density

from app.resources import Scene, SceneNode

class SparsityLoss(nn.Module):
    def __init__(
        self, 
        class_name_cfgs: Union[ConfigDict, float] = {}, 
        drawable_class_names: List[str] = [], 
        enable_after: int = 0,
        ) -> None:
        """ Sparsity regularization to encourage free spaces in unobserved regions.

        Args:
            class_name_cfgs (Union[ConfigDict, float], optional): Sparsity loss configuration for each corresponding model class_name. 
                Each configuration has the following format (for example):
                "Street": 
                {
                    'w': 1.0, # Loss weight, 
                    'anenal': ..., # Optional weight annealing configuration
                    'key': 'sdf, # The key to query the `uniform_samples` dict to get the sampled geometry values
                    'type': 'normalized_logistic_density', # The type of function to map the queried raw geometry values to loss values
                }
            drawable_class_names (List[str], optional): List of all possible class_names. Defaults to [].
            enable_after (int, optional): Enable this loss after this iteration. Defaults to 0.
        """
        
        super().__init__()
        self.class_name_cfgs: Dict[str, ConfigDict] = class_name_cfgs
        self.enable_after = enable_after
    
    def fn_normal(self, x: torch.Tensor, std: float = 1.0):
        return torch.exp(-x**std).mean()
    
    def fn_nld(self, x: torch.Tensor, inv_scale: float = 16.0):
        return normalized_logistic_density(x,inv_s=inv_scale).mean()
    
    def fn_density_reg(self, x: torch.Tensor, lamb: float = 0.05):
        return (1 - (-lamb * x).exp()).abs().mean()
    
    def forward_code_single(self, obj: SceneNode, ret: dict, uniform_samples: dict, sample: dict, ground_truth: dict, it: int) -> Dict[str, torch.Tensor]:
        ret_losses = {}
        if it < self.enable_after:
            return ret_losses
        
        class_name = ret['class_name']
        config = deepcopy(self.class_name_cfgs[class_name])
        w = config.pop('w', None)
        if (anneal_cfg:=config.pop('anneal', None)) is not None:
            w = get_anneal_val(it=it, **anneal_cfg)
        assert w is not None, f"Can not get w for {self.__class__.__name__}.{class_name}"
        
        fn_type = config.pop('type', 'normalized_logistic_density')
        val = uniform_samples[config.pop('key', 'sdf')]
        if fn_type == 'normal':
            loss = self.fn_normal(val, **config)
        elif fn_type == 'normalized_logistic_density':
            loss = self.fn_nld(val, **config)
        elif fn_type == 'density_reg':
            loss = self.fn_density_reg(val, **config)
        else:
            raise RuntimeError(f"Invalid type={fn_type}")
        ret_losses['loss_sparsity'] = w * loss
        return ret_losses
        
    def forward_code_multi(self, scene: Scene, ret: dict, uniform_samples: dict, sample: dict, ground_truth: dict, it: int) -> Dict[str, torch.Tensor]:
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
            assert class_name in uniform_samples.keys(), f"uniform_samples should contain {class_name}"
            w = config.pop('w', None)
            if (anneal_cfg:=config.pop('anneal', None)) is not None:
                w = get_anneal_val(it=it, **anneal_cfg)
            assert w is not None, f"Can not get w for {self.__class__.__name__}.{class_name}"
            
            val = uniform_samples[class_name][config.pop('key', 'sdf')]
            fn_type = config.pop('type', 'normalized_logistic_density')
            if fn_type == 'normal':
                loss = self.fn_normal(val, **config)
            elif fn_type == 'normalized_logistic_density':
                loss = self.fn_nld(val, **config)
            elif fn_type == 'density_reg':
                loss = self.fn_density_reg(val, **config)
            else:
                raise RuntimeError(f"Invalid type={fn_type}")
            ret_losses[f"loss_sparsity.{class_name}"] = w * loss
        return ret_losses            
