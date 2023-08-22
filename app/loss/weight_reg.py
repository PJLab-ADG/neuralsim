"""
@file   weight_reg.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Network weight regularization loss
"""

import numbers
from typing import Dict, List, Union

import torch
import torch.nn as nn

from nr3d_lib.config import ConfigDict
from nr3d_lib.models.loss.recon import *
from nr3d_lib.models.annealers import get_anneal_val

from app.resources import Scene, SceneNode

class WeightRegLoss(nn.Module):
    def __init__(
        self, 
        class_name_cfgs: Union[ConfigDict, float], 
        drawable_class_names: List[str], 
        ) -> None:
        """ Parameter regularization to prevent diversion
        NOTE: Will invoke each configured model's get_weight_reg() function to get a flattenned tensor of all params' norm.

        Args:
            class_name_cfgs (Union[ConfigDict, float], optional): Regularization configuration for each corresponding model class_name. 
                Defaults to {}.
            drawable_class_names (List[str], optional): List of all possible class_names. Defaults to [].
            enable_after (int, optional): Enable this loss after this iteration. Defaults to 0.
        """
        
        super().__init__()
        
        if isinstance(class_name_cfgs, numbers.Number):
            class_name_cfgs = {class_name: {'w': class_name_cfgs} for class_name in drawable_class_names}
        else:
            for k, v in class_name_cfgs.items():
                if isinstance(v, numbers.Number):
                    class_name_cfgs[k] = {'w' : v}
        self.class_name_cfgs: Dict[str, ConfigDict] = class_name_cfgs
    
    # def forward_code_single(self, obj: SceneNode, ret: dict, sample: dict, ground_truth: dict, it: int):
    #     model = obj.model
    #     class_name = obj.class_name
    #     config = self.class_name_cfgs[class_name].copy()
        
    #     w = config.pop('w', None)
    #     if (anneal_cfg:=config.get('anneal', None)) is not None:
    #         w = get_anneal_val(it=it, **anneal_cfg)
    #     assert w is not None, f"Can not get w for {self.__class__.__name__}.{class_name}"
    #     assert hasattr(model, 'get_weight_reg'), f"{obj.model.id} has no get_weight_reg"
    #     loss = w * model.get_weight_reg(**config).sum()
    #     return {'loss_weight_reg': loss}
    
    def forward(
        self, scene: Scene, ret: dict, sample: dict, ground_truth: dict, it: int
        ) -> Dict[str, torch.Tensor]:
        ret_losses = {}
        for class_name, config in self.class_name_cfgs.items():
            config = config.copy()
            w = config.pop('w', None)
            if (anneal_cfg:=config.pop('anneal', None)) is not None:
                w = get_anneal_val(it=it, **anneal_cfg)
            assert w is not None, f"Can not get w for {self.__class__.__name__}.{class_name}"

            obj = scene.get_drawable_groups_by_class_name(class_name)[0]
            model = obj.model
            assert hasattr(model, 'get_weight_reg'), f"{obj.model.id} has no get_weight_reg"
            
            loss = w * model.get_weight_reg(**config).sum()
            ret_losses[f"loss_weight_reg.{class_name}"] = loss
        return ret_losses