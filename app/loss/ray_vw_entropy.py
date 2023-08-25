"""
@file   ray_vw_entropy.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Entropy regularization
"""

from typing import Dict, List

import torch
import torch.nn as nn

from nr3d_lib.config import ConfigDict
from nr3d_lib.render.pack_ops import packed_mean
from nr3d_lib.models.annealers import get_annealer

from app.resources import Scene

class RayVisWeightEntropyRegLoss(nn.Module):
    def __init__(
        self, 
        w: float, anneal: ConfigDict = None, mode: str='total', 
        drawable_class_names: List[str] = []) -> None:
        super().__init__()
        self.w = w
        self.w_fn = None if anneal is None else get_annealer(**anneal)
        self.mode = mode
    
    def fn(self, volume_buffer: dict):
        vw = volume_buffer['vw']
        entropy = -vw*torch.log(vw+1e-8)
        if (buffer_type:=volume_buffer['buffer_type']) == 'packed':
            loss = packed_mean(entropy, volume_buffer['pack_infos_hit']).mean()
        elif buffer_type == 'batched':
            loss = entropy.mean()
        elif buffer_type == 'emtpy':
            loss = 0
        return loss
    
    def fn_in_total(self, volume_buffer: dict):
        if volume_buffer['buffer_type'] == 'empty':
            return 0
        else:
            vw = volume_buffer['vw_in_total']
            entropy = -vw*torch.log(vw+1e-8)
            return packed_mean(entropy, volume_buffer['pack_infos_collect']).mean()
    
    def forward_code_single(self, scene: Scene, ret: dict, sample: dict, ground_truth: dict, it: int) -> Dict[str, torch.Tensor]:
        w = self.w if self.w_fn is None else self.w_fn(it=it)
        
        ret_losses = dict()
        raw_per_obj_model = ret['raw_per_obj_model']
        if 'total' in self.mode:
            ret_losses['loss_entropy'] = w * self.fn(ret['volume_buffer'])
        if 'cr' in self.mode:
            main_class_name = scene.main_class_name
            cr_obj_id = scene.drawable_groups_by_class_name[main_class_name][0].id
            ret_losses[f'loss_entropy.{main_class_name}'] = w * self.fn_in_total(raw_per_obj_model[cr_obj_id]['volume_buffer'])
        if 'dv' in self.mode:
            dv_class_name = 'Distant'
            dv_obj_id = scene.drawable_groups_by_class_name[dv_class_name][0].id
            ret_losses[f'loss_entropy.{dv_class_name}'] = w * self.fn_in_total(raw_per_obj_model[dv_obj_id]['volume_buffer'])
        return ret_losses