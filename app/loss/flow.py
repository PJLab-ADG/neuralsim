"""
@file   flow.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Scene flow related loss
"""

from copy import deepcopy
from numbers import Number
from operator import itemgetter
from typing import Dict, List, Literal, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from nr3d_lib.config import ConfigDict
from nr3d_lib.models.annealers import get_anneal_val

from app.resources import Scene, SceneNode

class FlowLoss(nn.Module):
    def __init__(
        self, 
        class_name_cfgs: Union[ConfigDict, float], 
        drawable_class_names: List[str], 
        on_uniform_samples = True, 
        on_render_ratio: float = 1.0, 
        ):
        super().__init__()
        self.class_name_cfgs: Dict[str, ConfigDict] = class_name_cfgs

        self.on_uniform_samples = on_uniform_samples
        self.on_render_ratio = on_render_ratio

    def fn_cycle(
        self, 
        flow_fwd: torch.Tensor, flow_fwd_pred_bwd: torch.Tensor, 
        flow_bwd: torch.Tensor, flow_bwd_pred_fwd: torch.Tensor, 
        ) -> torch.Tensor:
        loss = 0.5 * (
            (flow_fwd.detach() + flow_fwd_pred_bwd) ** 2 # Sum should be zero
            + (flow_bwd.detach() + flow_bwd_pred_fwd) ** 2 # Sum should be zero
        ).mean()
        return loss

    def fn_sparsity(
        self, 
        flow_fwd: torch.Tensor, flow_fwd_pred_bwd: torch.Tensor, 
        flow_bwd: torch.Tensor, flow_bwd_pred_fwd: torch.Tensor, 
        ) -> torch.Tensor:
        loss = 0.25 * (
            flow_fwd.norm(dim=-1) + flow_fwd_pred_bwd.norm(dim=-1)
            + flow_bwd.norm(dim=-1) + flow_bwd_pred_fwd.norm(dim=-1)
        ).mean()
        return loss

    def forward(
        self, 
        scene: Scene, ret: dict, uniform_samples: dict, sample: dict, ground_truth: dict, it: int
        ) -> Dict[str, torch.Tensor]:
        ret_losses = {}
        
        for _, obj_raw_ret in ret['raw_per_obj_model'].items():
            if obj_raw_ret['volume_buffer']['type'] == 'empty':
                continue # Skip not rendered models
            class_name = obj_raw_ret['class_name']
            model_id = obj_raw_ret['model_id']
            # model = scene.asset_bank[model_id]
            if class_name not in self.class_name_cfgs.keys():
                continue

            config = deepcopy(self.class_name_cfgs[class_name])
            
            w_cycle = 0
            if 'cycle' in config:
                cfg_cycle: dict = config.pop('cycle')
                w_cycle = cfg_cycle.pop('w', None)
                if (anneal_cfg:=cfg_cycle.get('anneal', None)) is not None:
                    w_cycle = get_anneal_val(it=it, **anneal_cfg)
                assert w_cycle is not None, f"Can not get cycle.w for {self.__class__.__name__}.{class_name}"
            
            w_sparsity = 0
            if 'sparsity' in config:
                cfg_sparsity: dict = config.pop('sparsity')
                w_sparsity = cfg_sparsity.pop('w', None)
                if (anneal_cfg:=cfg_sparsity.get('anneal', None)) is not None:
                    w_sparsity = get_anneal_val(it=it, **anneal_cfg)
                assert w_sparsity is not None, f"Can not get w_sparsity for {self.__class__.__name__}.{class_name}"
            
            alpha_uniform = 1.0
            if self.on_uniform_samples:
                assert class_name in uniform_samples.keys(), f"uniform_samples should contain {class_name}"
                cls_samples = uniform_samples[class_name]
                #---- Collect flow from uniform_samples
                for k in ['flow_fwd', 'flow_fwd_pred_bwd', 'flow_bwd', 'flow_bwd_pred_fwd']:
                    assert k in cls_samples, f"uniform_samples[{class_name}] should contains '{k}'"
                inputs = itemgetter('flow_fwd', 'flow_fwd_pred_bwd', 'flow_bwd', 'flow_bwd_pred_fwd')(cls_samples)
                
                if w_cycle > 0:
                    l = self.fn_cycle(*inputs)
                    ret_losses[f"loss_flow_cycle.{class_name}.uniform"] = w_cycle * alpha_uniform * l
                
                if w_sparsity > 0:
                    l = self.fn_sparsity(*inputs)
                    ret_losses[f"loss_flow_sparsity.{class_name}.uniform"] = w_sparsity * alpha_uniform * l

            alpha_render = self.on_render_ratio
            alpha_render = config.pop('on_render_ratio', alpha_render)
            if (alpha_render > 0) and (obj_raw_ret['volume_buffer']['type'] != 'empty'):
                volume_buffer = obj_raw_ret['volume_buffer']
                inputs = itemgetter('flow_fwd', 'flow_fwd_pred_bwd', 'flow_bwd', 'flow_bwd_pred_fwd')(volume_buffer)
                
                if w_cycle > 0:
                    l = self.fn_cycle(*inputs)
                    ret_losses[f"loss_flow_cycle.{class_name}.render"] = \
                        w_cycle * cfg_cycle.get('on_render_ratio', alpha_render) * l
                
                if w_sparsity > 0:
                    l = self.fn_sparsity(*inputs)
                    ret_losses[f"loss_flow_sparsity.{class_name}.render"] = \
                        w_sparsity * cfg_sparsity.get('on_render_ratio', alpha_render) * l
                
        return ret_losses
