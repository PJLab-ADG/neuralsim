"""
@file   sdf_curvature.py
@author Jianfei Guo, Shanghai AI Lab
@brief  SDF curvature smoothness loss. 
        Borrowed from 
            PermutoSDF: Fast Multi-View Reconstruction with Implicit Surfaces using Permutohedral Lattices, 
            Radu Alexandru Rosu and Sven Behnke
"""

import numbers
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
        if isinstance(class_name_cfgs, numbers.Number):
            class_name_cfgs = {class_name: {'w': class_name_cfgs} for class_name in drawable_class_names}
        else:
            for k, v in class_name_cfgs.items():
                if isinstance(v, numbers.Number):
                    class_name_cfgs[k] = {'w' : v}
        self.class_name_cfgs: Dict[str, ConfigDict] = class_name_cfgs
        self.on_uniform_samples = on_uniform_samples
        self.eps = eps
        self.enable_after = enable_after
    
    def fn(self, curvature: torch.Tensor):
        return curvature.clamp_max_(0.5).abs().mean() 
    
    def forward_code_single(self, obj: SceneNode, ret: dict, uniform_samples: dict, sample: dict, ground_truth: dict, it: int) -> Dict[str, torch.Tensor]:
        device = obj.device
        model = obj.model
        class_name = ret['class_name']
        config = self.class_name_cfgs[class_name].copy()
        
        w = config.pop('w', None)
        if (anneal_cfg:=config.get('anneal', None)) is not None:
            w = get_anneal_val(it=it, **anneal_cfg)
        assert w is not None, f"Can not get w for {self.__class__.__name__}.{class_name}"
        
        if w <= 0:
            return {'loss_sdf_curvature_reg': torch.tensor([0.], device=device)}
        
        if (alpha:=config.get('alpha_loss_on_render', 0)) > 0:
            volume_buffer = ret['volume_buffer']
            if volume_buffer['buffer_type'] != 'empty':
                loss_on_render = alpha * self.fn(model.get_sdf_curvature_1d(volume_buffer['net_x'], volume_buffer['nablas'], eps=self.eps))
            else:
                loss_on_render = torch.tensor([0.], device=device)
        else:
            loss_on_render = torch.tensor([0.], device=device)
        
        if self.on_uniform_samples:
            assert 'nablas' in uniform_samples, f"uniform_samples should contains nablas"
            loss_on_uniform = self.fn(model.get_sdf_curvature_1d(uniform_samples['net_x'], uniform_samples['nablas'], eps=self.eps))
        else:
            loss_on_uniform = torch.tensor([0.], device=device)
        
        loss = w * (loss_on_render + loss_on_uniform)
        
        return {'loss_sdf_curvature_reg': loss}
    
    def forward_code_multi(self, scene: Scene, ret: dict, uniform_samples: dict, sample: dict, ground_truth: dict, it: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
        if it < self.enable_after:
            return {}
        ret_losses = {}
        for class_name, config in self.class_name_cfgs.items():
            obj = scene.get_drawable_groups_by_class_name(class_name)[0]
            device = obj.device
            model = obj.model
            assert hasattr(model, 'get_sdf_curvature_1d'), f"{obj.model.id} has no get_sdf_curvature_1d"
            
            config = config.copy()
            w = config.pop('w', None)
            if (anneal_cfg:=config.get('anneal', None)) is not None:
                w = get_anneal_val(it=it, **anneal_cfg)
            assert w is not None, f"Can not get w for {self.__class__.__name__}.{class_name}"
            
            if (alpha:=config.get('alpha_loss_on_render', 0)) > 0:
                volume_buffer = ret['volume_buffer']
                if volume_buffer['buffer_type'] != 'empty':
                    loss_on_render = alpha * self.fn(model.get_sdf_curvature_1d(volume_buffer['net_x'], volume_buffer['nablas'], eps=self.eps))
                else:
                    loss_on_render = torch.tensor([0.], device=device)
            else:
                loss_on_render = torch.tensor([0.], device=device)
