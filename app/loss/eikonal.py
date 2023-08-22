"""
@file   eikonal.py
@author Jianfei Guo, Shanghai AI Lab
@brief  SDF eikonal regularization loss
"""

import numbers
from typing import Dict, List, Literal, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from nr3d_lib.config import ConfigDict
from nr3d_lib.logger import Logger
from nr3d_lib.models.loss.safe import safe_mse_loss
from nr3d_lib.models.annealers import get_anneal_val

from app.resources import Scene, SceneNode
from nr3d_lib.render.pack_ops.pack_ops import packed_sum
from nr3d_lib.utils import tensor_statistics

class EikonalLoss(nn.Module):
    def __init__(
        self, 
        class_name_cfgs: Union[ConfigDict, float], 
        drawable_class_names: List[str], 
        #
        on_uniform_samples = True, 
        on_occ_ratio: float = 1.0, 
        on_render_type: Literal['equal', 'weighted', 'both']='both', 
        on_render_ratio: float = 0.1, 
        # 
        with_noise: float = 1.0e-3, 
        alpha_reg_zero: float = 0, 
        safe_mse=True, safe_mse_err_limit: float = 1.0, 
        log_every: int = -1) -> None:
        """ Apply eikonal loss to each model if needed.
        It is possible to apply eikonal loss on three types of points. You can choose one or multiple of them.
            1. `on_uniform_samples`: The uniformly sampled points in the whole valid space.
                Usually generated using model.uniform_sample(...)
            2. `on_occ_ratio`: The uniformly sampled points within occupancy grids. 
                Assumed to be near-surface areas. 
                In practice, this siginificanly ensures a whole valid SDF representation.
            3. `on_render_ratio`: The points from rendering buffer.

        Args:
            class_name_cfgs (Union[ConfigDict, float]): Eikonal loss configuration for each corresponding model class_name.
                Each configuration has the following format (for example):
                "Street": 
                {
                    'w': 1.0, # Loss weight, 
                    'anenal': ..., # Optional weight annealing configuration
                    'on_render_ratio_xxx': ..., # Optional special on_render_ratio for xxx train_step, 
                        where xxx can be one of ['pixel', 'image_step', 'lidar']
                }
            drawable_class_names (List[str]): List of all possible class_names.
            on_uniform_samples (bool, optional): If true, apply on uniformly sampled points in the whole valid space. Defaults to True.
            on_occ_ratio (float, optional): If true, apply on uniformly sampled points within occupancy grids (assumed to be near surface areas). Defaults to 1.0.
            on_render_type (Literal['equal', 'weighted', 'both'], optional): How to apply eikonal loss on points from rendering buffer. 
                'weighted': For each hit ray's rendering buffer, sum the eikonal loss of each point weighted by visibility weights, and then average all rays.
                'equal': Average the eikonal loss across all points.
                'both': Sum of 'weighted' and 'equal' methods.
                Defaults to 'both'.
            on_render_ratio (float, optional): Weight of the eikonal loss on the render buffer. Defaults to 0.1.
            with_noise (float, optional): The size of random noise added before calculating nablas_norm from nablas (mainly to prevent falling into zero vector nablas with no gradient). Defaults to 1.0e-3.
            alpha_reg_zero (float, optional): Additional weight of the regularization that penalizes zero vector nablas. Defaults to 0.
            safe_mse (bool, optional): Whether to use the safer mse loss with clipped gradients (mainly to prevent the training from collapsing due to extremely large nablas vectors). Defaults to True.
            safe_mse_err_limit (float, optional): If using safe mse loss, control how much first-order error is truncated to ensure safety. Defaults to 1.0.
            log_every (int, optional): Optionally, control the output frequency of debug log information (in terms of iterations). Defaults to -1.
        """
        super().__init__()
        
        if isinstance(class_name_cfgs, numbers.Number):
            class_name_cfgs = {class_name: {'w': class_name_cfgs} for class_name in drawable_class_names}
        else:
            for k, v in class_name_cfgs.items():
                if isinstance(v, numbers.Number):
                    class_name_cfgs[k] = {'w' : v}
        self.class_name_cfgs: Dict[str, ConfigDict] = class_name_cfgs
        
        self.on_uniform_samples = on_uniform_samples
        self.on_occ_ratio = on_occ_ratio
        self.on_render_type = on_render_type
        self.on_render_ratio = on_render_ratio
        
        self.with_noise = with_noise
        self.safe_mse = safe_mse
        self.safe_mse_err_limit = safe_mse_err_limit
        self.alpha_reg_zero = alpha_reg_zero
        
        self.log_every = log_every
        # self.fn = lambda pred: ((pred**2).sum(dim=-1) - pred.new_ones(pred.shape[:-1])).abs()

    def fn(self, nablas: torch.Tensor):
        nablas_norm = (nablas + self.with_noise * torch.randn_like(nablas)).norm(dim=-1, p=2)
        gt = nablas_norm.new_ones(nablas_norm.shape)
        if self.safe_mse:
            loss = safe_mse_loss(nablas_norm, gt, reduction='none', limit=(-1.1, self.safe_mse_err_limit))
        else:
            loss = F.mse_loss(nablas_norm, gt, reduction='none')
        if self.alpha_reg_zero > 0:
            loss += self.alpha_reg_zero / (0.01 + nablas_norm)
        return loss

    def forward_code_single(
        self, 
        obj: SceneNode, ret: dict, uniform_samples: dict, sample: dict, ground_truth: dict, it: int, 
        mode: Literal['pixel', 'lidar', 'image_patch'] = ..., logger: Logger=None) -> Dict[str, torch.Tensor]:
        
        device = obj.device
        class_name = ret['class_name']
        config = self.class_name_cfgs[class_name].copy()
        
        config = config.copy()
        w = config.pop('w', None)
        if (anneal_cfg:=config.get('anneal', None)) is not None:
            w = get_anneal_val(it=it, **anneal_cfg)
        assert w is not None, f"Can not get w for {self.__class__.__name__}.{class_name}"
        
        #----------------------------------------
        #---- Apply Eikonal loss on uniform samples in the whole valid space
        #----------------------------------------
        if self.on_uniform_samples:
            #---- Collect nablas from uniform_samples
            assert 'nablas' in uniform_samples, f"uniform_samples should contains nablas"
            nablas = uniform_samples['nablas']
            if (self.log_every > 0) and (it % self.log_every == 0) and (logger is not None):
                logger.add_nested_dict(f'train_step_{mode}.debug', 'uniform.nablas_norm', tensor_statistics(nablas.norm(dim=-1)), it=it)
            
            loss_on_uniform = self.fn(uniform_samples['nablas'].flatten(0, -2)).mean()
        else:
            loss_on_uniform = torch.tensor([0.], device=device)
        
        #----------------------------------------
        #---- Optionally, apply Eikonal loss on samples in the occupancy grids (near surface samples)
        #----------------------------------------
        alpha_occ = self.on_occ_ratio
        if (alpha_occ > 0) and (obj.model.accel is not None):
            #---- Collect nablas from occupancy grids
            occ_samples = obj.model.uniform_sample_on_occ(nablas[...,0].numel())
            nablas = occ_samples['nablas']
            if (self.log_every > 0) and (it % self.log_every == 0) and (logger is not None):
                logger.add_nested_dict(f'train_step_{mode}.debug', 'occ.nablas_norm', tensor_statistics(nablas.norm(dim=-1)), it=it)
            
            loss_on_occ = self.fn(nablas).mean()
        else:
            loss_on_occ = torch.tensor([0.], device=device)
        
        #----------------------------------------
        #---- Optionally, apply Eikonal loss on samples of `volume_buffer` when rendering
        #----------------------------------------
        alpha_render = self.on_render_ratio
        alpha_render = config.get(f'on_render_ratio_{mode}', alpha_render)
        if (self.on_render_type is not None) and (alpha_render > 0) and ret['volume_buffer']['buffer_type'] != 'empty':
            #---- Collect nablas from volume_buffer
            volume_buffer = ret['volume_buffer']
            nablas = volume_buffer['nablas'].flatten(0, -2)
            
            if (self.log_every > 0) and (it % self.log_every == 0) and (logger is not None):
                logger.add_nested_dict(f'train_step_{mode}.debug', 'render.nablas_norm', tensor_statistics(nablas.norm(dim=-1)), it=it)
            
            if self.on_render_type == 'weighted':
                loss_on_render = packed_sum(self.fn(nablas) * volume_buffer['vw_in_total'].data, volume_buffer['pack_infos_collect']).mean()
            elif self.on_render_type == 'equal':
                loss_on_render = self.fn(nablas).mean()
            elif self.on_render_type == 'both':
                loss_on_render = self.fn(nablas).mean() + \
                    packed_sum(self.fn(nablas) * volume_buffer['vw_in_total'].data, volume_buffer['pack_infos_collect']).mean()
            else:
                raise RuntimeError(f"Invalid on_render_type={self.on_render_type}")
        else:
            loss_on_render = torch.tensor([0.], device=device)
        
        return {
            'loss_eikonal.uniform': w * loss_on_uniform, 
            'loss_eikonal.occ': w * alpha_occ * loss_on_occ, 
            'loss_eikonal.render': w * alpha_render * loss_on_render, 
        }
    
    def forward_code_multi(
        self, 
        scene: Scene, ret: dict, uniform_samples: dict, sample: dict, ground_truth: dict, it: int, 
        mode: Literal['pixel', 'lidar', 'image_patch'] = ...) -> Dict[str, torch.Tensor]:
        device = scene.device
        
        cls_n_pts = {}
        cls_losses = {}
        ret_losses = {}
        
        for obj_id, obj_raw_ret in ret['raw_per_obj'].items():
            class_name = obj_raw_ret['class_name']
            if class_name in self.class_name_cfgs.keys():
                if (alpha:=self.class_name_cfgs[class_name].get('alpha_loss_on_render', 0)) > 0:
                    volume_buffer = obj_raw_ret['volume_buffer']
                    if 'nablas_norm' not in volume_buffer:
                        volume_buffer['nablas_norm'] = nablas_norm = volume_buffer['nablas'].norm(dim=-1)
                    else:
                        nablas_norm = volume_buffer['nablas_norm']
                    loss_on_render = alpha * self.fn(nablas_norm)
                    cls_n_pts.setdefault(class_name, []).append(nablas_norm.numel())
                    cls_losses.setdefault(class_name, []).append(loss_on_render)
                else:
                    assert class_name in uniform_samples, f"uniform_samples should contains {class_name}"
        
        for class_name, config in self.class_name_cfgs.items():
            config = config.copy()
            w = config.pop('w', None)
            if (anneal_cfg:=config.get('anneal', None)) is not None:
                w = get_anneal_val(it=it, **anneal_cfg)
            assert w is not None, f"Can not get w for {self.__class__.__name__}.{class_name}"
            
            if class_name in uniform_samples:
                cls_samples = uniform_samples[class_name]
                if 'nablas_norm' not in cls_samples:
                    cls_samples['nablas_norm'] = nablas_norm = cls_samples['nablas'].norm(dim=-1)
                else:
                    nablas_norm = cls_samples['nablas_norm']
                loss_on_uniform = self.fn(nablas_norm)
                cls_n_pts.setdefault(class_name, []).append(nablas_norm.numel())
                cls_losses.setdefault(class_name, []).append(loss_on_uniform)

            n_pts = cls_n_pts.get(class_name, [])
            if len(n_pts) > 0:
                losses = cls_losses[class_name]
                if len(n_pts) > 1:
                    if config.get('is_weighted_by_counts', True):
                        n_pts = torch.tensor(n_pts, dtype=torch.float32, device=device)
                        weight = n_pts / n_pts.sum()
                        loss = (torch.stack(losses)*weight).sum()
                    else:
                        loss = torch.stack(losses).mean()
                else:
                    loss = losses[0]

                ret_losses[f"loss_eikonal.{class_name}"] = w * loss
        
        return ret_losses            
