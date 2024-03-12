"""
@file   mahattan.py
@brief  Loss with extracted mono cues (mono depth prior and mono normals prior)
        Modified from https://github.com/autonomousvision/monosdf
"""

import functools
from typing import List, Literal, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from nr3d_lib.config import ConfigDict
from nr3d_lib.models.loss.recon import *

from app.resources import Scene
from app.resources.observers import Camera

class RoadNormalLoss(nn.Module):
    """
    Mahattan assumption for street views: constraints to road normals
    """
    def __init__(
        self, 
        w_l1: float = 0.03, w_cos: float = 0.03, 
        distant_mode: Literal['crdv', 'cr_only'] = 'crdv', 
        mask_pred_thresh: float = 0.95, 
        enable_after: int = 0, 
        apply_in_pixel_train_step=False, 
        detach_mean = True, 
        use_l1 = True, 
        ):
        super().__init__()
        self.w_l1 = w_l1
        self.w_cos = w_cos
        self.mask_pred_thresh = mask_pred_thresh
        self.distant_mode = distant_mode
        self.enable_after = enable_after
        self.detach_mean = detach_mean
        self.use_l1 = use_l1
        self.apply_in_pixel_train_step = apply_in_pixel_train_step

        if self.distant_mode == 'cr_only':
            self.requires_render_per_class = True
        else:
            self.requires_render_per_class = False

    def forward(self, scene: Scene, cam: Camera, ret: dict, sample: dict, ground_truth: dict, it: int):
        if it < self.enable_after:
            return {}
        
        device = scene.device
        
        if self.distant_mode == 'crdv':
            rendered = ret['rendered']
        elif self.distant_mode == 'cr_only':
            rendered = ret['rendered_per_obj_in_scene']['street']
        else:
            raise RuntimeError(f"Invalid distant_mode={self.distant_mode}")
        
        normal_pred = rendered['normals_volume']
        mask_pred = (rendered['mask_volume'].data > self.mask_pred_thresh) # detached
        
        #---- Road
        assert 'image_road_mask' in ground_truth
        road_mask = ground_truth['image_road_mask'].view(mask_pred.shape)
        
        mask = mask_pred & road_mask
        if self.detach_mean:
            road_normals_pred = F.normalize(normal_pred.data[mask], dim=-1) # Must be [N,3]
        else:
            road_normals_pred = F.normalize(normal_pred[mask], dim=-1) # Must be [N,3]
        
        if road_normals_pred.numel() > 0:
            # Encourage road to have same normals 
            #   (penalize difference with their mean)
            mean = F.normalize(road_normals_pred.mean(dim=0), dim=-1) # Detached ?
            if self.use_l1:
                loss_l1 = torch.abs(road_normals_pred - mean).sum(dim=-1).mean()
            else:
                loss_l1 = torch.tensor([0.], device=device)
            loss_cos = (1. - torch.sum(road_normals_pred * mean, dim = -1)).mean()
        else:
            loss_l1 = torch.tensor([0.], device=device)
            loss_cos = torch.tensor([0.], device=device)
        
        ret_losses = {
            'loss_road_normal.l1': self.w_l1 * loss_l1, 
            'loss_road_normal.cos': self.w_cos * loss_cos
        }
        return ret_losses

class MahattanLoss(nn.Module):
    """
    The original mahattan assumption on indoor datasets
    """
    pass