"""
@file   mask.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Mask supervision on the opacity / occupancy of pixels / images
"""

from typing import Dict, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from nr3d_lib.config import ConfigDict
from nr3d_lib.models.annealers import get_annealer
from nr3d_lib.models.loss.safe import safe_binary_cross_entropy

from app.resources import Scene

class MaskOccupancyLoss(nn.Module):
    def __init__(
        self, w: float = 1.0, anneal: ConfigDict = None, 
        pred_clip: float = 1.0e-3, 
        safe_bce=True, bce_limit=0.1, 
        w_on_errmap: float = 0, 
        special_mask_mode: Literal['always_occupied', 'only_cull_non_occupied', 'only_preserve_occupied'] = None
        ) -> None:
        """ Image occupancy (opacity) loss

        Args:
            w (float, optional): Loss weight. Defaults to 1.0.
            anneal (ConfigDict, optional): Annealing config of loss weight. Defaults to None.
            pred_clip (float, optional): Epsilon clip value to the predicted mask (for safer BCE). Defaults to 1.0e-3.
            safe_bce (bool, optional): Whether to use safe_bce implementation. Defaults to True.
            bce_limit (float, optional): safe_bce limit config. Defaults to 0.1.
            w_on_errmap (float, optional): The weight of mask error accumated to error_map for pixel importance sampling. Defaults to 0.
            special_mask_mode (Literal['always_occupied', 'only_cull_non_occupied', 'only_preserve_occupied'], optional): 
                Optional special mask loss types. 
                'always_occupied': All pixels areas are occupied.
                'only_cull_non_occupied': Only penalize opacity on ground truth non-occupied pixel areas.
                'only_preserve_occupied': Only encourage opacity on ground truth occupied pixel areas.
                Defaults to None: Using the ground truth mask with BCE loss.
        """
        
        super().__init__()
        
        self.w = w
        self.w_fn = None if anneal is None else get_annealer(**anneal)
        self.w_on_errmap = w_on_errmap
        
        self.special_mask_mode = special_mask_mode

        if not safe_bce:
            self.fn = lambda pred,gt,reduction='mean': F.binary_cross_entropy(pred.clip(pred_clip, 1.0-pred_clip), gt, reduction=reduction)
        else:
            if pred_clip is not None and pred_clip > 0:
                self.fn = lambda pred,gt,reduction='mean': safe_binary_cross_entropy(pred.clip(pred_clip, 1.0-pred_clip), gt, limit=bce_limit, reduction=reduction)
            else:
                self.fn = lambda pred,gt,reduction='mean': safe_binary_cross_entropy(pred, gt, limit=bce_limit, reduction=reduction)

    def forward(self, scene: Scene, ret: dict, sample: dict, ground_truth: dict, it: int) -> Dict[str, torch.Tensor]:
        device = scene.device
        ret_losses = {}
        w = self.w if self.w_fn is None else self.w_fn(it=it)
        
        mask_pred = ret['rendered']['mask_volume']
        if self.special_mask_mode is None:
            mask_gt = ground_truth['rgb_mask'].to(dtype=torch.float, device=device).view(*mask_pred.shape) # 1 for occupied, 0 for non-occupied
            loss = w * self.fn(mask_pred, mask_gt)
            with torch.no_grad():
                # NOTE: Error map from different losses should be in the same scale. 
                #       For now, we use [0,1] range for phtometric's error_map and mask's error_map
                err = (self.w_on_errmap * F.l1_loss(mask_pred, mask_gt, reduction="none")) if self.w_on_errmap > 0 else 0
        elif self.special_mask_mode == 'always_occupied':
            mask_gt = mask_pred.new_ones(mask_pred.shape)
            loss = w * self.fn(mask_pred, mask_gt)
            with torch.no_grad():
                err = (self.w_on_errmap * F.l1_loss(mask_pred, mask_gt, reduction="none")) if self.w_on_errmap > 0 else 0
        elif self.special_mask_mode == 'only_cull_non_occupied':
            non_occ_gt = ground_truth['rgb_mask'].to(device=device).view(*mask_pred.shape) < 0.5
            mask_pred_masked = mask_pred[non_occ_gt]
            mask_gt_masked = torch.zeros_like(mask_pred_masked)
            # NOTE: Use full image pixel count as denominator
            loss = w * self.fn(mask_pred_masked, mask_gt_masked, reduction='none').sum() / mask_pred.numel()
            with torch.no_grad():
                if self.w_on_errmap > 0:
                    err = torch.zeros_like(mask_pred)
                    err[non_occ_gt] = F.l1_loss(mask_pred_masked, mask_gt_masked, reduction='none')
                else:
                    err = 0
        elif self.special_mask_mode == 'only_preserve_occupied':
            occ_gt = ground_truth['rgb_mask'].to(device=device).view(*mask_pred.shape) > 0.5
            mask_pred_masked = mask_pred[occ_gt]
            mask_gt_masked = torch.ones_like(mask_pred_masked)
            # NOTE: Use full image pixel count as denominator
            loss = w * self.fn(mask_pred_masked, mask_gt_masked, reduction='none').sum() / mask_pred.numel()
            with torch.no_grad():
                if self.w_on_errmap > 0:
                    err = torch.zeros_like(mask_pred)
                    err[occ_gt] = F.l1_loss(mask_pred_masked, mask_gt_masked, reduction='none')
                else:
                    err = 0
        else:
            raise RuntimeError(f"Invalid special_mask_mode={self.special_mask_mode}")
        
        ret_losses['loss_mask'] = loss
        
        # class_name = scene.main_class_name
        # obj = scene.all_nodes_by_class_name[class_name][0]
        # obj_rendererd = ret['rendered_per_obj'][obj.id]
        # ret_losses[f'loss_mask.{class_name}'] = w * self.fn(obj_rendererd['mask_volume'], mask_gt)
        
        return ret_losses, err