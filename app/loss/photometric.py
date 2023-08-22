"""
@file   photometric.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Photometric loss
"""

import functools
from typing import Dict, Literal, Union

import torch
import torch.nn as nn

from nr3d_lib.config import ConfigDict
from nr3d_lib.models.loss.recon import *
from nr3d_lib.models.annealers import get_annealer

from app.resources import Scene

class PhotometricLoss(nn.Module):
    def __init__(
        self, w=1.0, anneal: ConfigDict=None, 
        respect_ignore_mask=False, 
        should_seperate_occ_areas=False, 
        non_occupied_rgb_gt: Literal['white', 'black']=None, 
        fn_type: Union[str, ConfigDict]='mse', fn_param: dict={}) -> None:
        """ Loss to utilize pixel supervision.

        Args:
            w (float, optional): Loss weight. Defaults to 1.0.
            anneal (ConfigDict, optional): Configuration for weight annealing. Defaults to None.
            respect_ignore_mask (bool, optional): 
                By default, all rendered pixels will be used. 
                If true, the 'rgb_ignore_mask' in the ground truth will be respected to ignore certain part(s) of pixels, 
                    e.g. non-occupied areas or areas belonging to dynamic objects.
                Defaults to False.
                NOTE: The 'rgb_ignore_mask' component in the ground truth is configured by `training.dataloader.tags.rgb_ignore_mask`.
            should_separate_occ_areas (bool, optional): 
                By default, all pixels will be compared as a whole with the ground truth to calculate loss. 
                If true, the 'rgb_mask' in the ground truth will be used to separately calculate loss for occupied and non-occupied areas.
                Defaults to False.
            non_occupied_rgb_gt (Literal['white', 'black'], optional): 
                Optionally, you can manually specify the pixel values for the non-occupied areas in the ground truth. 
                If not specified, the non-occupied part of the ground truth image will be used as is.
                Defaults to None.
            fn_type (Union[str, ConfigDict], optional): The type of loss function. Defaults to 'mse'.
            fn_param (dict, optional): Additional parameters for the loss function. Defaults to {}.
        """
        
        super().__init__()

        self.respect_ignore_mask = respect_ignore_mask
        
        self.non_occupied_rgb_gt = non_occupied_rgb_gt
        self.should_seperate_occ_areas = should_seperate_occ_areas
        if non_occupied_rgb_gt is None:
            self.non_occupied_rgb_gt_value = None
        elif non_occupied_rgb_gt == 'white':
            self.non_occupied_rgb_gt_value = 1.
        elif non_occupied_rgb_gt == 'black':
            self.non_occupied_rgb_gt_value = 0.
        else:
            raise RuntimeError(f"Invalid non_occupied_rgb_gt={non_occupied_rgb_gt}")

        self.w = w
        self.w_fn = None if anneal is None else get_annealer(**anneal)

        if fn_type == 'mse' or fn_type == 'l2':
            self.fn = functools.partial(mse_loss, **fn_param)
        elif fn_type == 'l1':
            self.fn = functools.partial(l1_loss, **fn_param)
        elif fn_type == 'smooth_l1':
            self.fn = functools.partial(smooth_l1_loss, **fn_param)
        elif fn_type == 'relative_l1':
            self.fn = functools.partial(relative_l1_loss, **fn_param)
        elif fn_type == 'mape':
            self.fn = functools.partial(mape_loss, **fn_param)
        elif fn_type == 'smape':
            self.fn = functools.partial(smape_loss, **fn_param)
        elif fn_type == 'relative_l2':
            self.fn = functools.partial(relative_l2_loss, **fn_param)
        elif fn_type == 'relative_l2_luminance':
            self.fn = functools.partial(relative_l2_luminance_loss, **fn_param)
        elif fn_type == 'huber':
            self.fn = functools.partial(huber_loss, **fn_param)
        else:
            raise RuntimeError(f'Invalid fn_type={fn_type}')

    def forward(self, scene: Scene, ret: dict, sample: dict, ground_truth: dict, it: int) -> Dict[str, torch.Tensor]:
        device = scene.device
        losses = dict()
        rgb_pred = ret['rendered']['rgb_volume']
        rgb_gt = ground_truth['rgb'].clone().to(device).view(rgb_pred.shape)
        w = self.w if self.w_fn is None else self.w_fn(it=it)
        
        rgb_remain_mask = None
        if self.respect_ignore_mask:
            assert 'rgb_ignore_mask' in ground_truth
            rgb_remain_mask = ~ground_truth['rgb_ignore_mask'].view(*rgb_pred.shape[:-1])
        
        if not self.should_seperate_occ_areas:
            """
            Use full-image prediction and full-image GT (considers ignore mask)
            """
            if self.non_occupied_rgb_gt_value is not None:
                assert 'rgb_mask' in ground_truth
                gt_occ_mask = ground_truth['rgb_mask'].view(*rgb_pred.shape[:-1])
                rgb_gt[gt_occ_mask] = self.non_occupied_rgb_gt_value
            # losses['loss_rgb'] = w * self.fn(rgb_pred, rgb_gt, mask=rgb_remain_mask, reduction='mean_in_mask')
            losses['loss_rgb'] = w * self.fn(rgb_pred, rgb_gt, mask=rgb_remain_mask, reduction='mean')
            with torch.no_grad():
                err = self.fn(rgb_pred.data, rgb_gt.data, mask=rgb_remain_mask, reduction='none')
        else:
            """
            Seperately consider occupied area's pred/GT and non-occupied area's (e.g. sky's) pred/GT
            """
            assert 'rgb_mask' in ground_truth
            assert 'rgb_volume_non_occupied' in ret['rendered']
            gt_occ_mask = ground_truth['rgb_mask'].view(*rgb_pred.shape[:-1])
            gt_not_occ_mask = gt_occ_mask.logical_not()
            
            #---- For occupied part
            rgb_pred_occupied = ret['rendered']['rgb_volume_occupied']
            if self.non_occupied_rgb_gt_value is not None:
                rgb_gt_occupied = rgb_gt.clone()
                rgb_gt_occupied[gt_not_occ_mask] = self.non_occupied_rgb_gt_value
                mask_rgb_occupied = rgb_remain_mask
            else:
                rgb_gt_occupied = rgb_gt
                mask_rgb_occupied = gt_occ_mask if rgb_remain_mask is None else (gt_occ_mask * rgb_remain_mask)
            # losses['loss_rgb.occupied'] = w * self.fn(rgb_pred_occupied, rgb_gt_occupied, mask=mask_rgb_occupied, reduction='mean_in_mask')
            losses['loss_rgb.occupied'] = w * self.fn(rgb_pred_occupied, rgb_gt_occupied, mask=mask_rgb_occupied, reduction='mean')
            with torch.no_grad():
                err1 = self.fn(rgb_pred_occupied.data, rgb_gt_occupied.data, mask=mask_rgb_occupied, reduction='none')
            
            #---- For non occupied part
            rgb_pred_non_occupied = ret['rendered']['rgb_volume_non_occupied']
            mask_rgb_non_occupied = gt_not_occ_mask if rgb_remain_mask is None else (gt_not_occ_mask * rgb_remain_mask)
            # losses['loss_rgb.not_occupied'] = w * self.fn(rgb_pred_non_occupied, rgb_gt, mask=mask_rgb_non_occupied, reduction='mean_in_mask')
            losses['loss_rgb.not_occupied'] = w * self.fn(rgb_pred_non_occupied, rgb_gt, mask=mask_rgb_non_occupied, reduction='mean')
            with torch.no_grad():
                err2 = self.fn(rgb_pred_non_occupied.data, rgb_gt.data, mask=mask_rgb_non_occupied, reduction='none')
            
            err = err1 + err2
            
        return losses, err.data.mean(dim=-1)