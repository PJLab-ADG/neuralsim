
"""
@file   lidar.py
@author Jianfei Guo, Shanghai AI Lab & Chenjing Ding, Sensetime
@brief  LiDAR supervisions (as sparse depth sensors)
"""

from operator import itemgetter
from typing import Dict, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from nr3d_lib.logger import Logger
from nr3d_lib.models.annealers import get_annealer, get_anneal_val
from nr3d_lib.graphics.pack_ops import packed_sum, packed_geq, packed_leq, packed_lt
from nr3d_lib.models.loss.recon import huber_loss, l2_loss, relative_l2_loss, l1_loss

from app.resources import Scene

class DepthLoss(nn.Module):
    def __init__(
        self, 
        w: float = 1.0, anneal: dict = None, 
        fn_type: Literal['l2', 'l2_relative', 'l2_normalized', 'l1_log', 'l1', 'huber'] = 'l1_log', 
        fn_param: dict = {}, 
        ) -> None:
        super().__init__()
        
        self.w = w
        self.w_fn = None if anneal is None else get_annealer(**anneal)

        if fn_type == 'l2':
            self.fn = lambda pred,gt,mask,far: l2_loss(pred, gt, mask, reduction='mean', **fn_param)
        elif fn_type == 'l2_relative':
            self.fn = lambda pred,gt,mask,far: relative_l2_loss(pred, gt, mask, reduction='mean', **fn_param)
        elif fn_type == 'l2_normalized':
            self.fn = lambda pred,gt,mask,far: l2_loss(pred/far, gt/far, mask, reduction='mean', **fn_param)
        elif fn_type == 'l1':
            self.fn = lambda pred,gt,mask,far: l1_loss(pred, gt, mask, reduction='mean', **fn_param)
        elif fn_type == 'l1_log':
            self.fn = lambda pred,gt,mask,far: l1_loss(torch.log(pred+1), torch.log(gt+1), mask, reduction='mean', **fn_param)
        elif fn_type == 'huber':
            fn_param.setdefault('alpha', 1.0)
            self.fn = lambda pred,gt,mask,far: huber_loss(pred, gt, mask, reduction='mean', **fn_param)
        else:
            raise ValueError(f"Invalid fn_type={fn_type}")

    def forward(self, depth_pred: torch.Tensor, depth_gt: torch.Tensor, mask: torch.BoolTensor, far: float, it: int):
        w = self.w if self.w_fn is None else self.w_fn(it=it)
        depth_loss = self.fn(depth_pred, depth_gt, mask, far)
        return {'lidar_loss.depth': w * depth_loss}

class LineOfSightLoss(nn.Module):
    # LoS (Line of Sight) losses from Urban Radiance Fields (Rematas et al., 2022).
    def __init__(
        self, 
        w: float = 1.0, anneal: dict = None, 
        fn_type: str = 'nerf', fn_param = dict(), 
        ) -> None:
        super().__init__()
        self.w = w
        self.w_fn = None if anneal is None else get_annealer(**anneal)
        self.fn_type = fn_type
        self.fn_param = fn_param

    def forward(
        self, 
        scene: Scene, ret: dict, sample: dict, ground_truth: dict, *, it: int, mask: torch.BoolTensor
        ) -> Dict[str, torch.Tensor]:
        if self.fn_type == 'nerf':
            return self.fn_for_nerf(scene, ret, sample, ground_truth, it=it, mask=mask, **self.fn_param)
        elif self.fn_type == 'neus_urban':
            return self.fn_for_neus_urban(scene, ret, sample, ground_truth, it=it, mask=mask, **self.fn_param)
        elif self.fn_type == 'neus_unisim':
            return self.fn_for_neus_unisim(scene, ret, sample, ground_truth, it=it, mask=mask, **self.fn_param)
        else:
            raise RuntimeError(f"Invalid fn_type={self.fn_type}")

    def fn_for_nerf(
        self, scene: Scene, ret: dict, sample: dict, ground_truth: dict, *, it: int,  
        mask: torch.BoolTensor,
        sigma: float = 1., # Uncertainty around depth values.
        sigma_scale_factor: float = 3.0, 
        ):
        
        volume_buffer = ret['volume_buffer']
        if (buffer_type:=volume_buffer['type']) == 'empty':
            return {}
        
        w = self.w if self.w_fn is None else self.w_fn(it=it)
        target_distribution = torch.distributions.normal.Normal(0.0, sigma / sigma_scale_factor)
        rays_inds_hit, depth_samples, vw = itemgetter('rays_inds_hit', 't', 'vw')(volume_buffer)
        
        mask_on_hit = mask[rays_inds_hit]
        depth_pred = ret['rendered']['depth_volume']
        depth_gt_hit = ground_truth['ranges'].to(depth_pred).view(depth_pred.shape)[rays_inds_hit]

        if buffer_type == 'packed':
            pack_infos = volume_buffer['pack_infos_hit']
            depth_gt_hit_ex = torch.repeat_interleave(depth_gt_hit, pack_infos[:,1], dim=0)
            neighbor_mask = (depth_samples <= depth_gt_hit_ex + sigma) & (depth_samples >= depth_gt_hit_ex - sigma)
            # neighbor_mask = packed_leq(depth_samples, depth_gt+sigma) & packed_geq(depth_samples, depth_gt-sigma)
            neighbor_loss = (vw - torch.exp(target_distribution.log_prob(depth_samples - depth_gt_hit_ex))) ** 2
            neighbor_loss = packed_sum(neighbor_mask * neighbor_loss, pack_infos) # Ray-wise sum -> [num_rays_hit,]
            
            empty_mask = depth_samples < depth_gt_hit_ex - sigma
            # empty_mask = packed_lt(depth_samples, depth_gt-sigma)
            empty_loss = packed_sum(empty_mask * (vw**2), pack_infos) # Ray-wise sum -> [num_rays_hit,]
        elif buffer_type == 'batched':
            depth_gt_hit_ex = depth_gt_hit.unsqueeze(-1)
            neighbor_mask = (depth_samples <= depth_gt_hit_ex + sigma) & (depth_samples >= depth_gt_hit_ex - sigma)
            neighbor_loss = (vw - torch.exp(target_distribution.log_prob(depth_samples - depth_gt_hit_ex))) ** 2
            neighbor_loss = (neighbor_mask * neighbor_loss).sum(-1) # Ray-wise sum -> [num_rays_hit,]
            
            empty_mask = depth_samples < depth_gt_hit_ex - sigma
            empty_loss = (empty_mask * vw**2).sum(-1) # Ray-wise sum -> [num_rays_hit,]
        else:
            raise RuntimeError(f"Invalid buffer_type={buffer_type}")
        
        # [num_rays_hit,],
        return {
            'lidar_loss.los.neighbor': w * (neighbor_loss*mask_on_hit).mean(), 
            'lidar_loss.los.empty': w * (empty_loss*mask_on_hit).mean()
        }

    def fn_for_neus_urban(
        self, scene: Scene, ret: dict, sample: dict, ground_truth: dict, *, it: int, 
        mask: torch.BoolTensor,
        sigma: float = 1., # Uncertainty around depth values.
        sigma_scale_factor: float = 3.0, 
        ):
        volume_buffer = ret['volume_buffer']
        if (buffer_type:=volume_buffer['type']) == 'empty':
            return {}
        
        w = self.w if self.w_fn is None else self.w_fn(it=it)
        target_distribution = torch.distributions.normal.Normal(0.0, sigma / sigma_scale_factor)
        rays_inds_hit, depth_samples, vw = itemgetter('rays_inds_hit', 't', 'vw')(volume_buffer)
        
        mask_on_hit = mask[rays_inds_hit]
        depth_pred = ret['rendered']['depth_volume']
        depth_gt_hit = ground_truth['ranges'].to(depth_pred).view(depth_pred.shape)[rays_inds_hit]

        if buffer_type == 'packed':
            pack_infos = volume_buffer['pack_infos_hit']
            depth_gt_hit_ex = torch.repeat_interleave(depth_gt_hit, pack_infos[:,1], dim=0)
            neighbor_mask = (depth_samples <= depth_gt_hit_ex + sigma) & (depth_samples >= depth_gt_hit_ex - sigma)
            # neighbor_mask = packed_leq(depth_samples, depth_gt_hit+sigma) & packed_geq(depth_samples, depth_gt_hit-sigma)
            neighbor_loss = (vw - torch.exp(target_distribution.log_prob(depth_samples - depth_gt_hit_ex))) ** 2
            neighbor_loss = packed_sum(neighbor_mask * neighbor_loss, pack_infos) # Ray-wise sum -> [num_rays_hit,]
            
            empty_mask = depth_samples < depth_gt_hit_ex - sigma
            # empty_mask = packed_lt(depth_samples, depth_gt_hit-sigma)
            empty_loss = packed_sum(empty_mask * (vw**2), pack_infos) # Ray-wise sum -> [num_rays_hit,]
        elif buffer_type == 'batched':
            depth_gt_hit_ex = depth_gt_hit.unsqueeze(-1)
            neighbor_mask = (depth_samples <= depth_gt_hit_ex + sigma) & (depth_samples >= depth_gt_hit_ex - sigma)
            neighbor_loss = (vw - torch.exp(target_distribution.log_prob(depth_samples - depth_gt_hit_ex))) ** 2
            neighbor_loss = (neighbor_mask * neighbor_loss).sum(-1) # Ray-wise sum -> [num_rays_hit,]
            
            empty_mask = depth_samples < depth_gt_hit_ex - sigma
            empty_loss = (empty_mask * vw**2).sum(-1) # Ray-wise sum -> [num_rays_hit,]
        else:
            raise RuntimeError(f"Invalid buffer_type={buffer_type}")
        
        # [num_rays_hit,], [num_rays_hit,]
        return {
            'lidar_loss.los.neighbor': w * (neighbor_loss*mask_on_hit).mean(), 
            'lidar_loss.los.empty': w * (empty_loss*mask_on_hit).mean()
        }

    def fn_for_neus_unisim(
        self, 
        scene: Scene, ret: dict, sample: dict, ground_truth: dict, *, it: int, 
        mask: torch.BoolTensor,
        epsilon: float = 1.0, # Neighbor boundary
        epsilon_anneal: dict = None, 
        ):
        # Inspired by unisim https://waabi.ai/wp-content/uploads/2023/05/UniSim-paper.pdf
        volume_buffer = ret['volume_buffer']
        if (buffer_type:=volume_buffer['type']) == 'empty':
            return {}

        epsilon = epsilon if epsilon_anneal is None else get_anneal_val(**epsilon_anneal, it=it)

        w = self.w if self.w_fn is None else self.w_fn(it=it)
        rays_inds_hit, depth_samples, vw = itemgetter('rays_inds_hit', 't', 'vw')(volume_buffer)
        
        mask_on_hit = mask[rays_inds_hit]
        depth_pred = ret['rendered']['depth_volume']
        depth_gt_hit = ground_truth['ranges'].to(depth_pred).view(depth_pred.shape)[rays_inds_hit]
        
        if buffer_type == 'packed':
            pack_infos = volume_buffer['pack_infos_hit']
            depth_gt_hit_ex = torch.repeat_interleave(depth_gt_hit, pack_infos[:,1], dim=0)
            empty_mask = (depth_samples - depth_gt_hit_ex).abs() > epsilon
            empty_loss = packed_sum(empty_mask * (vw**2), pack_infos) # [num_rays_hit,]
        elif buffer_type == 'batched':
            depth_gt_hit_ex = depth_gt_hit.unsqueeze(-1)
            empty_mask = (depth_samples - depth_gt_hit_ex).abs() > epsilon
            empty_loss = (empty_mask * (vw**2)).sum(-1) # [num_rays_hit, ]
        else:
            raise RuntimeError(f"Invalid buffer_type={buffer_type}")
        
        # [num_rays_hit,],
        return {
            'lidar_loss.los.empty': w * (empty_loss * mask_on_hit).mean()
        }

class LidarLoss(nn.Module):
    def __init__(
        self, 
        depth: dict = None,
        line_of_sight: dict = None,  
        discard_toofar: float = None, 
        discard_outliers: float = 0,
        discard_outliers_median: float = 100.0, 
        mask_pred_thresh: float = 1.0e-7
        ) -> None:
        """ LiDAR supervision treating LiDARs as sparse depth sensors. 

        Args:
            depth (dict, optional): The configuration for depth supervision. Defaults to None.
            line_of_sight (dict, optional): The configuration for line of sight regularization. Defaults to None.
            discard_toofar (float, optional): Lidar beams with GT depth exceeding this value will be discarded. Defaults to None.
            discard_outliers (float, optional): A value ranging from 0 to 1, representing a ratio. 
                Optionally discard a fixed proportion of lidar beams with large L1 errors. Defaults to 0.
            discard_outliers_median (float, optional): Optionally discard lidar beams whose L1 errors exceed a this multiple of the median L1 error. 
                In practice, compared to `discard_outliers`, `discard_outliers_median` is found to yield more precise results 
                    (since the proportion discarded is not fixed), and can also correctly ignore outliers in GT to ensure safety.
                Defaults to 100.0.
            mask_pred_thresh (float, optional): Only retain lidar beams where mask_pred surpasses this threshold. Defaults to 1.0e-7.
        """
        
        super().__init__()

        if depth is not None:
            self.depth_loss = DepthLoss(**depth)
        else:
            self.depth_loss = None
        
        if line_of_sight is not None:
            self.line_of_sight_loss = LineOfSightLoss(**line_of_sight)
        else:
            self.line_of_sight_loss = None
        
        self.discard_toofar = discard_toofar
        self.discard_outliers = discard_outliers
        self.discard_outliers_median = discard_outliers_median
        self.mask_pred_thresh = mask_pred_thresh
        
    def forward(
        self, 
        scene: Scene, ret: dict, sample: dict, ground_truth: dict, *, 
        it: int, far: float = None, logger: Logger = None) -> Dict[str, torch.Tensor]:
        
        depth_pred = ret['rendered']['depth_volume']
        mask_pred = ret['rendered']['mask_volume']
        depth_gt = ground_truth['ranges'].to(depth_pred).view(depth_pred.shape)
        
        # Binary overall validness mask
        mask = (mask_pred.data > self.mask_pred_thresh) & (depth_gt > 0)
        if self.discard_toofar is not None and self.discard_toofar > 0:
            mask = depth_gt <= self.discard_toofar

        if self.discard_outliers > 0:
            # Optionally discard a percentage of beam with largest depth errors
            with torch.no_grad():
                depth_err_l1 = l1_loss(depth_pred, depth_gt, mask, reduction='none')
                _, sort_inds = torch.sort(depth_err_l1.data) # depth_err[sort_inds]: From small to large
                                
                dicard_rays_inds = sort_inds[-int(depth_pred.numel() * self.discard_outliers):]
                mask[dicard_rays_inds] = False

        if self.discard_outliers_median > 0:
            # Optionally discard beams that have errors exceeding `self.discard_outliers_median` times the median error.
            with torch.no_grad():
                depth_err_l1 = l1_loss(depth_pred, depth_gt, mask, reduction='none')
                sort_values, sort_inds = torch.sort(depth_err_l1.data) # depth_err[sort_inds]: From small to large
                
                median = sort_values[depth_pred.numel()//2]
                mask[depth_err_l1 > median * self.discard_outliers_median] = False

        losses = {}
        
        if self.depth_loss is not None:
            losses.update(self.depth_loss(depth_pred, depth_gt, mask, far=far, it=it))
        
        if self.line_of_sight_loss is not None:
            losses.update(self.line_of_sight_loss(scene, ret, sample, ground_truth, it=it, mask=mask))
        
        return losses