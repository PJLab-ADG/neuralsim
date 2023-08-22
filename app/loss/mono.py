"""
@file   mono.py
@brief  Loss with extracted mono cues (mono depth prior and mono normals prior)
        Modified from https://github.com/autonomousvision/monosdf
"""

import functools
from typing import List, Literal, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia

from nr3d_lib.logger import Logger
from nr3d_lib.config import ConfigDict
from nr3d_lib.models.loss.recon import *
from nr3d_lib.models.annealers import get_annealer

from app.resources import Scene
from app.resources.observers import Camera
from app.renderers.utils import rotate_volume_buffer_nablas

from nr3d_lib.models.loss.utils import reduce
from nr3d_lib.render.pack_ops import packed_sum

def compute_scale_and_shift(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, sum_dims=(-2,-1)):
    prediction, target, mask = prediction.float(), target.float(), mask.float()
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, sum_dims)
    a_01 = torch.sum(mask * prediction, sum_dims)
    a_11 = torch.sum(mask, sum_dims)

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, sum_dims)
    b_1 = torch.sum(mask * target, sum_dims)

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    
    if det.numel() > 1:
        valid = det.nonzero()
        x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
        x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]
    elif det.item() != 0:
        x_0 = (a_11 * b_0 - a_01 * b_1) / det
        x_1 = (-a_01 * b_0 + a_00 * b_1) / det

    return x_0, x_1

def gradient_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    diff = (prediction - target) * mask
    
    grad_x = (mask[..., :,  1:] & mask[..., :, :-1]) * (diff[..., :, 1:] - diff[..., :, :-1]).abs()
    grad_y = (mask[..., 1:, :] & mask[..., :-1, :]) * (diff[..., 1:, :] - diff[..., :-1, :]).abs()
    
    image_loss = torch.sum(grad_x, (-2, -1)) + torch.sum(grad_y, (-2, -1))
    return image_loss

class MonoSSIDepthLoss(nn.Module):
    """
    Scale and Shift Invariant Depth loss, modified from MonoSDF (https://github.com/autonomousvision/monosdf)
    
    NOTE: This loss had better only be used on full or downsampled whole images, or image patches, rather than pixels,
        since the scale and shift esimation might not work well on sampled pixels.
    """
    def __init__(
        self, 
        # Depth loss function config
        w=1.0, anneal: ConfigDict=None, 
        fn_type: Union[str, ConfigDict]='mse', fn_param: dict={}, 
        # Regularization: depth gradient(difference)
        w_grad_reg=0.01, grad_reg_scales=4,
        # Predicted depth config
        far: float = None, 
        distant_mode: Literal['clamp_far', 'cr_only'] = None,
        gt_pre_scale: float = 50., gt_pre_shift: float = 0.5,
        scale_gt_to_pred=False, detach_scale_shift=False, 
        # Predicted mask config
        ignore_mask_list: List[str] = [],
        mask_pred_thresh = 0.95,
        mask_erode: int = 0,
        # discard_outliers_median: float = 0, 
        # Misc 
        debug_val_every: int = 0, 
        enable_after: int = 0):
        """ Loss to utilize monocular depth supervision.

        Args:
            w (float, optional): Weight for depth supervision. Defaults to 1.0.
            anneal (ConfigDict, optional): Configuration for weight annealing. Defaults to None.
            fn_type (Union[str, ConfigDict], optional): The type of loss function for depths. Defaults to 'mse'.
            fn_param (dict, optional): Additional parameters for the loss function. Defaults to {}.
            w_grad_reg (float, optional): Weight for depth-gradient regularization. Defaults to 0.01.
            grad_reg_scales (int, optional): Number of different scales to calculate the depth gradient. Defaults to 4.
            far (float, optional): The far value of the current rendering. Defaults to None.
            distant_mode (Literal['clamp_far', 'cr_only'], optional): Method to handle the close-range and distant-view models. 
                None: Directly use the jointly rendered depth of cr-dv models.
                'clamp_far': Use the jointly rendered depth of cr-dv models, but clamp the predicted depth with the given far value.
                'cr_only': Use the close-range's contributed part in the joint depth rendered by the cr-dv model.
                Defaults to None.
            gt_pre_scale (float, optional): Scale applied to monocular depth prior to the estimation of scale & shift. Defaults to 50..
            gt_pre_shift (float, optional): Shift applied to monocular depth prior to the estimation of scale & shift. Defaults to 0.5.
            scale_gt_to_pred (bool, optional): By default, will scale the predicted monocular depth to the scale of the GT monocular depth.
                If true, will scale the GT monocular depth to the scale of the predicted depth instead before calculating loss. 
                Defaults to False.
            detach_scale_shift (bool, optional): If true, will discard the gradients on the estimated scale & shift. 
                In practice, the gradients on the estimated scale & shift have been found to be helpful.
                Defaults to False.
            ignore_mask_list (List[str], optional): Specify which part(s) of the rendered pixels to ignore. Defaults to [].
            mask_pred_thresh (float, optional): Only retain pixels where mask_pred exceeds this threshold. Defaults to 0.95.
            mask_erode (int, optional): The number of pixels to erode the remaining binary mask after all ignores. 
                In practice, mask erosion has been found to be essential for street views to correctly retain only close-range pixels for calculating loss.
                Defaults to 0.
            debug_val_every (int, optional): Controls the logging frequency of the debug images in the 'image_patch' train step. Defaults to 0.
            enable_after (int, optional): Enable this loss after this iteration. Defaults to 0.
        """
        
        super().__init__()
        
        # NOTE: The ignore list in mono depth loss might be different from photometric loss
        self.ignore_mask_list = ignore_mask_list
        
        self.w = w
        self.w_grad_reg = w_grad_reg
        self.w_fn = None if anneal is None else get_annealer(**anneal)
        
        self.gt_pre_scale = gt_pre_scale
        self.gt_pre_shift = gt_pre_shift
        self.mask_pred_thresh = mask_pred_thresh
        self.mask_erode = mask_erode
        self.enable_after = enable_after
        self.distant_mode = distant_mode
        self.scale_gt_to_pred = scale_gt_to_pred
        self.detach_scale_shift = detach_scale_shift
        if self.distant_mode == 'cr_only':
            self.require_render_per_obj = True
        else:
            self.require_render_per_obj = False

        if fn_type == 'mse' or fn_type == 'l2':
            self.fn = functools.partial(mse_loss, **fn_param)
        elif fn_type == 'l1':
            self.fn = functools.partial(l1_loss, **fn_param)
        elif fn_type == 'log_l1':
            self.fn = lambda pred,gt,mask: l1_loss(torch.log((pred+1).clamp_min(1e-3)), torch.log(gt+1), mask)
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
        elif fn_type == 'huber':
            self.fn = functools.partial(huber_loss, **fn_param)
        else:
            raise RuntimeError(f'Invalid fn_type={fn_type}')

        self.far = far
        # self.discard_outliers_median = discard_outliers_median
        
        self.grad_reg_scales = grad_reg_scales
        self.debug_val_every = debug_val_every

    def fn_grad_reg(self, prediction, target, mask):
        total = 0
        for scale in range(self.grad_reg_scales):
            step = pow(2, scale)
            total += gradient_loss(prediction[..., ::step, ::step], target[..., ::step, ::step], mask[..., ::step, ::step])
        return total

    def fn_los_reg(self, prediction, target, mask):
        pass

    def forward(
        self, scene: Scene, ret: dict, sample: dict, ground_truth: dict, far: float=None, it: int = ..., 
        mode: Literal['pixel', 'lidar', 'image_patch'] = ..., logger: Logger = None):
        if it < self.enable_after:
            return {}
        
        ret_losses = {}
        device = scene.device
        w = self.w if self.w_fn is None else self.w_fn(it=it)
        
        if self.far is not None:
            far = self.far
        
        #---- Option 1: Clamp to discard depth points too much further (especially when there is distant-view).
        if self.distant_mode is None:
            rendered = ret['rendered']
            depth_pred = rendered['depth_volume']
        elif self.distant_mode == 'clamp_far':
            assert far is not None, "Mono depth need far"
            rendered = ret['rendered']
            depth_pred = rendered['depth_volume'].clamp_max(far)
        #---- Option 2: Use only close-range render in total
        elif self.distant_mode == 'cr_only':
            rendered = ret['rendered_per_obj_in_total']['street']
            depth_pred = rendered['depth_volume']
        else:
            raise RuntimeError(f"Invalid distant_mode={self.distant_mode}")
        
        mask_pred = rendered['mask_volume'].data > self.mask_pred_thresh # detached
        depth_gt = ground_truth['rgb_mono_depth'].to(device).reshape(depth_pred.shape)
        
        if len(self.ignore_mask_list) > 0:
            ignore_mask = torch.zeros(rendered['depth_volume'].shape, device=device, dtype=torch.bool)
            
            # Ignore too far away
            if 'toofar' in self.ignore_mask_list:
                assert far is not None, "Mono depth need far"
                ignore_mask |= rendered['depth_volume'] > far
            
            # Ignore not occupied in prediction
            if 'pred_not_occupied' in self.ignore_mask_list:
                ignore_mask |= (~mask_pred)
            # Ignore not occupied in GT
            if 'not_occupied' in self.ignore_mask_list:
                assert 'rgb_mask' in ground_truth
                ignore_mask |= (~ground_truth['rgb_mask'].view(mask_pred.shape))
            # Ignore dynamic
            if 'dynamic' in self.ignore_mask_list:
                assert 'rgb_dynamic_mask' in ground_truth
                ignore_mask |= ground_truth['rgb_dynamic_mask'].view(mask_pred.shape)
            # Ignore human
            if 'human' in self.ignore_mask_list:
                assert 'rgb_human_mask' in ground_truth
                ignore_mask |= ground_truth['rgb_human_mask'].view(mask_pred.shape)
            # Ignore "distant" in prediction
            if 'pred_distant' in self.ignore_mask_list:
                ignore_mask |= (depth_pred.data > (far * 0.8)).view(mask_pred.shape)
            # Ignore "distant" in monocular estimation
            if 'mono_distant' in self.ignore_mask_list:
                ignore_mask |= (depth_gt.data > depth_gt.data.mean()-1e-5).view(mask_pred.shape)

            # NOTE: The remained mask after all the "ignore"s above
            remain_mask = ignore_mask.logical_not()
            # NOTE: Erode the remained mask to prevent loss being too sensitive on object edges
            if self.mask_erode > 0:
                kernel_size = self.mask_erode * 2
                remain_mask = kornia.morphology.erosion(remain_mask[None,None].float(), torch.ones([kernel_size,kernel_size], device=remain_mask.device))[0,0].bool()
        else:
            remain_mask = torch.ones(rendered['depth_volume'].shape, device=device, dtype=torch.bool)

        #---- Check mask empty
        if not remain_mask.any():
            return {}
        
        #---- Estimate the scale and shift mono depth
        # Pre-scale GT and scale prediction (just for safer esitmation)
        depth_gt = self.gt_pre_scale * depth_gt + self.gt_pre_shift
        if self.detach_scale_shift:
            with torch.no_grad():
                scale, shift = compute_scale_and_shift(depth_pred, depth_gt, remain_mask)
        else:
            scale, shift = compute_scale_and_shift(depth_pred, depth_gt, remain_mask)
        
        if self.scale_gt_to_pred:
            gt_ssi = (depth_gt - shift) / scale
            if w > 0:
                loss_depth = self.fn(depth_pred, gt_ssi, remain_mask)
                ret_losses['loss_mono_depth.depth'] = w * loss_depth
            if self.w_grad_reg > 0:
                loss_grad_reg = self.fn_grad_reg(depth_pred, gt_ssi, remain_mask)
                ret_losses['loss_mono_depth.reg'] = self.w_grad_reg * loss_grad_reg
            # [2, H, W] -> [2, 1, H, W] for tensorboard logging
            depth_to_log = torch.stack([depth_pred.data * remain_mask, gt_ssi.data * remain_mask], dim=0).unsqueeze(1)
            depth_to_log = depth_to_log / depth_to_log.max().clamp_min(1e-5)
        else:
            prediction_ssi = scale * depth_pred + shift
            if w > 0:
                loss_depth = self.fn(prediction_ssi, depth_gt, remain_mask)
                ret_losses['loss_mono_depth.depth'] = w * loss_depth
            if self.w_grad_reg > 0:
                loss_grad_reg = self.fn_grad_reg(prediction_ssi, depth_gt, remain_mask)
                ret_losses['loss_mono_depth.reg'] = self.w_grad_reg * loss_grad_reg
            # [2, H, W] -> [2, 1, H, W] for tensorboard logging
            depth_to_log = torch.stack([prediction_ssi.data * remain_mask, depth_gt.data * remain_mask], dim=0).unsqueeze(1)
            depth_to_log = depth_to_log / depth_to_log.max().clamp_min(1e-5)

        if self.debug_val_every > 0:
            it_from_enable = it - self.enable_after
            debug_val_every = (max(self.debug_val_every//5, 200) if it_from_enable <= 1000 \
                else (max(self.debug_val_every, 200) if it_from_enable <= 10000 \
                else max(self.debug_val_every, 2000)))
            if (it_from_enable % debug_val_every == 0) and (logger is not None) and (mode == 'image_patch'):
                logger.add_imgs(f"train_step_{mode}.{scene.id}", "mono_depth_loss", depth_to_log, it)

        return ret_losses

class MonoNormalLoss(nn.Module):
    def __init__(
        self, 
        w_l1=1.0, w_cos=1.0, 
        should_apply_per_pts=False, 
        distant_mode: Literal['crdv', 'cr_only'] = 'crdv', 
        ignore_mask_list: List[str] = ['pred_not_occupied', 'not_occupied', 'human'], 
        mask_pred_thresh: float=0.5, mask_erode: int = 0, 
        apply_in_pixel_train_step=False, 
        enable_after: int = 0, 
        debug_val_every: int = -1, 
        ) -> None:
        """ Loss to utilize monocular normals supervision.

        Args:
            w_l1 (float, optional): Weight for L1 loss. Defaults to 1.0.
            w_cos (float, optional): Weight for cosine angle difference loss. Defaults to 1.0.
            should_apply_per_pts (bool, optional): 
                If true, supervision will be applied to each sampled point on each ray/pixel, 
                and them conduct weighted sum by visibility weights,
                instead of the rendered pixel normals by default. Defaults to False.
            distant_mode (Literal['crdv', 'cr_only'], optional): Method to handle the close-range and distant-view models. 
                'crdv': Supervise the rendered normals of joint cr-dv. The distant-view part has no gradients.
                'cr_only': Supervise the rendered normals of the close-range's contributed part in the joint cr-dv model.
                Defaults to 'crdv'.
            ignore_mask_list (List[str], optional): Specify which part(s) of the rendered pixels to ignore. 
                Defaults to ['pred_not_occupied', 'not_occupied', 'human'].
            mask_pred_thresh (float, optional): Only retain pixels where mask_pred exceeds this threshold. 
                Defaults to 0.5.
            mask_erode (int, optional): The number of pixels to erode the remaining binary mask after all ignores. 
                Defaults to 0.
            apply_in_pixel_train_step (bool, optional): If true, this loss will also be applied in the 'pixel' train step. 
                By default, this loss is only applied in 'image_patch' train step.
                Defaults to False.
            enable_after (int, optional): Enable this loss after this iteration. Defaults to 0.
            debug_val_every (int, optional): Controls the logging frequency of the debug images in the 'image_patch' train step. Defaults to -1.
        """
        
        super().__init__()
        self.w_l1 = w_l1
        self.w_cos = w_cos
        self.should_apply_per_obj = should_apply_per_pts
        self.distant_mode = distant_mode
        if self.distant_mode == 'cr_only':
            self.require_render_per_obj = True
        else:
            self.require_render_per_obj = False
        self.ignore_mask_list = ignore_mask_list
        self.mask_pred_thresh = mask_pred_thresh
        self.mask_erode = mask_erode
        self.apply_in_pixel_train_step = apply_in_pixel_train_step
        self.enable_after = enable_after
        self.debug_val_every = debug_val_every
    
    def fn(self, normal_pred: torch.Tensor, normal_gt: torch.Tensor, mask: torch.Tensor = None):
        normal_pred = F.normalize(normal_pred, dim=-1)
        normal_gt = F.normalize(normal_gt, dim=-1)
        l1 = reduce((normal_pred - normal_gt).abs().sum(dim=-1), mask=mask, reduction='mean')
        cos = reduce((1. - (normal_pred * normal_gt).sum(dim=-1)), mask=mask, reduction='mean')
        return l1, cos
    
    def forward(
        self, scene: Scene, cam: Camera, ret: dict, sample: dict, ground_truth: dict, it: int, 
        mode: Literal['pixel', 'lidar', 'image_patch'] = ..., logger: Logger = None):
        if it < self.enable_after:
            return {}
        
        ret_losses = {}
        device = scene.device
        if self.distant_mode == 'crdv':
            rendered = ret['rendered']
        elif self.distant_mode == 'cr_only':
            rendered = ret['rendered_per_obj_in_total']['street']
        else:
            raise RuntimeError(f"Invalid distant_mode={self.distant_mode}")

        normal_pred = rendered['normals_volume']
        mask_pred = (rendered['mask_volume'].data > self.mask_pred_thresh) # detached
        
        # Transform from world coordinate system to camera local coordinate system 
        # NOTE: rendered['normals_volume'] is already rotated from obj to world
        normal_pred = cam.world_transform.detach().rotate(normal_pred, inv=True)
        normal_gt = F.normalize(ground_truth['rgb_mono_normals'], dim=-1)
        
        if 'pred_not_occupied' in self.ignore_mask_list:
            ignore_mask = ~mask_pred
        else:
            ignore_mask = torch.zeros_like(mask_pred)
        if 'not_occupied' in self.ignore_mask_list:
            assert 'rgb_mask' in ground_truth
            ignore_mask |= (~ground_truth['rgb_mask'].view(mask_pred.shape))
        if 'dynamic' in self.ignore_mask_list:
            assert 'rgb_dynamic_mask' in ground_truth
            ignore_mask |= ground_truth['rgb_dynamic_mask'].view(mask_pred.shape)
        if 'human' in self.ignore_mask_list:
            assert 'rgb_human_mask' in ground_truth
            ignore_mask |= ground_truth['rgb_human_mask'].view(mask_pred.shape)
        
        remain_mask = ignore_mask.logical_not()
        # NOTE: Erode the remained mask to prevent loss being too sensitive on object edges
        if self.mask_erode > 0:
            kernel_size = self.mask_erode * 2
            remain_mask = kornia.morphology.erosion(remain_mask[None,None].float(), torch.ones([kernel_size,kernel_size], device=remain_mask.device))[0,0].bool()

        #---- Check mask empty
        if not remain_mask.any():
            return {}

        loss_l1, loss_cos = self.fn(normal_pred, normal_gt, remain_mask)
        ret_losses['loss_mono_normal.l1'] = self.w_l1 * loss_l1
        ret_losses['loss_mono_normal.cos'] = self.w_cos * loss_cos

        # for oid, oret in ret['raw_per_obj'].items():
        #     if 'nablas_in_world' not in oret['volume_buffer'].keys():
        #         continue
        #     # o2w = scene.all_nodes[oid].world_transform.detach() # Removing gradients on nablas can eliminate interference with the pose gradient.
        #     obj_nablas_in_world = oret['volume_buffer']['nablas_in_world']
        #     w2c_rot = cam.world_transform.rotation().data.T
        #     obj_nablas_in_cam = rotate_volume_buffer_nablas(w2c_rot, obj_nablas_in_world, oret['volume_buffer'])
        #     remain_mask_o = remain_mask.flatten()[oret['volume_buffer']['ray_inds_hit_collect']]
        #     normal_gt_o = normal_gt.flatten(0,-2)[oret['volume_buffer']['ray_inds_hit_collect']]
        #     normal_gt_per_pts = torch.repeat_interleave(normal_gt_o, oret['volume_buffer']['pack_infos_collect'][:,1], dim=0)
        #     loss_l1_o = packed_sum(
        #         oret['volume_buffer']['vw_in_total'].data * (obj_nablas_in_cam - normal_gt_per_pts).abs().sum(dim=-1).flatten(), 
        #         oret['volume_buffer']['pack_infos_collect'])
        #     loss_cos_o = packed_sum(
        #         oret['volume_buffer']['vw_in_total'].data * (1. - (obj_nablas_in_cam * normal_gt_per_pts).sum(dim=-1)).flatten(), 
        #         oret['volume_buffer']['pack_infos_collect'])
        #     ret_losses[f'loss_mono_normal.{oid}.l1'] = self.w_l1 * reduce(loss_l1_o, mask=remain_mask_o, reduction='mean')
        #     ret_losses[f'loss_mono_normal.{oid}.cos'] = self.w_cos * reduce(loss_cos_o, mask=remain_mask_o, reduction='mean')

        if self.debug_val_every > 0:
            # [2, H, W, C] -> [2, C, H, W] for tensorboard logging
            normal_to_log = torch.stack([(normal_pred.data / 2 + 0.5) * remain_mask.unsqueeze(-1), (normal_gt.data / 2 + 0.5) * remain_mask.unsqueeze(-1)], dim=0).movedim(-1, 1).clamp_(0,1)
            
            it_from_enable = it - self.enable_after
            debug_val_every = (max(self.debug_val_every//5, 200) if it_from_enable <= 1000 \
                else (max(self.debug_val_every, 200) if it_from_enable <= 10000 \
                else max(self.debug_val_every, 2000)))
            if (it_from_enable % debug_val_every == 0) and (logger is not None) and (mode == 'image_patch'):
                logger.add_imgs(f"train_step_{mode}.{scene.id}", "mono_normals_loss", normal_to_log, it)

        return ret_losses