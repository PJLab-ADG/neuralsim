"""
@file   mono.py
@brief  Losses using monocular depth and normals priors as weak / relative supervision
"""

import functools
from typing import List, Literal, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from torchmetrics import PearsonCorrCoef
from torchmetrics.functional.regression import pearson_corrcoef

from nr3d_lib.logger import Logger
from nr3d_lib.config import ConfigDict
from nr3d_lib.models.loss.recon import *
from nr3d_lib.models.loss.utils import reduce
from nr3d_lib.models.annealers import get_annealer
from nr3d_lib.graphics.pack_ops import packed_sum

from app.resources import Scene
from app.resources.observers import Camera
from app.renderers.utils import rotate_volume_buffer_nablas

class MonoSDFDepthLoss(nn.Module):
    """
    Scale and Shift Invariant Depth loss.
    Modified from MonoSDF (https://github.com/autonomousvision/monosdf)
    """
    def __init__(
        self, 
        fn_type: Union[str, ConfigDict]='mse', 
        fn_param: dict={}, 
        gt_pre_scale: float = 1., 
        gt_pre_shift: float = 0.,
        scale_gt_to_pred=False, 
        detach_scale_shift=False, 
        alpha_grad_reg=0.01, grad_reg_scales=4,
        ) -> None:
        super().__init__() 

        """
        Args:
            fn_type (Union[str, ConfigDict], optional): The type of loss function for depths. Defaults to 'mse'.
            fn_param (dict, optional): Additional parameters for the loss function. Defaults to {}.
            gt_pre_scale (float, optional): Scale applied to monocular depth prior to the estimation of scale & shift. Defaults to 50..
            gt_pre_shift (float, optional): Shift applied to monocular depth prior to the estimation of scale & shift. Defaults to 0.5.
            scale_gt_to_pred (bool, optional): By default, will scale the predicted monocular depth to the scale of the GT monocular depth.
                If true, will scale the GT monocular depth to the scale of the predicted depth instead before calculating loss. 
                Defaults to False.
            detach_scale_shift (bool, optional): If true, will discard the gradients on the estimated scale & shift. 
                In practice, the gradients on the estimated scale & shift have been found to be helpful.
                Defaults to False.
            alpha_grad_reg (float, optional): Weight for depth-gradient regularization. Defaults to 0.01.
            grad_reg_scales (int, optional): Number of different scales to calculate the depth gradient. Defaults to 4.
        """
        if fn_type == 'mse' or fn_type == 'l2':
            self.regression = functools.partial(mse_loss, **fn_param)
        elif fn_type == 'l1':
            self.regression = functools.partial(l1_loss, **fn_param)
        elif fn_type == 'log_l1':
            self.regression = lambda pred,gt,mask: l1_loss(torch.log((pred+1).clamp_min(1e-3)), torch.log(gt+1), mask)
        elif fn_type == 'smooth_l1':
            self.regression = functools.partial(smooth_l1_loss, **fn_param)
        elif fn_type == 'relative_l1':
            self.regression = functools.partial(relative_l1_loss, **fn_param)
        elif fn_type == 'mape':
            self.regression = functools.partial(mape_loss, **fn_param)
        elif fn_type == 'smape':
            self.regression = functools.partial(smape_loss, **fn_param)
        elif fn_type == 'relative_l2':
            self.regression = functools.partial(relative_l2_loss, **fn_param)
        elif fn_type == 'huber':
            self.regression = functools.partial(huber_loss, **fn_param)
        else:
            raise RuntimeError(f'Invalid fn_type={fn_type}')

        self.gt_pre_scale = gt_pre_scale
        self.gt_pre_shift = gt_pre_shift
        self.scale_gt_to_pred = scale_gt_to_pred
        self.detach_scale_shift = detach_scale_shift
        self.alpha_grad_reg = alpha_grad_reg
        self.grad_reg_scales = grad_reg_scales

    @staticmethod
    def compute_scale_and_shift(prediction, target, mask, sum_dims=(-2,-1)):
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

    @staticmethod
    def image_gradient_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        """ 
        L1 regression loss on image gradients (w.r.t. x and w.r.t. y)
           (pred[1:]-pred[:-1]) - (gt[1:]-gt[:-1]) 
         = (pred[1:]-gt[1:])-(pred[:-1]-gt[:-1])
         = diff[1:] - diff[:-1]
        """
        diff = (prediction - target) * mask
        grad_x = (mask[..., :,  1:] & mask[..., :, :-1]) * (diff[..., :, 1:] - diff[..., :, :-1]).abs()
        grad_y = (mask[..., 1:, :] & mask[..., :-1, :]) * (diff[..., 1:, :] - diff[..., :-1, :]).abs()
        image_loss = torch.sum(grad_x, (-2, -1)) + torch.sum(grad_y, (-2, -1))
        return image_loss

    def forward_grad_reg(self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        total = 0
        for scale in range(self.grad_reg_scales):
            step = pow(2, scale)
            total += self.image_gradient_loss(prediction[..., ::step, ::step], target[..., ::step, ::step], mask[..., ::step, ::step])
        return total

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, debug=False):
        # Optionally scale gt mono depth in advance (just for safer loss calcultion)
        target = self.gt_pre_scale * target + self.gt_pre_shift
        
        if self.detach_scale_shift:
            with torch.no_grad():
                scale, shift = self.compute_scale_and_shift(prediction, target, mask)
        else:
            scale, shift = self.compute_scale_and_shift(prediction, target, mask)
        
        gt_in_pred_scale = (target - shift) / scale
        pred_in_gt_scale = scale * prediction + shift
        loss_reg = None
        if self.scale_gt_to_pred:
            loss_depth = self.regression(prediction, gt_in_pred_scale, mask)
            if self.alpha_grad_reg > 0:
                loss_reg = self.alpha_grad_reg * self.forward_grad_reg(prediction, gt_in_pred_scale, mask)
        else:
            loss_depth = self.regression(pred_in_gt_scale, target, mask)
            if self.alpha_grad_reg > 0:
                loss_reg = self.alpha_grad_reg * self.forward_grad_reg(pred_in_gt_scale, target, mask)
        
        return loss_depth, loss_reg

class PearsonCorrDepthLoss(nn.Module):
    """
    Pearson Correlation monocular depth loss.
    Modified from fsgs (https://github.com/VITA-Group/FSGS)
    """
    def __init__(self, alpha_grad_reg=0.01, grad_reg_scales=4) -> None:
        super().__init__()
        """
        Args:
            alpha_grad_reg (float, optional): Weight for depth-gradient regularization. Defaults to 0.01.
            grad_reg_scales (int, optional): Number of different scales to calculate the depth gradient. Defaults to 4.
        """
        self.alpha_grad_reg = alpha_grad_reg
        self.grad_reg_scales = grad_reg_scales

    @staticmethod
    def image_gradient_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        """ 
        Pearson loss on image gradients (w.r.t. x and w.r.t. y)
        """
        grad_x_mask = mask[..., :,  1:] & mask[..., :, :-1]
        if grad_x_mask.any():
            grad_x_pred = prediction[..., :, 1:] - prediction[..., :, :-1]
            grad_x_gt = target[..., :, 1:] - target[..., :, :-1]
            loss_grad_x = 1 - pearson_corrcoef(grad_x_pred[grad_x_mask].view(-1), grad_x_gt[grad_x_mask].view(-1))
        else:
            loss_grad_x = 0
        
        grad_y_mask = mask[..., 1:, :] & mask[..., :-1, :]
        if grad_y_mask.any():
            grad_y_pred = prediction[..., 1:, :] - prediction[..., :-1, :]
            grad_y_gt = target[..., 1:, :] - target[..., :-1, :]
            loss_grad_y = 1 - pearson_corrcoef(grad_y_pred[grad_y_mask].view(-1), grad_y_gt[grad_y_mask].view(-1))
        else:
            loss_grad_y = 0
        
        ## DEBUG
        # import matplotlib
        # matplotlib.use('TkAgg')
        # import matplotlib.pyplot as plt
        # plt.subplot(2,2,1)
        # plt.imshow(grad_x_pred.data.cpu().numpy())
        # plt.title("grad_x_pred")
        # plt.subplot(2,2,2)
        # plt.imshow(grad_x_gt.data.cpu().numpy())
        # plt.title("grad_x_gt")
        # plt.subplot(2,2,3)
        # plt.imshow(grad_y_pred.data.cpu().numpy())
        # plt.title("grad_y_pred")
        # plt.subplot(2,2,4)
        # plt.imshow(grad_y_gt.data.cpu().numpy())
        # plt.title("grad_y_gt")
        # plt.show()
        return loss_grad_x + loss_grad_y
    
    def forward_grad_reg(self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        total = 0
        for scale in range(self.grad_reg_scales):
            step = pow(2, scale)
            total += self.image_gradient_loss(prediction[..., ::step, ::step], target[..., ::step, ::step], mask[..., ::step, ::step])
        return total

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, reverse_gt=False):
        # NOTE: Need to compute only remained pixels; 
        #   Other wise, the pixels that should be discarded will interfere with pearson correlation (?)
        
        loss_reg = None
        if self.alpha_grad_reg > 0:
            loss_reg = self.alpha_grad_reg * self.forward_grad_reg(prediction, target, mask)
        
        prediction = prediction[mask].view(-1)
        target = target[mask].view(-1)

        if not reverse_gt:
            """
            NOTE: Pearson correlation:
                = -1: total negative correlation
                =  0: no correlation
                =  1: total positive correlation
            """
            loss_depth = (1 - pearson_corrcoef(prediction, target))
        else: # For example, the direct depth output of midas is invert
            loss_depth = torch.minimum(
                (1 - pearson_corrcoef(prediction, -target)), 
                (1 - pearson_corrcoef(prediction, 1 / (target + 200.)))
            )
        return loss_depth, loss_reg

class MonoDepthLoss(nn.Module):
    """
    NOTE: This loss had better only be used on full or downsampled whole images, or image patches, rather than pixels,
        since the scale and shift esimation might not work well on sampled pixels.
    """
    def __init__(
        self, 
        w=1.0, anneal: dict=None, 
        # Loss function config
        loss_type: Literal['monosdf', 'pearson'] = 'pearson', 
        loss_param: dict = {}, 
        # Predicted depth config
        far: float = None, 
        distant_mode: Literal['clamp_far', 'cr_only'] = None,
        # Predicted mask config
        ignore_mask_list: List[str] = [],
        mask_pred_thresh = 0.95,
        mask_erode: int = 0,
        # Misc 
        debug_val_every: int = 0, 
        enable_after: int = 0
        ):
        """ Loss to utilize monocular depth supervision.

        Args:
            w (float, optional): Weight for depth supervision. Defaults to 1.0.
            anneal (dict, optional): Configuration for weight annealing. Defaults to None.
            loss_type (Literal['monosdf', 'pearson'], optional): The type of the underlying loss. Defaults to 'pearson'.
            loss_param (dict, optional): Additional parameters for the underlying loss module. Defaults to {}.
            far (float, optional): The far value of the current rendering. Defaults to None.
            distant_mode (Literal['clamp_far', 'cr_only'], optional): Method to handle the close-range and distant-view models. 
                None: Directly use the jointly rendered depth of cr-dv models.
                'clamp_far': Use the jointly rendered depth of cr-dv models, but clamp the predicted depth with the given far value.
                'cr_only': Use the close-range's contributed part in the joint depth rendered by the cr-dv model.
                Defaults to None.
            ignore_mask_list (List[str], optional): Specify which part(s) of the rendered pixels to ignore. Defaults to [].
            mask_pred_thresh (float, optional): Only retain pixels where mask_pred exceeds this threshold. Defaults to 0.95.
            mask_erode (int, optional): The number of pixels to erode the remaining binary mask after all ignores. 
                In practice, mask erosion has been found to be essential for street views to correctly retain only close-range pixels for calculating loss.
                Defaults to 0.
            debug_val_every (int, optional): Controls the logging frequency of the debug images in the 'image_patch' train step. Defaults to 0.
            enable_after (int, optional): Enable this loss after this iteration. Defaults to 0.
        """
        
        super().__init__()
        
        self.w = w
        self.w_fn = None if anneal is None else get_annealer(**anneal)
        
        # NOTE: The ignore list in mono depth loss might be different from photometric loss
        self.ignore_mask_list = ignore_mask_list
        self.mask_pred_thresh = mask_pred_thresh
        self.mask_erode = mask_erode
        self.distant_mode = distant_mode
        self.far = far

        # The underlying loss function
        if loss_type == 'monosdf':
            loss_fn = MonoSDFDepthLoss(**loss_param)
        elif loss_type == 'pearson':
            loss_fn = PearsonCorrDepthLoss(**loss_param)
        else:
            raise RuntimeError(f"Invalid loss_type={loss_type}")
        self.loss_type = loss_type
        self.loss_fn = loss_fn
        
        # Framework configs
        if self.distant_mode == 'cr_only':
            self.requires_render_per_class = True
        else:
            self.requires_render_per_class = False
        
        # Misc
        self.debug_val_every = debug_val_every
        self.enable_after = enable_after

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
        
        #---- The original rendered depth
        depth_pred0 = ret['rendered']['depth_volume']
        depth_gt = ground_truth['image_mono_depth'].to(device).reshape(depth_pred0.shape)

        #---- How do we deal with potential distant-view depths?
        if self.distant_mode is None:
            # Directly use the rendered depth
            rendered = ret['rendered']
            depth_pred = depth_pred0
        elif self.distant_mode == 'clamp_far':
            # Clamp the maximum predicted depth (all depths further than `far` will be clamped to `far`)
            assert far is not None, "Mono depth need far"
            rendered = ret['rendered']
            depth_pred = depth_pred0.clamp_max(far)
        elif self.distant_mode == 'cr_only':
            # Use only close-range render in total
            rendered = ret['rendered_per_obj_in_scene']['street']
            depth_pred = rendered['depth_volume']
        else:
            raise RuntimeError(f"Invalid distant_mode={self.distant_mode}")
        
        mask_pred = rendered['mask_volume'].data > self.mask_pred_thresh # detached
        
        #---- More extra rules on which parts of pixels are not wanted when applying the loss.
        if len(self.ignore_mask_list) > 0:
            ignore_mask = torch.zeros(mask_pred.shape, device=device, dtype=torch.bool)
            
            # Ignore too far away
            if 'toofar' in self.ignore_mask_list:
                assert far is not None, "Mono depth need far"
                ignore_mask |= depth_pred0 > far
            # Ignore not occupied in prediction
            if 'pred_not_occupied' in self.ignore_mask_list:
                ignore_mask |= (~mask_pred)
            # Ignore not occupied in GT
            if 'not_occupied' in self.ignore_mask_list:
                assert 'image_occupancy_mask' in ground_truth
                ignore_mask |= (~ground_truth['image_occupancy_mask'].view(mask_pred.shape))
            # Ignore dynamic
            if 'dynamic' in self.ignore_mask_list:
                assert 'image_dynamic_mask' in ground_truth
                ignore_mask |= ground_truth['image_dynamic_mask'].view(mask_pred.shape)
            # Ignore human
            if 'human' in self.ignore_mask_list:
                assert 'image_human_mask' in ground_truth
                ignore_mask |= ground_truth['image_human_mask'].view(mask_pred.shape)
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
            remain_mask = torch.ones(mask_pred.shape, device=device, dtype=torch.bool)

        #---- Check if remain_mask is empty (i.e. all the pixels in the image are marked as "not-wanted")
        if not remain_mask.any():
            return {}
        
        #---- Compute the loss!
        loss_depth, loss_grad_reg = self.loss_fn(depth_pred, depth_gt, remain_mask)
        ret_losses['loss_mono_depth.depth'] = w * loss_depth
        if loss_grad_reg is not None:
            ret_losses['loss_mono_depth.reg'] = w * loss_grad_reg
        
        #---- Validate the monodepth
        if self.debug_val_every > 0:
            with torch.no_grad():
                it_from_enable = it - self.enable_after
                debug_val_every = (max(self.debug_val_every//5, 200) if it_from_enable <= 1000 \
                    else (max(self.debug_val_every, 200) if it_from_enable <= 10000 \
                    else max(self.debug_val_every, 2000)))
                if (it_from_enable % debug_val_every == 0) and (logger is not None) and (mode == 'image_patch'):
                    scale, shift = MonoSDFDepthLoss.compute_scale_and_shift(depth_pred, depth_gt, remain_mask)
                    gt_in_pred_scale = (depth_gt - shift) / scale
                    depth_to_log = torch.stack([depth_pred.data * remain_mask, gt_in_pred_scale.data * remain_mask], dim=0).unsqueeze(1)
                    depth_to_log /= depth_to_log.max().clamp_min(1e-5)
                    logger.add_imgs(f"train_step_{mode}.{scene.id}", "mono_depth_loss", depth_to_log, it)

        return ret_losses

class MonoNormalLoss(nn.Module):
    def __init__(
        self, 
        w_l1=1.0, w_cos=1.0, 
        loss_per: Literal['pixels', 'points'] = 'pixels', 
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
            loss_per (Literal['pixels', 'points'], optional): 
                - 'pixels': The loss is computed per rendered pixel
                - 'points': The loss is computed per sample points, and sum up weighted by vw
                Defaults to 'pixels'.
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
        self.loss_per = loss_per
        self.distant_mode = distant_mode
        if self.distant_mode == 'cr_only':
            self.requires_render_per_class = True
        else:
            self.requires_render_per_class = False
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
        rendered = ret['rendered']
        # if self.distant_mode == 'crdv':
        #     rendered = ret['rendered']
        # elif self.distant_mode == 'cr_only':
        #     rendered = ret['rendered_per_obj_in_scene']['street']
        # else:
        #     raise RuntimeError(f"Invalid distant_mode={self.distant_mode}")

        normal_pred = rendered['normals_volume']
        mask_pred = (rendered['mask_volume'].data > self.mask_pred_thresh) # detached
        
        # Transform from world coordinate system to camera local coordinate system 
        # NOTE: rendered['normals_volume'] is already rotated from obj to world
        normal_pred = cam.world_transform.detach().rotate(normal_pred, inv=True)
        normal_gt = F.normalize(ground_truth['image_mono_normals'], dim=-1)
        
        if 'pred_not_occupied' in self.ignore_mask_list:
            ignore_mask = ~mask_pred
        else:
            ignore_mask = torch.zeros_like(mask_pred)
        if 'not_occupied' in self.ignore_mask_list:
            assert 'image_occupancy_mask' in ground_truth
            ignore_mask |= (~ground_truth['image_occupancy_mask'].view(mask_pred.shape))
        if 'dynamic' in self.ignore_mask_list:
            assert 'image_dynamic_mask' in ground_truth
            ignore_mask |= ground_truth['image_dynamic_mask'].view(mask_pred.shape)
        if 'human' in self.ignore_mask_list:
            assert 'image_human_mask' in ground_truth
            ignore_mask |= ground_truth['image_human_mask'].view(mask_pred.shape)
        
        remain_mask = ignore_mask.logical_not()
        # NOTE: Erode the remained mask to prevent loss being too sensitive on object edges
        if self.mask_erode > 0:
            kernel_size = self.mask_erode * 2
            remain_mask = kornia.morphology.erosion(remain_mask[None,None].float(), torch.ones([kernel_size,kernel_size], device=remain_mask.device))[0,0].bool()

        #---- Check mask empty
        if not remain_mask.any():
            return {}

        if self.loss_per == 'pixels':
            loss_l1, loss_cos = self.fn(normal_pred, normal_gt, remain_mask)
            ret_losses['loss_mono_normal.l1'] = self.w_l1 * loss_l1
            ret_losses['loss_mono_normal.cos'] = self.w_cos * loss_cos
        elif self.loss_per == 'points':
            raise NotImplementedError
            # for oid, oret in ret['raw_per_obj_model'].items():
            #     if 'nablas_in_world' not in oret['volume_buffer'].keys():
            #         continue
            #     # o2w = scene.all_nodes[oid].world_transform.detach() # Removing gradients on nablas can eliminate interference with the pose gradient.
            #     obj_nablas_in_world = oret['volume_buffer']['nablas_in_world']
            #     w2c_rot = cam.world_transform.rotation().data.T
            #     obj_nablas_in_cam = rotate_volume_buffer_nablas(w2c_rot, obj_nablas_in_world, oret['volume_buffer'])
            #     remain_mask_o = remain_mask.flatten()[oret['volume_buffer']['rays_inds_collect']]
            #     normal_gt_o = normal_gt.flatten(0,-2)[oret['volume_buffer']['rays_inds_collect']]
            #     normal_gt_per_pts = torch.repeat_interleave(normal_gt_o, oret['volume_buffer']['pack_infos_collect'][:,1], dim=0)
            #     loss_l1_o = packed_sum(
            #         oret['volume_buffer']['vw_in_total'].data * (obj_nablas_in_cam - normal_gt_per_pts).abs().sum(dim=-1).flatten(), 
            #         oret['volume_buffer']['pack_infos_collect'])
            #     loss_cos_o = packed_sum(
            #         oret['volume_buffer']['vw_in_total'].data * (1. - (obj_nablas_in_cam * normal_gt_per_pts).sum(dim=-1)).flatten(), 
            #         oret['volume_buffer']['pack_infos_collect'])
            #     ret_losses[f'loss_mono_normal.{oid}.l1'] = self.w_l1 * reduce(loss_l1_o, mask=remain_mask_o, reduction='mean')
            #     ret_losses[f'loss_mono_normal.{oid}.cos'] = self.w_cos * reduce(loss_cos_o, mask=remain_mask_o, reduction='mean')
        else:
            raise RuntimeError(f"Invalid loss_per={self.loss_per}")

        # Debug the rendered normals vs. gt mono normals
        if self.debug_val_every > 0:           
            it_from_enable = it - self.enable_after
            debug_val_every = (max(self.debug_val_every//5, 200) if it_from_enable <= 1000 \
                else (max(self.debug_val_every, 200) if it_from_enable <= 10000 \
                else max(self.debug_val_every, 2000)))
            if (it_from_enable % debug_val_every == 0) and (logger is not None) and (mode == 'image_patch'):
                # [2, H, W, C] -> [2, C, H, W] for tensorboard logging
                normal_to_log = torch.stack([(normal_pred.data / 2 + 0.5) * remain_mask.unsqueeze(-1), (normal_gt.data / 2 + 0.5) * remain_mask.unsqueeze(-1)], dim=0).movedim(-1, 1).clamp_(0,1)
                logger.add_imgs(f"train_step_{mode}.{scene.id}", "mono_normals_loss", normal_to_log, it)

        return ret_losses

if __name__ == "__main__":
    def unit_test_pearson():
        a = torch.randn([10*7,], requires_grad=True) # Will be viewed as 10 data points with 7 dimensions.
        # b = a * 2 - 3 + 0.01 * torch.randn_like(a)
        b = torch.randn_like(a)
        # b = a * -2 - 3 + 0.01 * torch.randn_like(a)
        b.requires_grad_(True)
        c1 = pearson_corrcoef(a,b) # [7,], Return the coefficients in 7 dimensions
        c1.mean().backward()
        a_grad_1 = a.grad.data.clone()
        b_grad_1 = b.grad.data.clone()
        a.grad = None
        b.grad = None
        
        c2 = pearson_corrcoef(b,a)
        c2.mean().backward()
        a_grad_2 = a.grad.data.clone()
        b_grad_2 = b.grad.data.clone( )
        
        # Experiment confirmed: both x and y have gradients & the gradients are the same after swapping.
        _  = 1
        
    unit_test_pearson()