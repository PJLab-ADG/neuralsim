"""
@file   perceptual.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Perceptual / Structural loss
"""

__all__ = [
    'PerceptualLoss', 
    # 'SSIMLoss', 
    'S3IMLoss', 
]

from math import sqrt
from typing import Dict, List, Literal, Union

import torch
import torch.nn as nn

from nr3d_lib.logger import Logger
from nr3d_lib.models.annealers import get_annealer

from app.resources import Scene

class PerceptualLoss(nn.Module):
    def __init__(
        self, 
        # Depth loss function config
        w=1.0, anneal: dict=None, 
        # Loss function config
        loss_type: str = 'lpips', 
        loss_param: dict = {}, 
        # Misc 
        device=None,
        enable_after: int = 0, 
        ) -> None:
        
        super().__init__()
        
        self.w = w
        self.w_fn = None if anneal is None else get_annealer(**anneal)
        self.available_modes = ['image_patch']
        
        self.loss_type = loss_type
        if self.loss_type == 'lpips':
            from nr3d_lib.models.loss.lpipsPyTorch import LPIPS
            # Already detached (no requries_grad parameters)
            self.loss_module = LPIPS(**loss_param, device=device).to(device)
            self.available_modes = ['image_patch']
            def loss_fn(x: torch.Tensor, y: torch.Tensor, mask: torch.BoolTensor = None):
                if mask is not None:
                    x = x * mask
                    y = y * mask
                # From (B)HWC to BCHW; from [0,1] to [-1,1]
                *_, H, W, C = x.shape
                x = x.movedim(-3,-1).reshape(-1,C,H,W) * 2 - 1
                y = y.movedim(-3,-1).reshape(-1,C,H,W) * 2 - 1
                loss = self.loss_module(x,y)
                return loss.mean()
        elif self.loss_type == 'ssim':
            from nr3d_lib.models.loss.ssim import ssim_module
            loss_param.setdefault('window_size', 11)
            loss_param.setdefault('stride', 1) # Usually the patches are small; 1 is good enough.
            self.loss_module = ssim_module(**loss_param, device=device)
            self.available_modes = ['image_patch']
            def loss_fn(x: torch.Tensor, y: torch.Tensor):
                # From (B)HWC to BCHW
                *_, H, W, C = x.shape
                x = x.movedim(-3,-1).reshape(-1,C,H,W)
                y = y.movedim(-3,-1).reshape(-1,C,H,W)
                return (1 - self.loss_module(x,y))
        else:
            raise RuntimeError(f"Invalid loss_type={self.loss_type}")
        
        self.loss_fn = loss_fn
    
        # Misc
        self.enable_after = enable_after
    
    def forward(
        self, scene: Scene, ret: dict, sample: dict, ground_truth: dict, it: int = ..., 
        mode: Literal['pixel', 'lidar', 'image_patch'] = ..., logger: Logger = None):
        assert mode in self.available_modes, \
            f"{self.__class__.__name__} only works with {self.available_modes}, but got mode={mode}"

        if it < self.enable_after:
            return {}

        w = self.w if self.w_fn is None else self.w_fn(it=it)
        if w <= 0:
            return {}        

        ret_losses = {}
        device = scene.device

        rgb_pred = ret['rendered']['rgb_volume']
        rgb_gt = ground_truth['image_rgb'].clone().to(device).view(rgb_pred.shape)
        
        ret_losses[f'loss_{self.loss_type}'] = w * self.loss_fn(rgb_pred, rgb_gt)
        return ret_losses

class S3IMLoss(nn.Module):
    """
    Stochastic Structural SIMilarity(S3IM) algorithm.
    Modified from:
    https://github.com/Madaoer/S3IM-Neural-Fields/blob/main/model_components/s3im.py

    @inproceedings{xie2023s3im,
        title = {S3IM: Stochastic Structural SIMilarity and Its Unreasonable Effectiveness for Neural Fields},
        author = {Xie, Zeke and Yang, Xindi and Yang, Yujie and Sun, Qi and Jiang, Yixiang and Wang, Haoran and Cai, Yunfeng and Sun, Mingming},
        booktitle = {International Conference on Computer Vision},
        year = {2023}
    }

    Arguments:
        kernel_size (int): kernel size in ssim's convolution(default: 4)
        stride (int): stride in ssim's convolution(default: 4)
        repeat_time (int): repeat time in re-shuffle virtual patch(default: 10)
        patch_height (height): height of virtual patch(default: 64)
        patch_width (height): width of virtual patch(default: 64)
    """
    def __init__(
        self, 
        w: float=1.0, 
        kernel_size: int=4, stride: int=4, repeat_time: int=10, 
        patch_height: int=64, patch_width: int=64, patch_pixels: int = None, 
        device=None) -> None:
        from nr3d_lib.models.loss.ssim import ssim_module
        
        super().__init__()

        if patch_pixels is not None:
            patch_height = patch_width = int(sqrt(patch_pixels))
            if patch_height * patch_width > patch_pixels:
                patch_height = patch_width = patch_width - 1

        self.w = w
        self.kernel_size = kernel_size
        self.stride = stride
        self.repeat_time = repeat_time
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.ssim_module = ssim_module(channel=3, window_size=self.kernel_size, stride=self.stride, device=device)
        self.register_buffer('idx', torch.arange(self.patch_height*self.patch_width, device=device, dtype=torch.long))

    def forward(self, scene: Scene, ret: dict, sample: dict, ground_truth: dict, it: int) -> Dict[str, torch.Tensor]:
        device = scene.device
        losses = dict()
        rgb_pred = ret['rendered']['rgb_volume'].view(-1,3)[:self.patch_height*self.patch_width]
        rgb_gt = ground_truth['image_rgb'].clone().to(device).view(-1,3)[:self.patch_height*self.patch_width]
        
        index_list = [self.idx.to(device)] + [torch.randperm(self.patch_height * self.patch_width, device=device) for _ in range(self.repeat_time-1)]
        res_index = torch.cat(index_list).to(device)
        
        # [1, 3, H_, W_ * self.repeat_time]
        rgb_gt_patch = rgb_gt[res_index].permute(1, 0).view(1, 3, self.patch_height, self.patch_width * self.repeat_time)
        rgb_pred_patch = rgb_pred[res_index].permute(1, 0).view(1, 3, self.patch_height, self.patch_width * self.repeat_time)
        losses['rgb_s3im'] = self.w * (1 - self.ssim_module(rgb_pred_patch, rgb_gt_patch))
        
        return losses

if __name__ == "__main__":
    def unit_test_lpips():
        device = torch.device('cuda')
        # Test for [B,H,W,C] case and [H,W,C] case
        loss = PerceptualLoss(loss_type='lpips', loss_param=dict(net_type='vgg'), device=device)
        gt = torch.rand([7, 240, 240, 3], device=device)
        l = loss.loss_fn(gt,gt) # Should be zero
        
        x = torch.rand([7, 240, 240, 3], device=device, requires_grad=True)
        y = torch.rand([7, 240, 240, 3], device=device)
        l = loss.loss_fn(x, y)
        l.backward()
        x_grad = x.grad.data.clone()
        
        x = torch.rand([240, 240, 3], device=device, requires_grad=True)
        y = torch.rand([240, 240, 3], device=device)
        l = loss.loss_fn(x, y)
        l.backward()
        x_grad = x.grad.data.clone()

    def unit_test_ssim():
        device = torch.device('cuda')
        # Test for [B,H,W,C] case and [H,W,C] case
        loss = PerceptualLoss(loss_type='ssim', device=device)
        
        gt = torch.rand([7, 240, 240, 3], device=device)
        l = loss.loss_fn(gt,gt) # Should be zero
        
        x = torch.rand([7, 240, 240, 3], device=device, requires_grad=True)
        y = torch.rand([7, 240, 240, 3], device=device)
        l = loss.loss_fn(x, y)
        l.backward()
        x_grad = x.grad.data.clone()
        
        x = torch.rand([240, 240, 3], device=device, requires_grad=True)
        y = torch.rand([240, 240, 3], device=device)
        l = loss.loss_fn(x, y)
        l.backward()
        x_grad = x.grad.data.clone()

    # unit_test_lpips()
    unit_test_ssim()