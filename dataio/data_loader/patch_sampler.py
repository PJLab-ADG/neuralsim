"""
@file   patch_sampler.py
@author Jianfei Guo, Shanghai AI Lab
@brief  A generic image patch sampler

Supported patch types:

- Whole image scale, interval sampler with random local perturbation
- Interval pixels at a certain downsampling rate

Consider whether to support the existence of gradients in some random pixels in a whole image?

The core output of the sampler: a group or batch_size groups of pixel positions in [0,1] and pixel indices (or batch)
The batched function still normally relies on the design of the batched function of pytorch's dataset
-> Here, it needs to be noted that in the process of view ind sampling, it may be necessary to design for more dependent multi-frame nvs

Configurable items:
- scale
- offset (none, all pixels fixed, all pixels random)

The internal calculation of xy has nothing to do with the specific H,W -> HW only serves the final calculation of hw and the control of pixel interval

"""

import math
import functools
import random
import numpy as np
from copy import deepcopy
from numbers import Number
from typing import List, Literal, Tuple, Union

import torch
from torch._tensor import Tensor

from nr3d_lib.models.annealers import get_anneal_val, get_anneal_val_milestones
from nr3d_lib.utils import check_to_torch


def get_scale_annealed(
    it: int, *, min_scale: float = 0.25, max_scale: float = 1.0, 
    is_increasing=False, anneal_type: str = 'linear', **anneal_kwargs):
    
    if is_increasing: # From min_scale to max_scale (from local to global)
        scale = get_anneal_val(it=it, type=anneal_type, start_val=min_scale, stop_val=max_scale, **anneal_kwargs)
    else:
        scale = get_anneal_val(it=it, type=anneal_type, start_val=max_scale, stop_val=min_scale, **anneal_kwargs)
    return scale

def get_scale_annealed_random(
    it: int, *, min_scale: float = 0.25, max_scale: float = 1.0, 
    is_increasing=False, anneal_on_min=True, anneal_type: str = 'linear', **anneal_kwargs):
    
    if anneal_on_min: # min_scale is annealed; max_scale is fixed.
        if is_increasing: # From min_scale to max_scale (from local to global)
            min_scale = get_anneal_val(it=it, type=anneal_type, start_val=min_scale, stop_val=max_scale*0.9, **anneal_kwargs)
        else:
            min_scale = get_anneal_val(it=it, type=anneal_type, start_val=max_scale*0.9, stop_val=min_scale, **anneal_kwargs)
    else: # max_scale is annealed; min_scale is fixed.
        if is_increasing: # From min_scale to max_scale (from local to global)
            max_scale = get_anneal_val(it=it, type=anneal_type, start_val=min_scale*1.1, stop_val=max_scale, **anneal_kwargs)
        else:
            max_scale = get_anneal_val(it=it, type=anneal_type, start_val=max_scale, stop_val=min_scale*1.1, **anneal_kwargs)
    scale = np.random.uniform(min_scale, max_scale)
    return scale

def get_scale_random(it: int, *, min_scale: float = 0.25, max_scale: float = 1.0, ):
    return np.random.uniform(min_scale, max_scale)

def get_scale_fixed(it: int, *, scale: float = 1.0, ):
    return scale

def get_patch_hw(num_pixels: int, aspect_ratio: float = 1.0) -> Tuple[int,int]:
    """_summary_

    Args:
        num_pixels (int): _description_
        aspect_ratio (float, optional): The ratio of width/height. Defaults to 1.0.

    Returns:
        Tuple[int,int]: _description_
    """
    H = math.sqrt(num_pixels * aspect_ratio)
    W = H * aspect_ratio
    return int(H), int(W)

class PatchSamplerBase:
    def sample_pixels(self, return_hw=True, device=None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        xy: torch.Tensor = None
        hw: torch.Tensor = None
        dbg_infos: dict = None
        return xy, hw, dbg_infos

class PatchSamplerFlexGrid(PatchSamplerBase):
    def __init__(
        self, 
        patch_hw: Union[Tuple[int,int], int] = 256, 
        aspect_ratio: Number = 1.0, # width / height
        num_pixels: int = None, 
        
        scale_cfg: Union[Number, dict, str] = 1.0, 
        # shift_cfg: Union[bool, str, dict] = True, 
        random_shift_per_pixel: bool = True, 
        ensure_pixel_interval: bool = False, # Ensure the pixel interval sampled is always 1
        
        device=None, 
        ) -> None:
        
        self.device = device

        #---- Config patch size
        if num_pixels is None:
            if isinstance(patch_hw, Number):
                patch_hw = (patch_hw,patch_hw)
            H, W = int(patch_hw[0]), int(patch_hw[1])
            aspect_ratio = W/H
        else:
            H, W = get_patch_hw(num_pixels, aspect_ratio)
        self.HW = (H,W)
        self.aspect_ratio = aspect_ratio

        #---- 
        # torch.FloatTensor, shape=[H,W], pre-stored normalized pixel locations
        self.y, self.x = torch.meshgrid([
            torch.linspace(0, 1, self.HW[0]+2, device=self.device)[1:-1], 
            torch.linspace(0, 1, self.HW[1]+2, device=self.device)[1:-1]], indexing='ij')

        #---- Config patch scale
        if isinstance(scale_cfg, Number):
            scale_cfg = dict(type='fixed', scale=scale_cfg)
        elif isinstance(scale_cfg, str):
            scale_cfg = dict(type=scale_cfg)
        self.scale_cfg = scale_cfg
        scale_cfg = deepcopy(scale_cfg)
        scale_type = scale_cfg.pop('type').lower()
        if scale_type == 'fixed':
            scale_fn = functools.partial(get_scale_fixed, **scale_cfg)
        elif scale_type == 'random':
            scale_fn = functools.partial(get_scale_random, **scale_cfg)
        elif scale_type == 'annealed':
            scale_fn = functools.partial(get_scale_annealed, **scale_cfg)
        elif scale_type == 'annealed_random':
            scale_fn = functools.partial(get_scale_annealed_random, **scale_cfg)
        else:
            raise RuntimeError(f"Invalid scale_cfg['type']={scale_type}")
        self.scale_fn = scale_fn

        #---- Config patch shift
        self.random_shift_per_pixel = random_shift_per_pixel
        self.ensure_pixel_interval = ensure_pixel_interval


    def sample_pixels(self, it: int, HW0: Tuple[int,int], num_pixels: int = None, device=None):
        """_summary_

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: `xy` and `hw`
            - `x`, `y`: [H, W], torch.FloatTensor in range [0,1], The pixel locations
            - `h`, `w`: [H, W], torch.LongTensor where `h` in range [0,H] and `w` in range [0,W]
        """
        
        device = device or self.device
        
        #---- Original image size
        if isinstance(HW0, torch.Tensor):
            HW0 = HW0.tolist()
        H0, W0 = HW0 
        
        #---- The target patch size
        if num_pixels is None:
            H, W = self.HW
            y, x = self.y.to(device), self.x.to(device)
        else:
            H, W = get_patch_hw(num_pixels, W0.item()/H0.item())
            y, x = torch.meshgrid([
                torch.linspace(0, 1, H+2, device=device)[1:-1], 
                torch.linspace(0, 1, W+2, device=device)[1:-1]], indexing='ij')
        
        #---- Patch at a certain scale and a random location
        self.scale = scale = self.scale_fn(it)
        if scale != 1:
            offset_x, offset_y = (np.random.rand(2) - 0.5) * (1-scale)
            y = (y - 0.5) * scale + (0.5 + offset_x)
            x = (x - 0.5) * scale + (0.5 + offset_y)
        
        #---- Random shift for each pixel
        max_shift_x = (1./(W+1)) * scale / 2.
        max_shift_y = (1./(H+1)) * scale / 2.
        if self.random_shift_per_pixel and max(max_shift_x * W0, max_shift_y * H0) >= 1: # Larger than one pixel
            shift_x = max_shift_x * torch.empty([H,W], dtype=torch.float, device=device).uniform_(-1, 1)
            shift_y = max_shift_y * torch.empty([H,W], dtype=torch.float, device=device).uniform_(-1, 1)
            x = x + shift_x
            y = y + shift_y
        xy = torch.stack((x,y),dim=-1)
        
        h, w = (y * H0).long(), (x * W0).long()
        if self.ensure_pixel_interval:
            delta_h, delta_w = h.diff(dim=0), w.diff(dim=1)
            delta_h = torch.where(delta_h==0, delta_h + torch.randint_like(delta_h, 2)*2-1 , delta_h)
            delta_w = torch.where(delta_w==0, delta_w + torch.randint_like(delta_w, 2)*2-1 , delta_w)
            h[1:,:] = h[0:-1,:] + delta_h
            w[:,1:] = w[:,0:-1] + delta_w

        h.clamp_(0, H0-1)
        w.clamp_(0, W0-1)
        hw = torch.stack((h,w),dim=-1)
        
        dbg_infos = {'scale':scale}
        return xy, hw, dbg_infos

def get_stride_random(it: int, *, stride_list: List[int]):
    return random.choice(stride_list)

def get_stride_milestones(it: int, *, milestones: List[int], stride_list: List[int]):
    return get_anneal_val_milestones(it, milestones, stride_list)

def get_stride_milestones_random(it: int, *, milestones: List[int], stride_list: List[int]):
    """ `milestones` mark the interval endings (or, the beginnings of the next intervals)
    e.g. 
        milestones: [100, 300]
        stride_list: [4, 2, 1]
        >>> random choice from [4],  if it < 100
        >>> random choice from [4,2],  if 100 <= it < 300
        >>> random choice from [4,2,1], if it >= 300
    """
    length_list = np.arange(len(stride_list))+1
    length = get_anneal_val_milestones(it, milestones, length_list)
    return random.choice(stride_list[:length])

def get_stride_fixed(it: int, *, stride: int=1):
    return stride

class PatchSamplerSlideWindow(PatchSamplerBase):
    def __init__(
        self, 
        patch_hw: Union[Tuple[int,int], int] = 256, # [width, height]
        aspect_ratio: Number = 1.0, # width / height
        num_pixels: int = None, 
        
        stride_cfg: Union[Number, dict] = 1, # The stride of the sliding window
        
        device = None
        ) -> None:
        
        self.device = device

        #---- Config patch size
        if num_pixels is None:
            if isinstance(patch_hw, Number):
                patch_hw = (patch_hw,patch_hw)
            H, W = int(patch_hw[0]), int(patch_hw[1])
            aspect_ratio = W/H
        else:
            H, W = get_patch_hw(num_pixels, aspect_ratio)
        self.HW = (H,W)
        self.aspect_ratio = aspect_ratio
        
        # torch.LongTensor, shape=[H,W], pre-stored base h,w indices
        self.h, self.w = torch.meshgrid([
            torch.arange(self.HW[0], device=self.device, dtype=torch.long), 
            torch.arange(self.HW[1], device=self.device, dtype=torch.long)], indexing='ij')

        if isinstance(stride_cfg, Number):
            stride_cfg = dict(type='fixed', stride=stride_cfg)
        self.stride_cfg = stride_cfg
        stride_cfg = deepcopy(stride_cfg)
        stride_type = stride_cfg.pop('type').lower()
        if stride_type == 'fixed':
            stride_fn = functools.partial(get_stride_fixed, **stride_cfg)
        elif stride_type == 'milestones':
            stride_fn = functools.partial(get_stride_milestones, **stride_cfg)
        elif stride_type == 'random':
            stride_fn = functools.partial(get_stride_random, **stride_cfg)
        elif stride_type == 'milestones_random':
            stride_fn = functools.partial(get_stride_milestones_random, **stride_cfg)
        else:
            raise RuntimeError(f"Invalid stride_cfg['type']={stride_type}")
        self.stride_fn = stride_fn

    def sample_pixels(self, it: int, HW0: Tuple[int,int], device=None) -> Tuple[Tensor, Tensor, dict]:
        device = device or self.device
        
        #---- Original image size
        if isinstance(HW0, torch.Tensor):
            HW0 = HW0.tolist()
        H0, W0 = HW0
        
        #---- The target patch size
        H, W = self.HW
        h, w = self.h.to(device), self.w.to(device)
        stride = self.stride_fn(it=it)
        
        H_, W_ = H*stride, W*stride # Strided patch size
        assert (H_ <= H0) and (W_ <= W0), f"stride={stride} is too large for patch_hw={self.HW} when image_hw={HW0}"
        
        #---- Patch at a random location
        offset_h = np.random.randint(0, H0-H_) if (H0-H_ > 0) else 0
        offset_w = np.random.randint(0, W0-W_) if (W0-W_ > 0) else 0
        h = h * stride + offset_h
        w = w * stride + offset_w
        hw = torch.stack((h,w),dim=-1)
        
        # (+0.5): Snap to pixel centers
        y = (h.float()+0.5) / H0
        x = (w.float()+0.5) / W0
        xy = torch.stack((x,y),dim=-1)
        
        dbg_infos = {'stride':stride}
        return xy, hw, dbg_infos

def get_patch_sampler(type: Literal['flex_grid', 'slide_window'], **kwargs):
    if type == 'flex_grid':
        return PatchSamplerFlexGrid(**kwargs)
    elif type == 'slide_window':
        return PatchSamplerSlideWindow(**kwargs)
    else:
        raise RuntimeError(f"Invalid patch_sampler type={type}")

if __name__ == "__main__":
    def unit_test():
        from torch.utils.benchmark import Timer
        device = torch.device('cuda')
        patch_sampler = PatchSamplerFlexGrid(
            scale_cfg=dict(
                type='annealed_random', anneal_on_min=True, is_increasing=False, 
                anneal_type='linear', stop_it=1000), 
            shift_type='local', 
            num_pixels=8192, 
            aspect_ratio=1920/1280, 
            ensure_pixel_interval=False, 
            device=device)
        patch_sampler.sample_pixels(500, (1280, 1920))
        print(Timer(
            stmt="patch_sampler.sample_pixels(500, (1280, 1920))", 
            globals={'patch_sampler': patch_sampler}
        ).blocked_autorange())
        
    unit_test()