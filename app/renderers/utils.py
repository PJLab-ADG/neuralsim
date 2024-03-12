"""
@file   utils.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Utilities for volume renderers
"""

__all__ = [
    'rotate_volume_buffer_nablas'
]

from typing import Tuple

import torch

from nr3d_lib.graphics.pack_ops import packed_matmul

def rotate_volume_buffer_nablas(rotation: torch.Tensor, nablas: torch.Tensor, volume_buffer: dict = None):
    if rotation.dim() == 2: # Single rotation
        nablas_in_world = (rotation * nablas.unsqueeze(-2)).sum(-1)
        return nablas_in_world
    else: # Multi-frame rotation (When obj is frozen at multi-frame)
        assert volume_buffer is not None, "Requires volume_buffer's `rays_inds_hit` and `pack_infos_hit` for rotating nablas"
        rotation_on_hitrays = rotation[volume_buffer['rays_inds_hit']]
        if (buffer_type:=volume_buffer['type']) == 'packed':
            nablas_in_world = packed_matmul(nablas, rotation_on_hitrays, volume_buffer['pack_infos_hit'])
        elif buffer_type == 'batched':
            nablas_in_world = (rotation_on_hitrays * nablas.unsqueeze(-2)).sum(-1)
        return nablas_in_world

def prepare_empty_rendered(
    prefix: Tuple[int], dtype=torch.float, device=None, 
    with_rgb=True, with_normal=True, with_feature_dim: int = 0):
    rendered = dict(
        mask_volume = torch.zeros(prefix, dtype=dtype, device=device),
        depth_volume = torch.zeros(prefix, dtype=dtype, device=device)
    )
    if with_rgb:
        rendered['rgb_volume'] = torch.zeros([*prefix, 3], dtype=dtype, device=device)
    if with_normal:
        rendered['normals_volume'] = torch.zeros([*prefix, 3], dtype=dtype, device=device)
    if with_feature_dim:
        rendered['feature_volume'] = torch.zeros([*prefix, with_feature_dim], dtype=dtype, device=device)
    return rendered

