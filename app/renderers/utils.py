"""
@file   utils.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Utilities for volume renderers
"""

__all__ = [
    'rotate_volume_buffer_nablas'
]

import torch

from nr3d_lib.render.pack_ops import packed_matmul

def rotate_volume_buffer_nablas(rotation: torch.Tensor, nablas: torch.Tensor, volume_buffer: dict = None):
    if rotation.dim() == 2: # Single rotation
        nablas_in_world = (rotation * nablas.unsqueeze(-2)).sum(-1)
        return nablas_in_world
    else: # Multi-frame rotation (When obj is frozen at multi-frame)
        assert volume_buffer is not None, "Requires volume_buffer's `ray_inds_hit` and `pack_infos_hit` for rotating nablas"
        rotation_on_hitrays = rotation[volume_buffer['ray_inds_hit']]
        if (buffer_type:=volume_buffer['buffer_type']) == 'packed':
            nablas_in_world = packed_matmul(nablas, rotation_on_hitrays, volume_buffer['pack_infos_hit'])
        elif buffer_type == 'batched':
            nablas_in_world = (rotation_on_hitrays * nablas.unsqueeze(-2)).sum(-1)
        return nablas_in_world