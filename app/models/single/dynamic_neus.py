"""
@file   neus.py
@author Jianfei Guo, Shanghai AI Lab
@brief  neuralsim's API for NeuS models.
"""

__all__ = [
    'DynamicPermutoConcatNeuSObj', 
]


import numpy as np
from typing import List

import torch
import torch.nn as nn

from nr3d_lib.logger import Logger
from nr3d_lib.config import ConfigDict
from nr3d_lib.models.embeddings import SeqEmbedding
from nr3d_lib.models.accelerations import get_accel_class, accel_types_dynamic
from nr3d_lib.models.fields_dynamic.neus import DynamicPermutoConcatNeuSModel

from app.models.asset_base import AssetAssignment, AssetMixin
from app.resources import Scene, SceneNode
from app.resources.observers import Camera

class DynamicPermutoConcatNeuSObj(AssetMixin, DynamicPermutoConcatNeuSModel):
    """
    MRO:
    -> AssetMixin
    -> DynamicPermutoConcatNeuSModel
    -> NeusRendererMixinDynamic
    -> DynamicPermutoConcatNeuS
    -> ModelMixin
    -> nn.Module
    """
    assigned_to = AssetAssignment.OBJECT
    is_ray_query_supported: bool = True
    use_ts: bool = True

    """ Asset functions """
    @classmethod
    def asset_compute_id(cls, scene: Scene = None, obj: SceneNode = None, class_name: str = None) -> str:
        return f"{cls.__name__}#{class_name or obj.class_name}#{scene.id}#{obj.id}"

    def asset_populate(
        self, 
        scene: Scene = None, obj: SceneNode = None, 
        config: dict = None, device=None, **kwargs):
        assert isinstance(obj, list) and isinstance(obj[0], SceneNode), f"Input `obj` for populate should be a list of SceneNode"
        
        ts_keyframes = obj.frame_global_ts.data.clone() # No gradients
        z_time_all = SeqEmbedding(ts_keyframes, **self.latents_cfg['z_time'], dtype=torch.float, device=device)
        self.z_time_all = z_time_all
        self.z_time_single = None
        
        #---- Dynamic Accel
        if self.accel_cfg is not None:
            accel_cls = get_accel_class(self.accel_cfg['type'])
            # NOTE: `config` is from `self.populate_cfg`
            accel_n_jump_frames = int(config.get('accel_n_jump_frames', 2))
            if accel_cls in accel_types_dynamic:
                self.accel_cfg.update(ts_keyframes=ts_keyframes[::accel_n_jump_frames].contiguous())
        
        #---- Model network's populate
        super().populate(config=config, device=device, **kwargs)

    def asset_training_initialize(self, scene: Scene, obj: SceneNode, config: dict, logger: Logger=None, log_prefix: str=None):
        # self.grad_guard_when_render.logger = logger
        # self.grad_guard_when_uniform.logger = logger
        return super().training_initialize(config, logger=logger, log_prefix=log_prefix)

    """ New define or model functions overwrite """
    def set_z_time(self, z_time_single: torch.Tensor):
        self.z_time_single = z_time_single
        assert z_time_single.dim() == 1, "Only support manually specifying one z_time"
    
    def clean_z_time(self):
        self.z_time_single = None
    
    def _check_or_get_z_time_per_x(
        self, x: torch.Tensor, ts: torch.Tensor = None, z_time: torch.Tensor = None
        ) -> torch.Tensor:
        x_prefix = [*x.shape[:-1]]
        z_time = self.z_time_all.get_z_per_input(x_prefix, ts_per_input=ts, z_single=self.z_time_single, z_per_input=z_time)
        return z_time

    def query_sdf(
        self, x: torch.Tensor, *, z_time: torch.Tensor = None, 
        ts: torch.Tensor = None, **kwargs):
        z_time = self._check_or_get_z_time_per_x(x, ts=ts, z_time=z_time)
        return super().query_sdf(x, z_time=z_time, ts=ts, **kwargs)

    def forward_sdf(
        self, x: torch.Tensor, *, z_time: torch.Tensor = None, 
        ts: torch.Tensor = None, **kwargs):
        z_time = self._check_or_get_z_time_per_x(x, ts=ts, z_time=z_time)
        return super().forward_sdf(x, z_time=z_time, ts=ts, **kwargs)

    def forward_sdf_nablas(
        self, x: torch.Tensor, *, z_time: torch.Tensor = None, 
        ts: torch.Tensor = None, **kwargs):
        z_time = self._check_or_get_z_time_per_x(x, ts=ts, z_time=z_time)
        return super().forward_sdf_nablas(x, z_time=z_time, ts=ts, **kwargs)

    def forward(
        self, x: torch.Tensor, *, z_time: torch.Tensor = None, 
        ts: torch.Tensor = None, **kwargs):
        z_time = self._check_or_get_z_time_per_x(x, ts=ts, z_time=z_time)
        return super().forward(x, z_time=z_time, ts=ts, **kwargs)

if __name__ == "__main__":
    def unit_test():
        pass
