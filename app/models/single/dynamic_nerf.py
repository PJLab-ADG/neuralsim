
__all__ = [
    'EmerNerfStreet', 
    'EmerNerfStreetOnlyDynamic'
]

from typing import List
import torch
import torch.nn as nn

from nr3d_lib.logger import Logger
from nr3d_lib.config import ConfigDict
from nr3d_lib.utils import check_to_torch

from nr3d_lib.models.embedders import get_embedder
from nr3d_lib.models.attributes import TransformMat4x4
from nr3d_lib.models.accelerations import get_accel_class, accel_types_dynamic
from nr3d_lib.models.fields_dynamic.nerf import EmerNeRFModel, EmerNeRFOnlyDynamicModel

from app.models.asset_base import AssetAssignment, AssetMixin
from app.resources import Scene, SceneNode

class EmerNerfStreet(AssetMixin, EmerNeRFModel):
    """
    NeRF network for single street-view scene, represented by LoTD encodings.
    
    MRO:
    -> AssetMixin
    -> EmerNeRFModel
    -> EmerNeRF
    -> ModelMixin
    -> nn.Module
    """
    assigned_to = AssetAssignment.OBJECT
    is_ray_query_supported = True
    use_ts: bool = True
    use_view_dirs: bool = True
    
    """ Asset functions """
    @torch.no_grad()
    def asset_populate(
        self, 
        scene: Scene = None, obj: SceneNode = None, config: dict = None, 
        device=None, **kwargs):
        """
        1. Use the range of observers in the scene to determine the pose and scale of the obj, so that the network input is automatically unit input
            For more descriptions, please refer to StreetSurf paper section 3.1
        2. If there is a need to change or additionally assign some attributes to the obj
        """
        ts_keyframes = obj.frame_global_ts.data.clone() # No gradients
        
        with torch.no_grad():
            ret = scene.process_observer_infos(far_clip=config['extend_size'])
            xyz_extend = ret.all_frustum_pts.view(-1,3)

        # NOTE: Apply pre-computed transform from dataset (usually for orientation alignment)
        scene.frozen_at_global_frame(0)
        xyz_extend = obj.world_transform.forward(xyz_extend, inv=True) / obj.scale.vec_3() # From world to street_obj
        scene.unfrozen()

        """
        The more scientific approach here is to find the smallest enclosing rectangular prism of these view cones (the axes of the BB do not necessarily have to be aligned).
        Currently, a rather crude method is used, which may waste a some space on certain sequences.
        """
        bmin = xyz_extend.min(0).values
        bmax = xyz_extend.max(0).values

        if config.get('use_cuboid', True):
            aabb = torch.stack([bmin, bmax], 0)
        else:
            radius = (bmax - bmin).max().item() / 2.
            center = (bmax + bmin) / 2.
            aabb = torch.stack([center - radius, center + radius], 0)

        #---- Dynamic Accel
        if self.accel_cfg is not None:
            accel_cls = get_accel_class(self.accel_cfg['type'])
            # NOTE: `config` is from `self.populate_cfg`
            accel_n_jump_frames = int(config.get('accel_n_jump_frames', 2))
            if accel_cls in accel_types_dynamic:
                self.accel_cfg.update(ts_keyframes=ts_keyframes[::accel_n_jump_frames].contiguous())

        #---- Model network's populate
        super().populate(aabb=aabb, ts_keyframes=ts_keyframes, device=device, **kwargs)

    @classmethod
    def asset_compute_id(cls, scene: Scene = None, obj: SceneNode = None, class_name: str = None) -> str:
        return f"{cls.__name__}#{class_name or obj.class_name}#{scene.id}#{obj.id}"

    def asset_training_initialize(self, scene: Scene, obj: SceneNode, config: dict, logger: Logger=None, log_prefix: str=None):
        return super().training_initialize(config, logger=logger, log_prefix=log_prefix)

class EmerNerfStreetOnlyDynamic(AssetMixin, EmerNeRFOnlyDynamicModel):
    """
    
    MRO:
    -> AssetMixin
    -> EmerNeRFOnlyDynamicModel
    -> EmerNeRFOnlyDynamic
    -> ModelMixin
    -> nn.Module
    """
    assigned_to = AssetAssignment.OBJECT
    is_ray_query_supported = True
    use_ts: bool = True
    use_view_dirs: bool = True
    
    """ Asset functions """
    @torch.no_grad()
    def asset_populate(
        self, 
        scene: Scene = None, obj: SceneNode = None, config: dict = None, 
        device=None, **kwargs):
        """
        1. Use the range of observers in the scene to determine the pose and scale of the obj, so that the network input is automatically unit input
            For more descriptions, please refer to StreetSurf paper section 3.1
        2. If there is a need to change or additionally assign some attributes to the obj
        """
        ts_keyframes = obj.frame_global_ts.data.clone() # No gradients
        
        with torch.no_grad():
            ret = scene.process_observer_infos(far_clip=config['extend_size'])
            xyz_extend = ret.all_frustum_pts.view(-1,3)

        # NOTE: Apply pre-computed transform from dataset (usually for orientation alignment)
        scene.frozen_at_global_frame(0)
        xyz_extend = obj.world_transform.forward(xyz_extend, inv=True) / obj.scale.vec_3() # From world to street_obj
        scene.unfrozen()

        """
        The more scientific approach here is to find the smallest enclosing rectangular prism of these view cones (the axes of the BB do not necessarily have to be aligned).
        Currently, a rather crude method is used, which may waste a some space on certain sequences.
        """
        bmin = xyz_extend.min(0).values
        bmax = xyz_extend.max(0).values

        if config.get('use_cuboid', True):
            aabb = torch.stack([bmin, bmax], 0)
        else:
            radius = (bmax - bmin).max().item() / 2.
            center = (bmax + bmin) / 2.
            aabb = torch.stack([center - radius, center + radius], 0)

        #---- Dynamic Accel
        if self.accel_cfg is not None:
            accel_cls = get_accel_class(self.accel_cfg['type'])
            # NOTE: `config` is from `self.populate_cfg`
            accel_n_jump_frames = int(config.get('accel_n_jump_frames', 2))
            if accel_cls in accel_types_dynamic:
                self.accel_cfg.update(ts_keyframes=ts_keyframes[::accel_n_jump_frames].contiguous())

        #---- Model network's populate
        super().populate(aabb=aabb, ts_keyframes=ts_keyframes, device=device, **kwargs)

    @classmethod
    def asset_compute_id(cls, scene: Scene = None, obj: SceneNode = None, class_name: str = None) -> str:
        return f"{cls.__name__}#{class_name or obj.class_name}#{scene.id}#{obj.id}"

    def asset_training_initialize(self, scene: Scene, obj: SceneNode, config: dict, logger: Logger=None, log_prefix: str=None):
        return super().training_initialize(config, logger=logger, log_prefix=log_prefix)
