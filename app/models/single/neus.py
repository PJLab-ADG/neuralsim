"""
@file   neus.py
@author Jianfei Guo, Shanghai AI Lab
@brief  neuralsim's API for NeuS models.
"""

__all__ = [
    'LoTDNeuSObj', 
    'MLPNeuSObj',
    'PermutoNeuSObj', 
    'LoTDNeuSStreet', 
    'MLPNeuSStreet', 
]


import numpy as np
from typing import List

import torch
import torch.nn as nn

from nr3d_lib.logger import Logger
from nr3d_lib.config import ConfigDict
from nr3d_lib.models.fields.neus import MlpPENeuSModel, LoTDNeuSModel, PermutoNeuSModel
from nr3d_lib.models.fields.sdf import pretrain_sdf_capsule, pretrain_sdf_road_surface

from app.models.asset_base import AssetAssignment, AssetMixin
from app.resources import Scene, SceneNode
from app.resources.observers import Camera

class LoTDNeuSObj(AssetMixin, LoTDNeuSModel):
    """
    NeuS network for single object-centric scene or indoor scene, represented by LoTD encodings
    
    MRO:
    -> LoTDNeuSObj
    -> AssetMixin
    -> LoTDNeuSModel
    -> NeusRendererMixin
    -> LoTDNeuS
    -> ModelMixin
    -> nn.Module
    """
    assigned_to = AssetAssignment.OBJECT
    is_ray_query_supported = True
    
    """ Asset functions """
    @classmethod
    def asset_compute_id(cls, scene: Scene = None, obj: SceneNode = None, class_name: str = None) -> str:
        return f"{cls.__name__}#{class_name or obj.class_name}#{scene.id}#{obj.id}"
    
    @torch.no_grad()
    def asset_populate(
        self, 
        scene: Scene = None, obj: SceneNode = None, config: dict = None, 
        device=None, **kwargs):
        super().populate(device=device)
    
    def asset_training_initialize(self, scene: Scene, obj: SceneNode, config: dict, logger: Logger=None, log_prefix: str=None):
        # self.grad_guard_when_render.logger = logger
        # self.grad_guard_when_uniform.logger = logger
        return super().training_initialize(config, logger=logger, log_prefix=log_prefix)

class PermutoNeuSObj(AssetMixin, PermutoNeuSModel):
    """
    NeuS network for single object-centric scene or indoor scene, represented by Permutohedral lattice encodings
    
    MRO:
    -> LoTDNeuSObj
    -> AssetMixin
    -> PermutoNeuSModel
    -> NeusRendererMixin
    -> PermutoNeuS
    -> ModelMixin
    -> nn.Module
    """
    assigned_to = AssetAssignment.OBJECT
    is_ray_query_supported = True

    """ Asset functions """
    @classmethod
    def asset_compute_id(cls, scene: Scene = None, obj: SceneNode = None, class_name: str = None) -> str:
        return f"{cls.__name__}#{class_name or obj.class_name}#{scene.id}#{obj.id}"
    
    @torch.no_grad()
    def asset_populate(
        self, 
        scene: Scene = None, obj: SceneNode = None, config: dict = None, 
        device=None, **kwargs):
        super().populate(device=device)
    
    def asset_training_initialize(self, scene: Scene, obj: SceneNode, config: dict, logger: Logger=None, log_prefix: str=None):
        # self.grad_guard_when_render.logger = logger
        # self.grad_guard_when_uniform.logger = logger
        return super().training_initialize(config, logger=logger, log_prefix=log_prefix)

class MLPNeuSObj(AssetMixin, MlpPENeuSModel):
    """
    NeuS network for single object-centric or indoor scene, represented by MLP
    
    MRO:
    -> MLPNeuSObj
    -> AssetMixin
    -> MlpPENeuSModel
    -> NeusRendererMixin
    -> MlpPENeuS
    -> ModelMixin
    -> nn.Module
    """
    assigned_to = AssetAssignment.OBJECT
    is_ray_query_supported = True
    
    """ Asset functions """
    @classmethod
    def asset_compute_id(cls, scene: Scene = None, obj: SceneNode = None, class_name: str = None) -> str:
        return f"{cls.__name__}#{class_name or obj.class_name}#{scene.id}#{obj.id}"
    
    @torch.no_grad()
    def asset_populate(
        self, 
        scene: Scene = None, obj: SceneNode = None, config: dict = None, 
        device=None, **kwargs):
        super().populate(device=device)
    
    def asset_training_initialize(self, scene: Scene, obj: SceneNode, config: dict, logger: Logger=None, log_prefix: str=None):
        return super().training_initialize(config, logger=logger, log_prefix=log_prefix)

class LoTDNeuSStreet(AssetMixin, LoTDNeuSModel):
    """
    NeuS network for single street-view scene, reprensented by LoTD encodings
    
    MRO:
    -> LoTDNeuSStreet
    -> AssetMixin
    -> LoTDNeuSModel
    -> NeusRendererMixin
    -> LoTDNeuS
    -> ModelMixin
    -> nn.Module
    """
    assigned_to = AssetAssignment.OBJECT
    is_ray_query_supported = True
    
    """ Asset functions """
    @classmethod
    def asset_compute_id(cls, scene: Scene = None, obj: SceneNode = None, class_name: str = None) -> str:
        return f"{cls.__name__}#{class_name or obj.class_name}#{scene.id}#{obj.id}"
    
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
        
        extend_size = config.get('extend_size')
        use_cuboid = config.get('use_cuboid', True)
        
        frustum_extend_pts = []
        cams = scene.get_cameras(only_valid=False)
        
        scene.frozen_at_full_global_frame()
        for cam in cams:
            frustum = cam.get_view_frustum_pts(near=0., far=extend_size)
            frustum_extend_pts.append(frustum)
        frustum_extend_pts = torch.stack(frustum_extend_pts, 0)
        scene.unfrozen()
        
        xyz_extend = frustum_extend_pts.view(-1,3)
        
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
        
        if use_cuboid:
            print(f"=> {obj.id} using cuboid space")
            aabb = torch.stack([bmin, bmax], 0)
        else:
            print(f"=> {obj.id} using cubic space")
            radius = (bmax - bmin).max().item()
            center = (bmax + bmin) / 2.
            aabb = torch.stack([center - radius, center + radius], 0)
        
        super().populate(aabb=aabb, device=device)

    def asset_training_initialize(self, scene: Scene, obj: SceneNode, config: dict, logger=None, log_prefix: str=None):
        if config is None: 
            config = dict()
        
        updated = False
        geo_init_method = self.implicit_surface.geo_init_method
        if geo_init_method == 'pretrain' or ('pretrain_after' in geo_init_method):
            if not self.implicit_surface.is_pretrained:
                if log_prefix is None:
                    log_prefix = obj.id
                
                config = config.copy()
                obs_ref = config.pop('obs_ref')
                target_shape = config.pop('target_shape', 'capsule')
                
                with torch.no_grad():
                    obs = scene.observers[obs_ref]
                    scene.frozen_at_full_global_frame()
                    tracks = obs.world_transform.translation()
                    scene.unfrozen()
                    
                    scene.frozen_at_global_frame(0)
                    tracks_in_obj = obj.world_transform(tracks, inv=True) / obj.scale.vec_3()
                    scene.unfrozen()

                if target_shape == 'capsule':
                    pretrain_sdf_capsule(self.implicit_surface, tracks_in_obj, **config, logger=logger, log_prefix=log_prefix)
                elif target_shape == 'road_surface':
                    pretrain_sdf_road_surface(self.implicit_surface, tracks_in_obj, **config, logger=logger, log_prefix=log_prefix)
                else:
                    raise RuntimeError(f'Invalid target_shape={target_shape}')

                self.implicit_surface.is_pretrained = ~self.implicit_surface.is_pretrained
                updated = True
        
        if self.accel is not None:
            self.accel.init(self.query_sdf, logger=logger)
        
        return updated

    @torch.no_grad()
    def asset_val(self, scene: Scene = None, obj: SceneNode = None, it: int = ..., logger: Logger = None, log_prefix: str = ''):
        pass
        # mesh = self.accel.debug_get_mesh()
        # logger.add_open3d(scene.id, ".".join([obj.model.id, "accel"]), mesh, it)
        
        # # verts, faces = self.accel.debug_get_mesh()
        # mesh = geometries[0]
        # verts = torch.tensor(mesh.vertices, dtype=torch.float, device=self.device)
        # faces = torch.tensor(mesh.triangles, dtype=torch.float, device=self.device)
        # logger.add_mesh(scene.id, ".".join([obj.model.id, "accel"]), verts, faces=faces, it=it)

class MLPNeuSStreet(AssetMixin, MlpPENeuSModel):
    """
    NeuS network for single street-view scene, reprensented by LoTD encodings
    
    MRO:
    -> MLPNeuSStreet
    -> AssetMixin
    -> MlpPENeuSModel
    -> NeusRendererMixin
    -> MlpPENeuS
    -> ModelMixin
    -> nn.Module
    """
    assigned_to = AssetAssignment.OBJECT
    is_ray_query_supported = True

    """ Asset functions """
    @classmethod
    def asset_compute_id(cls, scene: Scene = None, obj: SceneNode = None, class_name: str = None) -> str:
        return f"{cls.__name__}#{class_name or obj.class_name}#{scene.id}#{obj.id}"

    @torch.no_grad()
    def asset_populate(
        self, 
        scene: Scene = None, obj: SceneNode = None, config: dict = None, 
        device=None, **kwargs):
        
        extend_size = config.get('extend_size')
        use_cuboid = config.get('use_cuboid', True)
        
        frustum_extend_pts = []
        cams = scene.get_cameras(only_valid=False)
        scene.frozen_at_full_global_frame()
        for cam in cams:
            frustum = cam.get_view_frustum_pts(near=0., far=extend_size)
            frustum_extend_pts.append(frustum)
        frustum_extend_pts = torch.stack(frustum_extend_pts, 0)
        scene.unfrozen()
        
        xyz_extend = frustum_extend_pts.view(-1,3)
        
        # NOTE: Apply pre-computed transform from dataset (usually for orientation alignment)
        scene.frozen_at_global_frame(0)
        xyz_extend = obj.world_transform.forward(xyz_extend, inv=True) / obj.scale.vec_3() # From world to street_obj
        scene.unfrozen()

        bmin = xyz_extend.min(0).values
        bmax = xyz_extend.max(0).values

        if use_cuboid:
            print(f"=> {obj.id} using cuboid space")
            aabb = torch.stack([bmin, bmax], 0)
        else:
            print(f"=> {obj.id} using cubic space")
            radius = (bmax - bmin).max().item()
            center = (bmax + bmin) / 2.
            aabb = torch.stack([center - radius, center + radius], 0)

        super().populate(aabb=aabb, device=device)

    def asset_training_initialize(self, scene: Scene, obj: SceneNode, config: dict, logger=None, log_prefix: str=None):
        if config is None: 
            config = dict()
        
        updated = False
        geo_init_method = self.implicit_surface.geo_init_method
        if geo_init_method == 'pretrain' or ('pretrain_after' in geo_init_method):
            if not self.implicit_surface.is_pretrained:
                if log_prefix is None:
                    log_prefix = obj.id
                
                config = config.copy()
                obs_ref = config.pop('obs_ref')
                target_shape = config.pop('target_shape', 'capsule')
                
                with torch.no_grad():
                    obs = scene.observers[obs_ref]
                    scene.frozen_at_full_global_frame()
                    tracks = obs.world_transform.translation()
                    scene.unfrozen()
                    
                    scene.frozen_at_global_frame(0)
                    tracks_in_obj = obj.world_transform(tracks, inv=True) / obj.scale.vec_3()
                    scene.unfrozen()

                if target_shape == 'capsule':
                    pretrain_sdf_capsule(self.implicit_surface, tracks_in_obj, **config, logger=logger, log_prefix=log_prefix)
                elif target_shape == 'road_surface':
                    pretrain_sdf_road_surface(self.implicit_surface, tracks_in_obj, **config, logger=logger, log_prefix=log_prefix)
                else:
                    raise RuntimeError(f'Invalid target_shape={target_shape}')

                self.implicit_surface.is_pretrained = ~self.implicit_surface.is_pretrained
                updated = True
        
        if self.accel is not None:
            self.accel.init(self.query_sdf, logger=logger)
        
        return updated

if __name__ == "__main__":
    def unit_test():
        pass