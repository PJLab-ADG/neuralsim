"""
@file   nerf.py
@author Jianfei Guo, Shanghai AI Lab
@brief  neuralsim's API for NeRF models.
"""

__all__ = [
    'NeRFObj', 
    'LoTDNeRFObj', 
    'LoTDNeRFStreet', 
    'LoTDNeRFDistant', 
    'PermutoNeRFDistant', 
    'NeRFDistant'
]

from logging import Logger
import torch
import torch.nn as nn

from nr3d_lib.config import ConfigDict
from nr3d_lib.utils import check_to_torch

from nr3d_lib.models.embedders import get_embedder
from nr3d_lib.models.attributes import TransformMat4x4

from nr3d_lib.models.fields.nerf import NeRFModel, LoTDNeRFModel
from nr3d_lib.models.fields_distant.nerf import NeRFDistantModel, LoTDNeRFDistantModel, PermutoNeRFDistantModel

from app.models.asset_base import AssetAssignment, AssetMixin
from app.resources import Scene, SceneNode


class LoTDNeRFObj(AssetMixin, LoTDNeRFModel):
    """
    NeRF network for single object-centric scene or indoor scene, represented by LoTD encodings.
    
    MRO:
    -> AssetMixin
    -> LoTDNeRFModel
    -> NeRFRendererMixin
    -> LoTDNeRF
    -> ModelMixin
    -> nn.Module
    """
    assigned_to = AssetAssignment.OBJECT
    is_ray_query_supported = True
    @torch.no_grad()
    def asset_populate(
        self, 
        scene: Scene = None, obj: SceneNode = None, config: dict = None, 
        device=None, **kwargs):
        super().populate(device=device)

    def asset_training_initialize(self, scene: Scene, obj: SceneNode, config: dict, logger: Logger=None, log_prefix: str=None):
        return super().training_initialize(config, logger=logger, log_prefix=log_prefix)

    @classmethod
    def asset_compute_id(cls, scene: Scene = None, obj: SceneNode = None, class_name: str = None) -> str:
        return f"{cls.__name__}#{class_name or obj.class_name}#{scene.id}#{obj.id}"

class NeRFObj(AssetMixin, NeRFModel):
    """
    NeRF network for single object-centric scene or indoor scene, represented by MLP.
    
    MRO:
    -> AssetMixin
    -> NeRFModel
    -> NeRFRendererMixin
    -> EmbededNeRF
    -> ModelMixin
    -> nn.Module
    """
    assigned_to = AssetAssignment.OBJECT
    is_ray_query_supported = True
    @torch.no_grad()
    def asset_populate(
        self, 
        scene: Scene = None, obj: SceneNode = None, config: dict = None, 
        device=None, **kwargs):
        super().populate(device=device)

    def asset_training_initialize(self, scene: Scene, obj: SceneNode, config: dict, logger: Logger=None, log_prefix: str=None):
        return super().training_initialize(config, logger=logger, log_prefix=log_prefix)

    @classmethod
    def asset_compute_id(cls, scene: Scene = None, obj: SceneNode = None, class_name: str = None) -> str:
        return f"{cls.__name__}#{class_name or obj.class_name}#{scene.id}#{obj.id}"

class LoTDNeRFStreet(AssetMixin, LoTDNeRFModel):
    """
    NeRF network for single street-view scene, represented by LoTD encodings.
    
    MRO:
    -> AssetMixin
    -> LoTDNeRFModel
    -> NeRFRendererMixin
    -> LoTDNeRF
    -> ModelMixin
    -> nn.Module
    """
    assigned_to = AssetAssignment.OBJECT
    is_ray_query_supported = True
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
        
        super().populate(aabb=aabb, device=device)

    @classmethod
    def asset_compute_id(cls, scene: Scene = None, obj: SceneNode = None, class_name: str = None) -> str:
        return f"{cls.__name__}#{class_name or obj.class_name}#{scene.id}#{obj.id}"

    def asset_training_initialize(self, scene: Scene, obj: SceneNode, config: dict, logger: Logger=None, log_prefix: str=None):
        return super().training_initialize(config, logger=logger, log_prefix=log_prefix)

class LoTDNeRFDistant(AssetMixin, LoTDNeRFDistantModel):
    """
    NeRF++ network for distant-view models, represented by LoTD encodings.
    
    MRO:
    -> AssetMixin
    -> LoTDNeRFDistantModel
    -> NeRFRendererMixinDistant
    -> LoTDNeRF
    -> ModelMixin
    -> nn.Module
    """
    assigned_to = AssetAssignment.OBJECT
    is_ray_query_supported = True
    @torch.no_grad()
    def asset_populate(
        self, 
        scene: Scene = None, obj: SceneNode = None, config: dict = None, 
        device=None, **kwargs):
        if cr_obj_id:=config.get('cr_obj_id', None):
            self.cr_obj = scene.all_nodes[cr_obj_id]
        elif cr_obj_classname:=config.get('cr_obj_classname', None):
            self.cr_obj = scene.all_nodes_by_class_name[cr_obj_classname][0]
        elif config.get('ignore_cr', True):
            self.cr_obj = None
        else:
            raise RuntimeError(
                "Unable to decide close-range scene node for nerf++ distant-view model. \n"
                "Please specify one of `cr_obj_id` or `cr_obj_classname` in populate_cfg.")
        if self.cr_obj is not None:
            assert self.cr_obj.model is not None, f"Close-range object {self.cr_obj} has no model.\n"\
                "Please instantiate the close-range model in advance of the distant-view model."
            aabb = self.cr_obj.model.space.aabb
        else:
            aabb = None
        super().populate(aabb=aabb, device=device)

        # Determine whether renderer_mixin should include inf distance if not specified in config
        if self.include_inf_distance is None:
            self.include_inf_distance = "Sky" not in scene.all_nodes_by_class_name.keys()

    def ray_test(self, *args, **kwargs):
        if self.cr_obj is not None:
            # NOTE: nerf++ background should always directly use foreground's ray_test results.
            return self.cr_obj.model.ray_test(*args, **kwargs)

    def asset_training_initialize(self, scene: Scene, obj: SceneNode, config: dict, logger: Logger=None, log_prefix: str=None):
        return super().training_initialize(config, logger=logger, log_prefix=log_prefix)

    @classmethod
    def asset_compute_id(cls, scene: Scene = None, obj: SceneNode = None, class_name: str = None) -> str:
        return f"{cls.__name__}#{class_name or obj.class_name}#{scene.id}#{obj.id}"

class PermutoNeRFDistant(AssetMixin, PermutoNeRFDistantModel):
    """
    NeRF++ network for distant-view models, represented by Permutohedral lattice encodings.
    
    MRO:
    -> AssetMixin
    -> PermutoNeRFDistantModel
    -> NeRFRendererMixinDistant
    -> PermutoNeRF
    -> ModelMixin
    -> nn.Module
    """
    assigned_to = AssetAssignment.OBJECT
    is_ray_query_supported = True
    @torch.no_grad()
    def asset_populate(
        self, 
        scene: Scene = None, obj: SceneNode = None, config: dict = None, 
        device=None, **kwargs):
        if cr_obj_id:=config.get('cr_obj_id', None):
            self.cr_obj = scene.all_nodes[cr_obj_id]
        elif cr_obj_classname:=config.get('cr_obj_classname', None):
            self.cr_obj = scene.all_nodes_by_class_name[cr_obj_classname][0]
        elif config.get('ignore_cr', True):
            self.cr_obj = None
        else:
            raise RuntimeError(
                "Unable to decide close-range scene node for nerf++ distant-view model. \n"
                "Please specify one of `cr_obj_id` or `cr_obj_classname` in populate_cfg.")
        if self.cr_obj is not None:
            assert self.cr_obj.model is not None, f"Close-range object {self.cr_obj} has no model.\n"\
                "Please instantiate the close-range model in advance of the distant-view model."
            aabb = self.cr_obj.model.space.aabb
        else:
            aabb = None
        super().populate(aabb=aabb, device=device)

        # Determine whether renderer_mixin should include inf distance if not specified in config
        if self.include_inf_distance is None:
            self.include_inf_distance = "Sky" not in scene.all_nodes_by_class_name.keys()

    def ray_test(self, *args, **kwargs):
        if self.cr_obj is not None:
            # NOTE: nerf++ background should always directly use foreground's ray_test results.
            return self.cr_obj.model.ray_test(*args, **kwargs)

    def asset_training_initialize(self, scene: Scene, obj: SceneNode, config: dict, logger: Logger=None, log_prefix: str=None):
        return super().training_initialize(config, logger=logger, log_prefix=log_prefix)

    @classmethod
    def asset_compute_id(cls, scene: Scene = None, obj: SceneNode = None, class_name: str = None) -> str:
        return f"{cls.__name__}#{class_name or obj.class_name}#{scene.id}#{obj.id}"

class NeRFDistant(AssetMixin, NeRFDistantModel):
    """
    NeRF++ network for distant-view models, represented by MLP.
    
    MRO:
    -> AssetMixin
    -> NeRFDistantModel
    -> NeRFRendererMixinDistant
    -> EmbededNeRF
    -> ModelMixin
    -> nn.Module
    """
    assigned_to = AssetAssignment.OBJECT
    is_ray_query_supported = True
    @torch.no_grad()
    def asset_populate(
        self, 
        scene: Scene = None, obj: SceneNode = None, config: dict = None, 
        device=None, **kwargs):
        if cr_obj_id:=config.get('cr_obj_id', None):
            self.cr_obj = scene.all_nodes[cr_obj_id]
        elif cr_obj_classname:=config.get('cr_obj_classname', None):
            self.cr_obj = scene.all_nodes_by_class_name[cr_obj_classname][0]
        elif config.get('ignore_cr', True):
            self.cr_obj = None
        else:
            raise RuntimeError(
                "Unable to decide close-range scene node for nerf++ distant-view model. \n"
                "Please specify one of `cr_obj_id` or `cr_obj_classname` in populate_cfg.")
        
        aabb = self.cr_obj.model.space.aabb if self.cr_obj is not None else None
        super().populate(aabb=aabb, device=device)

        # Determine whether renderer_mixin should include inf distance if not specified in config
        if self.include_inf_distance is None:
            self.include_inf_distance = (len(scene.get_drawable_groups_by_class_name('Sky', only_valid=False))>0)

    def asset_training_initialize(self, scene: Scene, obj: SceneNode, config: dict, logger: Logger=None, log_prefix: str=None):
        return super().training_initialize(config, logger=logger, log_prefix=log_prefix)

    @classmethod
    def asset_compute_id(cls, scene: Scene = None, obj: SceneNode = None, class_name: str = None) -> str:
        return f"{cls.__name__}#{class_name or obj.class_name}#{scene.id}#{obj.id}"
