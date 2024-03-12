"""
@file   base.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Common API for neuralsim models
"""

from enum import Enum
from typing import List, Literal, Union

import torch
import torch.nn as nn

from nr3d_lib.logger import Logger
from nr3d_lib.config import ConfigDict
from nr3d_lib.models.spatial import AABBSpace
from nr3d_lib.models.model_base import ModelMixin
from nr3d_lib.models.autodecoder import AutoDecoderMixin

from app.resources import Scene, SceneNode

class AssetAssignment(Enum):
    OBJECT = 0
    SCENE = 1
    MULTI_OBJ_ONE_SCENE = 2
    MULTI_OBJ_MULTI_SCENE = 3
    MULTI_OBJ = 3 # The same with MULTI_OBJ_MULTI_SCENE
    MULTI_SCENE = 4
    MISC = 5

class AssetMixin:
    """
    Defines common APIs for drawable assets
    """
    id: str = None # The model's id. Unique identifier and key in the assetbank.
    
    assigned_to: AssetAssignment = AssetAssignment.OBJECT
    
    is_ray_query_supported: bool = False # Whether the model needs / supports xxx_ray_query
    is_batched_query_supported: bool # Whether the model supports batched_ray_query (batched render and inference)
    
    populate_cfg: dict = dict() # Config of `populate` process
    initialize_cfg: dict = dict() # Config for `initialize` process
    training_cfg: dict = None # Config of `training_setup` process (for optimizers)
    preload_cfg: dict = None

    def asset_init_config(
        self, 
        populate_cfg: dict = None, 
        initialize_cfg: dict = None, 
        training_cfg: dict = None, 
        preload_cfg: dict = None):
        self.populate_cfg = populate_cfg or dict()
        self.initialize_cfg = initialize_cfg or dict()
        self.training_cfg = training_cfg or dict()
        self.preload_cfg = preload_cfg or None

    def asset_populate(
        self, 
        scene: Union[Scene, List[Scene]] = None, 
        obj: Union[SceneNode, List[SceneNode]] = None, 
        config: dict = None, 
        device=None, **kwargs):
        """
        Populate the model. Called immediate after `__init__`
        """
        pass
    
    def asset_training_initialize(
        self, 
        scene: Union[Scene, List[Scene]] = None, 
        obj: Union[SceneNode, List[SceneNode]] = None, 
        config: dict = None, 
        logger: Logger=None, log_prefix: str='', **kwargs) -> bool:
        """
        Initialize (usally pretraining) a model.
        """
        updated = False
        return updated
    
    def asset_preload(self, *args, **kwargs):
        """
        Alternative for pretraining: load from a trained model.
        """
        pass
    
    def asset_val(
        self, 
        scene: Scene = None, obj: SceneNode = None, 
        it: int = ..., logger: Logger = None, log_prefix: str=''):
        """
        Validate current model
        """
        pass

    @classmethod
    def asset_compute_id(cls, scene: Scene = None, obj: SceneNode = None, class_name: str = None) -> str:
        """
        Calculate model's id accordingly # NOTE: This must be implemented.
        """
        raise NotImplementedError
        return f"{cls.__name__}#{class_name or obj.class_name}#{scene.id}#{obj.id}"

class AssetModelMixin(AssetMixin, ModelMixin):
    """
    Everything that can be put into the assetbank should be an AssetModelMixin
    """
    pass

def wrap_an_instance(o):
    pass

class DummyBox(AssetModelMixin, nn.Module):
    """
    A dummy model that only has space
    """
    is_ray_query_supported = False
    assigned_to = AssetAssignment.OBJECT
    def __init__(self, bounding_size: float = 2.0, device=None) -> None:
        super().__init__()
        self.space = AABBSpace(bounding_size=bounding_size, device=device)
        self.dummy = 2

    @classmethod
    def asset_compute_id(cls, scene: Scene = None, obj: SceneNode = None, class_name: str = None) -> str:
        return f"{cls.__name__}#{class_name or obj.class_name}#{scene.id}#{obj.id}"

class AD_DummyBox(AutoDecoderMixin, DummyBox):
    """
    A dummy shared model that only has space
    """
    is_ray_query_supported = False
    assigned_to = AssetAssignment.MULTI_OBJ
    @classmethod
    def asset_compute_id(cls, scene: Scene = None, obj: SceneNode = None, class_name: str = None) -> str:
        return f"{cls.__name__}#{class_name or obj.class_name}"

if __name__ == "__main__":
    def unit_test():
        from icecream import ic
        class TestModel(AssetModelMixin, nn.Module):
            def __init__(self) -> None:
                super().__init__()
        
        m1 = TestModel()
        m2 = TestModel()
        
        m1.space = AABBSpace(bounding_size=3.)
        m1.ray_query_cfg = ConfigDict(hello="world")
        ic(getattr(m1, 'space', None))
        ic(getattr(m2, 'space', None)) # Still no attr
        ic(getattr(m1, 'ray_query_cfg', None))
        ic(getattr(m2, 'ray_query_cfg', None)) # Still be default {}
        
        m1.ray_query_cfg.update(test="nice")
        ic(getattr(m1, 'ray_query_cfg', None))
        ic(getattr(m2, 'ray_query_cfg', None)) # Still be default {}
        
        model_id = DummyBox.asset_compute_id(scene=Scene("hello"), obj=SceneNode("world"))
        ic(model_id)

        class TestSharedModel(AssetModelMixin, nn.Module):
            assigned_to = AssetAssignment.MULTI_OBJ
        
        m3 = TestSharedModel()
        ic(AssetModelMixin.assigned_to)
        ic(TestSharedModel.assigned_to)
        ic(m3.assigned_to)
        m3.assigned_to = AssetAssignment.SCENE 
        ic(m3.assigned_to)
        ic(TestSharedModel.assigned_to) # Remains true

    unit_test()