"""
@file   base.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Common API for neuralsim models
"""

from typing import Literal
from enum import Enum, auto

import torch
import torch.nn as nn

from nr3d_lib.logger import Logger
from nr3d_lib.config import ConfigDict
from nr3d_lib.models.base import ModelMixin
from nr3d_lib.models.spatial.aabb import AABBSpace
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
    
    populate_cfg: ConfigDict = ConfigDict() # Config of `populate` process
    initialize_cfg: ConfigDict = ConfigDict() # Config for `initialize` process
    optim_cfg: ConfigDict = None # Config of `get_param_group` process (for optimizers)
    preload_cfg: ConfigDict = None

    def init_asset_config(
        self, 
        populate_cfg: ConfigDict = None, 
        initialize_cfg: ConfigDict = None, 
        optim_cfg: ConfigDict = None, 
        preload_cfg: ConfigDict = None):
        self.populate_cfg = populate_cfg or ConfigDict()
        self.initialize_cfg = initialize_cfg or ConfigDict()
        self.optim_cfg = optim_cfg or ConfigDict()
        self.preload_cfg = preload_cfg or None

    def populate(
        self, scene: Scene = None, obj: SceneNode = None, config: ConfigDict = None, 
        dtype=torch.float, device=torch.device('cuda'), **kwargs):
        """
        Populate the model. Called immediate after `__init__`
        """
        pass
    
    def initialize(
        self, 
        scene: Scene = None, obj: SceneNode = None, config: ConfigDict = None, 
        logger: Logger=None, log_prefix: str='', **kwargs):
        """
        Initialize (usally pretraining) a model.
        """
        pass
    
    def preload(self, *args, **kwargs):
        """
        Alternative for pretraining: load from a trained model.
        """
        pass
    
    def val(
        self, scene: Scene = None, obj: SceneNode = None, 
        it: int = ..., logger: Logger = None, log_prefix: str=''):
        """
        Validate current model
        """
        pass

    @classmethod
    def compute_model_id(cls, scene: Scene = None, obj: SceneNode = None, class_name: str = None) -> str:
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
    def compute_model_id(cls, scene: Scene = None, obj: SceneNode = None, class_name: str = None) -> str:
        return f"{cls.__name__}#{class_name or obj.class_name}#{scene.id}#{obj.id}"

class AD_DummyBox(AutoDecoderMixin, DummyBox):
    """
    A dummy shared model that only has space
    """
    is_ray_query_supported = False
    assigned_to = AssetAssignment.MULTI_OBJ
    @classmethod
    def compute_model_id(cls, scene: Scene = None, obj: SceneNode = None, class_name: str = None) -> str:
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
        
        model_id = DummyBox.compute_model_id(scene=Scene("hello"), obj=SceneNode("world"))
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