"""
@file   image_embeddings.py
@author Nianchen Deng, Shanghai AI Lab
@brief  Learnable image embeddings
"""

__all__ = [
    'ImageEmbeddings'
]

import torch
import torch.nn as nn
from nr3d_lib.config import ConfigDict
from nr3d_lib.models.embeddings import SeqEmbedding

from nr3d_lib.utils import torch_dtype
from nr3d_lib.fmt import log

from app.resources import Scene, SceneNode
from app.models.asset_base import AssetAssignment, AssetModelMixin


class ImageEmbeddings(AssetModelMixin, nn.ModuleDict):
    assigned_to = AssetAssignment.SCENE
    def __init__(
        self, 
        dims: int,
        ego_node_id: str=None, ego_class_name: str="Camera",
        weight_init: str="normal", weight_init_std: float=1.0,
        dtype=torch.float, device=None) -> None:
        super().__init__()

        self.dims = dims
        self.ego_node_id = ego_node_id
        self.ego_class_name = ego_class_name
        self.weight_init = weight_init
        self.weight_init_std = weight_init_std
        self.dtype = torch_dtype(dtype)
        self.set_device = device

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def asset_populate(
        self, 
        scene: Scene = None, obj: SceneNode = None, config: dict = None, 
        device=None, **kwargs):

        device = device or self.set_device
        self.set_device = device

        self.scene = scene
        if self.ego_node_id is not None:
            ego_node_list = [self.scene.all_nodes[self.ego_node_id]]
        elif self.ego_class_name is not None:
            ego_node_list = self.scene.all_nodes_by_class_name[self.ego_class_name]
        else:
            raise RuntimeError(
                f"Invalid combination of arguments ego_node_id={self.ego_node_id}, "
                f"ego_class_name={self.ego_class_name}")

        self.exposures = nn.ModuleDict()
        for ego_node in ego_node_list:
            # NOTE: Different nodes might have different frame lengths
            embedding_weight = torch.empty(len(ego_node.frame_global_ts), self.dims, dtype=self.dtype)
            if self.weight_init == "uniform":
                embedding_weight.uniform_(-self.weight_init_std, self.weight_init_std)
            elif self.weight_init == "normal":
                embedding_weight.normal_(0., self.weight_init_std)
            elif self.weight_init == "zero":
                embedding_weight.zero_()
            else:
                raise ValueError("Unknown weight initial method")
            self[ego_node.id] = SeqEmbedding(ego_node.frame_global_ts, v_keyframes=embedding_weight, dim=self.dims, dtype=self.dtype, device=device)
        log.info(f"{self.scene.id} create image embeddings for {[node.id for node in ego_node_list]}")

    @classmethod
    def asset_compute_id(cls, scene: Scene = None, obj: SceneNode = None, class_name: str = None) -> str:
        return f"{cls.__name__}#{scene.id}"