__all__ = [
    "SimpleSky",
    "PureColorSky"
]

from typing import List

import torch
import torch.nn as nn
from app.resources import Scene, SceneNode

from nr3d_lib.models.blocks import get_blocks
from nr3d_lib.models.embedders import get_embedder

from app.models.asset_base import AssetAssignment, AssetModelMixin

class SimpleSky(AssetModelMixin, nn.Module):
    """
    A simple sky model represented by directional MLPs
    """
    assigned_to = AssetAssignment.OBJECT # NOTE: For now, sky is an unique object; might be changed to scene-level property
    is_ray_query_supported = False
    is_batched_query_supported = False
    def __init__(
        self, 
        dir_embed_cfg:dict={'type':'spherical', 'degree': 4}, 
        D=3, W=64, skips=[], activation='relu', output_activation='sigmoid', 
        n_appear_embedding: int = 0, appear_embed_cfg:dict={'type':'identity'}, 
        weight_norm=False, dtype=torch.float, device=None, use_tcnn_backend=False):
        super().__init__()

        dir_embed_cfg.setdefault('use_tcnn_backend', use_tcnn_backend)
        self.embed_fn_view, input_ch_views = get_embedder(dir_embed_cfg, 3)
        
        self.use_h_appear = n_appear_embedding > 0
        # h_appear
        if self.use_h_appear:
            self.embed_fn_appear, input_ch_h_appear = get_embedder(appear_embed_cfg, n_appear_embedding)
        else:
            input_ch_h_appear = 0
        
        self.blocks = get_blocks(
            input_ch_views + input_ch_h_appear, 3, 
            D=D,  W=W, skips=skips, activation=activation, output_activation=output_activation, 
            dtype=dtype, device=device, weight_norm=weight_norm, use_tcnn_backend=use_tcnn_backend)

    def forward(self, v: torch.Tensor, *, h_appear: torch.Tensor = None):
        network_input = self.embed_fn_view(v)
        if self.use_h_appear > 0:
            network_input = torch.cat([network_input, h_appear], dim=-1)
        return self.blocks(network_input)

    @classmethod
    def asset_compute_id(cls, scene: Scene = None, obj: SceneNode = None, class_name: str = None) -> str:
        return f"{cls.__name__}#{class_name or obj.class_name}#{scene.id}#{obj.id}"

class PureColorSky(AssetModelMixin, nn.Module):
    """
    A dummy pure color sky model
    """
    assigned_to = AssetAssignment.OBJECT
    is_ray_query_supported = False
    is_batched_query_supported = False
    def __init__(self, RGB: List[int]=[255,255,255]) -> None:
        super().__init__()
        self.register_buffer('RGB', torch.tensor(RGB, dtype=torch.float)/255., persistent=False)
    def forward(self, v: torch.Tensor, x: torch.Tensor=None):
        prefix = v.shape[:-1]
        return self.RGB.tile([*prefix,1])

    @classmethod
    def asset_compute_id(cls, scene: Scene = None, obj: SceneNode = None, class_name: str = None) -> str:
        return f"{cls.__name__}#{class_name or obj.class_name}#{scene.id}#{obj.id}"