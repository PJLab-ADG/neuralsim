"""
@file   color_transform.py
@author Nianchen Deng, Shanghai AI Lab
@brief  Learnable pixel transform module for image postprocessing
"""

__all__ = [
    'ColorTransform'
]

import torch
import torch.nn as nn

from nr3d_lib.utils import torch_dtype
from nr3d_lib.models.blocks import get_blocks

from app.resources import Scene, SceneNode
from app.models.base import AssetAssignment, AssetModelMixin

class ColorTransform(AssetModelMixin, nn.Module):
    assigned_to = AssetAssignment.MISC
    def __init__(self, embedding_dim: int, mode: str, dtype=torch.float, **block_params) -> None:
        super().__init__()
        self.dtype = torch_dtype(dtype)
        self.mode = mode
        
        if self.mode == "exposure" or self.mode == "exposure+brightness":
            return
        
        color_chns = 3
        coord_chns = 2
        n_affine_elements = 12
        if self.mode == "direct":
            decoder_input_chns = embedding_dim + color_chns + coord_chns
            decoder_output_chns = color_chns
        elif self.mode == "pixel_affine":
            decoder_input_chns = embedding_dim + coord_chns
            decoder_output_chns = n_affine_elements
        elif self.mode == "global_affine":
            decoder_input_chns = embedding_dim
            decoder_output_chns = n_affine_elements
        else:
            raise ValueError("Invalid value for argument \"mode\"")
        self.decoder = get_blocks(decoder_input_chns, decoder_output_chns, dtype=self.dtype, **block_params)

    def forward(self, h: torch.Tensor, xy: torch.Tensor, rgbs: torch.Tensor) -> torch.Tensor:
        """ Transform input colors by the affine matrices decoded from image embeddings.

        Args:
            h (torch.Tensor): [N, D] Features
            xy (torch.Tensor): [N, 2] pixel coordinates in range [0,1]
            rgbs (torch.Tensor): [N, 3] rgb colors in range [0,1]

        Returns:
            torch.Tensor: [N, 3] transformed rgb colors
        """
        if self.mode == "exposure":
            return rgbs * torch.pow(2., h)
        elif self.mode == "exposure+brightness":
            return rgbs * torch.pow(2., h[:, 0:1]) + h[:, 1:2]
        
        if self.mode == "direct":
            decoder_input = torch.cat([h, xy, rgbs], dim=-1)
        elif self.mode == "pixel_affine":
            decoder_input = torch.cat([h, xy], dim=-1)
        elif self.mode == "global_affine":
            decoder_input = h
        
        decoder_output = self.decoder(decoder_input)

        if self.mode == "direct":
            rgbs_output = decoder_output
        elif self.mode == "pixel_affine" or self.mode == "global_affine":
            affine_trs = decoder_output.reshape(-1, 3, 4) # (N, 3, 4)
            rgbs_output = (affine_trs[:, :3, :3] @ rgbs[..., None] + affine_trs[:, :3, 3:])[..., 0]
        return rgbs_output.to(rgbs.dtype)

    @classmethod
    def compute_model_id(cls, scene: Scene = None, obj: SceneNode = None, class_name: str = None) -> str:
        return f"{cls.__name__}#{scene.id}"