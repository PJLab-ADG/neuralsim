"""
@file   repr_compose_renderer.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Joint rendering of multiple objects in one scene.
        Implements a renderer that:
            first composes one holistic representation from each object representation, 
            then applies ray tracing or rasterization to volume render.
        
        FEATURES:
        
"""

import itertools
import functools
import numpy as np
from typing import Any, Dict, Tuple, Union, List

import torch
import torch.nn as nn

from nr3d_lib.profile import profile

from app.models.asset_base import AssetAssignment, AssetModelMixin
from app.resources import AssetBank, Scene, SceneNode, namedtuple_ind_id_obj
from app.renderers.utils import rotate_volume_buffer_nablas, prepare_empty_rendered
from app.resources.observers import Camera, MultiCamBundle, Lidar, RaysLidar, MultiRaysLidarBundle
from app.renderers.render_parallel import render_parallel, render_parallel_with_replicas, EvalParallelWrapper

class ReprComposeRenderer(nn.Module):
    """
    Joint rendering of multiple objects in one scene.
    Implements a renderer that:
        first composes one holistic representation from each object representation, 
        then applies ray tracing or rasterization to volume render.
    """
    
    @profile
    def compose_repr(self, drawables: List[SceneNode]):
        """
        Compose a holistic representation from all the seperate mode reprs from each node.
        The underlying representation can be anything that is "composable" (i.e. has the definition of compose() method)
        For example:
        - Composing multiple object's feature grids together into one holistic feature grid. e.g. unisim / GIRAFFE
        - Composing multiple objects' 3D gaussians or other types of kernels info one holistic group of kernels.
        """
        pass

    @profile
    def view_query(
        self, 
        observer: Camera, 
        #---- Keyword arguments
        drawable_ids: List[str] = None, 
        scene: Scene = ..., 
        #---- Keyword arguments (View query configs)
        ):
        """
        Rasterize the composed holistic representation.
        """
        assert isinstance(observer, Camera), "view_query() only supports observer type=Camera"
        assert scene.i_is_single, "view_query() requires the scene to be frozen at single frame index / timestamp"

        if drawable_ids is None:
            drawables = observer.filter_drawable_groups(scene.get_drawables())
        else:
            drawables = scene.drawables[drawable_ids]

        """
        Consider the behavior here that falls back to buffer compose -> various composed representations.
        """
        

    @profile
    def ray_query(
        self, 
        #---- Tensor inputs
        rays_o: torch.Tensor, 
        rays_d: torch.Tensor, 
        rays_ts: torch.Tensor = None, 
        rays_pix: torch.Tensor = None, 
        *, 
        #---- Keyword arguments
        scene: Scene, 
        observer: Union[Camera, MultiCamBundle, Lidar, RaysLidar, MultiRaysLidarBundle]=None, 
        ):
        """
        Ray trace the composed holistic representation.
        """

