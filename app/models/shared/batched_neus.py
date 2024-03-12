"""
@file   neus.py
@author Jianfei Guo, Shanghai AI Lab
@brief  neuralsim's API for categorical NeuS models.
"""

__all__ = [
    'StyleLoTDNeuSObj', 
    'AD_StyleLoTDNeuSObj', 
    'StyleNeuSLXYObj', 
    'AD_StyleNeuSLXYObj', 
    "AD_GenerativePermutoConcatNeuSObj", 
]

import functools
import numpy as np
from numbers import Number
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn

from nr3d_lib.logger import Logger
from nr3d_lib.config import ConfigDict
from nr3d_lib.utils import check_to_torch, check_per_batch_tensors
from nr3d_lib.models.embeddings import Embedding
from nr3d_lib.models.model_base import ModelMixin
from nr3d_lib.models.autodecoder import AutoDecoderMixin
from nr3d_lib.models.accelerations import get_accel_class, accel_types_batched, OccGridAccelBatched_Ema
from nr3d_lib.models.fields_conditional.neus import StyleLoTDNeuSModel, StyleNeuSLXYModel, GenerativePermutoConcatNeuSModel

from app.resources import Scene, SceneNode
from app.models.asset_base import AssetMixin, AssetAssignment

class StyleLoTDNeuSObj(AssetMixin, StyleLoTDNeuSModel):
    """
    StyleLoTDNeuS, generative model, for infinite objects.
    
    MRO:
    -> AssetMixin
    -> NeuSRendererMixinBatched
    -> StyleLoTDNeuS
    -> ModelMixin
    -> nn.Module
    """
    assigned_to = AssetAssignment.MULTI_OBJ
    is_ray_query_supported = True
    is_batched_query_supported = True
    
    """ Asset functions """
    @classmethod
    def asset_compute_id(cls, scene: Scene = None, obj: SceneNode = None, class_name: str = None) -> str:
        return f"{cls.__name__}#{class_name or obj.class_name}"
    
    """ New define or model functions overwrite """
    def set_condition(self, batched_infos: dict):
        if 'z_ins' in batched_infos:
            # Directly specify `z_ins_per_batch`
            z_ins_per_batch = batched_infos['z_ins']
        else:
            z_ins_per_batch = None
        self.z_ins_per_batch = z_ins_per_batch
        
        #---- Model network's set_condition
        super().set_condition(z=z_ins_per_batch)

    def clean_condition(self):
        super().clean_condition()

class AD_StyleLoTDNeuSObj(AutoDecoderMixin, AssetMixin, StyleLoTDNeuSModel):
    """
    StyleLoTDNeuS, generative model, auto-decoder framework for a specific list of objects.
    
    MRO:
    -> AutoDecoderMixin
    -> AssetMixin
    -> NeuSRendererMixinBatched
    -> StyleLoTDNeuS
    -> ModelMixin
    -> nn.Module
    """
    assigned_to = AssetAssignment.MULTI_OBJ
    is_ray_query_supported = True
    is_batched_query_supported = True

    @property
    def z_ins_all(self) -> Embedding:
        return self._latents['z_ins'] # Created by autodecoder_populate()

    """ Asset functions """
    @classmethod
    def asset_compute_id(cls, scene: Scene = None, obj: SceneNode = None, class_name: str = None) -> str:
        return f"{cls.__name__}#{class_name or obj.class_name}"
    
    def asset_populate(
        self, 
        scene: List[Scene] = None, obj: List[SceneNode] = None, 
        config: dict = None, device=None, **kwargs):
        assert isinstance(obj, list) and isinstance(obj[0], SceneNode), f"Input `obj` for populate should be a list of SceneNode"
        obj_full_unique_ids = [o.full_unique_id for o in obj]
        key_maps = {'ins_id': obj_full_unique_ids}
        self.num_objs = len(obj_full_unique_ids)
        z_ins_all = Embedding(self.num_objs, **self.latents_cfg['z_ins'], dtype=torch.float, device=device)
        latent_maps = {'z_ins': z_ins_all}
        n_latent_dim = z_ins_all.embedding_dim
        
        self.ins_inds_per_batch = None
        self.z_ins_per_batch = None
        # self.z_ins_all = None # Should be `self._latents['z_ins']`
        
        #---- Batched Accel
        if self.accel_cfg is not None:
            accel_cls = get_accel_class(self.accel_cfg['type'])
            if accel_cls in (OccGridAccelBatched_Ema, ):
                self.accel_cfg.update(num_batches=self.num_objs)
        #---- Autodecoder's populate
        super().autodecoder_populate(key_maps=key_maps, latent_maps=latent_maps)
        #---- Model network's populate
        super().populate(n_latent_dim=n_latent_dim, device=device, **kwargs)

    def asset_training_initialize(self, scene: Scene, obj: SceneNode, config: dict, logger: Logger=None, log_prefix: str=None):
        updated = super().training_initialize(config=config, logger=logger, log_prefix=log_prefix, skip_accel=True)
        if self.accel is not None:
            # NOTE: To init occ grid we need all instances
            ins_inds_all = torch.arange(self.num_objs, dtype=torch.long, device=self.device)
            self.set_condition({'ins_ind': ins_inds_all})
            self.accel.init(self.query_sdf, logger=logger)
            self.clean_condition()
        return updated

    """ New define or model functions overwrite """
    def set_condition(self, batched_infos: dict):
        if 'ins_id' in batched_infos:
            ins_ids = batched_infos['ins_id']
            if isinstance(ins_ids, str): ins_ids = [ins_ids]
            ins_inds_per_batch = [self._index_maps['ins_id'][i] for i in ins_ids]
            ins_inds_per_batch = check_to_torch(ins_inds_per_batch, dtype=torch.long, device=self.device)
        elif 'ins_ind' in batched_infos:
            ins_inds_per_batch = batched_infos['ins_ind']
        else:
            ins_inds_per_batch = None
        self.ins_inds_per_batch = ins_inds_per_batch
        
        if 'z_ins' in batched_infos:
            # Directly specify `z_ins_per_batch`
            z_ins_per_batch = batched_infos['z_ins']
        else:
            # Get `z_ins_per_batch` from `ins_id` list
            assert ins_inds_per_batch is not None, "Requires `ins_id` in batched_infos when no `z_ins` provided"
            z_ins_per_batch = self.z_ins_all.forward(ins_inds_per_batch)
        self.z_ins_per_batch = z_ins_per_batch
        
        batch_size = check_per_batch_tensors(ins_inds_per_batch, z_ins_per_batch)
        
        #---- Model network's set_condition
        super().set_condition(z=z_ins_per_batch, ins_inds_per_batch=ins_inds_per_batch)

    def clean_condition(self):
        super().clean_condition()

class StyleNeuSLXYObj(AssetMixin, StyleNeuSLXYModel):
    """
    StyleNeuSLXY, generative model, for infinite objects.
    
    MRO:
    -> AssetMixin
    -> NeuSRendererMixinBatched
    -> StyleNeuSLXY
    -> ModelMixin
    -> nn.Module
    """
    assigned_to = AssetAssignment.MULTI_OBJ
    is_ray_query_supported = True
    is_batched_query_supported = True
    
    """ Asset functions """
    @classmethod
    def asset_compute_id(cls, scene: Scene = None, obj: SceneNode = None, class_name: str = None) -> str:
        return f"{cls.__name__}#{class_name or obj.class_name}"
    
    """ New define or model functions overwrite """
    def set_condition(self, batched_infos: dict):
        if 'z_ins' in batched_infos:
            # Directly specify `z_ins_per_batch`
            z_ins_per_batch = batched_infos['z_ins']
        else:
            z_ins_per_batch = None
        self.z_ins_per_batch = z_ins_per_batch
        
        #---- Model network's set_condition
        super().set_condition(z=z_ins_per_batch)

    def clean_condition(self):
        super().clean_condition()

class AD_StyleNeuSLXYObj(AutoDecoderMixin, AssetMixin, StyleNeuSLXYModel):
    """
    StyleNeuSLXY, generative model, auto-decoder framework for a specific list of objects.
    
    MRO:
    -> AutoDecoderMixin
    -> AssetMixin
    -> NeuSRendererMixinBatched
    -> StyleNeuSLXY
    -> ModelMixin
    -> nn.Module
    """
    assigned_to = AssetAssignment.MULTI_OBJ
    is_ray_query_supported = True
    is_batched_query_supported = True
    
    @property
    def z_ins_all(self) -> Embedding:
        return self._latents['z_ins'] # Created by autodecoder_populate()

    """ Asset functions """
    @classmethod
    def asset_compute_id(cls, scene: Scene = None, obj: SceneNode = None, class_name: str = None) -> str:
        return f"{cls.__name__}#{class_name or obj.class_name}"
    
    def asset_populate(
        self, 
        scene: List[Scene] = None, obj: List[SceneNode] = None, 
        config: dict = None, device=None, **kwargs):
        
        assert isinstance(obj, list) and isinstance(obj[0], SceneNode), f"Input `obj` for populate should be a list of SceneNode"
        obj_full_unique_ids = [o.full_unique_id for o in obj]
        key_maps = {'ins_id': obj_full_unique_ids}
        self.num_objs = len(obj_full_unique_ids)
        z_ins_all = Embedding(self.num_objs, **self.latents_cfg['z_ins'], dtype=torch.float, device=device)
        latent_maps = {'z_ins': z_ins_all}
        n_latent_dim = z_ins_all.embedding_dim
        
        self.ins_inds_per_batch = None
        self.z_ins_per_batch = None
        # self.z_ins_all = None # Should be `self._latents['z_ins']`
        
        #---- Batched Accel
        if self.accel_cfg is not None:
            accel_cls = get_accel_class(self.accel_cfg['type'])
            if accel_cls in (OccGridAccelBatched_Ema, ):
                self.accel_cfg.update(num_batches=self.num_objs)
        #---- Autodecoder's populate
        super().autodecoder_populate(key_maps=key_maps, latent_maps=latent_maps)
        #---- Model network's populate
        super().populate(n_latent_dim=n_latent_dim, device=device, **kwargs)

    def asset_training_initialize(self, scene: Scene, obj: SceneNode, config: dict, logger: Logger=None, log_prefix: str=None):
        updated = super().training_initialize(config=config, logger=logger, log_prefix=log_prefix, skip_accel=True)
        if self.accel is not None:
            # NOTE: To init occ grid we need all instances
            ins_inds_all = torch.arange(self.num_objs, dtype=torch.long, device=self.device)
            self.set_condition({'ins_ind': ins_inds_all})
            self.accel.init(self.query_sdf, logger=logger)
            self.clean_condition()
        return updated

    """ New define or model functions overwrite """
    def set_condition(self, batched_infos: dict):
        if 'ins_id' in batched_infos:
            ins_ids = batched_infos['ins_id']
            if isinstance(ins_ids, str): ins_ids = [ins_ids]
            ins_inds_per_batch = [self._index_maps['ins_id'][i] for i in ins_ids]
            ins_inds_per_batch = check_to_torch(ins_inds_per_batch, dtype=torch.long, device=self.device)
        elif 'ins_ind' in batched_infos:
            ins_inds_per_batch = batched_infos['ins_ind']
        else:
            ins_inds_per_batch = None
        self.ins_inds_per_batch = ins_inds_per_batch
        
        if 'z_ins' in batched_infos:
            # Directly specify `z_ins_per_batch`
            z_ins_per_batch = batched_infos['z_ins']
        else:
            # Get `z_ins_per_batch` from `ins_id` list
            assert ins_inds_per_batch is not None, "Requires `ins_id` in batched_infos when no `z_ins` provided"
            z_ins_per_batch = self.z_ins_all.forward(ins_inds_per_batch)
        self.z_ins_per_batch = z_ins_per_batch
        
        batch_size = check_per_batch_tensors(ins_inds_per_batch, z_ins_per_batch)
        
        #---- Model network's set_condition
        super().set_condition(z=z_ins_per_batch, ins_inds_per_batch=ins_inds_per_batch)

class DITNeuS(AssetMixin):
    assigned_to = AssetAssignment.MULTI_OBJ
    is_ray_query_supported = True
    is_batched_query_supported = False
    
    """ Asset functions """
    @classmethod
    def asset_compute_id(cls, scene: Scene = None, obj: SceneNode = None, class_name: str = None) -> str:
        return f"{cls.__name__}#{class_name or obj.class_name}"

class AD_GenerativePermutoConcatNeuSObj(AutoDecoderMixin, AssetMixin, GenerativePermutoConcatNeuSModel):
    """
    GenerativePermutoConcatNeuS, generative model, auto-decoder framework for a specific list of objects.
    
    MRO:
    -> AutoDecoderMixin
    -> AssetMixin
    -> GenerativePermutoConcatNeuSModel
    -> NeuSRendererMixinBatched
    -> GenerativePermutoConcatNeuS
    -> ModelMixin
    -> nn.Module
    """
    assigned_to = AssetAssignment.MULTI_OBJ
    is_ray_query_supported = True
    is_batched_query_supported = True

    @property
    def z_ins_all(self) -> Embedding:
        return self._latents['z_ins'] # Created by autodecoder_populate()

    """ Asset functions """
    @classmethod
    def asset_compute_id(cls, scene: Scene = None, obj: SceneNode = None, class_name: str = None) -> str:
        return f"{cls.__name__}#{class_name or obj.class_name}"

    def asset_populate(
        self, 
        scene: List[Scene] = None, obj: List[SceneNode] = None, 
        config: dict = None, device=None, **kwargs):
        
        assert isinstance(obj, list) and isinstance(obj[0], SceneNode), f"Input `obj` for populate should be a list of SceneNode"
        obj_full_unique_ids = [o.full_unique_id for o in obj]
        key_maps = {'ins_id': obj_full_unique_ids}
        self.num_objs = len(obj_full_unique_ids)
        z_ins_all = Embedding(self.num_objs, **self.latents_cfg['z_ins'], dtype=torch.float, device=device)
        latent_maps = {'z_ins': z_ins_all}
        n_latent_dim = z_ins_all.embedding_dim
        
        self.ins_inds_per_batch = None
        self.z_ins_per_batch = None
        # self.z_ins_all = None # Should be `self._latents['z_ins']`
        
        #---- Batched Accel
        if self.accel_cfg is not None:
            accel_cls = get_accel_class(self.accel_cfg['type'])
            # NOTE: `config` is from `self.populate_cfg`
            accel_use_avg_resolution = config.get('accel_use_avg_resolution', True)
            accel_vox_size = config.get('accel_vox_size', 0.2)
            if accel_cls in accel_types_batched:
                self.accel_cfg.update(num_batches=self.num_objs)
            
            if accel_use_avg_resolution:
                # NOTE: Use the average resolution (aspect ratio) to configure the batched occ grid
                with torch.no_grad():
                    # NOTE: Compute the average resolution
                    for s in scene:
                        s.frozen_at_full_global_frame()
                    # Use the `o.scale` at the first valid frame `o.valid_fi[0]`
                    radius3d_list = torch.stack([o.scale.vec_3()[o.valid_fi[0]] for o in obj if o.i_valid], 0)
                    # Average object scale
                    avg_stretch3d = 2 * radius3d_list.mean(0)
                    for s in scene:
                        s.unfrozen()
                avg_resolution = (avg_stretch3d / accel_vox_size).long().tolist()
                self.accel_cfg.update(resolution=avg_resolution)
            
        #---- Autodecoder's populate
        super().autodecoder_populate(key_maps=key_maps, latent_maps=latent_maps)
        #---- Model network's populate
        super().populate(n_latent_dim=n_latent_dim, device=device, **kwargs)

    def asset_training_initialize(self, scene: Scene, obj: SceneNode, config: dict, logger: Logger=None, log_prefix: str=None) -> bool:
        z_sampler = ... # TODO: Use self.z_ins to define a sampler, might be better than using random normal z to pretrain
        z_ins_all = self.z_ins_all.weight # TODO: Use z_all? Not ideal, it's better to uniformly sample z during initialization.
        updated = super().training_initialize(config=config, logger=logger, log_prefix=log_prefix, skip_accel=True)
        if self.accel is not None:
            # NOTE: To init occ grid we need all instances
            ins_inds_all = torch.arange(self.num_objs, dtype=torch.long, device=self.device)
            self.set_condition({'ins_ind': ins_inds_all})
            self.accel.init(self.query_sdf, logger=logger)
            self.clean_condition()
        return updated

    """ New define or model functions overwrite """
    def set_condition(self, batched_infos: dict):
        if 'ins_id' in batched_infos:
            ins_ids = batched_infos['ins_id']
            if isinstance(ins_ids, str): ins_ids = [ins_ids]
            ins_inds_per_batch = [self._index_maps['ins_id'][i] for i in ins_ids]
            ins_inds_per_batch = check_to_torch(ins_inds_per_batch, dtype=torch.long, device=self.device)
        elif 'ins_ind' in batched_infos:
            ins_inds_per_batch = batched_infos['ins_ind']
        else:
            ins_inds_per_batch = None
        self.ins_inds_per_batch = ins_inds_per_batch
        
        if 'z_ins' in batched_infos:
            # Directly specify `z_ins_per_batch`
            z_ins_per_batch = batched_infos['z_ins']
        else:
            # Get `z_ins_per_batch` from `ins_id` list
            assert ins_inds_per_batch is not None, "Requires `ins_id` in batched_infos when no `z_ins` provided"
            z_ins_per_batch = self.z_ins_all.forward(ins_inds_per_batch)
        self.z_ins_per_batch = z_ins_per_batch

        batch_size = check_per_batch_tensors(ins_inds_per_batch, z_ins_per_batch)

        #---- Model network's set_condition
        super().set_condition(z=z_ins_per_batch, ins_inds_per_batch=ins_inds_per_batch)
    
    def clean_condition(self):
        super().clean_condition()
