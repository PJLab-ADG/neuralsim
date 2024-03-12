"""
@file   neus.py
@author Jianfei Guo, Shanghai AI Lab
@brief  neuralsim's API for categorical NeuS models.
"""

__all__ = [
    "AD_Dynamic_GenerativePermutoConcatNeuSObj_Decomp", 
    "AD_Dynamic_GenerativePermutoConcatNeuSObj_Mixed"
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
from nr3d_lib.models.autodecoder import AutoDecoderMixin
from nr3d_lib.models.embeddings import Embedding, MultiSeqEmbeddingShared, MultiSeqEmbeddingIndividual
from nr3d_lib.models.accelerations import get_accel_class, accel_types_batched_dynamic, OccGridAccelBatchedDynamic_Ema
from nr3d_lib.models.fields_conditional_dynamic.neus import DynamicGenerativePermutoConcatNeuSModel

from app.resources import Scene, SceneNode
from app.models.asset_base import AssetMixin, AssetAssignment

class AD_Dynamic_GenerativePermutoConcatNeuSObj_Decomp(AutoDecoderMixin, AssetMixin, DynamicGenerativePermutoConcatNeuSModel):
    """
    DynamicGenerativePermutoConcatNeuS, dynamic generative model, \
        auto-decoder framework for a specific list of objects \
        and a specific set of normalized temporal key frames. 
    
    MRO:
    -> AutoDecoderMixin
    -> AssetMixin
    -> DynamicGenerativePermutoConcatNeuSModel
    -> NeusRendererMixinBatchedDynamic
    -> DynamicGenerativePermutoConcatNeuS
    -> GenerativePermutoConcatNeuS
    -> ModelMixin 
    -> nn.Module
    """
    assigned_to = AssetAssignment.MULTI_OBJ
    is_ray_query_supported = True
    is_batched_query_supported = True
    use_ts: bool = True
    use_bidx: bool = True

    @property
    def z_time_all(self) -> Union[MultiSeqEmbeddingShared, MultiSeqEmbeddingIndividual]:
        return self._latents['z_time'] # Created by autodecoder_populate()
    
    @property
    def z_ins_all(self) -> Embedding:
        return self._latents['z_ins'] # Created by autodecoder_populate()

    """ Asset functions """
    @classmethod
    def asset_compute_id(cls, scene: Scene = None, obj: SceneNode = None, class_name: str = None) -> str:
        return f"{cls.__name__}#{class_name or obj.class_name}"

    # def __init__(self, latents_cfg: dict = None, **model_params):
    #     super().__init__(latents_cfg, **model_params)
    #     self.use_ts = True # NOTE: Config RendererMixin to use rays_ts for temporal rendering

    def asset_populate(
        self, 
        scene: List[Scene] = None, obj: List[SceneNode] = None, 
        config: dict = None, device=None, **kwargs):
        assert isinstance(obj, list) and isinstance(obj[0], SceneNode), f"Input `obj` for populate should be a list of SceneNode"

        latents_cfg = self.latents_cfg.copy()
        if isinstance(obj, SceneNode):
            obj = [obj]
        obj_full_unique_ids = [o.full_unique_id for o in obj]
        z_ins_all = Embedding(len(obj_full_unique_ids), **self.latents_cfg['z_ins'], dtype=torch.float, device=device)
        
        if isinstance(scene, Scene):
            scene = [scene]
        
        self.max_num_frames = max([len(s) for s in scene])

        self.num_objs = len(obj_full_unique_ids)
        ts_keyframes = scene[0].frame_global_ts.data.clone() # no gradients # TODO: Allow for multi-scene
        is_individual_z_time = latents_cfg['z_time'].pop('individual', False)
        if is_individual_z_time:
            z_time_all = MultiSeqEmbeddingIndividual(self.num_objs, ts_keyframes, **latents_cfg['z_time'])
        else:
            z_time_all = MultiSeqEmbeddingShared(self.num_objs, ts_keyframes, **latents_cfg['z_time'])
        
        key_maps = {'ins_id': obj_full_unique_ids}
        latent_maps = {'z_ins': z_ins_all, 'z_time': z_time_all}
        n_latent_dim = z_ins_all.embedding_dim + z_time_all.embedding_dim
        self.z_ins_dim = z_ins_all.embedding_dim
        self.z_time_dim = z_time_all.embedding_dim

        self.ins_inds_per_batch = None
        self.ts_per_batch = None
        self.z_ins_per_batch = None
        self.z_time_per_batch = None
        # self.z_ins_all = z_ins_all # Should be `self._latents['z_ins']`
        # self.z_time_all = z_time_all # Should be `self._latents['z_time']`

        #---- Batched Dynamic Accel
        if self.accel_cfg is not None:
            accel_cls = get_accel_class(self.accel_cfg['type'])
            # NOTE: `config` is from `self.populate_cfg`
            accel_n_jump_frames = int(config.get('accel_n_jump_frames', 2))
            accel_use_avg_resolution = config.get('accel_use_avg_resolution', True)
            accel_vox_size = config.get('accel_vox_size', 0.2)
            if accel_cls in accel_types_batched_dynamic:
                self.accel_cfg.update(ts_keyframes=ts_keyframes[::accel_n_jump_frames].contiguous())
            if accel_cls in (OccGridAccelBatchedDynamic_Ema, ):
                self.accel_cfg.update(num_batches=self.num_objs)
            
            if accel_use_avg_resolution:
                # NOTE: Use the average resolution (aspect ratio) to configure the batched occ grid
                with torch.no_grad():
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
            raise RuntimeError(f"Please specify at least one of 'ins_id' or 'ins_ind' in batched_infos for {type(self)}")
        self.ins_inds_per_batch = ins_inds_per_batch
        
        if 'z_ins' in batched_infos:
            # Directly specify `z_ins_per_batch`
            z_ins_per_batch = batched_infos['z_ins']
        else:
            # Get `z_ins_per_batch` from `ins_id` list
            z_ins_per_batch = self.z_ins_all.forward(ins_inds_per_batch)
        self.z_ins_per_batch = z_ins_per_batch
        
        # NOTE: Optional. By default, `ts` (per-batch) is not required in `batched_infos`, since \
        #       the ray_query() functions will automaticaly expand `rays_ts` (per-ray) to \
        #       per-input timestamps and feed it to model's forward().
        if 'ts' in batched_infos:
            z_time_per_batch = batched_infos['z_time']
            ts_per_batch = batched_infos['ts']
            ts_per_batch = check_to_torch(ts_per_batch, dtype=self.z_time_all.dtype, device=self.device)
        else:
            ts_per_batch = None
        self.ts_per_batch = ts_per_batch
        
        # NOTE: Optional. By default, `z_time` (per-batch) is not required in `batched_infos`, since: \
        #       - when 'ts' is given in `batched_infos`: 
        #           `z_time_per_batch` is obtained via indexing (interpolating) `self.z_time_all` with `ts_per_batch`
        #       - when 'ts' is not given in `batched_infos`: 
        #           when forward(), `z_time` will be obtained via indexing (interpolating) `self.z_time_all` with `ts` .
        if 'z_time' in batched_infos:
            z_time_per_batch = batched_infos['z_time']
        elif ts_per_batch is not None:
            z_time_per_batch = self.z_time_all.forward(ts_per_batch, ins_inds_per_batch)
        else:
            z_time_per_batch = None
        self.z_time_per_batch = z_time_per_batch
        
        batch_size = check_per_batch_tensors(ins_inds_per_batch, z_ins_per_batch, ts_per_batch, z_time_per_batch)

        self.implicit_surface.B = self.B = batch_size
        if self.accel is not None:
            self.accel.set_condition(
                batch_size=self.B, 
                ins_inds_per_batch=self.ins_inds_per_batch, 
                val_query_fn_normalized_x_bi_ts=self.query_sdf)
            if self.training and (self.it > 0):
                self.accel.cur_batch__step(self.it, self.query_sdf)

    def clean_condition(self):
        self.ins_inds_per_batch = None
        self.z_ins_per_batch = None
        self.z_time_per_batch = None
        self.ts_per_batch = None
        
        self.implicit_surface.B = self.B = None # batch-size
        if self.accel is not None:
            self.accel.clean_condition()

    def sample_pts_uniform(self, num_samples: int):
        num_pts_per_batch = num_samples // self.B
        # NOTE: Returns normalized `x` and normalized `ts`
        x, bidx, ts = self.space.cur_batch__sample_pts_uniform(self.B, num_pts_per_batch)
        
        # z_ins_per_x = self.z_ins_per_batch.view(self.B, 1, self.z_ins_dim).expand(self.B, num_pts_per_batch, self.z_ins_dim).detach()
        z_ins = self.z_ins_per_batch[bidx]
        
        #---- Opt1: For the current frame index / ts (only suitable for single frame ind; not for joint-frame-pixel sampling)
        # if self.ts_per_batch.numel() == 1:
        #     z_time = self.z_time_all(self.ts_per_batch).view(1,1,-1).expand(self.B, num_pts_per_batch,-1).detach()
        
        #---- Opt2: For any random frame index. More natural, as the object stays the same across different frames.
        # z_time = self.z_time_all(torch.randint(self.max_num_frames, (self.B, num_pts_per_batch), device=self.device)).detach()
        
        #---- Opt3:
        z_time = self.z_time_all.forward(ts) #, ins_inds=self.ins_inds_per_batch[bidx])
        
        ret = self.forward_sdf_nablas(x, z_ins=z_ins, z_time=z_time, bidx=bidx, ts=ts, skip_accel=True)
        for k, v in ret.items():
            ret[k] = v.to(x.dtype)
        ret['net_x'] = x # NOTE: in network's uniformed space; not in world space.
        if 'nablas' in ret:
            ret['nablas_norm'] = ret['nablas'].norm(dim=-1)
        return ret

    def sample_pts_in_occupied(self, num_samples: int):
        return super().sample_pts_in_occupied(num_samples=num_samples)

    def _check_or_get_z_per_x(
        self, x: torch.Tensor, 
        bidx: torch.Tensor = None, ts: torch.Tensor = None, 
        z_ins: torch.Tensor = None, z_time: torch.Tensor = None) -> torch.Tensor:
        
        x_prefix = [*x.shape[:-1]]
        z_ins = self.z_ins_all.check_or_get_z_per_input_batched(
            x_prefix, bidx=bidx, 
            z_per_input=z_ins, 
            z_per_batch=self.z_ins_per_batch
        )
        
        z_time = self.z_time_all.check_or_get_z_per_input_batched(
            x_prefix, bidx=bidx, 
            z_per_input=z_time, ts_per_input=ts, 
            ts_per_batch=self.ts_per_batch, 
            ins_inds_per_batch=self.ins_inds_per_batch, 
            z_per_batch=self.z_time_per_batch
        )
        z = torch.cat((z_ins, z_time), dim=-1)
        return z

    def query_sdf(
        self, x: torch.Tensor, *, z_ins: torch.Tensor = None, z_time: torch.Tensor = None, 
        bidx: torch.Tensor = None, ts: torch.Tensor = None, **kwargs):
        z = self._check_or_get_z_per_x(x=x, z_ins=z_ins, z_time=z_time, bidx=bidx, ts=ts)
        return super().query_sdf(x=x, z=z, bidx=bidx, ts=ts, **kwargs)

    def forward_sdf(
        self, x: torch.Tensor, *, z_ins: torch.Tensor = None, z_time: torch.Tensor = None, 
        bidx: torch.Tensor = None, ts: torch.Tensor = None, **kwargs):
        z = self._check_or_get_z_per_x(x=x, z_ins=z_ins, z_time=z_time, bidx=bidx, ts=ts)
        return super().forward_sdf(x=x, z=z, bidx=bidx, ts=ts, **kwargs)

    def forward_sdf_nablas(
        self, x: torch.Tensor, *, z_ins: torch.Tensor = None, z_time: torch.Tensor = None, 
        bidx: torch.Tensor = None, ts: torch.Tensor = None, **kwargs):
        z = self._check_or_get_z_per_x(x=x, z_ins=z_ins, z_time=z_time, bidx=bidx, ts=ts)
        return super().forward_sdf_nablas(x=x, z=z, bidx=bidx, ts=ts, **kwargs)

    def forward(
        self, x: torch.Tensor, *, z_ins: torch.Tensor = None, z_time: torch.Tensor = None, 
        bidx: torch.Tensor = None, ts: torch.Tensor = None, **kwargs):
        z = self._check_or_get_z_per_x(x=x, z_ins=z_ins, z_time=z_time, bidx=bidx, ts=ts)
        return super().forward(x=x, z=z, bidx=bidx, ts=ts, **kwargs)

class AD_Dynamic_GenerativePermutoConcatNeuSObj_Mixed(AutoDecoderMixin, AssetMixin, DynamicGenerativePermutoConcatNeuSModel):
    """
    DynamicGenerativePermutoConcatNeuS, dynamic generative model, \
        auto-decoder framework for a specific list of objects \
        and a specific set of normalized temporal key frames. 
    
    MRO:
    -> AutoDecoderMixin
    -> AssetMixin
    -> DynamicGenerativePermutoConcatNeuSModel
    -> NeusRendererMixinBatchedDynamic
    -> DynamicGenerativePermutoConcatNeuS
    -> GenerativePermutoConcatNeuS
    -> ModelMixin
    -> nn.Module
    """
    assigned_to = AssetAssignment.MULTI_OBJ
    is_ray_query_supported = True
    is_batched_query_supported = True
    use_ts: bool = True
    use_bidx: bool = True

    @property
    def z_mixed_all(self) -> MultiSeqEmbeddingIndividual:
        return self._latents['z_mixed'] # Created by autodecoder_populate()

    """ Asset functions """
    @classmethod
    def asset_compute_id(cls, scene: Scene = None, obj: SceneNode = None, class_name: str = None) -> str:
        return f"{cls.__name__}#{class_name or obj.class_name}"

    def asset_populate(
        self, 
        scene: List[Scene] = None, obj: List[SceneNode] = None, 
        config: dict = None, device=None, **kwargs):
        assert isinstance(obj, list) and isinstance(obj[0], SceneNode), f"Input `obj` for populate should be a list of SceneNode"

        if isinstance(obj, SceneNode):
            obj = [obj]
        if isinstance(scene, Scene):
            scene = [scene]

        obj_full_unique_ids = [o.full_unique_id for o in obj]
        self.max_num_frames = max_num_frames = max([len(s) for s in scene])
        self.num_objs = len(obj_full_unique_ids)
        ts_keyframes = scene[0].frame_global_ts.data.clone() # TODO: Allow for multi-scene
        z_mixed_all = MultiSeqEmbeddingIndividual(
            max_num_frames * self.num_objs,
            ts_keyframes,
            **self.latents_cfg['z_mixed'], dtype=torch.float, device=device)
        key_maps = {'ins_id': obj_full_unique_ids}
        latent_maps = {'z_mixed': z_mixed_all}
        n_latent_dim = z_mixed_all.embedding_dim

        self.ins_inds_per_batch = None
        self.ts_per_batch = None
        self.z_mixed_per_batch = None
        # self.z_mixed_all = z_mixed_all # Should be `self._latents['z_mixed']`

        #---- Batched Dynamic Accel
        if self.accel_cfg is not None:
            accel_cls = get_accel_class(self.accel_cfg['type'])
            # NOTE: `config` is from `self.populate_cfg`
            accel_n_jump_frames = int(config.get('accel_n_jump_frames', 2))
            accel_use_avg_resolution = config.get('accel_use_avg_resolution', True)
            accel_vox_size = config.get('accel_vox_size', 0.2)
            if accel_cls in accel_types_batched_dynamic:
                self.accel_cfg.update(ts_keyframes=ts_keyframes[::accel_n_jump_frames].contiguous())
            if accel_cls in (OccGridAccelBatchedDynamic_Ema, ):
                self.accel_cfg.update(num_batches=self.num_objs)
            
            if accel_use_avg_resolution:
                # NOTE: Use the average resolution (aspect ratio) to configure the batched occ grid
                with torch.no_grad():
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
        
        if 'ts' in batched_infos:
            ts_per_batch = check_to_torch(batched_infos['ts'], dtype=self.z_time_all.dtype, device=self.device)
        else:
            ts_per_batch = None
        self.ts_per_batch = ts_per_batch
        
        if 'z_mixed' in batched_infos:
            z_mixed_per_batch = batched_infos['z_mixed']
        else:
            assert ts_per_batch is not None and ins_inds_per_batch is not None, \
                f"Requires `ts` and `ins_id` in batched_infos when no `z_mixed` provided"
            z_mixed_per_batch = self.z_mixed_all.forward(ts_per_batch, ins_inds_per_batch) # [B, ...]
        self.z_mixed_per_batch = z_mixed_per_batch
        
        batch_size = check_per_batch_tensors(ins_inds_per_batch, ts_per_batch, z_mixed_per_batch)
        
        self.B = batch_size
        self.implicit_surface.set_condition(z=z_mixed_per_batch)
        if self.accel is not None:
            self.accel.set_condition(
                batch_size=self.B, 
                ins_inds_per_batch=self.ins_inds_per_batch, 
                val_query_fn_normalized_x_bi_ts=self.query_sdf)
            if self.training and (self.it > 0):
                self.accel.cur_batch__step(self.it, self.query_sdf)
        
        #---- Model network's set_condition
        # super().set_condition(z=z_mixed_per_batch, ins_inds_per_batch=ins_inds_per_batch)

    def clean_condition(self):
        self.B = None
        self.ins_inds_per_batch = None
        self.ts_per_batch = None
        self.z_mixed_per_batch = None
        self.implicit_surface.clean_condition()
        if self.accel is not None:
            self.accel.clean_condition()
        # super().clean_condition()

    def _check_or_get_z_per_x(
        self, x: torch.Tensor, 
        bidx: torch.Tensor = None, ts: torch.Tensor = None, 
        z: torch.Tensor = None, 
        ) -> torch.Tensor:
        
        x_prefix = [*x.shape[:-1]]
        z = self.z_mixed_all.check_or_get_z_per_input_batched(
            x_prefix, bidx=bidx, 
            z_per_input=z, ts_per_input=ts, 
            ts_per_batch=self.ts_per_batch, 
            z_per_batch=self.z_mixed_per_batch, 
            ins_inds_per_batch=self.ins_inds_per_batch, 
        )
        return z

    def query_sdf(
        self, x: torch.Tensor, *, z: torch.Tensor = None, 
        bidx: torch.Tensor = None, ts: torch.Tensor = None, **kwargs):
        z = self._check_or_get_z_per_x(x=x, z=z, bidx=bidx, ts=ts)
        return super().query_sdf(x=x, z=z, bidx=bidx, ts=ts, **kwargs)

    def forward_sdf(
        self, x: torch.Tensor, *, z: torch.Tensor = None, 
        bidx: torch.Tensor = None, ts: torch.Tensor = None, **kwargs):
        z = self._check_or_get_z_per_x(x=x, z=z, bidx=bidx, ts=ts)
        return super().forward_sdf(x=x, z=z, bidx=bidx, ts=ts, **kwargs)

    def forward_sdf_nablas(
        self, x: torch.Tensor, *, z: torch.Tensor = None, 
        bidx: torch.Tensor = None, ts: torch.Tensor = None, **kwargs):
        z = self._check_or_get_z_per_x(x=x, z=z, bidx=bidx, ts=ts)
        return super().forward_sdf_nablas(x=x, z=z, bidx=bidx, ts=ts, **kwargs)

    def forward(
        self, x: torch.Tensor, *, z: torch.Tensor = None, 
        bidx: torch.Tensor = None, ts: torch.Tensor = None, **kwargs):
        z = self._check_or_get_z_per_x(x=x, z=z, bidx=bidx, ts=ts)
        return super().forward(x=x, z=z, bidx=bidx, ts=ts, **kwargs)
