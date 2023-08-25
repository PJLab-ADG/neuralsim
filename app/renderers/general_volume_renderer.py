"""
@file   general_volume_renderer.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Multiple-object joint volume rendering. 
        
        FEATURES:
        - Multi-object buffer collection, sorting and merging, and volume integration.
        - For each object, support render buffer of both packed and batched type.
        - Built on general (batched_)ray_test and (batched_)ray_query APIs. 
          Hence, any work in neural rendering that works with these APIs can be supported.
"""

import itertools
import functools
import numpy as np
from typing import Dict, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_min

from nr3d_lib.config import ConfigDict
from nr3d_lib.models.utils import batchify_query
from nr3d_lib.render.volume_graphics import packed_alpha_to_vw
from nr3d_lib.render.pack_ops import get_pack_infos_from_n, interleave_linstep, packed_div, packed_matmul, packed_sum, packed_sort

from app.models.base import AssetAssignment, AssetModelMixin
from app.resources import AssetBank, Scene, SceneNode, namedtuple_ind_id_obj
from app.resources.observers import Camera, MultiCamBundle, Lidar, RaysLidar, MultiRaysLidarBundle
from app.renderers.utils import rotate_volume_buffer_nablas

class GeneralVolumeRenderer(nn.Module):
    def __init__(self, config: ConfigDict):
        super().__init__()
        self.training = False
        common = config.common.copy()
        train = config.get('train', ConfigDict()).copy()
        train.update(common)
        val = config.get('val', ConfigDict()).copy()
        val.update(common)
        self._config = val
        self._config_train = train

        self.image_keys = ['depth_volume', 'mask_volume', 'rgb_volume', 'rgb_volume_occupied', 'rgb_volume_non_occupied', 'normals_volume']

    @property
    def config(self):
        if self.training:
            return self._config_train
        else:
            return self._config

    def populate(self, asset_bank: AssetBank):
        if self.config.get("enable_postprocessor", False):
            self.image_postprocessor = asset_bank["ImagePostprocessor"]
        else:
            self.image_postprocessor = None

    def _prep_empty_rendered(
        self, prefix: Tuple[int], dtype=torch.float, device=None, 
        with_rgb=True, with_normal=True, with_feature_dim: int = 0):
        rendered = dict(
            mask_volume = torch.zeros(prefix, dtype=dtype, device=device),
            depth_volume = torch.zeros(prefix, dtype=dtype, device=device)
        )
        if with_rgb:
            rendered['rgb_volume'] = torch.zeros([*prefix, 3], dtype=dtype, device=device)
        if with_normal:
            rendered['normals_volume'] = torch.zeros([*prefix, 3], dtype=dtype, device=device)
        if with_feature_dim:
            rendered['feature_volume'] = torch.zeros([*prefix, with_feature_dim], dtype=dtype, device=device)
        return rendered

    # @profile
    def forward(
        self, 
        rays_o: torch.Tensor, rays_d: torch.Tensor, rays_xy: torch.Tensor = None, rays_fi: torch.Tensor = None, 
        *, 
        scene: Scene, drawables: List[SceneNode]=None, 
        observer: Union[Camera, MultiCamBundle, Lidar, RaysLidar, MultiRaysLidarBundle]=None, 
        # ray_query_config
        near=None, far=None, bypass_ray_query_cfg=ConfigDict(), 
        with_rgb: bool=None, with_normal: bool=None, with_feature_dim: int=None, with_env: bool=None, 
        return_buffer=False, return_details=False, render_per_obj=False, render_per_obj_in_total=False
        ) -> dict:
        
        assert rays_o.dim() == rays_d.dim() == 2, "rays_o and rays_d should have size of [N, 3]"
        
        sky_objs = scene.get_drawable_groups_by_class_name('Sky')
        
        total_num_rays = rays_o.shape[0]
        device, config = scene.device, self.config
        if with_rgb is None: with_rgb = config.get('with_rgb', True)
        if with_normal is None: with_normal = config.get('with_normal', False)
        if with_feature_dim is None: with_feature_dim = config.get('with_feature_dim', 0)
        if with_env is None: with_env = config.get('with_env', len(sky_objs) > 0)
        if near is None: near = config.get('near', None)
        if far is None: far = config.get('far', None)

        if drawables is None:
            drawables = scene.get_drawables()
            drawables = observer.filter_drawable_groups(drawables)

        if with_rgb and scene.image_embeddings is not None:
            assert isinstance(observer, (Camera, MultiCamBundle)), f'Expected camera observers, but got observer type={observer}'
            assert rays_fi is not None, 'Need per-ray frame index `rays_fi` input when scene.image_embeddings is present.'
            h_image_embed = scene.image_embeddings[observer.id](rays_fi)
        else:
            h_image_embed = None

        #---- Get rays in renderable objects [num_object, total_num_rays, 3]
        rays_o_o, rays_d_o = scene.convert_rays_in_nodes_list(rays_o, rays_d, drawables)
        #---- Group objects with their model types
        # drawable_model_id_dict = scene.group_drawables_by_model_id(drawables)
        #---- NOTE: Equivalent to model_id, as model_id and class_name have one-to-one correspondence.
        drawable_class_name_dict = scene.group_drawables_by_class_name(drawables)
        if 'Distant' in drawable_class_name_dict.keys():
            # Put distant at the last of query queue
            drawable_class_name_dict['Distant'] = drawable_class_name_dict.pop('Distant')

        class_ind_map = scene.get_drawable_class_ind_map()
        instance_ind_map = scene.get_drawable_instance_ind_map()

        #----------------------------------------------------
        #                 Prepare outputs
        #----------------------------------------------------
        total_rendered = self._prep_empty_rendered(
            [total_num_rays], device=device, 
            with_rgb=with_rgb, with_normal=with_normal, with_feature_dim=with_feature_dim)
        ray_visible_objects = torch.zeros([total_num_rays], dtype=torch.long, device=device)
        ray_visible_samples = torch.zeros([total_num_rays], dtype=torch.long, device=device)
        
        if return_buffer:
            total_volume_buffer = dict(buffer_type='empty')
        
        # (Optional) Prepare per-object seperate rendering
        if render_per_obj:
            rendered_per_obj = dict()
            # NOTE: Render instance segmentation & class segmentation
            segmentation_threshold = config.get('segmentation_threshold', 0.6)
            ins_seg_mask_buffer = torch.full([total_num_rays], -1, dtype=torch.long, device=device)
            class_seg_mask_buffer = torch.full([total_num_rays], -1, dtype=torch.long, device=device)
            z_buffer = torch.full([total_num_rays], np.inf, dtype=torch.float, device=device) # Nearest z; used to sort overlapped object's indices according to depth
        
        # (Optional) Prepare per-object's part in total rendering
        if render_per_obj_in_total:
            rendered_per_obj_in_total = dict()
        
        raw_per_obj_model = dict()
        rendered_objs_group_by_class_name: Dict[str, List[str]] = {}
        rendered_objs_group_by_model_id: Dict[str, List[str]] = {}
        
        def batched_query_shared(model: AssetModelMixin, group: namedtuple_ind_id_obj):
            """
            Query a shared model with `batched_ray_test` and `batched_ray_query`
            """
            class_name = group.objs[0].class_name
            class_index = class_ind_map[class_name]
            query_cfg = ConfigDict(**model.ray_query_cfg, **config)
            query_cfg.update(with_rgb=with_rgb, with_normal=with_normal, with_feature_dim=with_feature_dim) # Possible override
            for k, v in bypass_ray_query_cfg.get(class_name, {}).items():
                query_cfg[k] = v

            # Early z-test (batched)
            batched_ray_input = dict(
                rays_o=rays_o_o[group.inds], rays_d=rays_d_o[group.inds], near=near, far=far, 
                rays_fi=rays_fi.tile(len(group.inds), *[1]*rays_fi.dim()) if rays_fi is not None else None, 
                rays_xy=rays_xy.tile(len(group.inds), *[1]*rays_xy.dim()) if rays_xy is not None else None, 
                rays_h_appear_embed=h_image_embed.tile(len(group.inds), *[1]*h_image_embed.dim()) if h_image_embed is not None else None)
            batched_ray_tested = model.batched_ray_test(**batched_ray_input, compact_batch=True)
            
            full_batch_ind_map = batched_ray_tested['full_batch_ind_map'].tolist()
            num_rays, ray_inds = batched_ray_tested['num_rays'], batched_ray_tested['ray_inds']
            compact_obj_full_ids = [group.objs[i].full_unique_id for i in full_batch_ind_map] # Only for querying shared models;
            compact_obj_ids = [group.objs[i].id for i in full_batch_ind_map] # For scene-level render and stats
            
            # Prepare contional models' conditions
            if num_rays > 0:
                batched_infos = {'keys': compact_obj_full_ids}
                model.set_condition(batched_infos)
            
            # Query on tested rays
            raw_ret: dict = model.batched_ray_query(
                batched_ray_tested=batched_ray_tested, batched_ray_input=batched_ray_input, config=query_cfg, 
                return_buffer=True, render_per_obj=render_per_obj, return_details=return_details)
            
            # (Optional) Per-object seperate rendering
            if render_per_obj:
                rendered = raw_ret.pop('rendered')
                for i, (oid, obj) in enumerate(zip(group.ids, group.objs)):
                    _cur_rendered = rendered_per_obj[oid] = {k: v[i] for k, v in rendered.items()}
                    # NOTE: Rotate obj's normals to world
                    if 'normals_volume' in _cur_rendered:
                        _cur_rendered['normals_volume_in_world'] = obj.world_transform.rotate(_cur_rendered['normals_volume'])
                
                # NOTE: Render class segmentation and instance segmentation
                if num_rays > 0:
                    compact_instance_indexes_map = torch.tensor([instance_ind_map[oid] for oid in compact_obj_ids], dtype=torch.long, device=device)
                    # object index of each tested ray
                    instance_indexes = compact_instance_indexes_map[batched_ray_tested['batch_inds']] # [num_tested]
                    
                    # [num_tested]
                    inds = (batched_ray_tested['full_batch_inds'], ray_inds)
                    mask_volume = rendered['mask_volume'][inds]
                    depth_volume = rendered['depth_volume'][inds]

                    mask_mask = (mask_volume > segmentation_threshold)
                    mask_mask_inds = mask_mask.nonzero().long()[..., 0]
                    
                    if mask_mask.any():
                        ray_inds_masked = ray_inds[mask_mask_inds]
                        depth_volume_masked = depth_volume[mask_mask_inds]
                        instance_indexes_masked = instance_indexes[mask_mask_inds]
                        # To check for overlapping objects on the same pixel
                        _, inv_idx, cnt = torch.unique(ray_inds_masked, return_counts=True, return_inverse=True)
                        if (cnt>1).any():
                            # Sort instances according to depth
                            depth_max, argmax = scatter_min(depth_volume_masked, inv_idx, dim=0)
                            ray_inds_masked = ray_inds_masked[argmax]
                            # depth_volume_masked = depth_volume_masked[argmax] # equals to `depth_max`
                            depth_volume_masked = depth_max
                            instance_indexes_masked = instance_indexes_masked[argmax]
                        
                        z_nearer = depth_volume_masked < z_buffer[ray_inds_masked]
                        if z_nearer.any():
                            z_buffer[ray_inds_masked[z_nearer]] = depth_volume_masked[z_nearer]
                            ins_seg_mask_buffer[ray_inds_masked[z_nearer]] = instance_indexes_masked[z_nearer]
                            class_seg_mask_buffer[ray_inds_masked[z_nearer]] = class_index
            if render_per_obj_in_total:
                for i, (oid, obj) in enumerate(zip(group.ids, group.objs)):
                    # Allocate empty data. Will be filled when volume rendering
                    rendered_per_obj_in_total[oid] = self._prep_empty_rendered(
                        [total_num_rays], device=device, 
                        with_rgb=with_rgb, with_normal=with_normal, with_feature_dim=with_feature_dim)

            # Gather results
            raw_ret.update(
                class_name=class_name, obj_id=compact_obj_ids, model_id=model.id, 
                num_rays=num_rays, ray_inds=ray_inds, )
            rendered_objs_group_by_class_name.setdefault(class_name, []).extend(compact_obj_ids)
            rendered_objs_group_by_model_id.setdefault(model.id, []).extend(compact_obj_ids)
            if num_rays > 0:
                volume_buffer = raw_ret['volume_buffer']
                if (buffer_type:=volume_buffer['buffer_type']) != 'empty':
                    ray_inds_hit = volume_buffer['ray_inds_hit']
                    num_per_hit = torch.full_like(ray_inds_hit, volume_buffer['num_per_hit'], device=device, dtype=torch.long) \
                        if buffer_type == 'batched' else volume_buffer['pack_infos_hit'][:,1]
                    
                    # NOTE: Check for duplicated ridx caused by the same ray passing through multiple batched objects
                    # NOTE: [!!!] Requires ridx to be consecutive and monotonically increasing. 
                    #       Example: this is sufficed by nr3d_lib/models/spatial/batched.py::ray_test()
                    ray_inds_hit_collect, duplicate_cnt = torch.unique_consecutive(ray_inds_hit, return_counts=True)
                    if (duplicate_cnt > 1).any():
                        num_per_hit_collect = torch.zeros([total_num_rays], device=device, dtype=torch.long).index_add_(0, ray_inds_hit, num_per_hit)[ray_inds_hit_collect]
                        pack_infos_collect = get_pack_infos_from_n(num_per_hit_collect)
                    else:
                        num_per_hit_collect = num_per_hit
                        pack_infos_collect = get_pack_infos_from_n(num_per_hit) \
                            if buffer_type == 'batched' else volume_buffer['pack_infos_hit']
                    
                    volume_buffer.update(ray_inds_hit_collect=ray_inds_hit_collect, pack_infos_collect=pack_infos_collect)
                    ray_visible_objects.index_add_(0, ray_inds_hit, ray_inds_hit.new_ones(ray_inds_hit.shape))
                    ray_visible_samples.index_add_(0, ray_inds_hit_collect, num_per_hit_collect)
                    # ray_visible_samples.index_add_(0, ray_inds_hit, num_per_hit) # NOTE: equivalent

                    # NOTE: [demo-mix]
                    if query_cfg.get('normal_as_rgb', False):
                        volume_buffer['rgb'] = (F.normalize(volume_buffer['nablas'].view(-1,3).clamp_(-1,1), dim=-1)/2.+0.5)

                    # NOTE: Rotate obj's normals to world
                    if 'nablas' in volume_buffer:
                        nablas = volume_buffer['nablas']
                        compact_obj_rotations = torch.stack([group.objs[i].world_transform.rotation() for i in full_batch_ind_map], dim=0)
                        rotation_of_each_hit_ray = compact_obj_rotations[volume_buffer['batch_inds_hit']]
                        if (buffer_type:=volume_buffer['buffer_type']) == 'packed':
                            # nablas: [num_feats, 3]
                            nablas_in_world = packed_matmul(nablas, rotation_of_each_hit_ray, volume_buffer['pack_infos_hit'])
                        else:
                            # rotation: [num_rays_hit, 3, 3]
                            # nablas: [num_rays_hit, num_pts, 3]
                            nablas_in_world = (rotation_of_each_hit_ray.unsqueeze(1) * nablas.unsqueeze(-2)).sum(-1)
                        volume_buffer['nablas_in_world'] = nablas_in_world

            # Put everything into overall buffer
            raw_per_obj_model[group.ids[0]] = raw_ret

        def foreach_query_shared(model: AssetModelMixin, group: namedtuple_ind_id_obj):
            """
            Query a shared model with foreach
            """
            class_name = group.objs[0].class_name
            class_index = class_ind_map[class_name]
            query_cfg = ConfigDict(**model.ray_query_cfg, **config)
            query_cfg.update(with_rgb=with_rgb, with_normal=with_normal, with_feature_dim=with_feature_dim) # Possible override
            for k, v in bypass_ray_query_cfg.get(class_name, {}).items():
                query_cfg[k] = v

            for (ind, obj_id, obj) in zip(*group):
                # Early z-test
                ray_input = dict(
                    rays_o=rays_o_o[ind], rays_d=rays_d_o[ind], near=near, far=far, 
                    rays_fi=rays_fi, rays_xy=rays_xy, rays_h_appear_embed=h_image_embed)
                ray_tested = model.ray_test(**ray_input)
                num_rays, ray_inds = ray_tested['num_rays'], ray_tested['ray_inds']
                if num_rays > 0:
                    batched_infos = {'keys': [obj.full_unique_id]}
                    model.set_condition(batched_infos)
                
                # Query on tested rays
                raw_ret: dict = model.ray_query(
                    ray_tested=ray_tested, ray_input=ray_input, config=query_cfg,
                    return_buffer=True, render_per_obj=render_per_obj, return_details=return_details)
                
                # (Optional) Per-object seperate rendering
                if render_per_obj:
                    rendered = rendered_per_obj[obj_id] = raw_ret.pop('rendered')
                    # NOTE: Rotate obj's normals to world
                    if 'normals_volume' in rendered:
                        rendered['normals_volume_in_world'] = obj.world_transform.rotate(rendered['normals_volume'])
                    
                    # Render class segmentation and instance segmentation
                    if num_rays > 0:
                        instance_index = instance_ind_map[obj_id]
                        # [num_tested]
                        mask_volume = rendered['mask_volume'][ray_inds]
                        depth_volume = rendered['depth_volume'][ray_inds]
                        mask_mask = (mask_volume > segmentation_threshold)
                        
                        mask_mask_inds = mask_mask.nonzero().long()[..., 0]
                        if mask_mask.any():
                            ray_inds_masked = ray_inds[mask_mask_inds]
                            z_nearer = (depth_volume[mask_mask_inds] < z_buffer[ray_inds_masked])
                            z_nearer_inds = z_nearer.nonzero().long()[..., 0]
                            if z_nearer.any():
                                ray_inds_to_update = ray_inds_masked[z_nearer_inds]
                                z_buffer[ray_inds_to_update] = depth_volume[mask_mask_inds][z_nearer_inds]
                                ins_seg_mask_buffer[ray_inds_to_update] = instance_index
                                class_seg_mask_buffer[ray_inds_to_update] = class_index
                if render_per_obj_in_total:
                    # Allocate empty data. Will be filled when volume rendering
                    rendered_per_obj_in_total[obj_id] = self._prep_empty_rendered(
                        [total_num_rays], device=device, 
                        with_rgb=with_rgb, with_normal=with_normal, with_feature_dim=with_feature_dim)
                
                # Gather results
                raw_ret.update(
                    class_name=class_name, obj_id=obj_id, model_id=model.id, 
                    num_rays=num_rays, ray_inds=ray_inds, )
                rendered_objs_group_by_class_name.setdefault(class_name, []).append(obj_id)
                rendered_objs_group_by_model_id.setdefault(model.id, []).append(obj_id)
                if num_rays > 0:
                    volume_buffer = raw_ret['volume_buffer']
                    if (buffer_type:=volume_buffer['buffer_type']) != 'empty':
                        ray_inds_hit = volume_buffer['ray_inds_hit']
                        num_per_hit = torch.full_like(ray_inds_hit, volume_buffer['num_per_hit'], device=device, dtype=torch.long) \
                            if buffer_type == 'batched' else volume_buffer['pack_infos_hit'][:,1]
                        pack_infos_hit = get_pack_infos_from_n(num_per_hit) \
                            if buffer_type == 'batched' else volume_buffer['pack_infos_hit']
                        volume_buffer.update(ray_inds_hit_collect=ray_inds_hit, pack_infos_collect=pack_infos_hit)
                        ray_visible_objects[ray_inds_hit] += 1
                        ray_visible_samples[ray_inds_hit] += num_per_hit

                        # NOTE: [demo-mix]
                        if query_cfg.get('normal_as_rgb', False):
                            volume_buffer['rgb'] = (F.normalize(volume_buffer['nablas'].view(-1,3).clamp_(-1,1), dim=-1)/2.+0.5)

                        # NOTE: Rotate obj's normals to world
                        if 'nablas' in volume_buffer:
                            o2w_rot = obj.world_transform.rotation().detach() # Removing gradients on nablas can eliminate interference with the pose gradient.
                            volume_buffer['nablas_in_world'] = rotate_volume_buffer_nablas(o2w_rot, volume_buffer['nablas'], volume_buffer)

                # Put everything into overall buffer
                raw_per_obj_model[obj_id] = raw_ret
        
        def query_single(model: AssetModelMixin, group: namedtuple_ind_id_obj):
            """
            Query a single model
            """
            class_name = group.objs[0].class_name
            class_index = class_ind_map[class_name]
            query_cfg = ConfigDict(**model.ray_query_cfg, **config)
            query_cfg.update(with_rgb=with_rgb, with_normal=with_normal, with_feature_dim=with_feature_dim) # Possible override
            for k, v in bypass_ray_query_cfg.get(class_name, {}).items():
                query_cfg[k] = v
        
            assert len(group.ids) == 1, "Multiple objects sharing the same single model is not allowed for now."
            ind, obj_id, obj = group.inds[0], group.ids[0], group.objs[0]

            # Ray preparations and early z-test
            if class_name != 'Distant':
                ray_input = dict(
                    rays_o=rays_o_o[ind], rays_d=rays_d_o[ind], near=near, far=far, 
                    rays_fi=rays_fi, rays_xy=rays_xy, rays_h_appear_embed=h_image_embed)
                ray_tested = model.ray_test(**ray_input)
            else:
                #---- NOTE: What's below: special ray preparation for distant model
                ray_input = dict(
                    rays_o=rays_o_o[ind], rays_d=rays_d_o[ind], 
                    near=torch.full([total_num_rays], near, device=device, dtype=torch.float), far=None, 
                    rays_fi=rays_fi, rays_xy=rays_xy, rays_h_appear_embed=h_image_embed)
                
                if (cr_obj:=model.cr_obj) is not None:
                    # NOTE: If cr_obj is present, distant-view model should use rays in cr_obj
                    cr_raw_ret = raw_per_obj_model[cr_obj.id]
                    cr_ind = drawable_class_name_dict[cr_obj.class_name].inds[0]
                    ray_input.update(rays_o=rays_o_o[cr_ind], rays_d=rays_d_o[cr_ind])
                    # NOTE: For rays that pass cr's ray_test, the distant model's sampling starts from cr's `far`
                    if cr_raw_ret['num_rays'] > 0:
                        ray_input['near'][cr_raw_ret['ray_inds']] = cr_raw_ret['ray_far']
                
                # NOTE: Detach pose gradient on distant-view branch (if any). This significantly helps pose refinement.
                ray_input['rays_o'] = ray_input['rays_o'].detach()
                ray_input['rays_d'] = ray_input['rays_d'].detach()
                
                # NOTE: Distant model does not actually need to ray_test
                ray_tested = dict(**ray_input, num_rays=total_num_rays, ray_inds=torch.arange(total_num_rays, device=device))
                    
            num_rays, ray_inds = ray_tested['num_rays'], ray_tested['ray_inds']
            ray_near, ray_far = ray_tested.get('near', None), ray_tested.get('far', None)
            
            raw_ret: dict = model.ray_query(
                ray_tested=ray_tested, ray_input=ray_input, config=query_cfg, 
                return_buffer=True, render_per_obj=render_per_obj, return_details=return_details)
            
            # (Optional) Per-object seperate rendering
            if render_per_obj:
                rendered = rendered_per_obj[obj_id] = raw_ret.pop('rendered')
                # NOTE: Rotate obj's normals to world
                if 'normals_volume' in rendered:
                    rendered['normals_volume_in_world'] = obj.world_transform.rotate(rendered['normals_volume'])
                
                # Render class segmentation and instance segmentation
                if num_rays > 0:
                    instance_index = instance_ind_map[obj_id]
                    # [num_tested]
                    mask_volume = rendered['mask_volume'][ray_inds]
                    depth_volume = rendered['depth_volume'][ray_inds]
                    mask_mask = (mask_volume > segmentation_threshold)
                    
                    mask_mask_inds = mask_mask.nonzero().long()[..., 0]
                    if mask_mask.any():
                        ray_inds_masked = ray_inds[mask_mask_inds]
                        z_nearer = (depth_volume[mask_mask_inds] < z_buffer[ray_inds_masked])
                        z_nearer_inds = z_nearer.nonzero().long()[..., 0]
                        if z_nearer.any():
                            ray_inds_to_update = ray_inds_masked[z_nearer_inds]
                            z_buffer[ray_inds_to_update] = depth_volume[mask_mask_inds][z_nearer_inds]
                            ins_seg_mask_buffer[ray_inds_to_update] = instance_index
                            class_seg_mask_buffer[ray_inds_to_update] = class_index
            if render_per_obj_in_total:
                # Allocate empty data. Will be filled when volume rendering
                rendered_per_obj_in_total[obj_id] = self._prep_empty_rendered(
                    [total_num_rays], device=device, 
                    with_rgb=with_rgb, with_normal=with_normal, with_feature_dim=with_feature_dim)
            
            # Gather results
            raw_ret.update(
                class_name=class_name, obj_id=obj_id, model_id=model.id, 
                num_rays=num_rays, ray_inds=ray_inds, ray_near=ray_near, ray_far=ray_far)
            rendered_objs_group_by_class_name.setdefault(class_name, []).append(obj_id)
            rendered_objs_group_by_model_id.setdefault(model.id, []).append(obj_id)
            if num_rays > 0:
                volume_buffer = raw_ret['volume_buffer']
                if (buffer_type:=volume_buffer['buffer_type']) != 'empty':
                    ray_inds_hit = volume_buffer['ray_inds_hit']
                    num_per_hit = torch.full_like(ray_inds_hit, volume_buffer['num_per_hit'], device=device, dtype=torch.long) \
                        if buffer_type == 'batched' else volume_buffer['pack_infos_hit'][:,1]
                    pack_infos_hit = get_pack_infos_from_n(num_per_hit) \
                        if buffer_type == 'batched' else volume_buffer['pack_infos_hit']
                    volume_buffer.update(ray_inds_hit_collect=ray_inds_hit, pack_infos_collect=pack_infos_hit)
                    ray_visible_objects[ray_inds_hit] += 1
                    ray_visible_samples[ray_inds_hit] += num_per_hit
                    
                    # NOTE: [demo-mix]
                    if query_cfg.get('normal_as_rgb', False):
                        volume_buffer['rgb'] = (F.normalize(volume_buffer['nablas'].view(-1,3).clamp_(-1,1), dim=-1)/2.+0.5)

                    # NOTE: Rotate obj's normals to world
                    if 'nablas' in volume_buffer:
                        o2w_rot = obj.world_transform.rotation().detach() # Removing gradients on nablas can eliminate interference with the pose gradient.
                        volume_buffer['nablas_in_world'] = rotate_volume_buffer_nablas(o2w_rot, volume_buffer['nablas'], volume_buffer)
             
            # Put everything into overall buffer
            raw_per_obj_model[obj_id] = raw_ret
            
        #----------------------------------------------------
        #              Ray query each model
        #----------------------------------------------------
        for class_name, group in drawable_class_name_dict.items():
            model: AssetModelMixin = group.objs[0].model
            # model = scene.asset_bank[model_id]
            if not model.is_ray_query_supported:
                # Not a ray query-able model
                continue
            
            if model.assigned_to in [AssetAssignment.MULTI_OBJ, AssetAssignment.MULTI_OBJ_ONE_SCENE]:
                #----------------------------------------------------
                #--------- Ray query shared models
                if model.is_batched_query_supported:
                    # If a shared model support `batched_ray_query`
                    batched_query_shared(model, group)
                else:
                    # If a shared model does not support `batched_ray_query`, will call `ray_query` one by one
                    foreach_query_shared(model, group)
            else:
                #----------------------------------------------------
                #--------- Ray query single models
                query_single(model, group)
        
        total_ray_inds_hit = ray_visible_samples.nonzero().long()[..., 0]
        if total_ray_inds_hit.numel() > 0:
            #----------------------------------------------------
            #            Gather all returned buffer
            #----------------------------------------------------
            total_pack_infos_sparse = get_pack_infos_from_n(ray_visible_samples)
            total_pack_infos = total_pack_infos_sparse[total_ray_inds_hit]
            total_num_samples = total_pack_infos_sparse[-1,:].sum().item()
            total_depths = torch.zeros([total_num_samples], dtype=torch.float, device=device)
            total_alphas = torch.zeros([total_num_samples], dtype=torch.float, device=device)
            if with_rgb:
                total_rgbs = torch.zeros([total_num_samples, 3], dtype=torch.float, device=device)
            if with_normal:
                total_nablas = torch.zeros([total_num_samples, 3], dtype=torch.float, device=device)
            if with_feature_dim:
                total_feature = torch.zeros([total_num_samples, with_feature_dim], dtype=torch.float, device=device)
            
            current_pack_indices_buffer = total_pack_infos_sparse[:, 0].clone()
            
            for raw_ret in raw_per_obj_model.values():
                volume_buffer = raw_ret['volume_buffer']
                if (buffer_type:=volume_buffer['buffer_type']) != 'empty':
                    ray_inds_hit_collect, pack_infos_collect = volume_buffer['ray_inds_hit_collect'], volume_buffer['pack_infos_collect']
                    volume_buffer['pidx_in_total'] = pidx_in_total = interleave_linstep(current_pack_indices_buffer[ray_inds_hit_collect], pack_infos_collect[:,1], 1, return_idx=False)
                    
                    total_depths[pidx_in_total] = volume_buffer['t'].flatten()
                    total_alphas[pidx_in_total] = volume_buffer['opacity_alpha'].flatten()
                    if with_rgb:
                        total_rgbs[pidx_in_total] = volume_buffer['rgb'].flatten(0, -2)
                    if with_normal and ('nablas_in_world' in volume_buffer):
                        total_nablas[pidx_in_total] = volume_buffer['nablas_in_world'].flatten(0, -2)
                    if with_feature_dim:
                        total_feature[pidx_in_total] = volume_buffer['feature'].flatten(0, -2)

                    current_pack_indices_buffer.index_add_(0, ray_inds_hit_collect, pack_infos_collect[:,1])

            #----------------------------------------------------
            #            Sort all returned buffer
            #----------------------------------------------------
            total_volume_buffer = dict(buffer_type='packed', ray_inds_hit=total_ray_inds_hit, pack_infos_hit=total_pack_infos)
            total_volume_buffer['t'], total_sort_indices = packed_sort(total_depths, total_pack_infos)
            total_volume_buffer['opacity_alpha'] = total_alphas[total_sort_indices]
            if with_rgb:
                total_volume_buffer['rgb'] = total_rgbs[total_sort_indices]
            if with_normal:
                total_volume_buffer['nablas'] = total_nablas[total_sort_indices]
            if with_feature_dim:
                total_volume_buffer['feature'] = total_feature[total_sort_indices]
            
            #----------------------------------------------------
            #              Overall volume integration
            #----------------------------------------------------
            total_volume_buffer['vw'] = total_vw = packed_alpha_to_vw(total_volume_buffer['opacity_alpha'], total_pack_infos)
            total_rendered['mask_volume'][total_ray_inds_hit] = vw_sum = packed_sum(total_vw, total_pack_infos)
            total_volume_buffer['vw_normalized'] = total_vw_normalized = packed_div(total_vw, vw_sum+1e-10, total_pack_infos)
            if config.get('depth_use_normalized_vw', True):
                total_rendered['depth_volume'][total_ray_inds_hit] = packed_sum(total_vw_normalized * total_volume_buffer['t'], total_pack_infos)
            else:
                total_rendered['depth_volume'][total_ray_inds_hit] = packed_sum(total_vw * total_volume_buffer['t'], total_pack_infos)
            if with_rgb:
                total_rendered['rgb_volume'][total_ray_inds_hit] = packed_sum(total_vw.view(-1,1) * total_volume_buffer['rgb'].view(-1,3), total_pack_infos)
            if with_normal:
                if self.training:
                    total_rendered['normals_volume'][total_ray_inds_hit] = packed_sum(total_vw.view(-1,1) * total_volume_buffer['nablas'].view(-1,3), total_pack_infos)
                else:
                    total_rendered['normals_volume'][total_ray_inds_hit] = packed_sum(total_vw.view(-1,1) * F.normalize(total_volume_buffer['nablas'].view(-1,3).clamp_(-1,1), dim=-1), total_pack_infos)
            if with_feature_dim:
                total_rendered['feature_volume'][total_ray_inds_hit] = packed_sum(total_vw.view(-1,1) * total_volume_buffer['feature'].view(-1,3), total_pack_infos)

            if return_buffer or render_per_obj_in_total:
                for raw_ret in raw_per_obj_model.values():
                    volume_buffer = raw_ret['volume_buffer']
                    if (buffer_type:=volume_buffer['buffer_type']) != 'empty':
                        volume_buffer['vw_in_total'] = total_vw[volume_buffer['pidx_in_total']]
                        if render_per_obj_in_total:
                            raise NotImplementedError("To be finished...")

        #----------------------------------------------------
        #                    Sky model
        #----------------------------------------------------
        if with_rgb:
            total_rendered["rgb_volume_occupied"] = total_rendered["rgb_volume"]
            if with_env and len(sky_objs) > 0:
                sky_obj = sky_objs[0]
                sky_model = sky_obj.model
                env_rgb = sky_model(v=F.normalize(rays_d, dim=-1), h_appear_embed=h_image_embed)
                total_rendered['rgb_sky'] = env_rgb
                total_rendered['rgb_volume_non_occupied'] = env_rgb_blend = (1-total_rendered['mask_volume'].unsqueeze(-1)) * env_rgb
                # NOTE: Avoid inplace op
                total_rendered['rgb_volume'] = total_rendered['rgb_volume'] + env_rgb_blend

        #----------------------------------------------------
        #                    Image Post-processing
        #----------------------------------------------------
        if with_rgb and self.image_postprocessor:
            assert rays_xy is not None, "Need `rays_xy` input when image_postprocessor is present."
            total_rendered["rgb_volume"] = self.image_postprocessor(h_image_embed, rays_xy, total_rendered["rgb_volume"])

        #----------------------------------------------------
        #                       Return
        #----------------------------------------------------
        ret = dict(
            ray_intersections=dict(
                obj_cnt=ray_visible_objects,
                samples_cnt=ray_visible_samples
            ), 
            
            rendered=total_rendered, 
            
            # num_rays_hit=0 if total_volume_buffer['buffer_type'] == 'empty' else total_volume_buffer['ray_inds_hit'].numel(), 
            rendered_objs_group_by_class_name=rendered_objs_group_by_class_name,
            rendered_objs_group_by_model_id=rendered_objs_group_by_model_id,
            rendered_class_names=list(rendered_objs_group_by_class_name.keys()),
            rendered_model_ids=list(rendered_objs_group_by_model_id.keys()),
            rendered_obj_ids=itertools.chain.from_iterable([[v] if isinstance(v, str) else v for v in rendered_objs_group_by_class_name.values()]),
        )
        
        if render_per_obj:
            ret.update(
                rendered_per_obj=rendered_per_obj, 
                class_seg_mask_buffer=class_seg_mask_buffer,
                ins_seg_mask_buffer=ins_seg_mask_buffer
            )
        
        if render_per_obj_in_total:
            ret['rendered_per_obj_in_total'] = rendered_per_obj_in_total
        
        if return_buffer:
            ret['volume_buffer'] = total_volume_buffer

        if return_details:
            ret['raw_per_obj_model'] = raw_per_obj_model

        return ret
    
    def render(
        self, 
        scene: Scene, *, drawables: List[SceneNode]=None, 
        rays: List[torch.Tensor] = None, # [rays_o, rays_d] or [rays_o, rays_d, rays_fi, rays_xy]
        observer: Union[Camera, MultiCamBundle, Lidar, RaysLidar, MultiRaysLidarBundle] = None, 
        show_progress: bool=False, rayschunk: int=None, 
        near: float = None, far: float = None, bypass_ray_query_cfg=ConfigDict(), 
        with_rgb: bool=None, with_normal: bool=None, with_feature_dim: int=None, with_env: bool=None, 
        return_buffer=False, return_details=False, render_per_obj=False, render_per_obj_in_total=False
        ) -> dict:

        assert rays is not None or observer is not None, \
            'At least one of "rays" and "observer" should be specified'
        
        if drawables is None:
            drawables = scene.get_drawables()
        
        if observer is not None:
            # assert observer.id in scene.observers, f"Observer {observer.id} is not part of scene {scene.id}"
            scene.asset_bank.preprocess_per_render_frame(renderer=self, observer=observer)
            near = near or observer.near
            far = far or observer.far
            
            # Filter drawables using observers' view frustum
            drawables = observer.filter_drawable_groups(drawables)

        if rayschunk is None:
            rayschunk = self.config.get("rayschunk", 0)

        with torch.set_grad_enabled(self.training):
            if rays is None:
                prefix_shape = None
                if isinstance(observer, (Camera, MultiCamBundle)):
                    rays = observer.get_all_rays(return_xy=True)
                    rays_fi = observer.i if isinstance(observer.i, torch.Tensor) else \
                        torch.full(rays[0].shape[:-1], observer.i, dtype=torch.long, device=rays[0].device)
                    rays.append(rays_fi)
                else:
                    rays = observer.get_all_rays()
            else:
                # Flatten rays
                prefix_shape = rays[0].shape[:-1]
                rays = [ri.flatten(0, -2) if i <= 2 else ri.flatten(0, -1) for i, ri in enumerate(rays)]
            
            render_rayschunk = functools.partial(
                self.__call__,
                scene=scene, observer=observer, drawables=drawables, near=near, far=far, bypass_ray_query_cfg=bypass_ray_query_cfg, 
                with_rgb=with_rgb, with_normal=with_normal, with_feature_dim=with_feature_dim, with_env=with_env,
                return_buffer=return_buffer, return_details=return_details, render_per_obj=render_per_obj, render_per_obj_in_total=render_per_obj_in_total
            )
            
            if self.training or (not rayschunk) or (rays[0].shape[0] <= rayschunk):
                ret = render_rayschunk(*rays)
            else:
                assert (not return_buffer) and (not return_details), "batchify_query does not work when return_buffer is True or return_details is True"
                ret = batchify_query(render_rayschunk, *rays, chunk=rayschunk, show_progress=show_progress)
            ret.update(rays_o=rays[0], rays_d=rays[1])
            if prefix_shape is not None and len(prefix_shape)>1:
                for k in self.image_keys:
                    if k in ret['rendered']:
                        v = ret['rendered'][k]
                        ret['rendered'][k] = v.unflatten(0, prefix_shape)
                    # Restore shape if any
                    for oid, odict in ret.get('rendered_per_obj_in_total', {}).items():
                        if k in odict:
                            odict[k] = odict[k].unflatten(0, prefix_shape)
                    # Restore shape if any
                    for oid, odict in ret.get('rendered_per_obj', {}).items():
                        if k in odict:
                            odict[k] = odict[k].unflatten(0, prefix_shape)
        return ret

if __name__ == "__main__":
    def test_multi_buffer_collect_and_merge(device=torch.device('cuda')):
        from icecream import ic
        
        # NOTE: The duplicated ray inds should be viewed as the same pack when merging.
        volume_buffer_1 = dict(
            buffer_type="batched", 
            ray_inds_hit=torch.tensor([1,1,2,6,8], device=device), # Duplicated ray_inds 1,1 (the same ray hitting different batch-obj)
            batch_inds_hit=torch.tensor([0,1,0,0,0], device=device), 
            num_per_hit=3, 
            t=torch.tensor([[0.1,0.2,0.3], [1.1,1.2,1.3], [0.1,0.2,0.3], [0.1,0.2,0.3], [0.1,0.2,0.3]], device=device, dtype=torch.float)
        )
        
        volume_buffer_2 = dict(
            buffer_type="packed", 
            ray_inds_hit=torch.tensor([1,1,2,6,8], device=device),  # Duplicated ray_inds 1,1 (the same ray hitting different batch-obj)
            batch_inds_hit=torch.tensor([0,1,0,0,0], device=device), 
            pack_infos_hit=get_pack_infos_from_n(torch.tensor([2,3,2,2,1], device=device)), 
            t=torch.tensor([ 0.15,0.25,  0.11,0.12,0.21,  0.4,0.5,  0.05,0.15,  0.14 ], device=device, dtype=torch.float)
        )
        
        volume_buffer_3 = dict(
            buffer_type="packed", 
            ray_inds_hit=torch.tensor([0,1,2,6,8], device=device), 
            pack_infos_hit=get_pack_infos_from_n(torch.tensor([1,2,3,2,4], device=device)), 
            t=torch.tensor([ 0.05,  0.31,0.34,   0.24,0.26,0.28,  0.6,0.7,   0.5,0.6,0.7,0.71], device=device, dtype=torch.float)
        )
        
        volume_buffers = [volume_buffer_1, volume_buffer_2, volume_buffer_3]
        total_num_rays = 10
        ray_visible_samples = torch.zeros([total_num_rays], dtype=torch.long, device=device)
        for volume_buffer in volume_buffers:
            if (buffer_type:=volume_buffer['buffer_type']) != 'empty':
                ray_inds_hit = volume_buffer['ray_inds_hit']
                num_per_hit = torch.full_like(ray_inds_hit, volume_buffer['num_per_hit'], device=device, dtype=torch.long) if buffer_type == 'batched' else volume_buffer['pack_infos_hit'][:,1]
                
                ray_inds_hit_collect, cnt = torch.unique_consecutive(ray_inds_hit, return_counts=True)
                if (cnt > 1).any():
                    num_per_hit_collect = torch.zeros([total_num_rays], device=device, dtype=torch.long).index_add_(0, ray_inds_hit, num_per_hit)[ray_inds_hit_collect]
                    pack_infos_collect = get_pack_infos_from_n(num_per_hit_collect)
                else:
                    num_per_hit_collect = num_per_hit
                    pack_infos_collect = volume_buffer['pack_infos_hit'] if 'pack_infos_hit' in volume_buffer else get_pack_infos_from_n(num_per_hit)

                ray_visible_samples.index_add_(0, ray_inds_hit_collect, num_per_hit_collect)
                # ray_visible_samples.index_add_(0, ray_inds_hit, num_per_hit) # NOTE: equivalent
                
                volume_buffer.update(ray_inds_hit_collect=ray_inds_hit_collect, pack_infos_collect=pack_infos_collect)
                
                # When collecting, uniformly flatten into packed version of `xxx_collect`

        ic(ray_visible_samples)

        total_pack_infos_sparse = get_pack_infos_from_n(ray_visible_samples)
        total_ray_inds_hit = ray_visible_samples.nonzero().long()[..., 0]
        total_pack_infos = total_pack_infos_sparse[total_ray_inds_hit]
        total_num_samples = total_pack_infos_sparse[-1,:].sum().item()
        total_depths = torch.zeros([total_num_samples], dtype=torch.float, device=device)
        
        current_pack_indices_buffer = total_pack_infos_sparse[:, 0].clone()
        
        for volume_buffer in volume_buffers:
            if (buffer_type:=volume_buffer['buffer_type']) != 'empty':
                ray_inds_hit_collect = volume_buffer['ray_inds_hit_collect']
                pack_infos_collect = volume_buffer['pack_infos_collect']
                # idx = interleave_arange_simple(pack_infos_collect[:,1], return_idx=False)
                # pidx_in_total = packed_add(idx, current_pack_indices_buffer, pack_infos_collect)
                pidx_in_total = interleave_linstep(current_pack_indices_buffer[ray_inds_hit_collect], pack_infos_collect[:,1], 1, False)
                total_depths[pidx_in_total] = volume_buffer['t'].flatten()
                current_pack_indices_buffer.index_add_(0, ray_inds_hit_collect, pack_infos_collect[:,1])

        # If total_pack_infos_sparse is passed in here instead of total_pack_infos, 
        #   there will be illegal memory access.
        total_depths_sorted, indices = packed_sort(total_depths, total_pack_infos)
        
        print(torch.equal(total_depths_sorted, total_depths[indices]))

    test_multi_buffer_collect_and_merge()