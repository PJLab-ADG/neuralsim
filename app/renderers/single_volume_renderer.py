"""
@file   single_volume_renderer.py
@author Jianfei Guo, Shanghai AI Lab & Nianchen Deng, Shanghai AI Lab
@brief  Single-object scene volume rendering. 
"""

import functools
from operator import itemgetter
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from nr3d_lib.config import ConfigDict
from nr3d_lib.profile import profile
from nr3d_lib.models.utils import batchify_query
from nr3d_lib.render.volume_graphics import packed_alpha_to_vw, ray_alpha_to_vw
from nr3d_lib.render.pack_ops import get_pack_infos_from_n, merge_two_packs_sorted, packed_div, packed_sum

from app.resources import Scene, AssetBank
from app.resources.observers import Camera, MultiCamBundle, Lidar, RaysLidar, MultiRaysLidarBundle
from app.renderers.utils import rotate_volume_buffer_nablas


class SingleVolumeRenderer(nn.Module):
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
        return self._config_train if self.training else self._config

    def populate(self, asset_bank: AssetBank):
        if self.config.get("enable_postprocessor", False):
            self.image_postprocessor = asset_bank["ImagePostprocessor"]
        else:
            self.image_postprocessor = None

    def _volume_integration(self, volume_buffer, rendered: dict):
        buffer_type = volume_buffer['buffer_type']
        ray_inds_hit = volume_buffer['ray_inds_hit']
        depth_use_normalized_vw = self.config.get('depth_use_normalized_vw', True)

        if buffer_type == "batched":
            alpha_to_vw = ray_alpha_to_vw
            sum = torch.sum
        elif buffer_type == 'packed':
            pack_infos_hit = volume_buffer['pack_infos_hit']
            alpha_to_vw = functools.partial(packed_alpha_to_vw, pack_infos=pack_infos_hit)
            def sum(input, _): return packed_sum(input, pack_infos_hit)

        volume_buffer['vw'] = vw = alpha_to_vw(volume_buffer['opacity_alpha'])
        rendered['mask_volume'][ray_inds_hit] = vw_sum = sum(vw, -1)
        if depth_use_normalized_vw:
            depth_w = packed_div(vw, vw_sum + 1e-10, pack_infos_hit) if buffer_type == "packed" \
                else vw / (vw_sum[..., None] + 1e-10)
        else:
            depth_w = vw
        rendered['depth_volume'][ray_inds_hit] = sum(depth_w * volume_buffer['t'], -1)
        
        if "rgb_volume" in rendered and "rgb" in volume_buffer:
            rendered['rgb_volume'][ray_inds_hit] = sum(vw[..., None] * volume_buffer['rgb'], -2)

        if "normals_volume" in rendered and "nablas_in_world" in volume_buffer:
            nablas = volume_buffer["nablas_in_world"] if self.training \
                else F.normalize(volume_buffer['nablas_in_world'].clamp_(-1, 1), dim=-1)
            rendered['normals_volume'][ray_inds_hit] = sum(vw[..., None] * nablas, -2)
        return rendered

    def _volume_integration_in_total(self, volume_buffer: dict, rendered: dict):
        ray_inds_hit, pack_infos_collect = itemgetter('ray_inds_hit', 'pack_infos_collect')(volume_buffer)
        vw_in_total = volume_buffer['vw_in_total'].flatten()
        
        rendered['mask_volume'][ray_inds_hit] = vw_sum = packed_sum(vw_in_total, pack_infos_collect)
        depth_w = packed_div(vw_in_total, vw_sum + 1e-10, pack_infos_collect)\
            if self.config.get('depth_use_normalized_vw', True) else vw_in_total
        
        rendered['depth_volume'][ray_inds_hit] = packed_sum(depth_w * volume_buffer['t'].flatten(), pack_infos_collect)

        if "rgb_volume" in rendered and "rgb" in volume_buffer:
            rendered['rgb_volume'][ray_inds_hit] = packed_sum(vw_in_total.unsqueeze(-1) * volume_buffer['rgb'].flatten(0,-2), pack_infos_collect)

        if "normals_volume" in rendered and "nablas_in_world" in volume_buffer:
            nablas = volume_buffer["nablas_in_world"] if self.training else F.normalize(volume_buffer['nablas_in_world'].clamp_(-1, 1), dim=-1)
            rendered['normals_volume'][ray_inds_hit] = packed_sum(vw_in_total.unsqueeze(-1) * nablas.flatten(0,-2), pack_infos_collect)
        return rendered

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

    @profile
    def forward(
        self, 
        rays_o: torch.Tensor, rays_d: torch.Tensor, rays_xy: torch.Tensor = None, rays_fi: torch.Tensor = None, 
        *,
        scene: Scene, observer: Union[Camera, MultiCamBundle, Lidar, RaysLidar, MultiRaysLidarBundle]=None, 
        # ray_query_config
        near=None, far=None, bypass_ray_query_cfg = ConfigDict(), 
        with_rgb: bool=None, with_normal: bool=None, with_feature_dim: int=None, only_cr: bool=False, with_env: bool=None, 
        return_buffer=False, return_details=False, render_per_obj=False, render_per_obj_in_total=False):
        
        assert rays_o.dim() == rays_d.dim() == 2, "rays_o and rays_d should have size of [N, 3]"

        objs = scene.get_drawable_groups_by_class_name(scene.main_class_name)
        sky_objs = scene.get_drawable_groups_by_class_name('Sky') if not only_cr else []
        distant_objs = scene.get_drawable_groups_by_class_name('Distant') if not only_cr else []
        
        dtype, device, config = torch.float, scene.device, self.config
        if with_rgb is None: with_rgb = config.get('with_rgb', True)
        if with_normal is None: with_normal = config.get('with_normal', False)
        if with_feature_dim is None: with_feature_dim = config.get('with_feature_dim', 0)
        if with_env is None: with_env = config.get('with_env', len(sky_objs) > 0)
        if near is None: near = config.get('near', None)
        if far is None: far = config.get('far', None)

        if with_rgb and scene.image_embeddings is not None:
            assert isinstance(observer, (Camera, MultiCamBundle)), f'Expected camera observers, but got observer type={observer}'
            assert rays_fi is not None, 'Need per-ray frame index `rays_fi` input when scene.image_embeddings is present.'
            h_image_embed = scene.image_embeddings[observer.id](rays_fi)
        else:
            h_image_embed = None

        #----------------------------------------------------
        #                 Prepare outputs
        #----------------------------------------------------
        total_num_rays = rays_o.shape[0]
        raw_per_obj_model = dict()
        total_num_samples_per_ray = torch.zeros(total_num_rays, dtype=torch.long, device=device)
        total_rendered = self._prep_empty_rendered(
            [total_num_rays], device=device, 
            with_rgb=with_rgb, with_normal=with_normal, with_feature_dim=with_feature_dim)
        
        if return_buffer:
            total_volume_buffer = dict(buffer_type='empty')
        if render_per_obj:
            rendered_per_obj = dict()
        if render_per_obj_in_total:
            rendered_per_obj_in_total = dict()

        def collect_buffer(volume_buffer: dict):
            if (buffer_type := volume_buffer['buffer_type']) != 'empty':
                ray_inds_hit = volume_buffer['ray_inds_hit']
                num_per_hit = torch.full_like(ray_inds_hit, volume_buffer['num_per_hit'], device=device, dtype=torch.long) \
                    if buffer_type == 'batched' else volume_buffer['pack_infos_hit'][:,1]
                total_num_samples_per_ray[ray_inds_hit] += num_per_hit
                pack_infos_hit = get_pack_infos_from_n(num_per_hit) \
                    if buffer_type == 'batched' else volume_buffer['pack_infos_hit']
                volume_buffer.update(ray_inds_hit_collect=ray_inds_hit, pack_infos_collect=pack_infos_hit)

        #----------------------------------------------------
        #               Query foreground model
        #----------------------------------------------------
        cr_model = None
        cr_ret = None
        cr_ray_tested = None
        cr_volume_buffer = None
        if len(objs) > 0:
            with profile("Query foreground model"):
                cr_obj = objs[0]
                cr_model = cr_obj.model
                cr_rays_o, cr_rays_d = scene.convert_rays_in_node(rays_o, rays_d, cr_obj)
                cr_ray_input = dict(
                    rays_o=cr_rays_o, rays_d=cr_rays_d, near=near, far=far,
                    rays_fi=rays_fi, rays_xy=rays_xy, rays_h_appear_embed=h_image_embed)
                cr_ray_tested = cr_model.ray_test(**cr_ray_input)
                ray_query_config = ConfigDict(**cr_model.ray_query_cfg, **config)
                ray_query_config.update(with_rgb=with_rgb, with_normal=with_normal, with_feature_dim=with_feature_dim) # Possible override
                for key, value in bypass_ray_query_cfg.get(cr_obj.class_name, {}).items():
                    ray_query_config[key] = value
                cr_ret = cr_model.ray_query(
                    ray_input=cr_ray_input, ray_tested=cr_ray_tested, config=ray_query_config, 
                    return_buffer=True, return_details=return_details, render_per_obj=render_per_obj)
                cr_ret.update(class_name=cr_obj.class_name, model_id=cr_model.id)
                # Collect per obj rendering
                if render_per_obj:
                    cr_rendered = rendered_per_obj[cr_obj.id] = cr_ret.pop('rendered')
                    # Rotate obj's normals to world
                    if 'normals_volume' in cr_rendered:
                        cr_rendered['normals_volume_in_world'] = cr_obj.world_transform.rotate(cr_rendered['normals_volume'])
                if render_per_obj_in_total:
                    # Allocate empty data. Will be filled when volume rendering
                    rendered_per_obj_in_total[cr_obj.id] = self._prep_empty_rendered(
                        [total_num_rays], device=device, 
                        with_rgb=with_rgb, with_normal=with_normal, with_feature_dim=with_feature_dim)
                # Collect volume buffer for merging
                if cr_ret['volume_buffer']['buffer_type'] != 'empty':
                    cr_volume_buffer = cr_ret['volume_buffer']
                    collect_buffer(cr_volume_buffer)
                    # Rotate obj's normals to world
                    if 'nablas' in cr_volume_buffer:
                        o2w_rot = cr_obj.world_transform.rotation().detach() # Removing gradients on nablas can eliminate interference with the pose gradient.
                        cr_volume_buffer['nablas_in_world'] = rotate_volume_buffer_nablas(o2w_rot, cr_volume_buffer['nablas'], cr_volume_buffer)
                raw_per_obj_model[cr_obj.id] = cr_ret

        #----------------------------------------------------
        #               Query background model
        #----------------------------------------------------
        dv_ret = None
        dv_volume_buffer = None
        if (not only_cr) and len(distant_objs) > 0:
            with profile("Query background model"):
                dv_obj = distant_objs[0]
                dv_model = dv_obj.model
                # NOTE: If cr_obj is present, distant-view model should use rays in cr_obj
                if cr_model is not None:
                    dv_ray_input = cr_ray_input
                else:
                    dv_ray_input = dict(
                        rays_o=rays_o, rays_d=rays_d, 
                        rays_fi=rays_fi, rays_xy=rays_xy, rays_h_appear_embed=h_image_embed)
                dv_ray_input.update(near=rays_o.new_full([total_num_rays], near), far=None)
                
                # NOTE: For rays that pass cr's ray_test, the distant model's sampling starts from cr's `far`
                if cr_ray_tested is not None and cr_ray_tested['num_rays'] > 0:
                    dv_ray_input["near"][cr_ray_tested['ray_inds']] = cr_ray_tested["far"]
                
                # NOTE: Detach pose gradient on distant-view branch (if any). This significantly helps pose refinement.
                dv_ray_input['rays_o'] = dv_ray_input['rays_o'].detach()
                dv_ray_input['rays_d'] = dv_ray_input['rays_d'].detach()

                dv_ray_tested = dict(
                    **dv_ray_input, 
                    num_rays=total_num_rays, 
                    ray_inds=torch.arange(total_num_rays, device=device)
                )
                ray_query_config = ConfigDict(**dv_model.ray_query_cfg, **config)
                ray_query_config.update(with_rgb=with_rgb, with_normal=with_normal, with_feature_dim=with_feature_dim) # Possible override
                for key, value in bypass_ray_query_cfg.get(dv_obj.class_name, {}).items():
                    ray_query_config[key] = value
                dv_ret: dict = dv_model.ray_query(
                    ray_input=dv_ray_input, ray_tested=dv_ray_tested, config=ray_query_config, 
                    return_buffer=True, return_details=return_details, render_per_obj=render_per_obj)
                dv_ret.update(class_name=dv_obj.class_name, model_id=dv_model.id)
                if render_per_obj:
                    dv_rendered = rendered_per_obj[dv_obj.id] = dv_ret.pop('rendered')
                    # Rotate obj's normals to world
                    if 'normals_volume' in dv_rendered:
                        dv_rendered['normals_volume_in_world'] = dv_obj.world_transform.rotate(dv_rendered['normals_volume'])
                if render_per_obj_in_total:
                    # Allocate empty data. Will be filled when volume rendering
                    rendered_per_obj_in_total[dv_obj.id] = self._prep_empty_rendered(
                        [total_num_rays], device=device, 
                        with_rgb=with_rgb, with_normal=with_normal, with_feature_dim=with_feature_dim)
                # Collect information
                if dv_ret['volume_buffer']['buffer_type'] != 'empty':
                    dv_volume_buffer = dv_ret['volume_buffer']
                    collect_buffer(dv_volume_buffer)
                    # Rotate obj's normals to world
                    if 'nablas' in dv_volume_buffer:
                        o2w_rot = dv_obj.world_transform.rotation().detach() # Removing gradients on nablas can eliminate interference with the pose gradient.
                        dv_volume_buffer['nablas_in_world'] = rotate_volume_buffer_nablas(o2w_rot, dv_volume_buffer['nablas'], dv_volume_buffer)
                raw_per_obj_model[dv_obj.id] = dv_ret

        #----------------------------------------------------
        #            Make total voloume buffer
        #----------------------------------------------------
        with profile("Merge volume buffers"):
            if cr_volume_buffer is not None and dv_volume_buffer is not None:
                #---------- Merge two buffers (cr+dv)
                total_ray_inds_hit = total_num_samples_per_ray.nonzero().long()[..., 0]
                pidx_dv, pidx_cr, total_pack_infos = merge_two_packs_sorted(
                    dv_volume_buffer['t'].flatten(), dv_volume_buffer['pack_infos_collect'], dv_volume_buffer['ray_inds_hit'], 
                    cr_volume_buffer['t'].flatten(), cr_volume_buffer['pack_infos_collect'], cr_volume_buffer['ray_inds_hit'], 
                )
                total_num_samples = total_pack_infos[-1, :].sum().item()
                total_depths = torch.zeros([total_num_samples], dtype=torch.float, device=device)
                total_depths[pidx_dv], total_depths[pidx_cr] = dv_volume_buffer['t'].flatten(), cr_volume_buffer['t'].flatten()
                total_alphas = torch.zeros([total_num_samples], dtype=torch.float, device=device)
                total_alphas[pidx_dv], total_alphas[pidx_cr] = dv_volume_buffer['opacity_alpha'].flatten(), cr_volume_buffer['opacity_alpha'].flatten()
                
                total_volume_buffer = dict(
                    buffer_type='packed', ray_inds_hit=total_ray_inds_hit, pack_infos_hit=total_pack_infos, 
                    t=total_depths, opacity_alpha=total_alphas
                )
                if with_rgb:
                    total_rgbs = torch.zeros([total_num_samples, 3], dtype=torch.float, device=device)
                    if 'rgb' in cr_volume_buffer:
                        total_rgbs[pidx_cr] = cr_volume_buffer['rgb'].flatten(0, -2)
                    if 'rgb' in dv_volume_buffer:
                        total_rgbs[pidx_dv] = dv_volume_buffer['rgb'].flatten(0, -2)
                    total_volume_buffer['rgb'] = total_rgbs
                if with_normal:
                    total_nablas = torch.zeros([total_num_samples, 3], dtype=torch.float, device=device)
                    if 'nablas_in_world' in cr_volume_buffer:
                        total_nablas[pidx_cr] = cr_volume_buffer['nablas_in_world'].flatten(0, -2)
                    if 'nablas_in_world' in dv_volume_buffer:
                        total_nablas[pidx_dv] = dv_volume_buffer['nablas_in_world'].flatten(0, -2)
                    total_volume_buffer['nablas_in_world'] = total_nablas
                if with_feature_dim:
                    total_feature = torch.zeros([total_num_samples, with_feature_dim], dtype=torch.float, device=device)
                    if 'feature' in cr_volume_buffer:
                        total_feature[pidx_cr] = cr_volume_buffer['feature'].flatten(0, -2)
                    if 'feature' in dv_volume_buffer:
                        total_feature[pidx_dv] = dv_volume_buffer['feature'].flatten(0, -2)
                    total_volume_buffer['feature'] = total_feature
            elif cr_volume_buffer is not None:
                total_volume_buffer = cr_volume_buffer
            elif dv_volume_buffer is not None:
                total_volume_buffer = dv_volume_buffer
            else:
                total_volume_buffer = dict(buffer_type="empty")

        #----------------------------------------------------
        #                 Volume rendering
        #----------------------------------------------------
        with profile("Volume rendering"):
            total_num_rays_hit = 0
            if total_volume_buffer['buffer_type'] != 'empty' and \
                (total_num_rays_hit := total_volume_buffer['ray_inds_hit'].numel() ) > 0:
                self._volume_integration(total_volume_buffer, total_rendered)
                if return_buffer or render_per_obj_in_total:
                    if cr_volume_buffer is not None and dv_volume_buffer is not None:
                        cr_volume_buffer['vw_in_total'] = total_volume_buffer['vw'][pidx_cr]
                        dv_volume_buffer['vw_in_total'] = total_volume_buffer['vw'][pidx_dv]
                        if render_per_obj_in_total:
                            # NOTE: close-range's part when rendering total
                            self._volume_integration_in_total(cr_volume_buffer, rendered_per_obj_in_total[cr_obj.id])
                            # NOTE: distant-view's part when rendering total
                            self._volume_integration_in_total(dv_volume_buffer, rendered_per_obj_in_total[dv_obj.id])
                    elif cr_volume_buffer is not None:
                        cr_volume_buffer['vw_in_total'] = total_volume_buffer['vw']
                        if render_per_obj_in_total:
                            self._volume_integration_in_total(cr_volume_buffer, rendered_per_obj_in_total[cr_obj.id])
                    elif dv_volume_buffer is not None:
                        dv_volume_buffer['vw_in_total'] = total_volume_buffer['vw']
                        if render_per_obj_in_total:
                            self._volume_integration_in_total(dv_volume_buffer, rendered_per_obj_in_total[dv_obj.id])
        
        #----------------------------------------------------
        #                    Sky model
        #----------------------------------------------------
        if with_rgb:
            total_rendered["rgb_volume_occupied"] = total_rendered["rgb_volume"]
            if with_env and len(sky_objs) > 0:
                with profile("Query sky"):
                    sky_obj = sky_objs[0]
                    sky_model = sky_obj.model
                    env_rgb = sky_model(v=F.normalize(rays_d, dim=-1), h_appear_embed=h_image_embed)
                    total_rendered['rgb_sky'] = env_rgb
                    total_rendered['rgb_volume_non_occupied'] = env_rgb_blend = (1. - total_rendered['mask_volume'][..., None]) * env_rgb
                    # NOTE: Avoid inplace op
                    total_rendered['rgb_volume'] = total_rendered["rgb_volume"] + env_rgb_blend

        #----------------------------------------------------
        #                    Image Post-processing
        #----------------------------------------------------
        if with_rgb and self.image_postprocessor:
            with profile("Postprocess image"):
                assert rays_xy is not None, "Need `rays_xy` input when image_postprocessor is present."
                total_rendered["rgb_volume"] = self.image_postprocessor(h_image_embed, rays_xy, total_rendered["rgb_volume"])

        #----------------------------------------------------
        #                   Return results
        #----------------------------------------------------
        ret = dict(
            # total_num_rays_hit=total_num_rays_hit, 
            ray_intersections=dict(
                samples_cnt=total_num_samples_per_ray
            ), 
            rendered=total_rendered
        )
        if return_buffer:
            ret['volume_buffer'] = total_volume_buffer
        
        if return_details:
            ret['raw_per_obj_model'] = raw_per_obj_model

        if render_per_obj:
            ret["rendered_per_obj"] = rendered_per_obj
        
        if render_per_obj_in_total:
            ret["rendered_per_obj_in_total"] = rendered_per_obj_in_total
        
        return ret

    def render(
        self, scene: Scene, *, 
        rays: List[torch.Tensor] = None, # [rays_o, rays_d] or [rays_o, rays_d, rays_xy, rays_fi]
        observer: Union[Camera, MultiCamBundle, Lidar, RaysLidar, MultiRaysLidarBundle] = None, 
        near: float = None, far: float = None,
        show_progress: bool = False, rayschunk: int=None, 
        bypass_ray_query_cfg = ConfigDict(), 
        with_rgb: bool = None, with_normal: bool = None, with_feature_dim: int = None, only_cr=False, with_env: bool = None,
        return_buffer=False, return_details=False, render_per_obj=False, render_per_obj_in_total=False):
        
        assert rays is not None or observer is not None, \
            'At least one of "rays" and "observer" should be specified'

        if observer is not None:
            scene.asset_bank.preprocess_per_render_frame(renderer=self, observer=observer)
            near = near or observer.near
            far = far or observer.far

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
                self.__call__, # self.forward, 
                scene=scene, observer=observer, near=near, far=far, only_cr=only_cr, with_env=with_env,
                with_rgb=with_rgb, with_normal=with_normal, with_feature_dim=with_feature_dim, 
                bypass_ray_query_cfg=bypass_ray_query_cfg, 
                return_buffer=return_buffer, return_details=return_details, 
                render_per_obj=render_per_obj, render_per_obj_in_total=render_per_obj_in_total
            )

            if self.training or (not rayschunk) or (rays[0].shape[0] <= rayschunk):
                ret = render_rayschunk(*rays)
            else:
                assert (not return_buffer) and (not return_details), \
                    "batchify_query does not work when return_buffer=True or return_details=True"
                ret = batchify_query(render_rayschunk, *rays, chunk=rayschunk, show_progress=show_progress)
            ret.update(rays_o=rays[0], rays_d=rays[1])
            if prefix_shape is not None and len(prefix_shape)>1:
                for k in self.image_keys:
                    if k in ret['rendered']:
                        ret['rendered'][k] = ret['rendered'][k].unflatten(0, prefix_shape)
                    # Restore shape if any
                    for oid, odict in ret.get('rendered_per_obj_in_total', {}).items():
                        if k in odict:
                            odict[k] = odict[k].unflatten(0, prefix_shape)
                    # Restore shape if any
                    for oid, odict in ret.get('rendered_per_obj', {}).items():
                        if k in odict:
                            odict[k] = odict[k].unflatten(0, prefix_shape)
        return ret
