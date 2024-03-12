"""
@file   neus.py
@author Jianfei Guo, Shanghai AI Lab
@brief  neuralsim's API for large scale NeuS models.
"""

__all__ = [
    'LoTDForestNeuSObj', 
    'LoTDForestNeuSStreet', 
]

import pickle
from math import sqrt

import torch

from nr3d_lib.fmt import log
from nr3d_lib.config import ConfigDict

from nr3d_lib.models.fields_forest.utils import prepare_dense_grids
from nr3d_lib.models.fields_forest.neus import LoTDForestNeuSModel
from nr3d_lib.models.fields_forest.sdf import pretrain_forest_sdf_capsule, pretrain_forest_sdf_flat, pretrain_forest_sdf_road_surface

from app.models.asset_base import AssetAssignment, AssetMixin
from app.resources import Scene, SceneNode

class LoTDForestNeuSObj(AssetMixin, LoTDForestNeuSModel):
    assigned_to = AssetAssignment.OBJECT
    is_ray_query_supported = True

class LoTDForestNeuSStreet(AssetMixin, LoTDForestNeuSModel):
    assigned_to = AssetAssignment.OBJECT
    is_ray_query_supported = True
    @torch.no_grad()
    def asset_populate(
        self, 
        scene: Scene = None, obj: SceneNode = None, config: dict = None, 
        device=None, **kwargs):
        """
        Construct the forest representation
        """
        populate_mode = config.get('mode', 'from_dataset')
        
        log.info(f"Populate {self.__class__.__name__}, mode={populate_mode}, for obj={obj.id}")
        
        if populate_mode == 'from_dataset':
            assert 'split_block_info' in scene.metas.keys(), \
                f"populate_mode={populate_mode} requires `split_block_info` in `scene.metas`."\
                    " Make sure that `split_block_info` is generated when processing scene's scenario."
            assert scene.metas.get('align_orientation', False), \
                f"Rotation normalization is not compatible with populate_mode={populate_mode}"
            split_block_info = scene.metas['split_block_info']
            world_origin = split_block_info['world_origin']
            world_block_size = split_block_info['world_block_size']
            block_ks = split_block_info['block_ks']
            level = split_block_info.get('level', None)
            resolution = split_block_info.get('resolution', None)
        
        elif populate_mode == 'from_file':
            fpath = config['split_block_info_file']
            assert scene.metas.get('align_orientation', False), \
                f"Rotation normalization is not compatible with populate_mode={populate_mode}"
            with open(fpath, 'rb') as f:
                split_block_info = pickle.load(f)
            world_origin = split_block_info['world_origin']
            world_block_size = split_block_info['world_block_size']
            block_ks = split_block_info['block_ks']
            level = split_block_info.get('level', None)
            resolution = split_block_info.get('resolution', None)
        
        elif populate_mode == 'cuboid':
            # xyz_extend in world
            xyz_extend = scene.process_observer_infos(far_clip=config['extend_size']).all_frustum_pts
            # NOTE: !!! xyz extend in obj
            # NOTE: Apply pre-computed transform from dataset (usually for orientation alignment)
            scene.frozen_at_global_frame(0)
            xyz_extend = obj.world_transform.forward(xyz_extend, inv=True) / obj.scale.vec_3() # From world to street_obj
            scene.unfrozen()
            
            #---- Dense blocks
            bmin = xyz_extend.view(-1,3).min(0).values
            bmax = xyz_extend.view(-1,3).max(0).values
            aabb = torch.stack([bmin, bmax], 0)
            resolution, world_origin, world_block_size, level = prepare_dense_grids(aabb, **config['block_cfgs'])
            
            block_ks = torch.stack(torch.meshgrid([torch.arange(r, device=device) for r in resolution], indexing='ij'), -1).reshape(-1,3)

        elif populate_mode == 'inside_frustum':
            # xyz_extend in world
            xyz_extend = scene.process_observer_infos(far_clip=config['extend_size']).all_frustum_pts
            # NOTE: !!! xyz extend in obj
            # NOTE: Apply pre-computed transform from dataset (usually for orientation alignment)
            scene.frozen_at_global_frame(0)
            xyz_extend = obj.world_transform.forward(xyz_extend, inv=True) / obj.scale.vec_3() # From world to street_obj
            scene.unfrozen()
            
            #---- Dense blocks
            bmin = xyz_extend.view(-1,3).min(0).values
            bmax = xyz_extend.view(-1,3).max(0).values
            aabb = torch.stack([bmin, bmax], 0)
            resolution, world_origin, world_block_size, level = prepare_dense_grids(aabb, **config['block_cfgs'])
            
            #---- Inside-frustum checks
            gidx = torch.stack(torch.meshgrid([torch.arange(r, device=device) for r in resolution], indexing='ij'), -1).reshape(-1,3)
            block_centers = world_origin + world_block_size * (gidx+0.5)
            block_bounding_spheres = torch.zeros([gidx.shape[0], 4], device=device, dtype=torch.float)
            block_bounding_spheres[:, :3] = block_centers
            block_bounding_spheres[:, 3] = sqrt(3) * world_block_size/2.
            
            scene.frozen_at_full_global_frame()
            valid = torch.zeros([gidx.shape[0],], dtype=torch.bool, device=device)
            for cam in scene.get_cameras(only_valid=False):
                # [num_frames, num_blocks]
                inside_per_cam = cam.check_spheres_inside_frustum(block_bounding_spheres.unsqueeze(0), holistic_body=False)
                valid |= inside_per_cam.any(dim=0) # For all blocks, any frame would suffice
            scene.unfrozen() # Reset frozen status
            
            valid_inds = valid.nonzero().long()[..., 0]
            block_ks = gidx[valid_inds]
        else:
            raise RuntimeError(f"Invalid populate_mode={populate_mode}")
        
        super().populate(mode='from_corners', corners=block_ks, level=level, world_origin=world_origin, world_block_size=world_block_size)
    
    def asset_training_initialize(self, scene: Scene, obj: SceneNode, config: dict, logger=None, log_prefix: str=None):
        if config is None: config = dict()
        geo_init_method = self.implicit_surface.geo_init_method
        if geo_init_method == 'pretrain' or ('pretrain_after' in geo_init_method):
            if not self.implicit_surface.is_pretrained:
                if log_prefix is None:
                    log_prefix = obj.id
                  
                config = config.copy()
                obs_ref = config.pop('obs_ref')
                target_shape = config.pop('target_shape', 'capsule')
                
                with torch.no_grad():
                    scene.frozen_at_full_global_frame()
                    obs = scene.observers[obs_ref]
                    tracks = obs.world_transform.translation()
                    scene.unfrozen()
                    
                    scene.frozen_at_global_frame(0)
                    tracks_in_obj = obj.world_transform(tracks, inv=True) / obj.scale.vec_3()
                    scene.unfrozen()
                
                if target_shape == 'capsule':
                    pretrain_forest_sdf_capsule(self.implicit_surface, tracks_in_obj, **config, logger=logger, log_prefix=log_prefix)
                
                elif target_shape == 'flat':
                    floor_info = scene.metas['floor_info']
                    pretrain_forest_sdf_flat(self.implicit_surface, **floor_info, **config, logger=logger, log_prefix=log_prefix)
                
                elif target_shape == 'road_surface':
                    floor_info = scene.metas.get('floor_info', ConfigDict())
                    floor_info.update(config)
                    pretrain_forest_sdf_road_surface(self.implicit_surface, tracks_in_obj, **config, logger=logger, log_prefix=log_prefix)
                
                else:
                    raise RuntimeError(f"Invalid target_shape={target_shape}")
                
                self.implicit_surface.is_pretrained = ~self.implicit_surface.is_pretrained
                return True
        return False

    @classmethod
    def asset_compute_id(cls, scene: Scene = None, obj: SceneNode = None, class_name: str = None) -> str:
        return f"{cls.__name__}#{class_name or obj.class_name}#{scene.id}#{obj.id}"