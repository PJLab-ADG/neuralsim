"""
@file   lidar_loader.py
@author Jianfei Guo, Shanghai AI Lab
@brief  
- `LidarDataset`: Sampling returns individual rays from the lidar.
"""

__all__ = [
    'LidarDataset'
]

import random
import numpy as np
from typing import Dict, Iterator, List, Literal, Tuple, Union

import torch
import torch.utils.data as torch_data
from torch.utils.data.dataloader import DataLoader

from nr3d_lib.utils import collate_tuple_of_nested_dict

from .base_loader import SceneDataLoader
from .sampler import get_frame_sampler

class LidarDataset(torch_data.Dataset):
    def __init__(
        self, scene_loader: SceneDataLoader, *, 
        num_rays: int = 4096, num_points: int = None, 
        equal_mode: Literal['ray_batch', 'point_batch'] = 'ray_batch', 
        lidar_sample_mode: Literal['single_uniform', 'single_weighted', 'merged_random', 'merged_uniform', 'merged_weighted'] = 'merged_uniform', 
        multi_lidar_weight: List[float]=None, 
        frame_sample_mode: Literal['uniform', 'weighted_by_speed']='uniform', 
        ddp=False, **sampler_kwargs
        ) -> None:
        """ Sampler and data loader for LiDAR beams 

        Args:
            scene_loader (SceneDataLoader): The base SceneDataLoader
            num_rays (int, optional): Number of rays to sample. Defaults to 4096.
            num_points (int, optional): Number of ray points to sample (WIP). Defaults to None.
            equal_mode (Literal['ray_batch', 'point_batch'], optional): (WIP). Defaults to 'ray_batch'.
            lidar_sample_mode (Literal['single_uniform', 'single_weighted', 'merged_random', 'merged_uniform', 'merged_weighted'], optional):
                Determines sampling method for sampling beams from multi-LiDAR dataset. 
                - `single_uniform`: Uniformly sample one Lidar, then sample its beams.
                - `single_weighted`: Sample one Lidar weighted by `multi_lidar_weight`, then sample its beams.
                - `merged_random`: Merge all Lidar beams, sample `num_rays` beams with replacement.
                - `merged_equal`: From all Lidars, sample equal beams count each, with replacement.
                - `merged_weighted`: From all Lidars, sample different beams count each weighted by `multi_lidar_weight`, with replacement.
                Defaults to 'merged_uniform'.
            multi_lidar_weight (List[float], optional): Weight for different LiDARs in `single_weighted` and `merged_weighted` modes. Defaults to None. 
                Order corresponds to self.scene_loader.lidar_id_list (self.scene_loader.self.config.tags.lidar.list)
            frame_sample_mode (Literal['uniform', 'weighted_by_speed'], optional): Determines sampling method for sampling a frame from multiple frames. 
                - `uniform`:  equal probability
                - `weighted_by_speed`: probability based on motion speed
                Defaults to 'uniform'.
        """
        
        super().__init__()
        
        self.scene_loader = scene_loader
        self.scene_id_list = list(self.scene_loader.scene_bank.keys())
        
        self.lidar_sample_mode = lidar_sample_mode
        self.multi_lidar_weight: np.ndarray = None
        if 'weighted' in self.lidar_sample_mode:
            assert multi_lidar_weight is not None, f"Please specify `multi_lidar_weight`"
            multi_lidar_weight = np.array(multi_lidar_weight)
            self.multi_lidar_weight: np.ndarray = multi_lidar_weight / multi_lidar_weight.sum()

        self.equal_mode = equal_mode
        self.set_n_rays = num_rays
        self.set_n_pts = num_points
        
        # Init num_rays
        self.num_rays = num_rays

        self.frame_sample_mode = frame_sample_mode

        self.cur_it: int = None

        self.ddp = ddp
        self.sampler, self.scene_weights = get_frame_sampler(self.scene_loader, frame_sample_mode=self.frame_sample_mode, ddp=ddp, **sampler_kwargs)

    @property
    def device(self) -> torch.device:
        return self.scene_loader.device

    def record_prev(self, totol_n_pts: int):
        raise NotImplementedError("TODO in v0.4.2")
        if self.equal_mode == 'point_batch':
            self.num_rays = int(self.set_n_pts / (totol_n_pts/self.num_rays))

    def sample_single(self, scene_id: str, lidar_id: str, lidar_fi: int, num_rays: int = None):
        if num_rays is None: num_rays = self.num_rays
        data = self.scene_loader.get_lidar_gts(scene_id, lidar_id, lidar_fi, device=self.device, filter_if_configured=True)
        # Sample lidar on the filtered data according to certain rule
        inds = torch.randint(data['rays_o'].shape[0], [num_rays, ], device=self.device)
        scene = self.scene_loader.scene_bank[scene_id]
        lidar = scene.observers[lidar_id]
        
        rays_fidx = torch.full([num_rays, ], lidar_fi, dtype=torch.long, device=self.device)
        sample = dict(
            scene_id=scene_id, 
            lidar_id=lidar_id, lidar_fi=lidar_fi, # lidar_sel=None, 
            rays_o=data['rays_o'][inds], rays_d=data['rays_d'][inds], rays_fidx=rays_fidx, rays_sel=None
        )
        ground_truth = dict(ranges=data['ranges'][inds])
        
        # NOTE: Moved to trainer to allow for differentiable timestamps
        # if lidar.frame_global_ts is not None:
        #     # NOTE: `lidar_ts` is for freezing the scene at a certain timestamp;
        #     #       `rays_ts` is for network input.
        #     # sample['lidar_ts'] = sample['rays_ts'] = lidar.get_timestamps(fi=rays_fidx)
        #     sample['lidar_ts'] = lidar.frame_global_ts[lidar_fi]
        #     sample['rays_ts'] = lidar.get_timestamps(fi=rays_fidx)
        
        return sample, ground_truth

    def sample_merged(self, scene_id: str, lidar_fi: int, num_rays: int = None):
        if num_rays is None: num_rays = self.num_rays
        #---- Get raw merged lidar data
        data = self.scene_loader.get_merged_lidar_gts(scene_id, lidar_fi, device=self.device, filter_if_configured=True)
        
        # #---- DEBUG with box
        # scene = self.scene_loader.scene_bank[scene_id]
        # scene.slice_at(lidar_fi)
        # # Assemble multiLidarBundle node
        # lidars = [scene.observers[lid] for lid in self.scene_loader.lidar_id_list]
        # lidar = MultiRaysLidarBundle(lidars)
        # # Lidar points in local coordinates
        # pts = torch.addcmul(data['rays_o'], data['rays_d'], data['ranges'].unsqueeze(-1))
        # # Local to world transform of each point
        # l2w = lidar.world_transform[data['li']]
        # # Lidar points in world coordinates
        # pts = l2w.forward(pts)
        # filter_out_obj_dynamic_only = self.scene_loader.config.tags.lidar.filter_kwargs.get('filter_out_obj_dynamic_only', True)
        # if filter_out_obj_dynamic_only:
        #     obj_box_list = scene.metas['obj_box_list_per_frame_dynamic_only']
        # else:
        #     obj_box_list = scene.metas['obj_box_list_per_frame']
        # class_names = None or list(obj_box_list.keys())
        # data_frame_ind = lidar_fi + scene.data_frame_offset
        # all_box_list = [(obj_box_list[c][data_frame_ind] 
        #                     if (c in obj_box_list ) and (len(obj_box_list[c][data_frame_ind]) > 0)
        #                     else np.empty([0,15])) for c in class_names]
        # all_box_list = np.concatenate(all_box_list, axis=0)
        # # Only filter when all_box_list is not empty.
        # if len(all_box_list) > 0:
        #     # [num_obj, 15], where 15 = 12 (transform 3x4) + 3 (size)
        #     all_box_list = torch.tensor(all_box_list, dtype=torch.float, device=self.device)
        #     # DEBUG
        #     from nr3d_lib.plot import vis_lidar_and_boxes_o3d
        #     vis_lidar_and_boxes_o3d(pts.data.cpu().numpy(), all_box_list.data.cpu().numpy())
        
        #---- Sample lidar on the filtered data according to certain rule
        if self.lidar_sample_mode == 'merged_random':
            inds = torch.randint(data['rays_o'].shape[0], [num_rays, ], device=self.device)
        elif self.lidar_sample_mode in ['merged_equal', 'merged_weighted']:
            # NOTE: These modes are helpful when one of the lidar possesses significanly more data points. 
            #       e.g. in waymo: lidar_TOP 150k pts vs. others 3~5k pts
            cnt = torch.zeros([len(self.scene_loader.lidar_id_list),], dtype=torch.long, device=self.device)
            unique_i, unique_cnt = torch.unique_consecutive(data['li'], return_counts=True)
            cnt[unique_i] = unique_cnt
            cnt = cnt.data.cpu().numpy()
            
            # NOTE: Respect the given weight, but disregard LiDARs with zero beam count
            weight = np.full([len(cnt),], 1/len(cnt)) if self.multi_lidar_weight is None else self.multi_lidar_weight
            weight[cnt==0] = 0
            weight = weight / weight.sum()
            num_rays_each_lidar = np.array(num_rays * weight, dtype=int).tolist()
            # Make sure to be num_rays
            if sum(num_rays_each_lidar) != num_rays:
                for i, n in enumerate(num_rays_each_lidar):
                    if n != 0:
                        break
                num_rays_each_lidar[i] += (num_rays - sum(num_rays_each_lidar))
            cumu_cnt = [0, *np.cumsum(cnt).tolist()]
            inds = torch.cat([torch.randint(cumu_cnt[li], cumu_cnt[li+1], [num,], device=self.device, dtype=torch.long) for li,num in enumerate(num_rays_each_lidar) if num > 0])

            li = data['li'][inds]
            rays_o = data['rays_o'][inds]
            rays_d = data['rays_d'][inds]
        
        else:
            raise RuntimeError(f"Invalid lidar_sample_mode={self.lidar_sample_mode}")
        
        scene = self.scene_loader.scene_bank[scene_id]
        all_lid = self.scene_loader.lidar_id_list
        rays_fidx = torch.full([num_rays, ], lidar_fi, dtype=torch.long, device=self.device)
        
        sample = dict(
            scene_id=scene_id, 
            lidar_id=all_lid, lidar_fi=lidar_fi, # lidar_sel=li, 
            rays_o=rays_o, rays_d=rays_d, rays_fidx=rays_fidx, rays_sel=li
        )
        ground_truth = dict(ranges=data['ranges'][inds])
        
        # NOTE: Moved to trainer to allow for differentiable timestamps
        # all_li_ts_kf = [scene.observers[lid].frame_global_ts for lid in all_lid]
        # if all_li_ts_kf[0] is not None:
        #     all_li_ts_kf = torch.stack(all_li_ts_kf, dim=0)
        #     sample['lidar_ts'] = sample['rays_ts'] = all_li_ts_kf[li, lidar_fi]
        
        return sample, ground_truth

    def sample_lidar_id(self) -> str:
        assert 'merged' not in self.lidar_sample_mode, "Do not support `sample_cam_id` when `lidar_sample_mode` is merged"
        if self.multi_lidar_weight is not None:
            return np.random.choice(self.scene_loader.lidar_id_list, p=self.multi_lidar_weight)
        else:
            return random.choice(self.scene_loader.lidar_id_list)

    def __len__(self):
        # Total number of frames of all scenes
        return sum([len(scene) for scene in self.scene_loader.scene_bank]) 

    def __getitem__(self, index: int) -> Tuple[dict, dict]:
        # TODO: Allow for custom and different frame lengths across different sensors
        scene_idx, lidar_fi = self.scene_loader.get_scene_frame_idx(index)
        scene_id = self.scene_id_list[scene_idx]
        if 'merged' in self.lidar_sample_mode:
            ret = self.sample_merged(scene_id, lidar_fi)
        elif 'single' in self.lidar_sample_mode:
            lidar_id = self.sample_lidar_id()
            ret = self.sample_single(scene_id, lidar_id, lidar_fi)
        else:
            raise RuntimeError(f"Invalid lidar_sample_mode={self.lidar_sample_mode}")
        return ret

    def get_dataloader(self, num_workers: int=0):
        return DataLoader(
            self, sampler=self.sampler, collate_fn=collate_tuple_of_nested_dict, 
            num_workers=0 if (self.scene_loader.preload or not self.ddp) else num_workers)
