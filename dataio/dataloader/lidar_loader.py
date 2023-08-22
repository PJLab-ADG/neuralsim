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

from .base import SceneDataLoader, FrameRandomSampler

class LidarDataset(torch_data.Dataset):
    def __init__(
        self, dataset: SceneDataLoader, *, 
        num_rays: int = 4096, num_points: int = None, 
        equal_mode: Literal['ray_batch', 'point_batch'] = 'ray_batch', 
        lidar_sample_mode: Literal['single_uniform', 'single_weighted', 'merged_random', 'merged_uniform', 'merged_weighted'] = 'merged_uniform', 
        multi_lidar_weight: List[float]=None, 
        frame_sample_mode: Literal['uniform', 'weighted_by_speed']='uniform', 
        ) -> None:
        """ Sampler and data loader for LiDAR beams 

        Args:
            dataset (SceneDataLoader): The base SceneDataLoader
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
                Order corresponds to self.dataset.lidar_id_list (self.dataset.self.config.tags.lidar.list)
            frame_sample_mode (Literal['uniform', 'weighted_by_speed'], optional): Determines sampling method for sampling a frame from multiple frames. 
                - `uniform`:  equal probability
                - `weighted_by_speed`: probability based on motion speed
                Defaults to 'uniform'.
        """
        
        super().__init__()
        
        self.dataset = dataset
        self.scene_id_list = list(self.dataset.scene_bank.keys())
        
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

        self.cur_it: int = np.inf

    @property
    def device(self):
        return self.dataset.device

    def record_prev(self, totol_n_pts: int):
        raise NotImplementedError("TODO in v0.4.2")
        if self.equal_mode == 'point_batch':
            self.num_rays = int(self.set_n_pts / (totol_n_pts/self.num_rays))

    def sample_single(self, scene_id: str, lidar_id: str, frame_id: int, num_rays: int = None):
        if num_rays is None: num_rays = self.num_rays
        data = self.dataset.get_lidar_gts(scene_id, lidar_id, frame_id, device=self.device, filter_if_configured=True)
        # Sample lidar on the filtered data according to certain rule
        inds = torch.randint(data['rays_o'].shape[0], [num_rays, ], device=self.device)
        sample = dict(
            scene_id=scene_id, lidar_id=lidar_id, frame_id=frame_id, 
            selects=dict(rays_o=data['rays_o'][inds], rays_d=data['rays_d'][inds]))
        ground_truth = dict(ranges=data['ranges'][inds])
        return sample, ground_truth

    def sample_merged(self, scene_id: str, frame_id: int, num_rays: int = None):
        if num_rays is None: num_rays = self.num_rays
        sample = dict(scene_id=scene_id, frame_id=frame_id)
        #---- Get raw merged lidar data
        data = self.dataset.get_merged_lidar_gts(scene_id, frame_id, device=self.device, filter_if_configured=True)
        
        # NOTE: DEBUG with box
        # scene = self.dataset.scene_bank[scene_id]
        # scene.frozen_at(frame_id)
        # # Assemble multiLidarBundle node
        # lidars = [scene.observers[lid] for lid in self.dataset.lidar_id_list]
        # lidar = MultiRaysLidarBundle(lidars)
        # # Lidar points in local coordinates
        # pts = torch.addcmul(data['rays_o'], data['rays_d'], data['ranges'].unsqueeze(-1))
        # # Local to world transform of each point
        # l2w = lidar.world_transform[data['i']]
        # # Lidar points in world coordinates
        # pts = l2w.forward(pts)
        # filter_out_obj_dynamic_only = self.dataset.config.tags.lidar.filter_kwargs.get('filter_out_obj_dynamic_only', True)
        # if filter_out_obj_dynamic_only:
        #     obj_box_list = scene.metas['obj_box_list_per_frame_dynamic_only']
        # else:
        #     obj_box_list = scene.metas['obj_box_list_per_frame']
        # class_names = None or list(obj_box_list.keys())
        # data_frame_ind = frame_id + scene.data_frame_offset
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
            cnt = torch.zeros([len(self.dataset.lidar_id_list),], dtype=torch.long, device=self.device)
            unique_i, unique_cnt = torch.unique_consecutive(data['i'], return_counts=True)
            cnt[unique_i] = unique_cnt
            cnt = cnt.data.cpu().numpy()
            
            # NOTE: Respect the given weight, but disregard LiDARs with zero beam count
            weight = np.full([len(cnt),], 1/len(cnt)) if self.multi_lidar_weight is None else self.multi_lidar_weight
            weight[cnt==0] = 0
            weight = weight / weight.sum()
            num_rays_each_lidar = np.array(num_rays * weight, dtype=int).tolist()
            cumu_cnt = [0, *np.cumsum(cnt).tolist()]
            inds = torch.cat([torch.randint(cumu_cnt[i], cumu_cnt[i+1], [num,], device=self.device, dtype=torch.long) for i,num in enumerate(num_rays_each_lidar) if num > 0])
        else:
            raise RuntimeError(f"Invalid lidar_sample_mode={self.lidar_sample_mode}")
        #---- Gather on the sampled indices
        sample = dict(
            scene_id=scene_id, lidar_id=self.dataset.lidar_id_list, frame_id=frame_id, 
            selects=dict(i=data['i'][inds], rays_o=data['rays_o'][inds], rays_d=data['rays_d'][inds]))
        ground_truth = dict(ranges=data['ranges'][inds])
        return sample, ground_truth

    def sample_lidar_id(self) -> str:
        assert 'merged' not in self.lidar_sample_mode, "Do not support `sample_cam_id` when `lidar_sample_mode` is merged"
        if self.multi_lidar_weight is not None:
            return np.random.choice(self.dataset.lidar_id_list, p=self.multi_lidar_weight)
        else:
            return random.choice(self.dataset.lidar_id_list)

    def get_index(self, index: int):
        # From holistic index to scene_idx and frame_idx
        scene_idx = 0
        while index >= 0:
            index -= len(self.dataset.scene_bank[scene_idx])
            scene_idx += 1
        return (scene_idx - 1), int(index + len(self.dataset.scene_bank[scene_idx - 1]))

    def __len__(self):
        return sum([len(scene) for scene in self.dataset.scene_bank]) # Total number of frames of all scenes

    def __getitem__(self, index: int) -> Tuple[dict, dict]:
        scene_idx, frame_id = self.get_index(index)
        scene_id = self.scene_id_list[scene_idx]
        if 'merged' in self.lidar_sample_mode:
            ret = self.sample_merged(scene_id, frame_id)
        elif 'single' in self.lidar_sample_mode:
            lidar_id = self.sample_lidar_id()
            ret = self.sample_single(scene_id, lidar_id, frame_id)
        else:
            raise RuntimeError(f"Invalid lidar_sample_mode={self.lidar_sample_mode}")
        return ret

    def get_random_sampler(self, multi_scene_balance=True, replacement=True):
        return FrameRandomSampler(
            self.dataset, 
            multi_scene_balance=multi_scene_balance, replacement=replacement, frame_sample_mode=self.frame_sample_mode)

    def get_dataloader(self, ddp=False):
        return DataLoader(self, sampler=self.get_random_sampler(), collate_fn=collate_tuple_of_nested_dict, num_workers=0)
