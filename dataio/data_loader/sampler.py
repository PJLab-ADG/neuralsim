"""
@file   sampler.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Defines scene-frame sampler and scene samplers and their DDP variant.
"""

__all__ = [
    "get_frame_sampler", 
    "get_scene_sampler", 
]

import math
import numpy as np
from typing import Dict, Iterator, List, Literal, Optional, Sequence, Sized, Tuple, Union

import torch
import torch.distributed as dist
import torch.utils.data as torch_data

from nr3d_lib.maths import geometric_mean
from nr3d_lib.models.importance import ImpSampler

from .base_loader import SceneDataLoader

def get_frame_sampler(
    scene_loader: SceneDataLoader, 
    replacement: bool = True, 
    scene_sample_mode: Literal['uniform', 'weighted_by_len'] = 'weighted_by_len', 
    frame_sample_mode: Literal['uniform', 'fixed', 'weighted_by_speed', 'error_map'] = 'uniform', 
    ddp = False, seed: int = 0, drop_last = False, 
    
    imp_samplers: Dict[str, Dict[str,ImpSampler]] = None, 
    **kwargs, 
    ):
    """ A sampler that randomly samples scene indices and frame indices. 
    The sampled indices represent an overall index across all frames of all scenes. 
    You can use `scene_loader.get_scene_frame_idx` to retrieve the specific scene_idx and frame_idx.
    - Suitable for frame-wise dataset e.g. PixelDataset, LidarDataset, ImageDataset, ImagePatchDataset
    - Not suitable for scene-wise dataset e.g. JointFramePixelDataset

    Args:
        scene_loader (SceneDataLoader): The base SceneDataLoader
        replacement (bool, optional): Whether replace when sampling. Defaults to True.
        scene_sample_mode (Literal['uniform', 'weighted_by_len'], optional): 
            Determines the method for sampling a scene from multiple options.
            - `uniform`: Each scene will be sampled with equal probability.
            - `weighted_by_len`: Probability of a scene being chosen is based on its length.
            Only meaningful when sampling multiple scenes.
            Defaults to 'weighted_by_len'.
        frame_sample_mode (Literal['uniform', 'weighted_by_speed'], optional): 
            Determines the method for sampling a frame from multiple options.
            - `uniform`: Each frame has an equal probability of being chosen.
            - `weighted_by_speed`: Probability of a frame being chosen is based on motion speed.
                (Not very robust for now)
            - `error_map`: Probability of a frame being chosen is based on each frame's accumulated error map.
            - `fixed`: Fixed at one single frame
            Defaults to 'uniform'.
        ddp (bool, optional): Whether in DDP mode.
    """
    scene_bank = scene_loader.scene_bank
    scene_lengths = [len(scene) for scene in scene_bank]
    num_frames = sum(scene_lengths)
    scene_weights = get_scene_weights(scene_loader, scene_sample_mode=scene_sample_mode)
    
    if frame_sample_mode == 'uniform':
        frame_weights = get_frame_weights_uniform(scene_loader, scene_weights, **kwargs)
    elif frame_sample_mode == 'fixed':
        frame_weights = get_frame_weights_fixed(scene_loader, scene_weights, **kwargs)
    elif frame_sample_mode == 'weighted_by_speed':
        frame_weights = get_frame_weights_by_speed(scene_loader, scene_weights, **kwargs)
    elif frame_sample_mode == 'error_map':
        if imp_samplers is None:
            # If not given: use uniform to initialize instead, \
            #   and then update with real imp_samplers later via set_weights()
            frame_weights = get_frame_weights_uniform(scene_loader, scene_weights, **kwargs)
        else:
            frame_weights = get_frame_weights_from_error_map(scene_loader, scene_weights, imp_samplers=imp_samplers, **kwargs)
    else:
        raise RuntimeError(f"Invalid frame_sample_mode={frame_sample_mode}")
    if ddp:
        frame_sampler = DistributedWeightedRandomSamplerSampler(
            num_samples=num_frames, mode='weighted_random', weights=frame_weights, 
            replacement=replacement, seed=seed, drop_last=drop_last)
    else:
        frame_sampler = WeightedRandomSampler(num_samples=num_frames, weights=frame_weights, replacement=replacement)
    return frame_sampler, scene_weights

def get_scene_sampler(
    scene_loader: SceneDataLoader, 
    replacement: bool = True, 
    scene_sample_mode: Literal['uniform', 'weighted_by_len'] = 'weighted_by_len', 
    ddp = False, seed: int = 0, drop_last = False, 
    ):
    """
    A sampler that randomly samples scene indices. 
    Suitable for scene-wise dataset e.g. JointFramePixelDataset
    
    Args:
        scene_loader (SceneDataLoader): The base SceneDataLoader
        replacement (bool, optional): Whether replace when sampling. Defaults to True.
        scene_sample_mode (Literal['uniform', 'weighted_by_len'], optional): 
            Determines the method for sampling a scene from multiple options.
            - `uniform`: Each scene will be sampled with equal probability.
            - `weighted_by_len`: Probability of a scene being chosen is based on its length.
            Only meaningful when sampling multiple scenes.
            Defaults to 'weighted_by_len'.
        ddp (bool, optional): Whether in DDP mode.
    """
    scene_bank = scene_loader.scene_bank
    num_scene = len(scene_bank)
    scene_weights = get_scene_weights(scene_loader, scene_sample_mode=scene_sample_mode)
    if ddp:
        scene_sampler = DistributedWeightedRandomSamplerSampler(
            num_samples=num_scene, mode='weighted_random', weights=scene_weights, 
            replacement=replacement, seed=seed, drop_last=drop_last)
    else:
        scene_sampler = WeightedRandomSampler(
            num_samples=num_scene, weights=scene_weights, replacement=replacement)
    return scene_sampler

@torch.no_grad()
def get_scene_weights(
    scene_loader: SceneDataLoader, 
    scene_sample_mode: Literal['uniform', 'weighted_by_len'] = 'weighted_by_len'):
    scene_bank = scene_loader.scene_bank
    scene_lengths = [len(scene) for scene in scene_bank]
    if scene_sample_mode == 'weighted_by_len':
        scene_weights = torch.tensor(scene_lengths, dtype=torch.float)
        scene_weights = scene_weights / scene_weights.sum()
    elif scene_sample_mode == 'uniform':
        scene_weights = torch.full([len(scene_bank), ],  1./len(scene_bank), dtype=torch.float)
    else:
        raise RuntimeError(f"Invalid scene_sample_mode={scene_sample_mode}")
    return scene_weights

@torch.no_grad()
def get_frame_weights_uniform(
    scene_loader: SceneDataLoader, 
    scene_weights: torch.Tensor, 
    ):
    assert len(scene_weights) == len(scene_loader.scene_bank), \
        "`scene_weights` should have the same length as scene_bank"
    total_weights = []
    for i, scene in enumerate(iter(scene_loader.scene_bank)):
        weights = torch.full([len(scene),], 1./len(scene), dtype=torch.float)
        total_weights.append( weights * scene_weights[i] )
    total_weights = torch.cat(total_weights)
    return total_weights

@torch.no_grad()
def get_frame_weights_fixed(
    scene_loader: SceneDataLoader, 
    scene_weights: torch.Tensor, 
    fixed_frame_ind: Union[int, List[int]] = None
    ):
    assert len(scene_weights) == len(scene_loader.scene_bank), \
        "`scene_weights` should have the same length as scene_bank"
    assert fixed_frame_ind is not None, f"`fixed_frame_ind` is requried"
    if isinstance(fixed_frame_ind, int):
        fixed_frame_ind = [fixed_frame_ind] * len(scene_loader.scene_bank) # Use the same frame_ind across multiple scenes
    total_weights = []
    for i, (scene_id, scene) in enumerate(scene_loader.scene_bank.items()):
        weights = torch.zeros([len(scene),], dtype=torch.float)
        weights[fixed_frame_ind[i]] = 1.0
        total_weights.append( weights * scene_weights[i] )
    total_weights = torch.cat(total_weights)
    return total_weights

@torch.no_grad()
def get_frame_weights_by_speed(
    scene_loader: SceneDataLoader, 
    scene_weights: torch.Tensor, 
    algo: Literal['linear', 'trunc_linear'] = 'linear', 
    multiplier: float=4.0):
    assert len(scene_weights) == len(scene_loader.scene_bank), \
        "`scene_weights` should have the same length as scene_bank"
    total_weights = []
    for i, scene in enumerate(iter(scene_loader.scene_bank)):
        tracks = scene.process_observer_infos().tracks
        dtrans = tracks.new_zeros([*tracks.shape[:-1]])
        dtrans[...,:-1] = (tracks[...,1:,:] - tracks[...,:-1,:]).norm(dim=-1)
        dtrans[...,-1] = dtrans[...,-2]
        
        if algo == 'linear':
            weights = dtrans.clamp(1e-5)
            weights /= weights.sum()
        elif algo == 'trunc_linear':
            w_mean = geometric_mean(dtrans, dim=-1)
            weights = dtrans.clip(w_mean/np.sqrt(multiplier), w_mean*np.sqrt(multiplier))
            weights /= weights.sum()
        else:
            raise RuntimeError(f"Invalid algo={algo}")
        
        total_weights.append(weights * scene_weights[i])
    total_weights = torch.cat(total_weights)
    return total_weights

@torch.no_grad()
def get_frame_weights_from_error_map(
    scene_loader: SceneDataLoader, 
    scene_weights: torch.Tensor, 
    imp_samplers: Dict[str, Dict[str,ImpSampler]]
    ):
    assert len(scene_weights) == len(scene_loader.scene_bank), \
        "`scene_weights` should have the same length as scene_bank"
    total_weights = []
    for i, (scene_id, scene) in enumerate(scene_loader.scene_bank.items()):
        weights = torch.zeros([len(scene)], dtype=torch.float)
        for cam_id, imp_sampler in imp_samplers[scene_id].items():
            weights += imp_sampler.get_pdf_image()
        weights /= weights.sum()
        total_weights.append(weights * scene_weights[i])
    total_weights = torch.cat(total_weights)
    return total_weights

class WeightedRandomSampler(torch_data.Sampler[int]):
    """
    Modified from WeightedRandomSampler
    """
    weights: torch.Tensor
    num_samples: int
    replacement: bool
    def __init__(self, weights: Sequence[float], num_samples: int,
                 replacement: bool = True, generator=None) -> None:
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(num_samples))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        self.weights = torch.as_tensor(weights, dtype=torch.float)
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        rand_tensor = torch.multinomial(self.weights, self.num_samples, self.replacement, generator=self.generator)
        yield from iter(rand_tensor.tolist())

    def __len__(self) -> int:
        return self.num_samples

    def set_weights(self, weights: Sequence[float]):
        self.weights = torch.as_tensor(weights, dtype=torch.float)

class DistributedWeightedRandomSamplerSampler(torch_data.Sampler[int]):
    """
    Modified from WeightedRandomSampler & DistributedSampler
    """
    def __init__(
        self, num_samples: int,  
        num_replicas: Optional[int] = None, rank: Optional[int] = None, 
        mode: Literal['sequential', 'random', 'weighted_random']='weighted_random', 
        weights: Sequence[float] = None, replacement = False, 
        seed: int = 0, drop_last: bool = False) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.total_num_samples = num_samples
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and self.total_num_samples % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (self.total_num_samples - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(self.total_num_samples / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.mode = mode
        self.seed = seed
        self.weights = weights
        self.replacement = replacement

        if self.mode == 'weighted_random':
            assert self.weights is not None, f"`weights` is required for mode={self.mode}"
            self.weights = torch.as_tensor(self.weights, dtype=torch.float)
            assert len(self.weights) == self.total_num_samples, f"`weights` should have the same length with `dataset`"

    def __iter__(self) -> Iterator[int]:
        if self.mode == 'random':
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.total_num_samples, generator=g).tolist()  # type: ignore[arg-type]
        elif self.mode == 'weighted_random':
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.multinomial(self.weights, self.total_num_samples, self.replacement, generator=g).tolist()
        elif self.mode == 'sequential':
            indices = list(range(self.total_num_samples))  # type: ignore[arg-type]
        else:
            raise RuntimeError(f"Invalid mode={self.mode}")

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch

    def set_weights(self, weights: Sequence[float]):
        self.weights = torch.as_tensor(weights, dtype=torch.float)
