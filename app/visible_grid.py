"""
@file   visible_grid.py
@author Nianchen Deng, Shanghai AI Lab
@brief  Visible grid.
"""

import torch
import math
from collections import defaultdict
from kaolin.ops.spc import unbatched_query

from nr3d_lib.fmt import log
from nr3d_lib.config import ConfigDict
from nr3d_lib.models.accelerations import get_accel
from nr3d_lib.models.accelerations.occgrid_accel import *
from nr3d_lib.models.attributes import *
from nr3d_lib.models.spatial import AABBSpace, ForestBlockSpace


def voxel_indices_to_voxel_coords(voxel_indices: torch.Tensor, grid_size: torch.Tensor):
    strides = grid_size.new_tensor([grid_size[1] * grid_size[2], grid_size[2], 1])
    coord_x = voxel_indices // strides[0]
    coord_y = (voxel_indices - coord_x * strides[0]) // strides[1]
    coord_z = voxel_indices % strides[1]
    return torch.stack([coord_x, coord_y, coord_z], 1)


def voxel_coords_to_voxel_indices(voxel_coords: torch.Tensor, grid_size: torch.Tensor):
    strides = grid_size.new_tensor([grid_size[1] * grid_size[2], grid_size[2], 1])
    return (voxel_coords * strides).sum(-1)


class VisibleGrid:
    space: Union[AABBSpace, ForestBlockSpace]
    accel: Union[OccGridAccel, OccGridAccelForest]
    grid_center: torch.Tensor
    grid_size: torch.Tensor
    grid_extent: float
    grid_extent_in_world: float
    voxel_size: torch.Tensor
    voxel_size_in_world: torch.Tensor
    voxels_in_block: Dict[str, Union[List[torch.Tensor], torch.Tensor]]
    voxel_hits_in_block: Dict[str, Union[List[torch.Tensor], torch.Tensor]]

    def __init__(self, space, octree_depth: int = None, prefer_voxel_size: float = None) -> None:
        self.space = space
        self.accel = None
        if isinstance(space, AABBSpace):
            self.grid_center = space.center
            self.grid_extent_in_world = self.grid_extent = space.radius3d.max().item() * 2
        elif isinstance(space, ForestBlockSpace):
            # grid_center, grid_extent, voxel_size for ForestBlockSpace is in [0, 1] space (i.e. block space)
            self.grid_center = torch.zeros(3, device=space.device)
            self.grid_extent = 1.
            self.grid_extent_in_world = space.world_block_size.max().item()
            prefer_voxel_size = prefer_voxel_size and prefer_voxel_size / self.grid_extent_in_world
        else:
            raise NotImplementedError("Only support AABBSpace and ForestBlockSpace now")
        self.octree_depth = octree_depth or math.floor(
            math.log2(self.grid_extent / prefer_voxel_size))
        self.grid_size = self.grid_center.new_tensor([2 ** self.octree_depth] * 3, dtype=torch.long)
        self.voxel_size = self.grid_extent / self.grid_size
        self.voxel_size_in_world = self.grid_extent_in_world / self.grid_size
        self.voxels_in_block = defaultdict(list)
        self.voxel_hits_in_block = defaultdict(list)

        log.info(f"VisibleGrid initialized: octree_depth={self.octree_depth}, "
                 f"voxel_size_in_world={self.voxel_size_in_world.tolist()}")

    @staticmethod
    def load(file_path: str, space):
        state_dict = torch.load(file_path)
        visible_grid = VisibleGrid(space, state_dict["octree_depth"])
        visible_grid.voxels_in_block = state_dict["voxels_in_block"]
        return visible_grid

    def save(self, file: str):
        torch.save({
            "octree_depth": self.octree_depth,
            "voxels_in_block": self.voxels_in_block
        }, file)

    def reduce_points_and_add(self, pts: torch.Tensor):
        voxels_in_block = {}
        voxel_hits_in_block = {}
        # Reduce points to voxels
        if isinstance(self.space, AABBSpace):
            in_space_selector = self.space.contains(pts).nonzero()[:, 0]
            voxels, voxel_hits = self.reduce_to_voxels(pts[in_space_selector])
            voxels_in_block[0] = voxels
            voxel_hits_in_block[0] = voxel_hits
        elif isinstance(self.space, ForestBlockSpace):
            coords_in_block, blidx = self.space.normalize_coords_01(pts)
            in_space_selector = (blidx >= 0).nonzero()[:, 0]

            # Sort by block indices
            blidx, sort_indices = blidx[in_space_selector].sort()
            coords_in_block = coords_in_block[in_space_selector][sort_indices]

            unique_blidx, num_pts_in_blidx = blidx.unique_consecutive(return_counts=True)
            unique_blidx = unique_blidx.tolist()
            offset = 0
            for blidx, num_pts in zip(unique_blidx, num_pts_in_blidx):
                coords_in_blidx = coords_in_block[offset:offset + num_pts]
                voxels, voxel_hits = self.reduce_to_voxels(coords_in_blidx)
                voxels_in_block[blidx] = voxels
                voxel_hits_in_block[blidx] = voxel_hits
                offset += num_pts
        else:
            raise NotImplementedError("Only support AABBSpace and ForestBlockSpace now")

        for key in voxels_in_block:
            self.voxels_in_block[key].append(voxels_in_block[key])
            self.voxel_hits_in_block[key].append(voxel_hits_in_block[key])

        return voxels_in_block, voxel_hits_in_block

    def reduce_to_voxels(self, pts: torch.Tensor):
        voxel_coords = ((pts - self.grid_center) / self.voxel_size).to(torch.long)
        voxel_indices = voxel_coords_to_voxel_indices(voxel_coords, self.grid_size)
        voxel_indices, voxel_hits = voxel_indices.unique(return_counts=True)
        return voxel_indices, voxel_hits

    def reduce_voxels(self):
        # Reduce voxels across frames
        self.voxels_in_block = {
            key: torch.cat(value, 0)
            for key, value in self.voxels_in_block.items()
        }
        self.voxel_hits_in_block = {
            key: torch.cat(value, 0)
            for key, value in self.voxel_hits_in_block.items()
        }
        total_voxels = 0
        for blidx, voxel_indices in self.voxels_in_block.items():
            voxel_hits = self.voxel_hits_in_block[blidx]
            unique_voxel_indices, inverse_indices = voxel_indices.unique(return_inverse=True)
            unique_voxel_hits = torch.zeros_like(unique_voxel_indices)
            unique_voxel_hits.put_(inverse_indices, voxel_hits)
            self.voxels_in_block[blidx] = unique_voxel_indices
            self.voxel_hits_in_block[blidx] = unique_voxel_hits
            total_voxels += unique_voxel_indices.shape[0]
        log.info(f"VisibleGrid collect {total_voxels} voxels")
        return self

    def build_accel(self):
        if isinstance(self.space, AABBSpace):
            self.accel = get_accel("occ_grid", space=self.space, resolution=self.grid_size)
            self.accel.occ.occ_grid.flatten()[self.voxels_in_block[0]] = True
        elif isinstance(self.space, ForestBlockSpace):
            self.accel = get_accel("occ_grid_forest", space=self.space, resolution=self.grid_size)
            self.accel.populate()
            for blidx, voxel_indices in self.voxels_in_block.items():
                self.accel.occ.occ_grid[blidx].flatten()[voxel_indices] = True
        return self

    def update_voxels_in_block_from_occgrid(self):
        self.voxels_in_block = {}
        for blidx in range(self.space.n_trees):
            voxel_coords = (self.accel.occ.occ_grid[blidx] > 0).nonzero()
            if voxel_coords.shape[0] == 0:
                continue
            self.voxels_in_block[blidx] = voxel_coords_to_voxel_indices(voxel_coords, self.grid_size)
        return self

    def dilation_occ_grid(self):
        for blidx, voxels_ind in self.voxels_in_block.items():
            voxel_coord = voxel_indices_to_voxel_coords(voxels_ind, self.grid_size)
            neighbors_blidx, neighbors_coord = self.get_neighbors(blidx, voxel_coord)
            if isinstance(self.space, AABBSpace):
                neighbors_occ_ind = [neighbors_coord[:, ax] for ax in range(3)]
                self.accel.occ.occ_grid[neighbors_occ_ind] = True
            elif isinstance(self.space, ForestBlockSpace):
                neighbors_occ_ind = [neighbors_blidx, *[neighbors_coord[:, ax] for ax in range(3)]]
                self.accel.occ.occ_grid[neighbors_occ_ind] = True
            else:
                raise NotImplementedError("Only support AABBSpace and ForestBlockSpace now")

    def erosion_occ_grid(self):
        dilated_occ_grid = self.accel.occ.occ_grid.clone()
        # Erosion
        for blidx in range(self.space.n_trees):
            voxels_coord = (dilated_occ_grid[blidx] > 0).nonzero()
            if voxels_coord.shape[0] == 0:
                continue
            neighbors_blidx, neighbors_coord = self.get_neighbors(blidx, voxels_coord,
                                                                valid_only=False) # (N*26[,3])
            neighbors_blidx = neighbors_blidx.reshape(-1, 26)
            neighbors_coord = neighbors_coord.reshape(-1, 26, 3)
            
            if isinstance(self.space, AABBSpace):
                neighbors_occ_ind = [neighbors_coord[..., ax] for ax in range(3)]
                neighbors_occ = dilated_occ_grid[neighbors_occ_ind] \
                    .logical_and(neighbors_blidx >= 0)
                voxels_should_keep = neighbors_occ.all(dim=1)
                voxels_occ_ind = [voxels_coord[:, ax] for ax in range(3)]
                self.accel.occ.occ_grid[voxels_occ_ind] = voxels_should_keep
                # Original voxels lied at boundary of space match the condition of erosion, but they
                # should not be removed.
                # Here we simply reset the occ of all original voxels to avoid this.
                self.accel.occ.occ_grid.flatten()[self.voxels_in_block[0]] = True
            elif isinstance(self.space, ForestBlockSpace):
                neighbors_occ_ind = [neighbors_blidx, *[neighbors_coord[..., ax] for ax in range(3)]]
                neighbors_occ = dilated_occ_grid[neighbors_occ_ind] \
                    .logical_and(neighbors_blidx >= 0)
                voxels_should_keep = neighbors_occ.all(dim=1)
                voxels_occ_ind = [voxels_coord[:, ax] for ax in range(3)]
                self.accel.occ.occ_grid[blidx][voxels_occ_ind] = voxels_should_keep
                # Original voxels lied at boundary of space match the condition of erosion, but they
                # should not be removed.
                # Here we simply reset the occ of all original voxels to avoid this.
                if blidx in self.voxels_in_block:
                    self.accel.occ.occ_grid[blidx].flatten()[self.voxels_in_block[blidx]] = True
            else:
                raise NotImplementedError("Only support AABBSpace and ForestBlockSpace now")

    def postprocess(self, morphology_op="close"):
        assert morphology_op == "dilation" or morphology_op == "close" or morphology_op == "close2", \
            "Only support dilation, close, close2 operation"
        original_voxels_in_block = {key: value.clone() for key, value in self.voxels_in_block.items()}
        self.dilation_occ_grid()
        if morphology_op == "close2":
            self.update_voxels_in_block_from_occgrid()
            self.dilation_occ_grid()
        
        if morphology_op == "close" or morphology_op == "close2":
            self.voxels_in_block = original_voxels_in_block
            if morphology_op == "close2":
                self.erosion_occ_grid()
            self.erosion_occ_grid()
        
        return self.update_voxels_in_block_from_occgrid()

    def get_neighbors(self, voxel_blidxs: Union[int, torch.Tensor], voxel_coords: torch.Tensor,
                      valid_only: bool = True):
        neighbors = voxel_coords.new_tensor([
            [-1, -1, -1], [-1, -1, 0], [-1, -1, 1],
            [-1, 0, -1], [-1, 0, 0], [-1, 0, 1],
            [-1, 1, -1], [-1, 1, 0], [-1, 1, 1],
            [0, -1, -1], [0, -1, 0], [0, -1, 1],
            [0, 0, -1], [0, 0, 1],
            [0, 1, -1], [0, 1, 0], [0, 1, 1],
            [1, -1, -1], [1, -1, 0], [1, -1, 1],
            [1, 0, -1], [1, 0, 0], [1, 0, 1],
            [1, 1, -1], [1, 1, 0], [1, 1, 1]])  # (26, 3)
        neighbors_coords = (voxel_coords[:, None] + neighbors).flatten(0, 1)  # (N*26, 3)
        block_coords = self.space.block_ks[voxel_blidxs]  # ([N, ]3)
        if isinstance(voxel_blidxs, torch.Tensor):
            neighbors_block_coords = block_coords.repeat_interleave(26, dim=0)
        else:
            neighbors_block_coords = block_coords[None].expand_as(neighbors_coords).contiguous()
        for ax in range(3):
            low = (neighbors_coords[..., ax] < 0).nonzero()[:, 0]
            axes = torch.full_like(low, ax)
            neighbors_coords[low, axes] = self.grid_size[ax] - 1
            neighbors_block_coords[low, axes] -= 1
            high = (neighbors_coords[..., ax] >= self.grid_size[ax]).nonzero()[:, 0]
            axes = torch.full_like(high, ax)
            neighbors_coords[high, axes] = 0
            neighbors_block_coords[high, axes] += 1

        # get block indices of neighbors_block_coords
        if isinstance(self.space, AABBSpace):
            # Only (0, 0, 0) is valid block
            neighbors_blidxs = (neighbors_block_coords == 0).all(dim=1).to(torch.int) - 1
        elif isinstance(self.space, ForestBlockSpace):
            coords_in_forest = (neighbors_block_coords + .5).div_(2 ** self.space.level) \
                .mul_(2.).sub_(1.)
            neighbors_blidxs = self.space.pidx2blidx(
                unbatched_query(self.space.spc.octrees, self.space.spc.exsum, coords_in_forest,
                                self.space.level, with_parents=False)
            )
        if not valid_only:
            return neighbors_blidxs, neighbors_coords
        valid_neighbors_selector = (neighbors_blidxs >= 0).nonzero()[:, 0]
        valid_neighbors_blidxs = neighbors_blidxs[valid_neighbors_selector]
        valid_neighbors_coords = neighbors_coords[valid_neighbors_selector]
        return valid_neighbors_blidxs, valid_neighbors_coords

    def get_grid_center_in_world(self, block_index: int = 0):
        if isinstance(self.space, AABBSpace):
            return self.grid_center
        elif isinstance(self.space, ForestBlockSpace):
            return self.space.block_ks[block_index] * self.grid_extent_in_world + \
                self.space.world_origin
        else:
            raise NotImplementedError("Only support AABBSpace and ForestBlockSpace now")

    def get_grid_aabb_in_world(self, block_index: int = 0):
        grid_min = self.get_grid_center_in_world(block_index)
        grid_max = grid_min + self.grid_extent_in_world
        return grid_min, grid_max

    def get_voxel_aabb_in_world(self, voxel_indices: torch.Tensor, block_index: int = 0):
        voxel_coords = voxel_indices_to_voxel_coords(voxel_indices, self.grid_size)
        voxel_mins = voxel_coords * self.voxel_size_in_world + \
            self.get_grid_center_in_world(block_index)
        voxel_maxs = voxel_mins + self.voxel_size_in_world
        return voxel_mins, voxel_maxs
