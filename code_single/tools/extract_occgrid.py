
"""
@file   extract_occgrid.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Extract high-res occupancy grids, via sign-change detection on sub-sampled voxles.
"""
import os
import sys
def set_env(depth: int):
    # Add project root to sys.path
    current_file_path = os.path.abspath(__file__)
    project_root_path = os.path.dirname(current_file_path)
    for _ in range(depth):
        project_root_path = os.path.dirname(project_root_path)
    if project_root_path not in sys.path:
        sys.path.append(project_root_path)
        print(f"Added {project_root_path} to sys.path")
set_env(2)

import numpy as np
from itertools import product
from datetime import datetime

import torch
from tqdm import tqdm

from nr3d_lib.fmt import log
from nr3d_lib.checkpoint import sorted_ckpts
from nr3d_lib.config import ConfigDict, BaseConfig
from nr3d_lib.utils import IDListedDict, cond_mkdir
from nr3d_lib.models.grids.utils import offset_voxel
from nr3d_lib.models.spatial import AABBSpace, ForestBlockSpace

from app.resources import Scene, load_scene_bank, AssetBank

@torch.no_grad()
def main_function(args: ConfigDict):
    exp_dir = args.exp_dir
    device = torch.device('cuda', 0)
    dtype = torch.float32

    #---------------------------------------------
    #--------------     Load     -----------------
    #---------------------------------------------
    device = torch.device('cuda', 0)
    # Automatically load 'final_xxx.pt' or 'latest.pt'
    ckpt_file = sorted_ckpts(os.path.join(args.exp_dir, 'ckpts'))[-1]
    log.info("=> Use ckpt:" + str(ckpt_file))
    state_dict = torch.load(ckpt_file, map_location=device)

    #---------------------------------------------
    #-----------     Scene Bank     --------------
    #---------------------------------------------
    scene_bank: IDListedDict[Scene] = IDListedDict()
    scenebank_root = os.path.join(args.exp_dir, 'scenarios')
    scene_bank, _ = load_scene_bank(scenebank_root, device=device)
    
    #---------------------------------------------
    #-----------     Asset Bank     --------------
    #---------------------------------------------
    asset_bank = AssetBank(args.assetbank_cfg)
    asset_bank.create_asset_bank(scene_bank, load=state_dict['asset_bank'], device=device)
    asset_bank.to(device)
    log.info(asset_bank)

    #---------------------------------------------
    #---     Load assets to scene objects     ----
    #---------------------------------------------
    # for scene in scene_bank:
    scene = scene_bank[0]
    scene.load_assets(asset_bank)
    # !!! Only call preprocess_per_train_step when all assets are ready & loaded !
    asset_bank.preprocess_per_train_step(args.training.num_iters) # NOTE: Finished training.

    scene.frozen_at(0)

    if len(lst:=scene.get_drawable_groups_by_class_name(scene.main_class_name)) > 0:
        obj = lst[0]
    else:
        raise RuntimeError(f"Empty node_list of the given class_name={scene.main_class_name}")

    occ_res = args.occ_res
    side_length_chunk = args.side_length_chunk
    subsample_factor = int(args.subsample_factor)

    output_dir = os.path.join(exp_dir, args.dirname)
    cond_mkdir(output_dir)
    expname = os.path.split(args.exp_dir.rstrip("/"))[-1]
    global_step = state_dict['global_step']
    global_step_str = f"iter{global_step/1000}k" if global_step >= 1000 else f"iter{global_step}"
    name = f"{expname[0:64]}_{global_step_str}_{args.outbase}_occ_res={occ_res}"
    output_file = os.path.join(output_dir, f"{name}.npz")
    
    model = obj.model
    box_in_obj_net = offset_voxel(_0=-1., _1=1.).float().to(device)
    aabb_objnet = model.space.aabb
    aabb_objnet_space = AABBSpace(aabb=aabb_objnet, device=device, dtype=torch.float)
    box_in_obj = aabb_objnet_space.unnormalize_coords(box_in_obj_net)
    box_in_world = obj.world_transform(box_in_obj * obj.scale.ratio())
    aabb_in_world = torch.stack([box_in_world.min(dim=0).values, box_in_world.max(dim=0).values], dim=0)
    aabb_space = AABBSpace(aabb=aabb_in_world)
    resolution = ((aabb_in_world[1] - aabb_in_world[0]) / args.occ_res).long()
    resolution_list = resolution.tolist()
        
    def sdf_query_fn(x_in_world: torch.Tensor):
        # NOTE: x: [-1,1]
        x_in_obj = obj.world_transform(x_in_world, inv=True) / obj.scale.ratio()
        # NOTE: Put inf values for out-of-bound coordinates (caused by pose difference of obj and world)
        sdf = model.implicit_surface.forward_in_obj(x_in_obj, invalid_sdf=np.inf, return_h=False, with_normal=False)['sdf']
        # if isinstance(model.space, AABBSpace):
        #     x_in_objnet = model.space.normalize_coords(x_in_obj)
        #     sdf = model.forward_sdf(x_in_objnet, return_h=False)['sdf']
        # elif isinstance(model.space, ForestBlockSpace):
        #     sdf = model.forward_in_obj(x_in_obj)['sdf']
        # else:
        #     raise RuntimeError(f"Unsupported model.space type={type(model.space)}")
        return sdf
    
    # Total output
    occ_corners = []
    if args.verbose:
        occ_grid = np.zeros(resolution_list, dtype=bool)

    subsample_coords = [torch.arange(subsample_factor+1, device=device, dtype=torch.float) / subsample_factor for _ in range(3)]
    subsample_coords = torch.stack(torch.meshgrid(subsample_coords, indexing='ij'), dim=-1).view(-1, 3)
    
    inds = [np.arange(resolution_list[i]) for i in [0,1,2]]
    total_blocks = list(product(*[range(0, resolution_list[i], side_length_chunk) for i in (0,1,2)]))
    print(f"=> Start [extract_occgrid] with resolution: {resolution_list}, occ_res={occ_res}, in {exp_dir}")
    for (ix, iy, iz) in tqdm(total_blocks, 'generating occ_grid...'):
        block_inds = [inds[0][ix:ix+side_length_chunk], inds[1][iy:iy+side_length_chunk], inds[2][iz:iz+side_length_chunk]]
        block_inds = [torch.tensor(bi, dtype=torch.long, device=device) for bi in block_inds]
        block_inds_full = torch.stack(torch.meshgrid(block_inds, indexing='ij'), dim=-1)
        
        coords = block_inds_full.float().unsqueeze(-2) + subsample_coords[None, None, None, :, :]
        coords_in_world = aabb_space.unnormalize_coords((coords / resolution) * 2 - 1)
        sdf = sdf_query_fn(coords_in_world)
        sdf_sign = sdf > 0
        sdf_sign_sum = sdf_sign.sum(dim=-1)
        
        # If a sign change happens
        has_surface = (sdf_sign_sum < ((subsample_factor+1)**3)) & (sdf_sign_sum > 0)
        has_out_of_bound = sdf.isinf().any(dim=-1)
        has_surface = has_surface & has_out_of_bound.logical_not()
        
        occ_idx = has_surface.nonzero().long()
        occ_idx_in_all = occ_idx + torch.tensor([ix, iy, iz], dtype=torch.long, device=device)
        occ_corners.append(occ_idx_in_all.data.short().cpu().numpy())
        
        if args.verbose:
            occ_grid[ix:ix+side_length_chunk, iy:iy+side_length_chunk, iz:iz+side_length_chunk] = has_surface.data.cpu().numpy()
    
    occ_corners = np.concatenate(occ_corners, axis=0)
    num_occ_voxels = len(occ_corners)
    np.savez_compressed(
        output_file, 
        occ_corners=occ_corners, sidelength=resolution_list, occ_res=occ_res, 
        coord_min=aabb_in_world[0].data.cpu().numpy(), coord_offset=scene.metas['world_offset'], 
        meta={'scene_id': scene.id, 'start_frame': scene.data_frame_offset, 'num_frames': scene.metas['n_frames']})
    print(f"=> Extracted occ_grid contains {num_occ_voxels} voxels. Saved in {output_file}")
    
    # NOTE: To load:
    # import numpy as np
    # datadict = np.load("xxx.npz", allow_pickle=True)
    
    if args.verbose:
        from nr3d_lib.plot import vis_occgrid_voxels_o3d
        vis_occgrid_voxels_o3d( # Needs CPU tensor; otherwise easily GPU OOM; still might exceeds maximum CPU mem.
            torch.tensor(occ_grid, device=torch.device('cpu')), draw_lines=False, draw_mesh=True, 
            block_size=torch.tensor([occ_res]*3, dtype=torch.float, device=torch.device('cpu')), 
            origin=aabb_in_world[0].cpu())

def make_parser():
    bc = BaseConfig()
    
    bc.parser.add_argument("--occ_res", type=float, default=0.1, help="Defines the grid side length (in world units).")
    bc.parser.add_argument("--subsample_factor", type=int, default=2, 
                           help="Sets the side length of sub-voxel sample points to enhance occupancy estimation accuracy.")
    bc.parser.add_argument("--side_length_chunk", type=int, default=64, 
                           help='Specifies the chunk size for side length to avoid OOM issues when processing large cuboid grids.')
    bc.parser.add_argument("--verbose", action='store_true', help="If set, a visualization window will pop up once the extraction is complete.")
    bc.parser.add_argument("--outbase", type=str, default=datetime.now().strftime("%Y_%m_%d_%H_%M"), help="Sets the basename of the output file (without extension).")
    bc.parser.add_argument("--dirname", type=str, default='occgrid', help="Sets the output directory to /path/to/exp_dir/${dirname}`.")
    return bc

if __name__ == "__main__":
    bc = make_parser()
    main_function(bc.parse())