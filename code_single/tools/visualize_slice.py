"""
@file   visualize_slice.py
@author Nianchen Deng, Shanghai AI Lab & Jianfei Guo, Shanghai AI Lab
@brief  Visualize slice(s) of a SDF network.
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

import os
import imageio
import numpy as np
from tqdm import trange
from typing import List

import torch

from nr3d_lib.fmt import log
from nr3d_lib.checkpoint import sorted_ckpts
from nr3d_lib.config import ConfigDict, BaseConfig
from nr3d_lib.utils import IDListedDict, cond_mkdir

from app.resources import Scene, load_scene_bank, AssetBank


@torch.no_grad()
def sdf_slice(net, resolution: int, coordinate: List[str], dim=0, depth=0.0, device=torch.device('cuda')):

    def normalized_slice(height, width, coordinate: List[str], dim=0, depth=0.0, device=torch.device('cuda')):
        """Returns grid[x,y] -> coordinates for a normalized slice for some dim at some depth."""

        def normalized_grid(height, width, device=torch.device('cuda')):
            """Returns grid[x,y] -> coordinates for a normalized window.

            Args:
                width, height (int): grid resolution
            """

            # These are normalized coordinates
            # i.e. equivalent to 2.0 * (fragCoord / iResolution.xy) - 1.0
            window_x = torch.linspace(-1., 1., steps=width, device=device)
            window_x += torch.rand_like(window_x) * (1. / width)
            window_y = torch.linspace(-1., 1., steps=height, device=device)
            window_y += torch.rand_like(window_y) * (1. / height)
            coord = torch.stack(torch.meshgrid(window_x, window_y, indexing="xy"), dim=-1)
            return coord

        window = normalized_grid(height, width, device)
        depth_pts = torch.ones(height, width, 1, device=device) * depth

        if dim == 0:
            pts = [depth_pts, window[..., :1], -window[..., 1:]]
        elif dim == 1:
            pts = [-window[..., :1], depth_pts, -window[..., 1:]]
        elif dim == 2:
            pts = [-window[..., :1], -window[..., 1:], -depth_pts]
        else:
            raise ValueError("Invalid value for argument \"dim\"")
        pts_world = [None, None, None]
        for i, coord in enumerate(coordinate):
            if coord == "x":
                pts_world[0] = pts[i]
            elif coord == "-x":
                pts_world[0] = -pts[i]
            elif coord == "y":
                pts_world[1] = pts[i]
            elif coord == "-y":
                pts_world[1] = -pts[i]
            elif coord == "z":
                pts_world[2] = pts[i]
            elif coord == "-z":
                pts_world[2] = -pts[i]
            else:
                raise ValueError("Invalid value for argument \"coordinate\"")
        return torch.cat(pts_world, dim=-1)

    if getattr(net, 'space', None):
        aabb = net.space.aabb
    else:
        aabb = torch.tensor([[-1., -1., -1.], [1., 1., 1.]], device=device)
    
    dim_w = coordinate[1 if dim == 0 else 0][-1]
    dim_h = coordinate[1 if dim == 2 else 2][-1]
    dim_w = 0 if dim_w == "x" else 1 if dim_w == "y" else 2
    dim_h = 0 if dim_h == "x" else 1 if dim_h == "y" else 2
    world_w = (aabb[1][dim_w] - aabb[0][dim_w]).item()
    world_h = (aabb[1][dim_h] - aabb[0][dim_h]).item()
    if world_w < world_h:
        resolution = (int(resolution * world_h / world_w), resolution)
    else:
        resolution = (resolution, int(resolution * world_w / world_h))
    pts = normalized_slice(*resolution, coordinate, dim=dim, depth=depth, device=device)

    d: torch.Tensor = net.forward_sdf(pts.reshape(-1, 3))["sdf"].to(torch.float).reshape(*resolution)
    d = d.clip(-1.0, 1.0)
    blue = d.clip(0.0, 1.0)
    yellow = 1.0 - blue
    vis = torch.zeros(*d.shape, 3, device=device)
    vis[..., 2] = blue
    vis += yellow[..., None] * d.new_tensor([0.4, 0.3, 0.0])
    vis += 0.2
    vis[d < 0] = d.new_tensor([1.0, 0.38, 0.0])
    for i in range(-50, 51):
        vis[(d - 0.02 * i).abs() < 0.0025] = 0.8
    vis[d.abs() < 0.002] = 0.0
    return vis

@torch.no_grad()
def sdf_slice_video(net, depths: List[float], resolution: int, coordinate: List[str], dim=0, device=torch.device('cuda')):
    slice_img_seq = []
    for depth in depths:
        slice_img = sdf_slice(net, resolution, coordinate, dim=dim, depth=depth, device=device)
        slice_img_seq.append(slice_img.data.cpu().numpy()) # Always use cpu array list to save GPU mem
    return slice_img_seq

def main_function(args: ConfigDict):
    # ---------------------------------------------
    # --------------     Load     -----------------
    # ---------------------------------------------
    device = torch.device('cuda')
    if (ckpt_file := args.get('load_pt', None)) is None:
        # Automatically load 'final_xxx.pt' or 'latest.pt'
        ckpt_file = sorted_ckpts(os.path.join(args.exp_dir, 'ckpts'))[-1]
    log.info("=> Use ckpt:" + str(ckpt_file))
    state_dict = torch.load(ckpt_file, map_location=device)

    # ---------------------------------------------
    # -----------     Scene Bank     --------------
    # ---------------------------------------------
    scene_bank: IDListedDict[Scene] = IDListedDict()
    scenebank_root = os.path.join(args.exp_dir, 'scenarios')
    scene_bank, _ = load_scene_bank(scenebank_root, device=device)

    # ---------------------------------------------
    # -----------     Asset Bank     --------------
    # ---------------------------------------------
    asset_bank = AssetBank(args.assetbank_cfg)
    asset_bank.create_asset_bank(scene_bank, load_state_dict=state_dict['asset_bank'], device=device)
    # log.info(asset_bank)

    # ---------------------------------------------
    # ---     Load assets to scene objects     ----
    # ---------------------------------------------
    # for scene in scene_bank:
    scene = scene_bank[0]
    scene.load_assets(asset_bank)
    # !!! Only call training_before_per_step when all assets are ready & loaded !
    asset_bank.training_before_per_step(args.training.num_iters) # NOTE: Finished training.

    fg_node = scene.get_drawable_groups_by_class_name(scene.main_class_name)[0]

    output_root = os.path.join(args.exp_dir, args.dirname)
    cond_mkdir(output_root)

    if args.anim:
        def write_video(uri, frames, **kwargs):
            if ".mp4" not in uri:
                uri = f"{uri}.mp4"
            imageio.mimwrite(uri, frames, fps=args.fps, quality=args.quality, **kwargs)
            log.info(f"Video saved to {uri}")

        log.info(f"Start [visualize_slice].anim, in {args.exp_dir}")
        depths = np.linspace(-1., 1., 100)
        slice_img_seq = sdf_slice_video(fg_node.model, depths, args.resolution, args.coordinate, dim=args.dim, device=device)
        slice_img_seq = [(im * 255.).clip(0,255).astype(np.uint8) for im in slice_img_seq]
        output_path = os.path.join(output_root, f"slice{args.dim}.mp4")
        write_video(output_path, slice_img_seq)
        print(f"Slice video saved to {output_path}")
    else:
        slice_img = sdf_slice(fg_node.model, args.resolution, args.coordinate, dim=args.dim, depth=args.depth, device=device)
        slice_img = (slice_img * 255.).to(torch.uint8).cpu().numpy()
        output_path = os.path.join(output_root, f"slice{args.dim}_depth={args.depth}.png")
        imageio.imwrite(output_path, slice_img)
        print(f"Slice image saved to {output_path}")


if __name__ == "__main__":
    from nr3d_lib.config import BaseConfig
    bc = BaseConfig()
    bc.parser.add_argument("--load_pt", type=str, default=None, help="Typically unnecessary as the final or latest ckpt is loaded automatically. \n"\
        "Only specify the ckpt file path if indeed a non-final or non-latest ckpt needs to be loaded.")
    bc.parser.add_argument("--dirname", type=str, default="videos", help="Sets the output directory to /path/to/exp_dir/${dirname}`.")
    bc.parser.add_argument("--dim", type=int, default=0,
                           help="Slice along which direction, for waymo: 0: Right->Left, 1: Back->Forward, 2: Up->Down")
    bc.parser.add_argument("--depth", type=float, default=0.0,
                           help="The depth of output slice along the axis specified by \"dim\", "
                                "should between -1 and 1")
    bc.parser.add_argument("--anim", action="store_true",
                           help="Generate animation video for depth move from -1 to 1")
    bc.parser.add_argument("--fps", type=int, default=24)
    bc.parser.add_argument("--quality", type=int, default=None, help="Sets the quality for imageio.mimwrite (range: 0-10; 10 is the highest; default is 5).")
    bc.parser.add_argument("--resolution", type=int, default=512,
                           help="The resolution of short side")
    bc.parser.add_argument("--coordinate", type=str, nargs=3, default=["y", "x", "z"],
                           help="The world axis for left, forward and up direction, "
                                "defaults to y x z (i.e. waymo's coordinate)")
    main_function(bc.parse())

