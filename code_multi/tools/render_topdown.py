"""
@file   render_topdown.py
@author Nianchen Deng, Shanghai AI Lab
@brief  
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
from datetime import datetime

import torch

from nr3d_lib.models.attributes import *

from app.resources import Scene, load_scenes_and_assets
from app.resources.observers import OrthogonalCamera, OrthoCameraIntrinsics
from app.renderers import BufferComposeRenderer

DEBUG_LIDAR = False


def main_function(args, device=torch.device('cuda')):
    # ---------------------------------------------
    # --------------     Load     -----------------
    # ---------------------------------------------
    scene_bank, asset_bank, *_ = load_scenes_and_assets(**args, device=device, class_name_list=args.class_name)
    asset_bank.eval()

    observer = OrthogonalCamera("topdown_observer", device=device)
    observer.intr = OrthoCameraIntrinsics(
            phyW=torch.tensor([60.], device=device),
            phyH=torch.tensor([180.], device=device),
            W=torch.tensor([1600], device=device),
            H=torch.tensor([4800], device=device))
    observer.transform = TransformMat4x4(np.array([
        [1., 0., 0., 20.],
        [0., -1., 0., -90.],
        [0., 0., -1., 8.],
        [0., 0., 0., 1.]
    ]), device=observer.device, dtype=observer.dtype)
    scene.add_node(observer, scene.root)

    # ---------------------------------------------
    # ------------     Renderer     ---------------
    # ---------------------------------------------
    renderer = BufferComposeRenderer(args.renderer)
    renderer.populate(asset_bank)
    renderer.eval()
    asset_bank.eval()
    renderer.config.rayschunk = args.rayschunk
    renderer.config.with_normal = False
    for scene in scene_bank:
        for obs in scene.get_observers(False):
            obs.near = renderer.config.near
            obs.far = renderer.config.far

    with torch.no_grad():
        scene: Scene = scene_bank[0]
        scene.frozen_at_global_frame(0)
        ret = renderer.render(scene, observer=observer, render_per_obj_individual=True, show_progress=args.progress)

        def to_img(tensor):
            return (tensor * 255).clamp(0., 255.).to(torch.uint8)\
                .reshape(observer.intr.H, observer.intr.W, -1).cpu().numpy()

        rgb_volume = to_img(ret['rendered']['rgb_volume'])
        

        #------------- Background
        bg_obj_id = scene.get_drawable_groups_by_class_name(args.class_name)[0].id
        bg_rendered = ret['rendered_per_obj'][bg_obj_id]
        bgrgb_volume = to_img(bg_rendered['rgb_volume'])
        imageio.imwrite(os.path.join(args.exp_dir, "topdown.png"), bgrgb_volume)


if __name__ == "__main__":
    from nr3d_lib.config import BaseConfig
    bc = BaseConfig()
    bc.parser.add_argument("--class_name", type=str, default='Street', help="The class_name of the object you want to operate with.")
    bc.parser.add_argument("--progress", action='store_true', help="If set, shows per frame progress.")
    bc.parser.add_argument("--load_pt", type=str, default=None, help="Typically unnecessary as the final or latest ckpt is loaded automatically. \n"\
        "Only specify the ckpt file path if indeed a non-final or non-latest ckpt needs to be loaded.")
    bc.parser.add_argument("--quality", type=int, default=None, help="Sets the quality for imageio.mimwrite (range: 0-10; 10 is the highest; default is 5).")
    bc.parser.add_argument("--outbase", type=str, default=datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
                           help="Sets the basename of the output file (without extension).")
    bc.parser.add_argument("--rayschunk", type=int, default=4096)

    main_function(bc.parse())
