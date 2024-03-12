"""
@file   nvs.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Novel view synthesis of one scene. Travel through / inspect-look / etc.
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
import functools
import numpy as np
from tqdm import tqdm
from datetime import datetime


import torch

from nr3d_lib.fmt import log
from nr3d_lib.utils import cond_mkdir
from nr3d_lib.plot import get_n_ind_colors, color_depth, gallery

from nr3d_lib.models.attributes import *
from nr3d_lib.graphics.cameras import get_path_front_left_lift_then_spiral_forward

from app.resources import load_scenes_and_assets
from app.resources.observers import Lidar, Camera
from app.renderers import BufferComposeRenderer

from code_multi.tools.utils import draw_box

def main_function(args, device=torch.device('cuda')):
    #---------------------------------------------
    #--------------     Load     -----------------
    #---------------------------------------------
    scene_bank, asset_bank, *_ = load_scenes_and_assets(**args, device=device)
    asset_bank.eval()
    
    fg_class_names = args.fg_class_name.split(',')

    #---------------------------------------------
    #------------     Renderer     ---------------
    #---------------------------------------------
    renderer = BufferComposeRenderer(args.renderer)
    renderer.populate(asset_bank)
    renderer.eval()
    renderer.config.rayschunk = args.rayschunk
    renderer.config.with_normal = True
    if args.depth_max is None:
        depth_max = renderer.config.far
    else:
        depth_max = args.depth_max
    for scene in scene_bank:
        for obs in scene.get_observers(False):
            obs.near = renderer.config.near
            obs.far = renderer.config.far

    #---------------------------------------------
    #------     Scene Bank Dataset     -----------
    #---------------------------------------------
    cam_id_list = list(scene.get_cameras(False).keys()) if args.cam_id is None else [args.cam_id]

    #---------------------------------------------
    #--------------     Plot     -----------------
    #---------------------------------------------
    expname = os.path.split(args.exp_dir.rstrip("/"))[-1]
    name = f"{expname[0:64]}_{args.outbase}_ds={args.downscale}" + ("" if args.forward_inv_s is None else f"_s={int(args.forward_inv_s)}")
    vid_root = os.path.join(args.exp_dir, args.dirname)
    cond_mkdir(vid_root)
    def write_video(uri, frames, **kwargs):
        if len(frames) > 1:
            if ".mp4" not in uri:
                uri = f"{uri}.mp4"
            imageio.mimwrite(uri, frames, fps=args.fps, quality=args.quality, **kwargs)
            print(f"Video saved to {uri}")
        else:
            if ".mp4" in uri:
                uri = f"{os.path.splitext(uri)[0]}.png"
            imageio.imwrite(uri, frames[0], **kwargs)
            print(f"Image saved to {uri}")
    
    instance_id_map = scene.get_drawable_instance_ind_map()
    instance_cmap = np.array(get_n_ind_colors(len(instance_id_map)))
    
    classname_map = scene.get_drawable_class_ind_map()
    class_cmap = np.array(get_n_ind_colors(len(classname_map)))
    
    draw_box_fn = functools.partial(
        draw_box, thickness=(2 if args.downscale==1 else 1), fontscale=(1. if args.downscale==1 else 0.5), 
        instance_id_map=instance_id_map, instance_cmap=instance_cmap, classname_map=classname_map, class_cmap=class_cmap)
    
    collate_keys = ['rgb_volume', 'depth_volume', 'normals_volume', \
        'ins_seg_mask_buffer', 'bgrgb_volume', 'bgdepth_volume', 'bg_normals_volume']
    
    all_gather = dict()
    for cam_id in cam_id_list:
        all_gather[cam_id] = dict({k: [] for k in collate_keys})
    
    with torch.no_grad():
        if args.ref_stop_frame == -1:
            args.ref_stop_frame = len(scene)
        scene.slice_at(slice(args.ref_start_frame, args.ref_stop_frame, 1))
        ego_car = scene.all_nodes['ego_car']
        pose_ref = ego_car.world_transform.mat_4x4().data.cpu().numpy()
        scene.unfrozen()
        
        #-------- Define nvs travel tracks
        kwargs = dict(pose_ref=pose_ref, num_frames=args.num_nvs_frames, 
                      duration_frames=int(args.duration * args.fps), elongation=args.nvs_elongation)
        # Waymo convention: [+z] up; [+x] front, [+y] left
        kwargs.update(forward_vec=[1., 0., 0.], up_vec=[0., 0., 1.], left_vec=[0., 1., 0.])
        if args.nvs_param is not None:
            up_max, up_offset, left_max, left_offset = [float(i) for i in args.nvs_param.split(',')]
            kwargs.update(up_max=up_max, up_offset=up_offset, left_max=left_max, left_offset=left_offset)
        render_pose_all = get_path_front_left_lift_then_spiral_forward(**kwargs)
    
    with torch.no_grad():
        bg_obj = scene.get_drawable_groups_by_class_name(args.bg_class_name)[0]
        bg_obj_id = bg_obj.id
        bg_model = bg_obj.model
        if args.forward_inv_s is not None:
            bg_model.ray_query_cfg.forward_inv_s = args.forward_inv_s
        
        scene.slice_at(args.ref_start_frame if args.slice_at is None else args.slice_at)

        drawables = [bg_obj] + scene.get_drawable_groups_by_class_name('Sky', only_valid=True).to_list()
        if args.draw_objs:
            drawables += scene.get_drawable_groups_by_class_name_list(fg_class_names, only_valid=True)
        
        log.info(f"Start [nvs], ds={args.downscale}, in {args.exp_dir}")
        for i, render_pose in enumerate(tqdm(render_pose_all, 'rendering...')):
            ego_car.world_transform = TransformMat4x4(render_pose, device=device, dtype=torch.float)
            for node in ego_car.children:
                node.update()

            for cam_ind, cam_id in enumerate(cam_id_list):
                cam: Camera = scene.observers[cam_id]
                cam.intr.set_downscale(args.downscale)
                
                #-----------------------------------------------
                ret = renderer.render(scene, observer=cam, drawables=drawables, render_per_obj_individual=True, show_progress=args.progress)
                rendered = ret['rendered']
                #-----------------------------------------------
                
                cur_frame_dict = dict({k: [] for k in collate_keys})
                def to_img(tensor):
                    return tensor.reshape([cam.intr.H, cam.intr.W, -1]).data.cpu().numpy()
                
                mask_volume = to_img(rendered['mask_volume'])
                depth_volume = to_img(rendered['depth_volume'])
                depth_volume = mask_volume * np.clip(depth_volume/depth_max, 0, 1) + (1-mask_volume) * 1
                depth_volume = color_depth(depth_volume.squeeze(-1), scale=1, cmap='turbo')    # turbo_r, viridis, rainbow
                
                rgb_volume = np.clip((to_img(rendered['rgb_volume'])*255).astype(np.uint8),0,255)
                
                cur_frame_dict['rgb_volume'] = rgb_volume
                cur_frame_dict['depth_volume'] = depth_volume
                
                if 'normals_volume' in rendered:
                    normals_volume = np.clip((to_img(rendered['normals_volume']/2+0.5)*255).astype(np.uint8),0,255)
                    cur_frame_dict['normals_volume'] = normals_volume

                ins_seg_mask_buffer = np.take(instance_cmap, axis=0, indices=to_img(ret['ins_seg_mask_buffer'])[...,0]).astype(np.uint8)
                cur_frame_dict['ins_seg_mask_buffer'] = ins_seg_mask_buffer


                bg_rendered = ret['rendered_per_obj'][bg_obj_id]
                bgmask_volume = to_img(bg_rendered['mask_volume'])
                bgdepth_volume = to_img((bg_rendered['depth_volume']))
                bgdepth_volume = bgmask_volume * np.clip(bgdepth_volume/depth_max, 0, 1) + (1-bgmask_volume) * 1
                bgdepth_volume = color_depth(bgdepth_volume.squeeze(-1), scale=1, cmap='turbo')    # turbo_r, viridis, rainbow
                bgrgb_volume = np.clip((to_img(bg_rendered['rgb_volume'])*255).astype(np.uint8),0,255)
                
                cur_frame_dict['bgrgb_volume'] = bgrgb_volume
                cur_frame_dict['bgdepth_volume'] = bgdepth_volume
                
                if 'normals_volume' in bg_rendered:
                    bg_normals_volume = np.clip((to_img(bg_rendered['normals_volume']/2+0.5)*255).astype(np.uint8),0,255)
                    cur_frame_dict['bg_normals_volume'] = bg_normals_volume

                for k, v in cur_frame_dict.items():
                    all_gather[cam.id][k].append(v)
                
    #--------- Seperate video
    if args.save_seperate_keys:
        for cam_id, obs_dict in all_gather.items():
            for k, frames in obs_dict.items():
                write_video(os.path.join(vid_root, f"{name}_{cam_id}_{k}.mp4"), frames)

    #--------- 2 rows collection
    frames_per_obs_all = []
    for cam_id, obs_dict in all_gather.items():
        frames_per_obs = []
        for kvs in zip(*(obs_dict.values())):
            kvs = list(kvs)
            kvs.insert(0, np.zeros_like(kvs[0]))
            frames_per_obs.append(gallery(np.stack(kvs, 0), nrows=2))
        if not args.only_all:
            write_video(os.path.join(vid_root, f"{name}_{cam_id}.mp4"), frames_per_obs)
        frames_per_obs_all.append(np.array(frames_per_obs))
    if len(frames_per_obs_all) > 1 or args.only_all:
        # NOTE: only for waymo: different cameras has the same width
        write_video(os.path.join(vid_root, f"{name}_all.mp4"), np.concatenate(frames_per_obs_all, axis=1))

    #--------- 1 row collection
    keys_1l = ['rgb_volume', 'bgrgb_volume', 'depth_volume', 'bgdepth_volume', 'ins_seg_mask_buffer', 'normals_volume', 'bg_normals_volume']
    frames_per_obs_1l_all = []
    for cam_id, obs_dict in all_gather.items():
        frames_per_obs_1l = []
        new_obs_dict = dict()
        for k in keys_1l:
            new_obs_dict[k] = obs_dict[k]
        for kvs in zip(*(new_obs_dict.values())):
            frames_per_obs_1l.append(np.concatenate(kvs, axis=1))
        if not args.only_all:
            write_video(os.path.join(vid_root, f"{name}_{cam_id}_1l.mp4"), frames_per_obs_1l)
        frames_per_obs_1l_all.append(np.array(frames_per_obs_1l))
    if len(frames_per_obs_1l_all) > 1 or args.only_all:
        # NOTE: only for waymo: different cameras has the same width
        frames_per_obs_1l_all = np.concatenate(frames_per_obs_1l_all, axis=1)
        write_video(os.path.join(vid_root, f"{name}_1l_all.mp4"), frames_per_obs_1l_all)

if __name__ == "__main__":
    from nr3d_lib.config import BaseConfig
    bc = BaseConfig()
    bc.parser.add_argument("--bg_class_name", type=str, default='Street', help="The class_name of the background object.")
    bc.parser.add_argument("--fg_class_name", type=str, default='Vehicle,Pedestrian', help="The class_name of the foreground objects.")
    
    bc.parser.add_argument("--load_pt", type=str, default=None, help="Typically unnecessary as the final or latest ckpt is loaded automatically. \n"\
        "Only specify the ckpt file path if indeed a non-final or non-latest ckpt needs to be loaded.")
    bc.parser.add_argument("--downscale", type=float, default=1.0, help="Sets the side length downscale for rendering and output.")
    bc.parser.add_argument("--progress", action='store_true', help="If set, shows per frame progress.")
    bc.parser.add_argument("--save_seperate_keys", action='store_true')
    bc.parser.add_argument("--rayschunk", type=int, default=4096)
    # NeuS specific hacks
    bc.parser.add_argument("--forward_inv_s", type=float, default=None)
    
    bc.parser.add_argument("--draw_objs", action='store_true')
    
    bc.parser.add_argument("--fps", type=int, default=24)
    bc.parser.add_argument("--quality", type=int, default=None, help="Sets the quality for imageio.mimwrite (range: 0-10; 10 is the highest; default is 5).")
    bc.parser.add_argument("--dirname", type=str, default='vid_nvs', help="Sets the output directory to /path/to/exp_dir/${dirname}`.")
    bc.parser.add_argument("--outbase", type=str, default=datetime.now().strftime("%Y_%m_%d_%H_%M_%S"), help="Sets the basename of the output file (without extension).")
    bc.parser.add_argument("--only_all", action='store_true', help='render only one total video')
    
    bc.parser.add_argument("--cam_id", type=str, default=None, help="If set, uses a specific camera; otherwise, uses all available cameras.")
    bc.parser.add_argument("--slice_at", type=int, default=None, help="Specifies the frame at which the scene is frozen in advance of NVS.")
    bc.parser.add_argument('--ref_start_frame', type=int, default=0)
    bc.parser.add_argument('--ref_stop_frame', type=int, default=-1)

    bc.parser.add_argument("--nvs_param", type=str, default=None)
    bc.parser.add_argument("--nvs_elongation", type=float, default=1., help="whether to extrapolate and extend the ref track. [1.0] fo not")
    bc.parser.add_argument('--duration', type=float, default=2.0, help='how many seconds per round (seconds)')
    bc.parser.add_argument("--num_nvs_frames", type=int, default=200)

    main_function(bc.parse())