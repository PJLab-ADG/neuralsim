"""
@file   vis_anno.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Visualize a scene bank.
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
import cv2
import imageio
import functools
import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch

from nr3d_lib.fmt import log
from nr3d_lib.utils import import_str, cond_mkdir
from nr3d_lib.plot import color_depth, get_n_ind_colors

from app.resources import Scene, AssetBank, create_scene_bank
from app.resources.observers import Camera, RaysLidar

from dataio.scene_dataset import SceneDataset
from dataio.data_loader import SceneDataLoader

from code_multi.tools.utils import draw_box

def main_function(args):
    #---------------------------------------------
    #--------------     Load     -----------------
    #---------------------------------------------
    device = torch.device('cuda')

    #---------------------------------------------
    #-----------     Scene Bank     --------------
    #---------------------------------------------
    dataset_impl: SceneDataset = import_str(args.dataset_cfg.target)(args.dataset_cfg.param)
    asset_bank = AssetBank(args.assetbank_cfg)
    scenebank_root = os.path.join(args.exp_dir, 'scenarios')
    scene_bank, scenebank_meta = create_scene_bank(
        dataset=dataset_impl, device=device, 
        scenebank_root=scenebank_root,
        scenebank_cfg=args.scenebank_cfg, 
        drawable_class_names=asset_bank.class_name_configs.keys(),
        misc_node_class_names=asset_bank.misc_node_class_names, 
    )

    #---------------------------------------------
    #-----------     Asset Bank     --------------
    #---------------------------------------------
    asset_bank.create_asset_bank(scene_bank, scenebank_meta, device=device)

    #---------------------------------------------
    #---     Load assets to scene objects     ----
    #---------------------------------------------
    for scene in scene_bank:
        scene.load_assets(asset_bank)

    # Fallsback to regular Mat4x4 for convenience
    # with torch.no_grad():
    #     for scene in scene_bank:
    #         for o in scene.all_nodes:
    #             for seg in o.attr_segments:
    #                 if 'transform' in seg.subattr:
    #                     seg.subattr['transform'] = TransformMat4x4(seg.subattr['transform'].mat_4x4())
            
    #         for o in scene.get_observers(False):
    #             for seg in o.attr_segments:
    #                 if 'intr' in seg.subattr.keys():
    #                     seg.subattr['intr'] = PinholeCameraMatHW(
    #                         mat=seg.subattr['intr'].mat_4x4(),
    #                         H=seg.subattr['intr'].unscaled_wh()[...,1],
    #                         W=seg.subattr['intr'].unscaled_wh()[...,0]
    #                     )

    #---------------------------------------------
    #------     Scene Bank Dataset     -----------
    #---------------------------------------------
    dataset_impl: SceneDataset = import_str(args.dataset_cfg.target)(args.dataset_cfg.param)
    args.training.dataloader.preload = False
    scene_dataloader = SceneDataLoader(scene_bank, dataset_impl, config=args.training.val_dataloader, device=device)
    scene_dataloader.set_camera_downscale(args.downscale)

    # cam_id_list = scene_dataloader.cam_id_list if args.cam_id is None else [args.cam_id]
    cam_id_list = ['camera_FRONT_LEFT', 'camera_FRONT', 'camera_FRONT_RIGHT']

    #---------------------------------------------
    #--------------     Plot     -----------------
    #---------------------------------------------
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

    all_gather = dict()
    for cam_id in cam_id_list:
        all_gather[cam_id] = []

    with torch.no_grad():
        # for scene in scene_bank:
        scene: Scene = scene_bank[0]
        name = f"{scene.id}_ds={args.downscale}"
        if args.lidar_id is not None:
            name += f"_{args.lidar_id}"
        
        vid_root = os.path.join(args.exp_dir, args.dirname)
        cond_mkdir(vid_root)

        if args.num_frames is not None:
            num_frames = args.num_frames
            args.stop_frame = args.start_frame + num_frames
            print(f"=> args.stop_frame is set to {args.stop_frame}")
        else:
            if args.stop_frame is None:
                args.stop_frame = len(scene)-1
            num_frames = max(args.stop_frame - args.start_frame, 1)
        
        instance_id_map = scene.get_drawable_instance_ind_map()
        instance_cmap = np.array(get_n_ind_colors(len(instance_id_map)))
        
        classname_map = scene.get_drawable_class_ind_map()
        class_cmap = np.array(get_n_ind_colors(len(classname_map)))

        draw_box_fn = functools.partial(
            draw_box, thickness=(2 if args.downscale==1 else 1), fontscale=(1. if args.downscale==1 else 0.5), 
            instance_id_map=instance_id_map, instance_cmap=instance_cmap, classname_map=classname_map, class_cmap=class_cmap)

        # for frame_ind in tqdm(range(len(scene)), "rendering frames..."):
        log.info(f"Start [vis_anno], ds={args.downscale} in {args.exp_dir}")
        for frame_ind in tqdm(range(args.start_frame, args.stop_frame, 1), "rendering frames..."):
            scene.slice_at(frame_ind)
            for cam_id in cam_id_list:
                cam: Camera = scene.observers[cam_id]
                cam.intr.set_downscale(args.downscale)

                def to_img(tensor):
                    return tensor.reshape([cam.intr.H, cam.intr.W, -1]).data.cpu().numpy()

                ground_truth = scene_dataloader.get_image_and_gts(scene.id, cam.id, frame_ind)
                gt_rgb = np.clip((to_img(ground_truth['image_rgb'].contiguous())*255).astype(np.uint8),0,255)
                
                if args.lidar_id is not None:
                    lidar: RaysLidar = scene.observers[args.lidar_id]
                    lidar_gt = scene_dataloader.get_lidar_gts(scene.id, args.lidar_id, frame_ind, device=device)
                    lidar_rays_o, lidar_rays_d, lidar_ranges = lidar_gt['rays_o'], lidar_gt['rays_d'], lidar_gt['ranges']
                    lidar_mask = lidar_ranges > 0.1
                    local_lidar_pts = torch.addcmul(lidar_rays_o[lidar_mask], lidar_rays_d[lidar_mask], lidar_ranges[lidar_mask].unsqueeze(-1))
                    lidar_pts = lidar.world_transform(local_lidar_pts)
                    _, n, u, v, d = cam.project_pts_in_image(lidar_pts)
                    if n > 0:
                        color_d = color_depth(d.data.cpu().numpy(), scale=120.0, cmap='turbo')
                        u = u.data.long().cpu().numpy()
                        v = v.data.long().cpu().numpy()
                        # gt_rgb[v, u] = color_d
                        
                        for ui,vi,ci in zip(u,v,color_d):
                            cv2.circle(gt_rgb, (ui,vi), radius=2, color=ci.tolist(), thickness=2)
                        
                drawables = scene.get_drawables(True)
                drawables = cam.filter_drawable_groups(drawables)
                # drawables = [d for d in drawables if 'TBDn' in d.id]
                for obj in drawables:
                    if obj.i_valid and obj.id in instance_id_map.keys() and obj.class_name in classname_map.keys():
                        draw_box_fn(gt_rgb, obj, cam, inplace=True)
                all_gather[cam.id].append(gt_rgb)

        frames_per_obs_all = []
        for cam_id, frames in all_gather.items():
            write_video(os.path.join(vid_root, f"{name}_{cam_id}.mp4"), frames)
            frames_per_obs_all.append(np.array(frames))
        if len(frames_per_obs_all) > 1:
            write_video(os.path.join(vid_root, f"{name}_all.mp4"), np.concatenate(frames_per_obs_all, axis=2))

if __name__ == "__main__":
    from nr3d_lib.config import BaseConfig
    bc = BaseConfig()
    bc.parser.add_argument("--load_pt", type=str, default=None, help="Typically unnecessary as the final or latest ckpt is loaded automatically. \n"\
        "Only specify the ckpt file path if indeed a non-final or non-latest ckpt needs to be loaded.")
    bc.parser.add_argument("--downscale", type=float, default=1.0, help="Sets the side length downscale for rendering and output.")

    bc.parser.add_argument("--fps", type=int, default=24)
    bc.parser.add_argument("--quality", type=int, default=None, help="Sets the quality for imageio.mimwrite (range: 0-10; 10 is the highest; default is 5).")
    bc.parser.add_argument("--dirname", type=str, default='videos', help="Sets the output directory to /path/to/exp_dir/${dirname}`.")
    bc.parser.add_argument("--outbase", type=str, default=datetime.now().strftime("%Y_%m_%d_%H_%M_%S"), help="Sets the basename of the output file (without extension).")
    
    bc.parser.add_argument("--cam_id", type=str, default=None, help="If set, uses a specific camera; otherwise, uses all available cameras.")
    bc.parser.add_argument("--lidar_id", type=str, default=None, help="whether to draw a lidar on rgb")
    bc.parser.add_argument('--start_frame', type=int, default=0)
    bc.parser.add_argument("--num_frames", type=int, default=None)
    bc.parser.add_argument('--stop_frame', type=int, default=None)
    main_function(bc.parse())