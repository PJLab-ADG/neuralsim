"""
@file   eval.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Appearance evaluation evaluation on a single static scene.
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
import json
import imageio
import numpy as np
from tqdm import tqdm
from math import prod

import torch

from nr3d_lib.fmt import log
from nr3d_lib.plot import color_depth
from nr3d_lib.checkpoint import sorted_ckpts
from nr3d_lib.config import ConfigDict, BaseConfig
from nr3d_lib.utils import cond_mkdir, import_str
from nr3d_lib.render.utils import PSNR, SSIM

from nr3d_lib.models.spatial import AABBSpace
from nr3d_lib.models.spatial_accel import OccupancyGridAS
from nr3d_lib.models.loss.safe import safe_binary_cross_entropy


from app.resources.observers import Camera
from app.renderers import SingleVolumeRenderer
from app.resources import Scene, AssetBank, create_scene_bank, load_scene_bank

from dataio.dataloader import SceneDataLoader

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
    #----  Train - scene Bank & Asset Bank  ------
    #---------------------------------------------
    scene_bank_trainval, _ = load_scene_bank(os.path.join(args.exp_dir, 'scenarios'), device=device)
    
    #---------------------------------------------
    #-----------     Asset Bank     --------------
    #---------------------------------------------
    asset_bank = AssetBank(args.assetbank_cfg)
    asset_bank.create_asset_bank(scene_bank_trainval, load=state_dict['asset_bank'], device=device)
    asset_bank.to(device)
    log.info(asset_bank)

    #---------------------------------------------
    #---  Test - Scene Bank Dataset & Loader  ----
    #---------------------------------------------
    # NOTE: 1. Specify args.test_dataset_cfg (or args.dataset_cfg.test_param) if different from train/val dataset impl.
    #       Otherwise, directly use trainval's dataset_impl configuration
    if 'test_dataset_cfg' in args:
        dataset_impl_test = import_str(args.test_dataset_cfg.target)(args.test_dataset_cfg.param)
        is_dataset_impl_test = True
    elif 'test_param' in args.dataset_cfg:
        dataset_impl_test = import_str(args.dataset_cfg.target)(args.dataset_cfg.test_param)
        is_dataset_impl_test = True
    else:
        dataset_impl_test = import_str(args.dataset_cfg.target)(args.dataset_cfg.param)
        is_dataset_impl_test = False
    
    # NOTE: 2. Specify args.test_scenebank_cfg if different from train/val scenebank_cfg
    #       Otherwise, directly use trainval's scene_bank configuration.
    if 'test_scenebank_cfg' in args or is_dataset_impl_test:
        scene_bank_test, _ = create_scene_bank(
            dataset=dataset_impl_test, device=device, 
            scenebank_cfg=args.get('test_scenebank_cfg', args.scenebank_cfg), 
            drawable_class_names=asset_bank.class_name_configs.keys(),
            misc_node_class_names=asset_bank.misc_node_class_names
        )
    else:
        scene_bank_test = scene_bank_trainval
    
    # NOTE: 3. Specify args.training.test_dataloader if different from val_dataloader
    #       Otherwise, directly use val's dataloder configuration
    if 'test_dataloader' in args.training:
        dataloader_cfg_test = args.training.test_dataloader
    else:
        dataloader_cfg_test = args.training.val_dataloader
    
    dataloader_cfg_test.preload = False
    scene_dataloader = SceneDataLoader(scene_bank_test, dataset_impl=dataset_impl_test, config=dataloader_cfg_test, device=device)
    scene_dataloader.set_camera_downscale(args.downscale)

    #---------------------------------------------
    #---     Load assets to scene objects     ----
    #---------------------------------------------
    scene = scene_bank_test[0]
    scene.load_assets(asset_bank)
    # !!! Only call preprocess_per_train_step when all assets are ready & loaded !
    asset_bank.preprocess_per_train_step(args.training.num_iters) # NOTE: Finished training.

    # Fallsback to regular Mat4x4 for convenience
    # TODO

    #---------------------------------------------
    #------------     Renderer     ---------------
    #---------------------------------------------
    renderer = SingleVolumeRenderer(args.renderer)
    renderer.populate(asset_bank)
    renderer.eval()
    asset_bank.eval()
    renderer.config.rayschunk = args.rayschunk
    renderer.config.with_normal = True
    if args.depth_max is None:
        depth_max = renderer.config.far
    else:
        depth_max = args.depth_max
    assert depth_max is not None, "You need to specify at least one of the args.depth_max or renderer.config.far"
    for scene in scene_bank_test:
        # NOTE: When training, set all observer's near&far to a larger value
        for obs in scene.get_observers(False):
            obs.near = renderer.config.near
            obs.far = renderer.config.far

    cam_id_list = scene_dataloader.cam_id_list if args.cam_id is None else [args.cam_id]

    #---------------------------------------------
    #--------------     Plot     -----------------
    #---------------------------------------------
    expname = os.path.split(args.exp_dir.rstrip("/"))[-1]
    global_step = state_dict['global_step']
    global_step_str = f"iter{global_step/1000}k" if global_step >= 1000 else f"iter{global_step}"
    name = f"{expname[0:64]}_{global_step_str}_{args.outbase}_ds={args.downscale}"
    if args.forward_inv_s is not None:
        name += f"_s={int(args.forward_inv_s)}"
    
    vid_root = os.path.join(args.exp_dir, args.dirname)
    cond_mkdir(vid_root)
    if args.save_perframe_camera:
        vid_raw_root = os.path.join(vid_root, name)
        cond_mkdir(vid_raw_root)
    def write_video(uri, frames, **kwargs):
        if not args.no_output:
            if len(frames) > 1:
                if ".mp4" not in uri:
                    uri = f"{uri}.mp4"
                imageio.mimwrite(uri, frames, fps=args.fps, quality=args.quality, **kwargs)
                log.info(f"Video saved to {uri}")
            else:
                if ".mp4" in uri:
                    uri = f"{os.path.splitext(uri)[0]}.png"
                imageio.imwrite(uri, frames[0], **kwargs)
                log.info(f"Image saved to {uri}")
    
    collate_keys = ['rgb_gt', 'rgb_volume', 'depth_volume', 'normals_volume']
    all_gather = dict()
    for cam_id in cam_id_list:
        all_gather[cam_id] = dict({k: [] for k in collate_keys})

    # Test correct.
    ground_truth = scene_dataloader.get_rgb_gts(scene.id, cam_id_list[0], 0)
    # Collect rgb eval scores
    all_mask_metric = dict()
    all_fg_psnr_only_in_mask = dict()
    all_fg_psnr = dict()
    all_fg_ssim_only_in_mask = dict()
    all_fg_ssim = dict()
    all_bg_psnr = dict()
    all_bg_ssim = dict()
    all_full_psnr = dict()
    all_full_ssim = dict()
    for cam_id in cam_id_list:
        all_mask_metric[cam_id] = []
        all_fg_psnr_only_in_mask[cam_id] = []
        all_fg_psnr[cam_id] = []
        all_fg_ssim_only_in_mask[cam_id] = []
        all_fg_ssim[cam_id] = []
        all_bg_psnr[cam_id] = []
        all_bg_ssim[cam_id] = []
        all_full_psnr[cam_id] = []
        all_full_ssim[cam_id] = []

    with torch.no_grad():
        # for scene in scene_bank:
        scene: Scene = scene_bank_test[0]
        obj = scene.get_drawable_groups_by_class_name(scene.main_class_name)[0]
        model = obj.model
        if args.forward_inv_s is not None:
            model.ray_query_cfg.forward_inv_s = args.forward_inv_s

        log.info(f"Start [eval], ds={args.downscale}, in {exp_dir}")
        for frame_ind in tqdm(range(args.start_frame, args.stop_frame or len(scene), 1), "rendering frames..."):

            scene.frozen_at(frame_ind)
            
            for cam_id in cam_id_list:
                cam: Camera = scene.observers[cam_id]
                cam.intr.set_downscale(args.downscale)
                
                cur_frame_dict = dict({k: [] for k in collate_keys})
                #-----------------------------------------------
                ret = renderer.render(
                    scene, observer=cam, 
                    show_progress=args.progress, with_env=not args.no_sky, 
                    render_per_obj=True, render_per_obj_in_total=True)
                rendered = ret['rendered']
                main_rendered_in_total = ret['rendered_per_obj_in_total'][obj.id]
                #-----------------------------------------------
                
                def to_img(tensor):
                    return tensor.reshape([cam.intr.H, cam.intr.W, -1]).data.cpu().numpy()

                rgb_volume = (to_img(rendered['rgb_volume'])*255).clip(0,255).astype(np.uint8)
                
                ground_truth = scene_dataloader.get_rgb_gts(scene.id, cam.id, frame_ind)
                rgb_gt = (to_img(ground_truth['rgb'])*255).clip(0,255).astype(np.uint8)
                cur_frame_dict['rgb_gt'] = rgb_gt
                eval_rgb_pred = rendered['rgb_volume'].view(cam.intr.H, cam.intr.W, -1)
                eval_rgb_gt = ground_truth['rgb'].to(device).view(eval_rgb_pred.shape)
                
                full_psnr = PSNR(eval_rgb_pred, eval_rgb_gt).item()
                full_ssim = SSIM(eval_rgb_pred, eval_rgb_gt).item()
                all_full_psnr[cam_id].append(full_psnr)
                all_full_ssim[cam_id].append(full_ssim)
                
                if 'rgb_mask' in ground_truth:
                    # NOTE: If there is a ground truth mask, we can seperately evaluate fg & bg
                    eval_rgb_mask_pred = rendered['mask_volume'].view(cam.intr.H, cam.intr.W, 1)
                    eval_rgb_mask_gt = ground_truth['rgb_mask'].to(device).view(*eval_rgb_mask_pred.shape)
                    mean_bce = safe_binary_cross_entropy(eval_rgb_mask_pred, eval_rgb_mask_gt.float(), reduction='mean').item()
                    all_mask_metric[cam_id].append(mean_bce)
                    
                    fg_eval_pred = rendered['rgb_volume_occupied'].view(cam.intr.H, cam.intr.W, -1)
                    fg_eval_gt = eval_rgb_gt * eval_rgb_mask_gt.view(*eval_rgb_gt.shape[:-1], 1)
                    fg_psnr = PSNR(fg_eval_pred, fg_eval_gt, eval_rgb_mask_gt, only_in_mask=False).item()
                    fg_psnr_only_in_mask = PSNR(fg_eval_pred, fg_eval_gt, eval_rgb_mask_gt, only_in_mask=True).item()
                    fg_ssim = SSIM(fg_eval_pred, fg_eval_gt, eval_rgb_mask_gt, only_in_mask=False).item()
                    fg_ssim_only_in_mask = SSIM(fg_eval_pred, fg_eval_gt, eval_rgb_mask_gt, only_in_mask=True).item()
                    all_fg_psnr[cam_id].append(fg_psnr)
                    all_fg_psnr_only_in_mask[cam_id].append(fg_psnr_only_in_mask)
                    all_fg_ssim[cam_id].append(fg_ssim)
                    all_fg_ssim_only_in_mask[cam_id].append(fg_ssim_only_in_mask)
                    
                    non_occupied_mask = ~eval_rgb_mask_gt
                    # Decide bg_eval_gt
                    # non_occupied_rgb_gt_mode = args.training.losses.rgb.get('non_occupied_rgb_gt_mode', None)
                    non_occupied_rgb_gt_mode = args.eval_non_occupied_mode
                    if non_occupied_rgb_gt_mode is None:
                        bg_eval_gt = eval_rgb_gt * non_occupied_mask.view(*eval_rgb_gt.shape[:-1], 1)
                    elif non_occupied_rgb_gt_mode == 'black':
                        bg_eval_gt = eval_rgb_pred.new_zeros(eval_rgb_pred.shape)
                    elif non_occupied_rgb_gt_mode == 'white':
                        bg_eval_gt = eval_rgb_pred.new_ones(eval_rgb_pred.shape)
                    else:
                        raise RuntimeError(f"Invalid non_occupied_rgb_gt_mode={non_occupied_rgb_gt_mode}")
                    # Decide bg_eval_pred
                    if 'rgb_volume_non_occupied' in rendered:
                        bg_eval_pred = rendered['rgb_volume_non_occupied'].view(cam.intr.H, cam.intr.W, -1)
                    else:
                        bg_eval_pred = eval_rgb_pred * non_occupied_mask.view(*eval_rgb_pred.shape[:-1], 1)
                    bg_psnr = PSNR(bg_eval_pred, bg_eval_gt, non_occupied_mask).item()
                    bg_ssim = SSIM(bg_eval_pred, bg_eval_gt, non_occupied_mask).item()

                    all_bg_psnr[cam_id].append(bg_psnr)
                    all_bg_ssim[cam_id].append(bg_ssim)

                if not args.with_distant_depth:
                    # Since distant depth is usally messy and in-accurate
                    mask_volume = to_img(main_rendered_in_total['mask_volume'])
                    depth_volume = to_img(main_rendered_in_total['depth_volume'])
                else:
                    mask_volume = to_img(rendered['mask_volume'])
                    depth_volume = to_img(rendered['depth_volume'])
                
                depth_volume = mask_volume * np.clip(depth_volume/depth_max, 0, 1) + (1-mask_volume) * 1
                depth_volume = color_depth(depth_volume.squeeze(-1), scale=1, cmap='turbo')    # turbo_r, viridis, rainbow
                cur_frame_dict['rgb_volume'] = rgb_volume
                cur_frame_dict['depth_volume'] = depth_volume
                
                # Normals
                if 'normals_volume' in rendered:
                    if not args.with_distant_normal:
                        normals_volume = to_img(main_rendered_in_total['normals_volume'])
                    else:
                        normals_volume = to_img(rendered['normals_volume'])
                    normals_volume = ((normals_volume/2+0.5)*255).clip(0,255).astype(np.uint8)
                    cur_frame_dict['normals_volume'] = normals_volume
                
                for k, v in cur_frame_dict.items():
                    all_gather[cam.id][k].append(v)
                    if args.save_perframe_camera:
                        obs_dir = os.path.join(vid_raw_root, cam_id)
                        cond_mkdir(obs_dir)
                        k_dir = os.path.join(obs_dir, k)
                        cond_mkdir(k_dir)
                        imageio.imwrite(os.path.join(k_dir, f"{frame_ind:08d}.png"), v)

    # Metrics
    #--------------- PSNR
    total_full_psnr_per_cam = {cam_id: np.array(vals).mean() for cam_id, vals in all_full_psnr.items()}
    total_full_psnr = np.array(list((total_full_psnr_per_cam.values()))).mean()
    if len(next(iter(all_fg_psnr.values()))) > 0:
        total_fg_psnr_per_cam = {cam_id: np.array(vals).mean() for cam_id, vals in all_fg_psnr.items()}
        total_fg_psnr = np.array(list((total_fg_psnr_per_cam.values()))).mean()
        total_fg_psnr_only_in_mask_per_cam = {cam_id: np.array(vals).mean() for cam_id, vals in all_fg_psnr_only_in_mask.items()}
        total_fg_psnr_only_in_mask = np.array(list((total_fg_psnr_only_in_mask_per_cam.values()))).mean()
        total_bg_psnr_per_cam = {cam_id: np.array(vals).mean() for cam_id, vals in all_bg_psnr.items()}
        total_bg_psnr = np.array(list((total_bg_psnr_per_cam.values()))).mean()
    else:
        total_fg_psnr = None
        total_fg_psnr_only_in_mask = None
        total_bg_psnr = None
    
    psnr_f = os.path.join(vid_root, f'{name}.txt')
    with open(psnr_f, 'w') as f:
        f.write(f"full: {total_full_psnr:.4f}\n")
        if total_fg_psnr is not None:
            f.write(f"fg: {total_fg_psnr:.4f}\n")
            f.write(f"fg_only_in_mask: {total_fg_psnr_only_in_mask:.4f}\n")
            f.write(f"bg: {total_bg_psnr:.4f}\n")
        
        if total_fg_psnr is not None:
            f.write("fg".center(40, '=') + '\n')
            for cam_id, vals in all_fg_psnr.items():
                f.write(f"{cam_id}: {total_fg_psnr_per_cam[cam_id]:.4f}\n")
            for cam_id, vals in all_fg_psnr.items():
                f.write('='*40 + '\n')
                f.write(f"{cam_id}: {total_fg_psnr_per_cam[cam_id]:.4f}\n")
                f.writelines([f"{v:.4f}\n" for v in vals])
                f.write("\n")

            f.write("fg_only_in_mask".center(40, '=') + '\n')
            for cam_id, vals in all_fg_psnr.items():
                f.write(f"{cam_id}: {total_fg_psnr_only_in_mask_per_cam[cam_id]:.4f}\n")
            for cam_id, vals in all_fg_psnr.items():
                f.write('='*40 + '\n')
                f.write(f"{cam_id}: {total_fg_psnr_only_in_mask_per_cam[cam_id]:.4f}\n")
                f.writelines([f"{v:.4f}\n" for v in vals])
                f.write("\n")

            f.write("bg".center(40, '=') + '\n')
            for cam_id, vals in all_bg_psnr.items():
                f.write(f"{cam_id}: {total_bg_psnr_per_cam[cam_id]:.4f}\n")
            for cam_id, vals in all_bg_psnr.items():
                f.write('='*40 + '\n')
                f.write(f"{cam_id}: {total_bg_psnr_per_cam[cam_id]:.4f}\n")
                f.writelines([f"{v:.4f}\n" for v in vals])
                f.write("\n")
            
            f.write("full".center(40, '=') + '\n')
            for cam_id, vals in all_full_psnr.items():
                f.write(f"{cam_id}: {total_full_psnr_per_cam[cam_id]:.4f}\n")
            for cam_id, vals in all_full_psnr.items():
                f.write('='*40 + '\n')
                f.write(f"{cam_id}: {total_full_psnr_per_cam[cam_id]:.4f}\n")
                f.writelines([f"{v:.4f}\n" for v in vals])
                f.write("\n")
    log.info(f"PSNR saved to {psnr_f}")
    
    
    #--------------- SSIM
    total_full_ssim_per_cam = {cam_id: np.array(vals).mean() for cam_id, vals in all_full_ssim.items()}
    total_full_ssim = np.array(list((total_full_ssim_per_cam.values()))).mean()
    if len(next(iter(all_fg_ssim.values()))) > 0:
        total_fg_ssim_per_cam = {cam_id: np.array(vals).mean() for cam_id, vals in all_fg_ssim.items()}
        total_fg_ssim = np.array(list((total_fg_ssim_per_cam.values()))).mean()
        total_fg_ssim_only_in_mask_per_cam = {cam_id: np.array(vals).mean() for cam_id, vals in all_fg_ssim_only_in_mask.items()}
        total_fg_ssim_only_in_mask = np.array(list((total_fg_ssim_only_in_mask_per_cam.values()))).mean()
        total_bg_ssim_per_cam = {cam_id: np.array(vals).mean() for cam_id, vals in all_bg_ssim.items()}
        total_bg_ssim = np.array(list((total_bg_ssim_per_cam.values()))).mean()
    else:
        total_fg_ssim = None
        total_fg_ssim_only_in_mask = None
        total_bg_ssim = None
    
    ssim_f = os.path.join(vid_root, f'{name}.txt')
    with open(ssim_f, 'w') as f:
        f.write(f"full: {total_full_ssim:.4f}\n")
        if total_fg_ssim_only_in_mask is not None:
            f.write(f"fg: {total_fg_ssim:.4f}\n")
            f.write(f"fg_only_in_mask: {total_fg_ssim_only_in_mask:.4f}\n")
            f.write(f"bg: {total_bg_ssim:.4f}\n")
        
        if total_fg_ssim_only_in_mask is not None:
            f.write("fg".center(40, '=') + '\n')
            for cam_id, vals in all_fg_ssim.items():
                f.write(f"{cam_id}: {total_fg_ssim_per_cam[cam_id]:.4f}\n")
            for cam_id, vals in all_fg_ssim.items():
                f.write('='*40 + '\n')
                f.write(f"{cam_id}: {total_fg_ssim_per_cam[cam_id]:.4f}\n")
                f.writelines([f"{v:.4f}\n" for v in vals])
                f.write("\n")
            
            f.write("fg_only_in_mask".center(40, '=') + '\n')
            for cam_id, vals in all_fg_ssim.items():
                f.write(f"{cam_id}: {total_fg_ssim_only_in_mask_per_cam[cam_id]:.4f}\n")
            for cam_id, vals in all_fg_ssim.items():
                f.write('='*40 + '\n')
                f.write(f"{cam_id}: {total_fg_ssim_only_in_mask_per_cam[cam_id]:.4f}\n")
                f.writelines([f"{v:.4f}\n" for v in vals])
                f.write("\n")
            
            f.write("bg".center(40, '=') + '\n')
            for cam_id, vals in all_bg_ssim.items():
                f.write(f"{cam_id}: {total_bg_ssim_per_cam[cam_id]:.4f}\n")
            for cam_id, vals in all_bg_ssim.items():
                f.write('='*40 + '\n')
                f.write(f"{cam_id}: {total_bg_ssim_per_cam[cam_id]:.4f}\n")
                f.writelines([f"{v:.4f}\n" for v in vals])
                f.write("\n")
            
            f.write("full".center(40, '=') + '\n')
            for cam_id, vals in all_full_ssim.items():
                f.write(f"{cam_id}: {total_full_ssim_per_cam[cam_id]:.4f}\n")
            for cam_id, vals in all_full_ssim.items():
                f.write('='*40 + '\n')
                f.write(f"{cam_id}: {total_full_ssim_per_cam[cam_id]:.4f}\n")
                f.writelines([f"{v:.4f}\n" for v in vals])
                f.write("\n")
    log.info(f"SSIM saved to {ssim_f}")
    
    
    #--------------- MISC
    misc = {}
    misc['full_psnr'] = total_full_psnr
    misc['fg_psnr'] = total_fg_psnr
    misc['fg_psnr_only_in_mask'] = total_fg_psnr_only_in_mask
    misc['bg_psnr'] = total_bg_psnr
    
    misc['full_ssim'] = total_full_ssim
    misc['fg_ssim'] = total_fg_ssim
    misc['fg_ssim_only_in_mask'] = total_fg_ssim_only_in_mask
    misc['bg_ssim'] = total_bg_ssim
    
    total_mask_per_cam = {cam_id: (np.array(vals).mean() if len(vals) > 0 else 0) for cam_id, vals in all_mask_metric.items()}
    total_mask_metric = np.array(list((total_mask_per_cam.values()))).mean()
    misc['mask_metric'] = total_mask_metric
    
    if isinstance(model.space, AABBSpace):
        aabb = model.space.aabb
        misc['aabb'] = aabb.tolist()
        volume = prod((aabb[1] - aabb[0]).tolist())
        misc['aabb_volume'] = volume
        if isinstance((accel:=model.accel), OccupancyGridAS):
            misc['occ_ratio'] = accel.frac_occupied()
        
    misc_f = os.path.join(vid_root, f'{name}_misc.json')
    with open(misc_f, 'w') as f:
        json.dump(misc, f)
    log.info(f"MISC saved to {misc_f}")

    #--------- Seperate video
    if args.save_seperate_keys:
        for cam_id, obs_dict in all_gather.items():
            for k, frames in obs_dict.items():
                write_video(os.path.join(vid_root, f"{name}_{cam_id}_{k}.mp4"), frames)

    #--------- 1 row collection
    keys_1l = ['rgb_gt', 'rgb_volume', 'depth_volume', 'normals_volume']
    frames_per_obs_1l_all = []
    for cam_id, obs_dict in all_gather.items():
        frames_per_obs_1l = []
        new_obs_dict = dict()
        for k in keys_1l:
            new_obs_dict[k] = obs_dict[k]
        for kvs in zip(*(new_obs_dict.values())):
            frames_per_obs_1l.append(np.concatenate(kvs, axis=1))
        write_video(os.path.join(vid_root, f"{name}_{cam_id}_1l.mp4"), frames_per_obs_1l)
        frames_per_obs_1l_all.append(np.array(frames_per_obs_1l))
    if len(frames_per_obs_1l_all) > 1:
        # NOTE: Only when different cameras has the same width
        frames_per_obs_1l_all = np.concatenate(frames_per_obs_1l_all, axis=1)
        write_video(os.path.join(vid_root, f"{name}_1l_all.mp4"), frames_per_obs_1l_all)
    
    #--------- 3 cam concat
    cam_id_list = ['camera_FRONT_LEFT', 'camera_FRONT', 'camera_FRONT_RIGHT'] # For waymo's three frontal cameras
    if set(all_gather.keys()) == set(cam_id_list):
        keys_1c = ['rgb_gt', 'rgb_volume', 'normals_volume', 'depth_volume']
        frame_per_obs_1c_all = []
        for cam_id in cam_id_list:
            obs_dict = all_gather[cam_id]
            frames_per_obs_1c = []
            new_obs_dict = dict()
            for k in keys_1c:
                new_obs_dict[k] = obs_dict[k]
            for kvs in zip(*(new_obs_dict.values())):
                frames_per_obs_1c.append(np.concatenate(kvs, axis=0))
            frame_per_obs_1c_all.append(np.array(frames_per_obs_1c))
        frame_per_obs_1c_all = np.concatenate(frame_per_obs_1c_all, axis=2)
        write_video(os.path.join(vid_root, f"{name}_1c_all.mp4"), frame_per_obs_1c_all)
    
def make_parser():
    bc = BaseConfig()
    
    bc.parser.add_argument("--rayschunk", type=int, default=65536, help="Chunkify the rendering process.")
    bc.parser.add_argument("--downscale", type=float, default=1.0, help="Sets the side length downscale for rendering and output.")
    # NeuS specific hacks
    bc.parser.add_argument("--forward_inv_s", type=float, default=None, help="Bypasses the inv_s parameter for NeuS during rendering.")

    bc.parser.add_argument("--no_output", action='store_true', help="If set, the rendered video will not be saved.")
    bc.parser.add_argument("--no_sky", action='store_true', help="If set, the sky model will not be rendered.")
    bc.parser.add_argument("--eval_fg_only_in_mask", action='store_true')
    bc.parser.add_argument("--eval_non_occupied_mode", type=str, default='black')
    bc.parser.add_argument("--progress", action='store_true', help="If set, shows per frame progress.")
    bc.parser.add_argument("--save_seperate_keys", action='store_true', help="If set, saves raw per key camera renderings.")
    bc.parser.add_argument("--save_perframe_camera", action='store_true', help="If set, saves raw per frame camera renderings.")
    bc.parser.add_argument("--with_distant_depth", action='store_true', help="If set, uses joint depth of cr+dv for depth visualization, otherwise uses cr.")
    bc.parser.add_argument("--with_distant_normal", action='store_true', help="If set, uses joint depth of cr+dv for normal visualization (only if dv can output normal), otherwise uses cr.")
    
    bc.parser.add_argument("--fps", type=int, default=24)
    bc.parser.add_argument("--start_frame", type=int, default=0)
    bc.parser.add_argument("--stop_frame", type=int, default=None)
    bc.parser.add_argument("--quality", type=int, default=None, help="Sets the quality for imageio.mimwrite (range: 0-10; 10 is the highest; default is 5).")
    bc.parser.add_argument("--dirname", type=str, default='eval', help="Sets the output directory to /path/to/exp_dir/${dirname}`.")
    bc.parser.add_argument("--outbase", type=str, default='eval', help="Sets the basename of the output file (without extension).")
    # Camera Sim Argument
    bc.parser.add_argument("--cam_id", type=str, default=None, help="If set, uses a specific camera; otherwise, uses all available cameras.")
    return bc

if __name__ == "__main__":
    bc = make_parser()
    main_function(bc.parse())