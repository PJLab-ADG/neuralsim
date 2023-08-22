"""
@file   eval_lidar.py
@author Jianfei Guo, Shanghai AI Lab & Nianchen Deng, Shanghai AI Lab
@brief  Evaluate lidar quality by Chamfer Distance & Depth l1 error.
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
import torch
import imageio
import logging
import numpy as np
from typing import Dict
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm
from datetime import datetime
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont

from nr3d_lib.fmt import log
from nr3d_lib.checkpoint import sorted_ckpts
from nr3d_lib.utils import cond_mkdir, import_str
from nr3d_lib.config import ConfigDict, BaseConfig
from nr3d_lib.models.spatial import AABBSpace, ForestBlockSpace
from nr3d_lib.geometry import chamfer_distance_borrowed_from_pt3d as chamfer_distance

from app.renderers import SingleVolumeRenderer
from app.resources.observers import RaysLidar, Camera, MultiCamBundle
from app.resources import Scene, AssetBank, create_scene_bank, load_scene_bank

from dataio.dataloader import SceneDataLoader

LOG = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

@dataclass
class LidarData:
    pcl_world: torch.FloatTensor # LiDAR pointclouds in world coords
    pcl_local: torch.FloatTensor # LiDAR pointclouds in sensor local coords
    ranges: torch.FloatTensor # LiDAR ranges

@torch.no_grad()
def main_function(args: ConfigDict):
    exp_dir = args.exp_dir
    device = torch.device('cuda', 0)
    dtype = torch.float32

    output_root = os.path.join(args.exp_dir, args.dirname)
    cond_mkdir(output_root)
    # pcl_save_dir = os.path.join(output_root, 'pcls')
    # cond_mkdir(pcl_save_dir)

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
    if ('test_scenebank_cfg' in args) or is_dataset_impl_test:
        scene_bank_test, _ = create_scene_bank(
            dataset=dataset_impl_test, device=device, 
            scenebank_cfg=args.get('test_scenebank_cfg', args.scenebank_cfg), 
            drawable_class_names=asset_bank.class_name_configs.keys(),
            misc_node_class_names=asset_bank.misc_node_class_names
        )
        print("=> Creating new scene_bank for evaluation/testing")
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
    scene_dataloader.set_camera_downscale(1.)

    #---------------------------------------------
    #---     Load assets to scene objects     ----
    #---------------------------------------------
    scene = scene_bank_test[0]
    scene.load_assets(asset_bank)
    # !!! Only call preprocess_per_train_step when all assets are ready & loaded !
    asset_bank.preprocess_per_train_step(args.training.num_iters) # NOTE: Finished training.
    
    #---------------------------------------------
    #------------     Renderer     ---------------
    #---------------------------------------------
    renderer = SingleVolumeRenderer(args.renderer)
    renderer.populate(asset_bank)
    renderer.eval()
    asset_bank.eval()
    renderer.config.rayschunk = args.rayschunk
    renderer.config.with_normal = True
    for scene in scene_bank_test:
        # NOTE: When training, set all observer's near&far to a larger value
        for obs in scene.get_observers(False):
            obs.near = renderer.config.near
            obs.far = renderer.config.far

    #---------------------------------------------
    #--------------     Plot     -----------------
    #---------------------------------------------    
    global_step = state_dict['global_step']
    global_step_str = f"iter{global_step/1000}k" if global_step >= 1000 else f"iter{global_step}"
    name = f"{args.outbase}_{global_step_str}"

    forward_inv_s = args.forward_inv_s
    if forward_inv_s.lower() == 'none' or forward_inv_s.lower() == 'null':
        forward_inv_s = None
    else:
        forward_inv_s = float(forward_inv_s)
    if forward_inv_s is not None:
        name += f"_s={int(forward_inv_s)}"

    if args.num_frames is not None:
        num_frames = args.num_frames
        args.stop_frame = args.start_frame + num_frames
        log.info(f"=> args.stop_frame is set to {args.stop_frame}")
    else:
        if args.stop_frame is None:
            args.stop_frame = len(scene)
        num_frames = max(args.stop_frame - args.start_frame, 1)
    
    scene: Scene = scene_bank_test[0]
    scene.frozen_at(args.start_frame)
    obj = scene.get_drawable_groups_by_class_name(scene.main_class_name, only_valid=False)[0]
    assert args.lidar_id in scene.observers, f"Invalid lidar_id={args.lidar_id},\n"\
        f"Current LiDARs: {list(scene.get_observer_groups_by_class_name('RaysLidar', False).keys())+list(scene.get_observer_groups_by_class_name('Lidar', False).keys())}"
    lidar: RaysLidar = scene.observers[args.lidar_id]

    model = obj.model
    if forward_inv_s is not None:
        model.ray_query_cfg.forward_inv_s = forward_inv_s

    pcl_imgs = {}
    
    chamfers_pred = []
    chamfers_pred_99 = []
    chamfers_pred_97 = []
    chamfers_pred_95 = []
    
    chamfers_gt = []
    chamfers_gt_99 = []
    chamfers_gt_97 = []
    chamfers_gt_95 = []
    
    chamfers_all = []
    chamfers_all_99 = []
    chamfers_all_97 = []
    chamfers_all_95 = []
    
    depth_errors = []
    depth_errors_99 = []
    depth_errors_97 = []
    depth_errors_95 = []

    @torch.no_grad()
    def filter_lidar_gts_in_scene_space(lidar_gt:  Dict[str, torch.Tensor], frame_ind: int):
        assert scene.i == frame_ind
        pts_gt = torch.addcmul(lidar_gt['rays_o'], lidar_gt['rays_d'], lidar_gt['ranges'].unsqueeze(-1))
        pts_gt = lidar.world_transform(pts_gt)
        # From [world] to [obj]
        pts_gt_in_obj = obj.world_transform.forward(pts_gt, inv=True)
        # From [obj] to [obj-net]
        if isinstance(model.space, AABBSpace):
            pts_gt_in_net = model.space.normalize_coords(pts_gt_in_obj)
            lidar_filter_inds = ((pts_gt_in_net >= -1) & (pts_gt_in_net <= 1)).all(dim=-1).nonzero()[..., 0]
            lidar_gt = {k: v[lidar_filter_inds] for k,v in lidar_gt.items()}
            pts_gt = pts_gt[lidar_filter_inds]
        elif isinstance(model.space, ForestBlockSpace):
            raise NotImplementedError
        else:
            raise RuntimeError(f"Unsupported model.space type={type(model.space)}")
        return lidar_gt

    @torch.no_grad()
    def filter_lidar_gts_toofar(lidar_gt:  Dict[str, torch.Tensor], frame_ind: int, too_far: float):
        assert scene.i == frame_ind
        lidar_filter_inds = (lidar_gt['ranges'] <= too_far).nonzero()[..., 0]
        lidar_gt = {k: v[lidar_filter_inds] for k,v in lidar_gt.items()}
        return lidar_gt

    @torch.no_grad()
    def get_lidar_pcl(lidar_gt: Dict[str, torch.Tensor], pred_mask: float = 0.5):
        assert lidar.i is not None and scene.i is not None
        gt_rays_valid = lidar_gt['ranges'] > 0
        ranges_gt = lidar_gt['ranges'][gt_rays_valid]
        rays_o, rays_d = lidar_gt['rays_o'][gt_rays_valid], lidar_gt['rays_d'][gt_rays_valid]
        pcl_local_gt = torch.addcmul(rays_o, rays_d, ranges_gt.unsqueeze(-1))
        pcl_world_gt = lidar.world_transform(pcl_local_gt)
        
        gt = LidarData(
            pcl_world=pcl_world_gt, 
            pcl_local=pcl_local_gt, 
            ranges=ranges_gt
        )
        
        # Lidar rays in world
        rays_o_world, rays_d_world = lidar.get_selected_rays(rays_o=rays_o, rays_d=rays_d)
        
        lidar_pred = renderer.render(scene, rays=(rays_o_world, rays_d_world), observer=lidar, only_cr=True,
                                    with_rgb=False, with_normal=False, render_per_obj=False,
                                    rayschunk=args.rayschunk, show_progress=args.progress)
        lidar_rays_acc = lidar_pred['rendered']['mask_volume']
        lidar_rays_depth = lidar_pred['rendered']['depth_volume']
        pred_valid_in_gt = lidar_rays_acc > pred_mask # For NeRF, there are cases when all of the pred mask is between 0.1~0.85
        #-------- outputs
        ranges_pred = lidar_rays_depth[pred_valid_in_gt]
        pred = LidarData(
            pcl_world=torch.addcmul(rays_o_world[pred_valid_in_gt], rays_d_world[pred_valid_in_gt], ranges_pred.unsqueeze(-1)), 
            pcl_local=torch.addcmul(rays_o[pred_valid_in_gt], rays_d[pred_valid_in_gt], ranges_pred.unsqueeze(-1)), 
            ranges=ranges_pred
        )
        return pred, gt, pred_valid_in_gt

    def save_lidar_pcl(pcl_save_dir: str, frame_i: int, data: LidarData, prefix: str):
        lidar_pts_datas = torch.cat([data.pcl_local, data.ranges.unsqueeze(-1)], dim=-1).cpu().numpy()
        np.save(os.path.join(pcl_save_dir, f"{prefix}_{frame_i:08d}.npy"), lidar_pts_datas)

    def vis_lidar_pcl_mayavi(pcl_world: torch.Tensor, pcl_val: torch.Tensor, min: float=None, max: float=None):
        nonlocal fig
        # NOTE: Convert to a common coordinate system (OpenCV pinhole camera in this case)
        pcl_cam_ref = cam_ref.world_transform.forward(pcl_world, inv=True).cpu().numpy()
        if min is None:
            min = pcl_val.min().item()
        if max is None:
            max = pcl_val.max().item()
        pcl_val = pcl_val.cpu().numpy()

        mlab.clf()
        mlab.points3d(pcl_cam_ref[..., 0], pcl_cam_ref[... ,1], pcl_cam_ref[...,2], pcl_val,
                      mode="point", colormap='rainbow', vmin=min, vmax=max, figure=fig)

        # Top view
        mlab.view(focalpoint=np.array([0., 0., 15.]), distance=100.0, 
                  azimuth=90.0, elevation=-90.0, roll=-90.0)
        fig.scene._lift()
        im1 = mlab.screenshot(figure=fig, mode='rgb', antialiased=True)

        # Front view
        mlab.view(focalpoint=np.array([0., 0., 50.]), distance=70.0, 
                  azimuth=-90.0, elevation=176.0, roll=179.0)
        fig.scene._lift()
        im2 = mlab.screenshot(figure=fig, mode='rgb', antialiased=True)

        return im1, im2

    def vis_lidar_pcl_vedo(pcl_world: torch.Tensor, pcl_val: torch.Tensor, min: float=None, max: float=None):
        nonlocal plt_top, plt_front
        # NOTE: Convert to a common coordinate system (OpenCV pinhole camera in this case)
        pcl_cam_ref = cam_ref.world_transform.forward(pcl_world, inv=True).cpu().numpy()
        if min is None:
            min = pcl_val.min().item()
        if max is None:
            max = pcl_val.max().item()
        pcl_val = pcl_val.cpu().numpy()
        pts_c = (vedo.color_map(pcl_val, 'rainbow', vmin=min, vmax=max) * 255.).clip(0,255).astype(np.uint8)
        pts_c = np.concatenate([pts_c, np.full_like(pts_c[:,:1], 255)], axis=-1) # RGBA is ~50x faster
        lidar_pts = vedo.Points(pcl_cam_ref, c=pts_c, r=2)
        # Top view
        plt_top.clear()
        plt_top.show(lidar_pts, resetcam=False, size=[W_lidar_vis, H_lidar_vis], 
                 camera={'focal_point': [0., 0., 15.], 'pos': [0., -100., 15.], 'viewup': [-1,0,0]})
        im1 = plt_top.topicture().tonumpy()
        
        # Front view
        plt_front.clear()
        plt_front.show(lidar_pts, resetcam=False, size=[W_lidar_vis, H_lidar_vis], 
                 camera={'focal_point': [0., 0., 50.], 'pos': [0., -5, -19.82120022], 'viewup': [ 0., -0.99744572, 0.07142857]})
        im2 = plt_front.topicture().tonumpy()
        
        return im1, im2

    def draw_text(im, content):
        im = Image.fromarray(im)
        img_draw = ImageDraw.Draw(im)
        img_draw.text((int(W_lidar_vis * 0.05), int(W_lidar_vis * 0.05)), content, font=ttf, fill=font_color)
        im = np.asarray(im)
        return im

    lidar.far = args.lidar_far
    W_lidar_vis = args.video_width
    H_lidar_vis = W_lidar_vis * 9 // 16

    if args.video_backend is not None:
        cam_id_list = scene_dataloader.cam_id_list
        cam_ref_id = cam_id_list[0] if len(cam_id_list) == 1 else args.cam_ref
        assert cam_ref_id is not None and cam_ref_id in scene.observers, \
            f"A valid frontal reference camera is required.\nCurrent camera list={scene.get_observer_groups_by_class_name('Camera', False)}"
        cam_ref: Camera = scene.observers[cam_ref_id]
        
        if args.video_backend == 'mayavi':
            from mayavi import mlab
            mlab.options.offscreen = True
            
            vis_lidar_pcl = vis_lidar_pcl_mayavi
            if args.video_bg == 'black':
                bg_color = (0, 0, 0)
                font_color = (255, 255, 255)
            elif args.video_bg == 'white':
                bg_color = (255, 255, 255)
                font_color = (0, 0, 0)
            else:
                raise RuntimeError(f"Invalid video_bg={args.video_bg}")

            fig = mlab.figure(bgcolor=np.array(bg_color)/255., size=(W_lidar_vis, H_lidar_vis))
            # ttf_path = "/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf"
            ttf_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf" # sudo apt install msttcorefonts -qq
            ttf = ImageFont.truetype(ttf_path, int(W_lidar_vis * 0.05))
        
        elif args.video_backend == 'vedo':
            import vedo
            import vedo.utils
            
            vis_lidar_pcl = vis_lidar_pcl_vedo
            bg_color = np.array(vedo.get_color(args.video_bg)) * 255.
            font_color = tuple((255. - bg_color).clip(0, 255).astype(np.uint8).tolist())  # Complementary color
            bg_color = tuple(bg_color.clip(0, 255).astype(np.uint8).tolist())
            plt_top = vedo.Plotter(interactive=False, offscreen=not args.video_verbose, size=[W_lidar_vis, H_lidar_vis], bg=args.video_bg)
            plt_front = vedo.Plotter(interactive=False, offscreen=not args.video_verbose, size=[W_lidar_vis, H_lidar_vis], bg=args.video_bg)
            plt_demo = vedo.Plotter(interactive=False, offscreen=not args.video_verbose, shape=(1,2), size=[1200*2, 900], bg=args.video_bg)
            # NOTE: To debug camera pose: print out camera parameters when dragging
            #       Need to set plt_demo with interactive=True
            # def callbck(evt):
            #     nonlocal plt_demo
            #     print(f"focal={plt_demo.camera.GetFocalPoint()}, pos={plt_demo.camera.GetPosition()}, viewup={plt_demo.camera.GetViewUp()}")
            # plt_demo.add_callback("Interaction", callbck)
            
            ttf_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf" # sudo apt install msttcorefonts -qq
            ttf = ImageFont.truetype(ttf_path, int(W_lidar_vis * 0.05))
        
        else:
            raise RuntimeError(f"Invalid video_backend={args.video_backend}")

    if 'lidar' in scene_dataloader.config.tags:
        lidar_filter_kwargs = scene_dataloader.config.tags.lidar.filter_kwargs
    else:
        lidar_filter_kwargs = ConfigDict(
            filter_in_cams=True
        )
        scene_dataloader.config.tags.lidar = ConfigDict(
            list=[args.lidar_id], 
            multi_lidar_merge=True,
            filter_kwargs=lidar_filter_kwargs
        )
        
    if args.bypass_filter_in_cam is not None:
        lidar_filter_kwargs.filter_in_cams = args.bypass_filter_in_cam
    if args.bypass_filter_objs_dynamic is not None:
        lidar_filter_kwargs.filter_out_obj_dynamic_only = args.bypass_filter_objs_dynamic
    
    log.info(f"Start [eval_lidar] in {exp_dir}")
    with logging_redirect_tqdm():
        for frame_ind in trange(args.start_frame, args.stop_frame, 1):
            # Load lidar gt data; NOTE: disable originally configured filtering to allow for manual override.
            lidar_gt = scene_dataloader.get_lidar_gts(scene.id, lidar.id, frame_ind, device=device, filter_if_configured=False)
            lidar_gt = scene_dataloader.filter_lidar_gts(scene.id, lidar.id, frame_ind, lidar_gt, inplace=False, **lidar_filter_kwargs)

            scene.frozen_at(frame_ind)
            if not args.no_filter_in_scene_space:
                lidar_gt = filter_lidar_gts_in_scene_space(lidar_gt, frame_ind)
            if args.filter_toofar is not None:
                lidar_gt = filter_lidar_gts_toofar(lidar_gt, frame_ind, args.filter_toofar)

            pred, gt, pred_valid_in_gt = get_lidar_pcl(lidar_gt)
            
            # save_lidar_pcl(pcl_save_dir, frame_ind, gt, "gt")
            # save_lidar_pcl(pcl_save_dir, frame_ind, pred, "pred")
            
            #---- Metric: chamfer dis
            if args.cd_filter_by_pred_mask:
                cham_pred, cham_gt = chamfer_distance(pred.pcl_world, gt.pcl_world[pred_valid_in_gt])
            else:
                cham_pred, cham_gt = chamfer_distance(pred.pcl_world, gt.pcl_world)
            
            cham_pred_sorted = torch.sort(cham_pred).values
            cham_gt_sorted = torch.sort(cham_gt).values
        
            mean_cham_pred = cham_pred.mean().item()
            mean_cham_gt = cham_gt.mean().item()
            
            #---- Metric: depth mse (must filter_by_pred_mask)
            depth_pred = pred.ranges
            depth_gt_of_pred = gt.ranges[pred_valid_in_gt]
            depth_err_each_abs = (depth_pred-depth_gt_of_pred).abs()
            # RMSE: Root Mean Sqaure Error
            depth_err_rmse = depth_err_each_abs.square().mean().sqrt().item()

            #---- Test plot convergence curve mentioned in [NFL] (https://research.nvidia.com/labs/toronto-ai/nfl/)
            # x_ranges = torch.arange(0, 100, 1, device=device) / 100
            # y_percentages = []
            # plot_chunk = 10
            # for i in range(0, 100, plot_chunk):
            #     xi = x_ranges[i:i+plot_chunk]
            #     yi = (depth_err_each_abs.unsqueeze(-1) < xi).sum(dim=0) / len(depth_err_each_abs) * 100
            #     y_percentages.append(yi)
            # y_percentages = torch.cat(y_percentages)
            # import matplotlib.pyplot as plt
            # plt.plot(x_ranges.data.cpu().numpy(), y_percentages.data.cpu().numpy())
            
            mean_cham_pred_99 = cham_pred_sorted[0:int(cham_pred_sorted.numel()*0.99)].mean().item()
            mean_cham_pred_97 = cham_pred_sorted[0:int(cham_pred_sorted.numel()*0.97)].mean().item()
            mean_cham_pred_95 = cham_pred_sorted[0:int(cham_pred_sorted.numel()*0.95)].mean().item()
            
            mean_cham_gt_99 = cham_gt_sorted[0:int(cham_gt_sorted.numel()*0.99)].mean().item()
            mean_cham_gt_97 = cham_gt_sorted[0:int(cham_gt_sorted.numel()*0.97)].mean().item()
            mean_cham_gt_95 = cham_gt_sorted[0:int(cham_gt_sorted.numel()*0.95)].mean().item() # Small difference compared to 0.97
            
            depth_err_each_abs_sorted = torch.sort(depth_err_each_abs)[0]
            depth_err_rmse_99 = depth_err_each_abs_sorted[0:int(depth_err_each_abs_sorted.numel()*0.99)].square().mean().sqrt().item()
            depth_err_rmse_97 = depth_err_each_abs_sorted[0:int(depth_err_each_abs_sorted.numel()*0.97)].square().mean().sqrt().item()
            depth_err_rmse_95 = depth_err_each_abs_sorted[0:int(depth_err_each_abs_sorted.numel()*0.95)].square().mean().sqrt().item()

            if args.video_backend is not None:
                if args.video_backend == 'vedo':
                    #---- DEBUG Plot
                    lidar_pts_gt_ref = cam_ref.world_transform.forward(gt.pcl_world, inv=True).data.cpu().numpy()
                    pcl_gt = vedo.Points(lidar_pts_gt_ref, c='gray3', r=2)
                    anno_gt = vedo.Text2D(f"GT", font=ttf_path, s=2)
                    gt_plots = [anno_gt, pcl_gt]
                    
                    lidar_pts_pred_ref = cam_ref.world_transform.forward(pred.pcl_world, inv=True).data.cpu().numpy()
                    # pts_pred_c = (vedo.color_map(cham_pred.data.cpu().numpy(), 'rainbow', vmin=0.0, vmax=args.errormap_chamfer_max) * 255.).clip(0,255).astype(np.uint8)
                    pts_pred_c = (vedo.color_map(depth_err_each_abs.data.cpu().numpy(), 'rainbow', vmin=0.0, vmax=args.errormap_depth_max) * 255.).clip(0,255).astype(np.uint8)
                    pts_pred_c = np.concatenate([pts_pred_c, np.full_like(pts_pred_c[:,:1], 255)], axis=-1) # RGBA is ~50x faster
                    # pcl_pred = vedo.Points(lidar_pts_pred_ref, c=pts_pred_c, r=2)
                    pcl_pred = vedo.Points(lidar_pts_pred_ref, r=2)
                    pcl_pred.cmap('rainbow', depth_err_each_abs.data.cpu().numpy(), vmin=0., vmax=args.errormap_depth_max)
                    pcl_pred.add_scalarbar(title="", font_size=24, nlabels=6, c='black')
                    anno_pred = vedo.Text2D(f"Rendered/simulated\nChamfer Distance={mean_cham_pred:.3f}" , font=ttf_path, s=2)
                    pred_plots = [anno_pred, pcl_pred]
                    # vedo.show([gt_plots, pred_plots], shape=(1,2), new=True, size=(1200*2, 900), axes=1, 
                    #           camera={'focal_point': [0., 0., 50.]})
                    plt_demo.clear(at=0, deep=True)
                    plt_demo.clear(at=1, deep=True)
                    plt_demo.show(gt_plots, at=0, resetcam=False, camera={'focal_point': [-6.1,6.2,34.1], 'pos': [18.0,-22.1,-26.7], 'viewup': [0.006176,-0.905337,0.424648]})
                    plt_demo.show(pred_plots, at=1, resetcam=False)
                    im_demo = plt_demo.topicture().tonumpy()
                    pcl_imgs.setdefault("demo", []).append(im_demo)
                
                min_range = max(gt.ranges.quantile(0.05).item(), pred.ranges.quantile(0.05).item())
                max_range = max(gt.ranges.quantile(0.95).item(), pred.ranges.quantile(0.95).item())

                im1, im2 = vis_lidar_pcl(gt.pcl_world, gt.ranges, min=min_range, max=max_range)
                if args.video_label:
                    im1 = draw_text(im1, "Ground Truth")
                    im2 = draw_text(im2, "Ground Truth")
                pcl_imgs.setdefault("gt", []).append([im1, im2])
                
                im1, im2 = vis_lidar_pcl(pred.pcl_world, pred.ranges, min=min_range, max=max_range)
                if args.video_label:
                    im1 = draw_text(im1, "Predicted")
                    im2 = draw_text(im2, "Predicted")
                pcl_imgs.setdefault("pred", []).append([im1, im2])

                im1, im2 = vis_lidar_pcl(gt.pcl_world, cham_gt, min=0.0, max=args.errormap_chamfer_max)
                if args.video_label:
                    im1 = draw_text(im1, f"GT Chamfer: {mean_cham_gt:.3f}")
                    im2 = draw_text(im2, f"GT Chamfer: {mean_cham_gt:.3f}")
                pcl_imgs.setdefault("gt_err_chamfer", []).append([im1, im2])

                im1, im2 = vis_lidar_pcl(pred.pcl_world, cham_pred, min=0.0, max=args.errormap_chamfer_max)
                if args.video_label:
                    im1 = draw_text(im1, f"Pred Chamfer: {mean_cham_pred:.3f}")
                    im2 = draw_text(im2, f"Pred Chamfer: {mean_cham_pred:.3f}")
                pcl_imgs.setdefault("pred_err_chamfer", []).append([im1, im2])
                
                im1, im2 = vis_lidar_pcl(pred.pcl_world, depth_err_each_abs, min=0.0, max=args.errormap_depth_max)
                if args.video_label:
                    im1 = draw_text(im1, f"RMSE: {depth_err_rmse:.3f}")
                    im2 = draw_text(im2, f"RMSE: {depth_err_rmse:.3f}")
                pcl_imgs.setdefault("pred_err_depth", []).append([im1, im2])

            depth_errors.append(depth_err_rmse)
            depth_errors_99.append(depth_err_rmse_99)
            depth_errors_97.append(depth_err_rmse_97)
            depth_errors_95.append(depth_err_rmse_95)
            
            chamfers_pred.append(mean_cham_pred)
            chamfers_pred_99.append(mean_cham_pred_99)
            chamfers_pred_97.append(mean_cham_pred_97)
            chamfers_pred_95.append(mean_cham_pred_95)
            
            chamfers_gt.append(mean_cham_gt)
            chamfers_gt_99.append(mean_cham_gt_99)
            chamfers_gt_97.append(mean_cham_gt_97)
            chamfers_gt_95.append(mean_cham_gt_95)
            
            chamfers_all.append(mean_cham_pred + mean_cham_gt)
            chamfers_all_99.append(mean_cham_pred_99 + mean_cham_gt_99)
            chamfers_all_97.append(mean_cham_pred_97 + mean_cham_gt_97)
            chamfers_all_95.append(mean_cham_pred_95 + mean_cham_gt_95)
            LOG.debug(f"{frame_ind}: cham_pred={mean_cham_pred}, cham_gt={mean_cham_gt}")

    avg_depth_error = sum(depth_errors) / len(depth_errors)
    avg_depth_error_99 = sum(depth_errors_99) / len(depth_errors_99)
    avg_depth_error_97 = sum(depth_errors_97) / len(depth_errors_97)
    avg_depth_error_95 = sum(depth_errors_95) / len(depth_errors_95)
    
    avg_chamfer_pred = sum(chamfers_pred) / len(chamfers_pred)
    avg_chamfer_pred_99 = sum(chamfers_pred_99) / len(chamfers_pred_99)
    avg_chamfer_pred_97 = sum(chamfers_pred_97) / len(chamfers_pred_97)
    avg_chamfer_pred_95 = sum(chamfers_pred_95) / len(chamfers_pred_95)
    
    avg_chamfer_gt = sum(chamfers_gt) / len(chamfers_gt)
    avg_chamfer_gt_99 = sum(chamfers_gt_99) / len(chamfers_gt_99)
    avg_chamfer_gt_97 = sum(chamfers_gt_97) / len(chamfers_gt_97)
    avg_chamfer_gt_95 = sum(chamfers_gt_95) / len(chamfers_gt_95)
    
    avg_chamfer = sum(chamfers_all) / len(chamfers_all)
    avg_chamfer_99 = sum(chamfers_all_99) / len(chamfers_all_99)
    avg_chamfer_97 = sum(chamfers_all_97) / len(chamfers_all_97)
    avg_chamfer_95 = sum(chamfers_all_95) / len(chamfers_all_95)
    print("Average chamfer distance:")
    print("Pred, Gt, Overall")
    print(f"{avg_chamfer_pred}, {avg_chamfer_gt}, {avg_chamfer}")
    stat_file = os.path.join(output_root, 'chamfer_dis_and_depth_err.txt')
    with open(stat_file, 'w') as f:
        f.write("cham_pred, cham_gt, chamfer, cham_pred_99, cham_gt_99, chamfer_99, cham_pred_97, cham_gt_97, chamfer_97, cham_pred_95, cham_gt_95, chamfer_95, depth, depth_99, depth_97, depth_95\n")
        # f.write(f"{avg_chamfer_pred}, {avg_chamfer_gt}, {avg_chamfer}, {avg_depth_error}\n")
        # f.write("=> Per frame chamfer_dis on pred, gt, all, and mean depth_err\n")
        for chamfer_pred, chamfer_gt, chamfer, chamfer_pred_99, chamfer_gt_99, chamfer_99, \
            chamfer_pred_97, chamfer_gt_97, chamfer_97, chamfer_pred_95, chamfer_gt_95, chamfer_95, \
            depth_err, depth_err_99, depth_err_97, depth_err_95 in \
                zip(chamfers_pred, chamfers_gt, chamfers_all, 
                chamfers_pred_99, chamfers_gt_99, chamfers_all_99, 
                chamfers_pred_97, chamfers_gt_97, chamfers_all_97, 
                chamfers_pred_95, chamfers_gt_95, chamfers_all_95, 
                depth_errors, depth_errors_99, depth_errors_97, depth_errors_95):
            f.write(f"{chamfer_pred:.5f}, {chamfer_gt:.5f}, {chamfer:.5f}, {chamfer_pred_99:.5f}, {chamfer_gt_99:.5f}, {chamfer_99:.5f}, "\
                f"{chamfer_pred_97:.5f}, {chamfer_gt_97:.5f}, {chamfer_97:.5f}, {chamfer_pred_95:.5f}, {chamfer_gt_95:.5f}, {chamfer_95:.5f}, "\
                f"{depth_err:.5f}, {depth_err_99:.5f}, {depth_err_97:.5f}, {depth_err_95:.5f}\n")
    print(f"=> Eval results saved to {stat_file}")

    #--------------- MISC
    misc = {'depth_error': avg_depth_error, 'chamfer_pred': avg_chamfer_pred, 'chamfer_gt': avg_chamfer_gt, 'chamfer': avg_chamfer, 
            'depth_error_99': avg_depth_error_99, 'chamfer_pred_99': avg_chamfer_pred_99, 'chamfer_gt_99': avg_chamfer_gt_99, 'chamfer_99': avg_chamfer_99, 
            'depth_error_97': avg_depth_error_97, 'chamfer_pred_97': avg_chamfer_pred_97, 'chamfer_gt_97': avg_chamfer_gt_97, 'chamfer_97': avg_chamfer_97, 
            'depth_error_95': avg_depth_error_95, 'chamfer_pred_95': avg_chamfer_pred_95, 'chamfer_gt_95': avg_chamfer_gt_95, 'chamfer_95': avg_chamfer_95, 
            }
    misc_f = os.path.join(output_root, f'{name}_misc.json')
    with open(misc_f, 'w') as f:
        json.dump(misc, f)
    print(f"MISC saved to {misc_f}")

    if args.video_backend is not None:
        def write_video(uri, frames, **kwargs):
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
        
        video_frames = []
        video_frames_err = []
        for i in range(len(pcl_imgs["gt"])):
            video_frames.append(np.concatenate([
                np.concatenate([pcl_imgs[key][i][0] for key in ["gt", "pred"]], axis=1),
                np.concatenate([pcl_imgs[key][i][0] for key in ["gt_err_chamfer", "pred_err_chamfer"]], axis=1),
            ], axis=0))
            video_frames_err.append(
                np.concatenate([pcl_imgs[key][i][0] for key in ["pred_err_chamfer", "pred_err_depth"]], axis=1)
            )
        write_video(os.path.join(output_root, f"{name}_{args.lidar_id}_topdown.mp4"), video_frames)
        write_video(os.path.join(output_root, f"{name}_{args.lidar_id}_topdown_err.mp4"), video_frames_err)
        for key in pcl_imgs:
            if key == "demo": 
                continue
            video_frames = [item[0] for item in pcl_imgs[key]]
            write_video(os.path.join(output_root, f"{name}_{args.lidar_id}_topdown_{key}.mp4"), video_frames)

        video_frames = []
        video_frames_err = []
        for i in range(len(pcl_imgs["gt"])):
            video_frames.append(np.concatenate([
                np.concatenate([pcl_imgs[key][i][1] for key in ["gt", "pred"]], axis=1),
                np.concatenate([pcl_imgs[key][i][1] for key in ["gt_err_chamfer", "pred_err_chamfer"]], axis=1),
            ], axis=0))
            video_frames_err.append(
                np.concatenate([pcl_imgs[key][i][1] for key in ["pred_err_chamfer", "pred_err_depth"]], axis=1)
            )
        write_video(os.path.join(output_root, f"{name}_{args.lidar_id}_front.mp4"), video_frames)
        write_video(os.path.join(output_root, f"{name}_{args.lidar_id}_front_err.mp4"), video_frames_err)
        for key in pcl_imgs:
            if key == "demo": 
                continue
            video_frames = [item[1] for item in pcl_imgs[key]]
            write_video(os.path.join(output_root, f"{name}_{args.lidar_id}_front_{key}.mp4"), video_frames)

        if 'demo' in pcl_imgs.keys():
            write_video(os.path.join(output_root, f"{name}_{args.lidar_id}_demo.mp4"), pcl_imgs['demo'])

def make_parser():
    bc = BaseConfig()
    
    bc.parser.add_argument("--load_pt", type=str, default=None, 
                           help="Typically unnecessary as the final or latest ckpt is loaded automatically. \n"\
                           "Only specify the ckpt file path if indeed a non-final or non-latest ckpt needs to be loaded.")
    bc.parser.add_argument("--rayschunk", type=int, default=65536, 
                           help="Chunkify the rendering process.")
    # NeuS specific hacks
    bc.parser.add_argument("--forward_inv_s", type=str, default="64000.", 
                           help="Bypasses the inv_s parameter for NeuS during rendering.")

    # Basic arguments
    bc.parser.add_argument("--outbase", type=str, default=datetime.now().strftime("%Y_%m_%d_%H_%M_%S"), help="Sets the basename of the output file (without extension).")
    bc.parser.add_argument("--dirname", type=str, default='eval_lidar', help="Sets the output directory to /path/to/exp_dir/${dirname}`.")
    bc.parser.add_argument("--progress", action='store_true', help="If set, shows per frame progress.")

    bc.parser.add_argument('--start_frame', type=int, default=0)
    bc.parser.add_argument("--num_frames", type=int, default=None)
    bc.parser.add_argument('--stop_frame', type=int, default=None)
    
    # LiDAR evaluation arguments
    bc.parser.add_argument("--lidar_id", type=str, required=True, help="Specifies the lidar name")
    bc.parser.add_argument("--lidar_far", type=float, default=120.0)
    bc.parser.add_argument("--no_filter_in_scene_space", action='store_true', 
        help="By default (if not set), will use lidar gt points falling only in the scene AABB during evaluation.")
    bc.parser.add_argument("--cd_filter_by_pred_mask", action='store_true', 
        help="If set, GT rays whose prediction don't hit will not be filtered, \n"
        "thus the hit or miss also contributes to the accuracy metric.\n"
        "This accuracy can be improved with ray-drop probability prediction.")
    bc.parser.add_argument("--filter_toofar", type=float, default=None, help="Optionally sets a far limit for evaluation")
    bc.parser.add_argument("--bypass_filter_in_cam", type=bool, default=None, help="Optionally bypasses `filter_in_cam` option set during training.")
    bc.parser.add_argument("--bypass_filter_objs_dynamic", type=bool, default=None, help="Optinally bypasses `filter_objs_dynamic` option set during training. ")
    bc.parser.add_argument("--errormap_chamfer_max", type=float, default=1.0, help="Sets the maximum value for visualizing CD")
    bc.parser.add_argument("--errormap_depth_max", type=float, default=5.0, help="Sets the maximum value for visualizing depth error")
    
    # [Optional] Video arguments
    bc.parser.add_argument("--video_backend", type=str, default=None, 
                           help="Specifies the visualization backend. Options: [mayavi, vedo]. "\
                           "Setting this will output an optional visualization video for the GT and rendered point clouds. "\
                           "If not set, no video will be produced.")
    bc.parser.add_argument("--fps", type=int, default=24)
    bc.parser.add_argument("--quality", type=int, default=None, help="Sets the quality for imageio.mimwrite (range: 0-10; 10 is the highest; default is 5).")
    bc.parser.add_argument("--cam_ref", type=str, default='camera_FRONT', help="Reference camera for visulization of LiDAR, mesh, etc.")
    bc.parser.add_argument("--video_bg", type=str, default='white', help="Defines the background color of visualization.")
    bc.parser.add_argument("--video_verbose", action='store_true', help="If set, a visualization window will pop up during the rendering process.")
    bc.parser.add_argument("--video_width", type=int, default=1200, help="Sets the width of lidar visualization viewport (in pixels).")
    bc.parser.add_argument('--video_label', action="store_true", help="If set, labels will be drawn on the output video")
    
    return bc


if __name__ == "__main__":
    bc = make_parser()
    main_function(bc.parse())