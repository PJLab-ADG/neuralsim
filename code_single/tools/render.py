"""
@file   render.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Render (replay or nvs) a single static object / street / room etc. scene.
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
from math import prod
from tqdm import tqdm
from glob import glob
from icecream import ic
from copy import deepcopy
from datetime import datetime

import torch

from nr3d_lib.fmt import log
from nr3d_lib.plot import color_depth, scene_flow_to_rgb
from nr3d_lib.checkpoint import sorted_ckpts
from nr3d_lib.config import ConfigDict, BaseConfig
from nr3d_lib.utils import IDListedDict, cond_mkdir, import_str, cpu_resize, pad_images_to_same_size

from nr3d_lib.models.spatial import AABBSpace
from nr3d_lib.models.accelerations.occgrid_accel import OccGridAccel
from nr3d_lib.models.attributes.transform import TransformMat4x4

from app.renderers import SingleVolumeRenderer
from app.resources.observers import Lidar, RaysLidar, Camera
from app.resources import Scene, SceneNode, load_scene_bank, AssetBank, get_dataset_scenario

from dataio.scene_dataset import SceneDataset
from dataio.data_loader import SceneDataLoader

def main_function(args: ConfigDict):
    exp_dir = args.exp_dir
    device = torch.device('cuda')

    #---------------------------------------------
    #--------------     Load     -----------------
    #---------------------------------------------
    device = torch.device('cuda')
    if (ckpt_file:=args.get('load_pt', None)) is None:
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
    asset_bank.create_asset_bank(scene_bank, load_state_dict=state_dict['asset_bank'], device=device)
    # log.info(asset_bank)

    #---------------------------------------------
    #---     Load assets to scene objects     ----
    #---------------------------------------------
    # for scene in scene_bank:
    scene = scene_bank[0]
    scene.load_assets(asset_bank)
    # !!! Only call training_before_per_step when all assets are ready & loaded !
    asset_bank.training_before_per_step(args.training.num_iters) # NOTE: Finished training.

    # Fallsback to regular Mat4x4 for convenience
    # TODO

    #---------------------------------------------
    #------     Scene Bank Dataset     -----------
    #---------------------------------------------
    dataset_impl: SceneDataset = import_str(args.dataset_cfg.target)(args.dataset_cfg.param)
    if args.no_gt:
        args.training.dataloader.tags = {}
    args.training.dataloader.preload = False
    args.training.val_dataloader.preload = False
    scene_dataloader = SceneDataLoader(scene_bank, dataset_impl=dataset_impl, config=args.training.val_dataloader, device=device)
    scene_dataloader.set_camera_downscale(args.downscale)
    cam_id_list = scene_dataloader.cam_id_list if args.cam_id is None else [args.cam_id]
    cam_ref_id = cam_id_list[0] if len(cam_id_list) == 1 else args.cam_ref
    
    #---------------------------------------------
    #------------     Renderer     ---------------
    #---------------------------------------------
    renderer = SingleVolumeRenderer(args.renderer)
    renderer.populate(asset_bank)
    renderer.eval()
    asset_bank.eval()
    renderer.config.rayschunk = args.rayschunk
    with_normal = renderer.config.get('with_normal', False)
    with_flow = renderer.config.get('with_flow', False)
    with_static_dynamic = renderer.config.setdefault('with_static_dynamic', False)
    
    if args.depth_max is None:
        depth_max = renderer.config.far
    else:
        depth_max = args.depth_max
    assert depth_max is not None, "You need to specify at least one of the args.depth_max or renderer.config.far"
    for scene in scene_bank:
        # NOTE: When training, set all observer's near&far to a larger value
        for obs in scene.get_observers(False):
            obs.near = renderer.config.near
            obs.far = renderer.config.far

    if args.forward_inv_s is None or args.forward_inv_s.lower() == 'none' or args.forward_inv_s.lower() == 'null':
        forward_inv_s = None
    else:
        forward_inv_s = float(args.forward_inv_s)
    if args.lidar_forward_inv_s is None or args.lidar_forward_inv_s.lower() == 'none' or args.lidar_forward_inv_s.lower() == 'null':
        lidar_forward_inv_s = None
    else:
        lidar_forward_inv_s = float(args.lidar_forward_inv_s)

    #---------------------------------------------
    #--------------     Plot     -----------------
    #---------------------------------------------
    
    expname = os.path.split(args.exp_dir.rstrip("/"))[-1]
    global_step = state_dict['global_step']
    global_step_str = f"iter{global_step/1000}k" if global_step >= 1000 else f"iter{global_step}"
    name = f"{expname[0:64]}_{global_step_str}_{args.outbase}_ds={args.downscale}"
    if forward_inv_s is not None:
        name += f"_invs={int(forward_inv_s)}"
    if args.zoom_focal_scale is not None:
        name += f'_zoom_focal_scale={args.zoom_focal_scale}'
    
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
                    uri = f"{os.path.splitext(uri)[0]}.jpg"
                imageio.imwrite(uri, frames[0], **kwargs)
                log.info(f"Image saved to {uri}")

    # Original / reference frame_ind
    if args.num_frames is not None:
        args.stop_frame = args.start_frame + args.num_frames
    else:
        if args.stop_frame is None:
            args.stop_frame = len(scene)
    log.info(f"=> args.stop_frame is set to {args.stop_frame}")
    frame_ind_list = ref_frame_ind_list = np.arange(args.start_frame, args.stop_frame, 1).tolist()
    
    # The actual frame_ind used in rendering (for replay or nvs)
    if args.nvs_path is None:
        current_mode = 'replay'
        num_frames = len(frame_ind_list)
    else:
        assert args.nvs_num_frames is not None, "nvs_num_frames is required"
        num_frames = args.nvs_num_frames
        frame_ind_list = np.arange(num_frames).tolist()
        current_mode = 'nvs'

    if not args.no_gt:
        # Test correct.
        ground_truth = scene_dataloader.get_image_and_gts(scene.id, cam_id_list[0], 0)


    #---------------------------------------------
    #   Parepare and start rendering !
    #---------------------------------------------
    with torch.no_grad():
        # for scene in scene_bank:
        scene: Scene = scene_bank[0]
        scene.slice_at(frame_ind_list[0])
        obj = scene.get_drawable_groups_by_class_name(scene.main_class_name)[0]
        model = obj.model

        focals0 = {}
        for cam_id in cam_id_list:
            cam = scene.observers[cam_id]
            focals0[cam_id] = cam.intr.focal().clone()

        #---------------------------------------------
        #     [Optional] Prepare experimental fast render
        #---------------------------------------------
        if args.fast_render:
            if isinstance(model.space, AABBSpace):
                resolution = 128
                num_step_per_occvoxel = 8
                march_step_size = (model.space.radius3d.norm(dim=-1).item() / resolution / num_step_per_occvoxel) * obj.scale.vec_3().max()
                ic(march_step_size)
                accel = OccGridAccel(
                    model.space, 
                    resolution=128, occ_val_fn_cfg=ConfigDict(type='sdf', inv_s=256.0), occ_thre=0.8, 
                    init_cfg=ConfigDict(num_steps=128, num_pts=2**24, mode='from_net'))
                # NOTE: `step_size` in world space.
                model.accel = accel
                # Run init from net
                model.accel.init(model.query_sdf)
                forward_inv_s = 6400.0
                model.ray_query_cfg = ConfigDict(
                    forward_inv_s=6400., 
                    query_mode='march_occ_multi_upsample_compressed', 
                    query_param=ConfigDict(
                        nablas_has_grad=False, num_coarse=0, 
                        march_cfg=ConfigDict(
                            step_size=march_step_size, max_steps=2*resolution*num_step_per_occvoxel), 
                        num_fine=4, upsample_inv_s=64.0, upsample_inv_s_factors=[4, 16]
                    )
                )
            else:
                raise RuntimeError(f"Unsupported space type={type(model.space)}")

        #---------------------------------------------
        #     [Optional] Prepare gathering camera pointclouds     
        #---------------------------------------------
        if args.gather_cam_pcl:
            from nr3d_lib.graphics.pointcloud import export_pcl_ply
            cam_pcl = []
            cam_pcl_color = []

        #---------------------------------------------
        #     [Optional] Prepare lidar render     
        #---------------------------------------------
        if args.render_lidar:
            import vedo
            
            if (args.lidar_vis_rgb_choice == 'ins_seg'):
                raise NotImplementedError("ins seg is only available in multi-object rendering.")
            
            assert cam_ref_id is not None and cam_ref_id in scene.observers, \
                f"A valid frontal reference camera is required; current camera list={scene.get_observer_groups_by_class_name('Camera', False)}"
            
            cam0: Camera = scene.observers[cam_ref_id]
            
            if args.lidar_model == 'original' or args.lidar_model == 'original_reren':
                lidar: RaysLidar = scene.observers[args.lidar_id]
                lidar.far = args.lidar_far
            else:
                # Create new lidar to be simulated and make it a child of cam0
                lidar = Lidar('sim_lidar', lidar_model=args.lidar_model, lidar_name=args.lidar_id, scene=scene).to(device=device)
                scene.add_node(lidar, parent=cam0)

            cam0.intr.set_downscale(args.downscale)
            if not args.no_cam:
                W_lidar_vis = min(cam0.intr.W * 4, args.lidar_vis_width) # 4 is the number of column when joint render
            else:
                W_lidar_vis = args.lidar_vis_width
            H_lidar_vis = W_lidar_vis*9//12
            
            pcl_imgs = []
            # NOTE: top view, front view, slope (demo) view
            plt_lidar = vedo.Plotter(
                interactive=False, offscreen=not args.lidar_vis_verbose, 
                sharecam=False, resetcam=False, 
                shape=(1,3), size=[W_lidar_vis * 3, H_lidar_vis], bg='white')
            # ttf_path = "/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf"
            # ttf_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf" # sudo apt install msttcorefonts -qq
            
            if args.save_perframe_lidar:
                pcl_save_dir = os.path.join(vid_root, 'pcls')
                cond_mkdir(pcl_save_dir)
            
            def render_pcl(scene: Scene, frame_ind: int) -> np.ndarray:
                with torch.no_grad():
                    cam0: Camera = scene.observers[cam_ref_id]
                    
                    if args.lidar_model == 'original' or args.lidar_model == 'original_reren':
                        lidar_gt = scene_dataloader.get_lidar_gts(scene.id, args.lidar_id, frame_ind, device=device)
                        lidar_rays_o_local, lidar_rays_d_local = lidar_gt['rays_o'], lidar_gt['rays_d']
                        if args.lidar_model == 'original':
                            lidar_ranges_gt = lidar_gt['ranges']
                            valid = lidar_ranges_gt > 0
                            #-------- outputs
                            # Lidar ranges
                            lidar_ranges = lidar_ranges_gt[valid]
                            # Lidar pts in local
                            lidar_pts_local = lidar_rays_o_local[valid] + lidar_rays_d_local[valid] * lidar_ranges_gt[valid].unsqueeze(-1)
                            # Lidar pts in world
                            lidar_pts = lidar.world_transform(lidar_pts_local)
                            # Dummy all occupied mask just for compatibility.
                            lidar_mask = lidar_pts.new_ones([*lidar_pts.shape[:-1]])
                            assert args.lidar_vis_rgb_choice in ['height', 'grey'], f"lidar_model=original does not support lidar_vis_rgb_choice=={args.lidar_vis_rgb_choice}"
                            
                        elif args.lidar_model == 'original_reren':
                            # Lidar rays in world
                            lidar_rays_o, lidar_rays_d = lidar.get_selected_rays(rays_o=lidar_rays_o_local, rays_d=lidar_rays_d_local)
                            
                            #-----------------------------------------------
                            ret_lidar = renderer.render(
                                scene, rays=(lidar_rays_o,lidar_rays_d), near=lidar.near, far=lidar.far, 
                                only_cr=True, with_normal=False, 
                                with_rgb=(args.lidar_vis_rgb_choice == 'appearance'), 
                                render_per_obj_individual=(args.lidar_vis_rgb_choice == 'ins_seg'), 
                                rayschunk=args.rayschunk, show_progress=args.progress, 
                                bypass_ray_query_cfg=ConfigDict({obj.class_name: {'forward_inv_s': lidar_forward_inv_s}}))  
                            lidar_rays_acc = ret_lidar['rendered']['mask_volume']
                            lidar_rays_depth = ret_lidar['rendered']['depth_volume']
                            valid = lidar_rays_acc > 0.95
                            
                            #-----------------------------------------------
                            #-------- outputs
                            lidar_ranges = lidar_rays_depth[valid]
                            # Lidar pts in world
                            lidar_pts = lidar_rays_o[valid] + lidar_rays_d[valid] * lidar_ranges.unsqueeze(-1)
                            # Lidar pts in local
                            lidar_pts_local = lidar_rays_o_local[valid] + lidar_rays_d_local[valid] * lidar_ranges.unsqueeze(-1)
                            lidar_mask = lidar_rays_acc[valid]
                            if (args.lidar_vis_rgb_choice == 'appearance'):
                                lidar_rgb = ret_lidar['rendered']['rgb_volume'][valid]
                            elif (args.lidar_vis_rgb_choice == 'ins_seg'):
                                lidar_ins_seg_id = ret_lidar['ins_seg_mask_buffer'][valid]
                    else:
                        # lidar.transform = cam0.world_transform
                        # lidar.world_transform = lidar.transform
                        # TODO: Here we should use `lidar`, not `camera_0`! 
                        #   We need to make `rendering_before_per_view` more adaptable to accommodate lidar sensor parameters.
                        asset_bank.rendering_before_per_view(renderer=renderer, observer=cam0, scene_id=scene.id)
                        # Lidar rays in world
                        lidar_rays_o, lidar_rays_d, lidar_rays_ts = lidar.get_all_rays(return_ts=True)
                        
                        #-----------------------------------------------
                        ret_lidar = renderer.render(
                            scene, rays=(lidar_rays_o,lidar_rays_d,lidar_rays_ts), near=lidar.near, far=lidar.far, 
                            only_cr=True, with_normal=False, 
                            with_rgb=(args.lidar_vis_rgb_choice == 'appearance'), 
                            render_per_obj_individual=(args.lidar_vis_rgb_choice == 'ins_seg'), 
                            rayschunk=args.rayschunk, show_progress=args.progress, 
                            bypass_ray_query_cfg=ConfigDict({obj.class_name: {'forward_inv_s': lidar_forward_inv_s}}))  
                        lidar_rays_acc = ret_lidar['rendered']['mask_volume']
                        lidar_rays_depth = ret_lidar['rendered']['depth_volume']
                        valid = lidar_rays_acc > 0.95
                        #-----------------------------------------------
                        
                        #-------- outputs
                        lidar_ranges = lidar_rays_depth[valid]
                        # Lidar pts in world
                        lidar_pts = lidar_rays_o[valid] + lidar_rays_d[valid] * lidar_rays_depth[valid].unsqueeze(-1)
                        # Lidar pts in local
                        lidar_pts_local = lidar.world_transform(lidar_pts, inv=True)
                        lidar_mask = lidar_rays_acc[valid]
                        if (args.lidar_vis_rgb_choice == 'appearance'):
                            lidar_rgb = ret_lidar['rendered']['rgb_volume'][valid]
                        elif (args.lidar_vis_rgb_choice == 'ins_seg'):
                            lidar_ins_seg_id = ret_lidar['ins_seg_mask_buffer'][valid]

                    if args.save_perframe_lidar:
                        lidar_pts_datas = [lidar_pts_local]
                        if (args.lidar_vis_rgb_choice == 'appearance'):
                            lidar_pts_datas.append(lidar_rgb)
                        else:
                            lidar_pts_datas.append(lidar_mask.unsqueeze(-1))
                        lidar_pts_datas = torch.cat(lidar_pts_datas, dim=-1).contiguous().data.cpu().numpy()
                        np.save(os.path.join(pcl_save_dir, f"{frame_ind:08d}.npy"), lidar_pts_datas)

                    
                    # NOTE: Convert to a common coordinate system (OpenCV pinhole camera in this case)
                    lidar_pts_np = cam0.world_transform(lidar_pts, inv=True).data.cpu().numpy()
                    lidar_mask_np = lidar_mask.data.cpu().numpy()
                    # lidar_depth_np = lidar_mask_np * np.clip(lidar_ranges.data.cpu().numpy() / depth_max, 0, 1) + (1-lidar_mask_np) * 1
                    
                    if args.lidar_vis_rgb_choice == 'appearance':
                        #---- Use appearance radiance to colorize LiDAR pcl
                        lidar_vis_rgb = (lidar_rgb.data.cpu().numpy()*255.).clip(0,255).astype(np.uint8)
                    elif args.lidar_vis_rgb_choice == 'ins_seg':
                        #---- Use beam intersections to colorize LiDAR pcl
                        raise NotImplementedError("ins seg is only available in multi-object rendering.")
                    elif args.lidar_vis_rgb_choice == 'height':
                        #---- Use `-y` ("height" in openCV camera) to colorize LiDAR pcl
                        lidar_vis_rgb = (vedo.color_map(-lidar_pts_np[...,1], 'rainbow', vmin=args.lidar_vis_vmin, vmax=args.lidar_vis_vmax) * 255.).clip(0,255).astype(np.uint8)
                    else:
                        raise RuntimeError(f"Invalid args.lidar_vis_rgb_choice={args.lidar_vis_rgb_choice}")
                    
                    lidar_vis_alpha = (lidar_mask_np*255).clip(0,255).astype(np.uint8)
                    lidar_vis_rgba = np.concatenate([lidar_vis_rgb, lidar_vis_alpha[..., None]], axis=-1) # RGBA is ~50x faster
                    pcl_pred = vedo.Points(lidar_pts_np, c=lidar_vis_rgba, r=args.lidar_vis_radius)
                    
                    plt_lidar.clear(at=0, deep=True)
                    plt_lidar.clear(at=1, deep=True)
                    plt_lidar.clear(at=2, deep=True)
                    # Top view
                    plt_lidar.show(pcl_pred, at=0, resetcam=False, camera={'focal_point': [0., 0., 15.], 'pos': [0., -100., 15.], 'viewup': [-1,0,0]})
                    # Front view
                    plt_lidar.show(pcl_pred, at=1, resetcam=False, camera={'focal_point': [0., 0., 50.], 'pos': [0., -5, -19.82120022], 'viewup': [ 0., -0.99744572, 0.07142857]})
                    # Slope (demo) view
                    plt_lidar.show(pcl_pred, at=2, resetcam=False, camera={'focal_point': [-6.1,6.2,34.1], 'pos': [18.0,-22.1,-26.7], 'viewup': [0.006176,-0.905337,0.424648]})
                    
                    pcl_im = plt_lidar.topicture().tonumpy()
                    return pcl_im

        #---------------------------------------------
        #     [Optional] Prepare mesh render
        #---------------------------------------------
        should_collect_mesh_imgs = args.render_mesh and not args.render_mesh_verbose
        if args.render_mesh:
            import open3d as o3d
            
            assert cam_ref_id is not None and cam_ref_id in scene.observers, \
                f"A valid frontal reference camera is required.\nCurrent camera list={scene.get_observer_groups_by_class_name('Camera', False)}"
            
            cam0: Camera = scene.observers[cam_ref_id]
            cam0.intr.set_downscale(args.downscale)
            
            if args.render_mesh == 'local':
                args.render_mesh = glob(os.path.join(exp_dir, 'meshes', '*.ply'))[-1]
            log.info(f"=> Load mesh: {args.render_mesh}")
            read_geometry = o3d.io.read_triangle_mesh(args.render_mesh)
            # From obj's coordinates to world coordinates
            if args.render_mesh_transform == 'to_world':
                geometry = deepcopy(read_geometry).transform(obj.world_transform.mat_4x4().data.cpu().numpy())
            elif args.render_mesh_transform == 'identity':
                geometry = read_geometry
            else:
                raise RuntimeError(f"Invalid render_mesh_transform={args.render_mesh_transform}")
            geometry.compute_vertex_normals()
            o3d_W, o3d_H = cam0.intr.W, cam0.intr.H
            vis = o3d.visualization.Visualizer()
            vis.create_window(width=o3d_W, height=o3d_H, visible=args.render_mesh_verbose)
            ctrl = vis.get_view_control()
            vis.add_geometry(geometry)
            # opt = vis.get_render_option()
            # opt.mesh_show_back_face = True
            cam_model = ctrl.convert_to_pinhole_camera_parameters()
            
            def render_mesh(cam: Camera):
                intr = cam.intr.mat_3x3().data.cpu().numpy()
                W, H = cam.intr.W, cam.intr.H
                # cam.intrinsic.set_intrinsics(W, H, intr[0,0], intr[1,1], intr[0,2], intr[1,2])
                cam_model.intrinsic.set_intrinsics(W, H, intr[0,0], intr[1,1], W/2-0.5, H/2-0.5)
                ctrl.convert_from_pinhole_camera_parameters(cam_model)
                
                extr = np.linalg.inv(cam.world_transform.mat_4x4().data.cpu().numpy())
                cam_model.extrinsic = extr
                ctrl.convert_from_pinhole_camera_parameters(cam_model)
                vis.poll_events()
                vis.update_renderer()
                if should_collect_mesh_imgs:
                    rgb_mesh = vis.capture_screen_float_buffer(do_render=True)
                    return (np.asarray(rgb_mesh) * 255.).clip(0, 255).astype(np.uint8)

        #---------------------------------------------
        #     [Optional] Prepare nvs
        #---------------------------------------------
        if args.nvs_path is not None:
            args.no_gt = True # No GT comparison since we are rendering novel view
            assert args.nvs_node_id is not None, "`--nvs_node_id=` is required to manipulate the (ego node's) pose.\n"\
                "For single objects, 'camera' is typically used. Please refer to the 'id' field of the camera in xxx_dataset.py.\n"\
                "For street views, the typical value varies: 'camera_FRONT' for Waymo, ..."
            scene.slice_at(ref_frame_ind_list)
            # if args.nvs_should_recollect:
            #     assert args.nvs_node_id not in scene.all_nodes, f"The target nvs_node_id={args.nvs_node_id} is already in the scene graph, no need to re-collect"
            #     nvs_node = SceneNode(args.nvs_node_id, scene=scene, device=device, dtype=torch.float)
            #     scenebank_cfg = args.scenebank_cfg.deepcopy()
            #     scenebank_cfg.update(scenebank_cfg.pop('on_load', {}))
            #     # Only for waymo
            #     scenebank_cfg.scene_graph_has_ego_car = True
            #     scenebank_cfg.correct_extr_for_timestamp_difference = True
            #     data_scenario = get_dataset_scenario(dataset_impl, scene.id, args.scenebank_cfg)
            # else:
            
            nvs_node = scene.all_nodes[args.nvs_node_id]
            pose_ref = nvs_node.world_transform.mat_4x4().data.cpu().numpy()
            
            if args.nvs_path == 'spherical_spiral':
                from nr3d_lib.graphics.cameras import get_path_spherical_spiral
                view_ids = args.nvs_param.split(',')
                assert len(view_ids) == 3, 'please select three view indices on a small circle, in CCW order (looking from above)'
                view_ids = [int(v) for v in view_ids]
                centers = pose_ref[view_ids, :3, 3]
                render_pose_all = get_path_spherical_spiral(
                    centers, num_frames, n_rots=2.2, up_angle=np.pi/3, 
                    # verbose kwargs 
                    verbose=args.nvs_verbose, intrs=nvs_node.intr.mat_3x3().data.cpu().numpy(), 
                    H=nvs_node.intr.H, W=nvs_node.intr.W, cam_size=0.05, font_size=12)
            elif args.nvs_path == "small_circle":
                from nr3d_lib.graphics.cameras import get_path_small_circle
                view_ids = args.nvs_param.split(',')
                assert len(view_ids) == 3, 'please select three view indices on a small circle, in CCW order (looking from above)'
                view_ids = [int(v) for v in view_ids]
                centers = pose_ref[view_ids, :3, 3]
                render_pose_all = get_path_small_circle(
                    centers, num_frames, 
                    # verbose kwargs 
                    verbose=args.nvs_verbose, intrs=nvs_node.intr.mat_3x3().data.cpu().numpy(), 
                    H=nvs_node.intr.H, W=nvs_node.intr.W, cam_size=0.2, font_size=6
                )
            elif args.nvs_path == "interpolation":
                from nr3d_lib.graphics.cameras import get_path_interpolation
                render_pose_all = get_path_interpolation(pose_ref, num_frames)
            elif args.nvs_path == "street_view":
                from nr3d_lib.graphics.cameras import get_path_front_left_lift_then_spiral_forward
                kwargs = dict(pose_ref=pose_ref, num_frames=num_frames)
                kwargs.update(forward_vec=dataset_impl.forward_vec, up_vec=dataset_impl.up_vec, left_vec=-1*dataset_impl.right_vec)
                if args.nvs_param is not None:
                    # 2.0, 1.0, 
                    duration, elongation, up_max, up_min, left_max, left_min = [float(i) for i in args.nvs_param.split(',')]
                    kwargs.update(
                        duration_frames=int(duration * args.fps), elongation=elongation, 
                        up_max=up_max, up_min=up_min, left_max=left_max, left_min=left_min)
                render_pose_all = get_path_front_left_lift_then_spiral_forward(**kwargs)
            else:
                raise RuntimeError(f"Invalid nvs_path={args.nvs_path}")
            scene.unfrozen()
            scene.slice_at(args.start_frame) # Re-freeze at start reference frame 

        #---------------------------------------------
        #     Prepare results gathering
        #---------------------------------------------
        collate_keys = []
        if not args.no_gt:
            collate_keys.extend(['rgb_gt'])
        collate_keys.extend(['rgb_volume', 'depth_volume'])
        if with_normal:
            collate_keys.extend(['normals_volume'])
        if with_static_dynamic:
            if not with_normal:
                keys_static_dynamic = ['mask_static', 'rgb_static', 
                                       'mask_dynamic', 'rgb_dynamic', ]
            else:
                keys_static_dynamic = ['mask_static', 'rgb_static', 'normals_static', 
                                       'mask_dynamic', 'rgb_dynamic', 'normals_dynamic']
            for k in keys_static_dynamic:
                collate_keys.append(k)
        if with_flow:
            collate_keys.extend(['flow_fwd', 'flow_bwd'])
        if should_collect_mesh_imgs:
            collate_keys.extend(['mesh'])
        
        all_gather = dict()
        for cam_id in cam_id_list:
            all_gather[cam_id] = dict({k: [] for k in collate_keys})

        #---------------------------------------------
        #--------     Start rendering !!!    ---------
        #---------------------------------------------
        log.info(f"Start [{current_mode}], ds={args.downscale}, in {exp_dir}")
        for frame_ind in tqdm(frame_ind_list, "rendering frames..."):
            if args.nvs_path is None:
                # [replay]
                scene.slice_at(frame_ind)
            else:
                # [nvs]
                render_pose = render_pose_all[frame_ind]
                nvs_node.world_transform = TransformMat4x4(render_pose, device=device, dtype=torch.float)
                nvs_node.update_children()
            
            if args.render_lidar:
                im = render_pcl(scene, frame_ind)
                pcl_imgs.append(im)

            if args.no_cam:
                continue

            for cam_id in cam_id_list:
                cur_frame_dict = dict({k: [] for k in collate_keys})
                
                cam: Camera = scene.observers[cam_id]
                cam.intr.set_downscale(args.downscale)
                if args.zoom_focal_scale is not None:
                    new_focal = focals0[cam_id] / args.zoom_focal_scale
                    cam.intr.set_focal(*new_focal)
                    if not args.no_distortion and args.ultra_wide_angle:
                        # A dummy distortion param for 3.0 zoom_focal_scale's ultra wide-angle
                        cam.intr.subattr.distortion.tensor[..., 0] = -0.129
                        cam.intr.subattr.distortion.tensor[..., 1] = 0.0108
                        cam.intr.subattr.distortion.tensor[..., 4] = -0.00033
                if args.no_distortion:
                    cam.intr.subattr.distortion.tensor.zero_()
                
                if args.render_mesh:
                    im_mesh = render_mesh(cam)
                    if should_collect_mesh_imgs:
                        cur_frame_dict['mesh'] = im_mesh
                
                #-----------------------------------------------
                ret = renderer.render(scene, observer=cam, show_progress=args.progress, with_env=not args.no_sky, 
                                      render_per_obj_individual=True, render_per_obj_in_scene=True, only_cr=args.only_cr, 
                                      bypass_ray_query_cfg=ConfigDict({obj.class_name: {'forward_inv_s': forward_inv_s}}))
                rendered = ret['rendered']
                main_rendered = ret['rendered_per_obj'][obj.id]
                main_rendered_in_scene = ret['rendered_per_obj_in_scene'][obj.id]
                #-----------------------------------------------
                
                if args.gather_cam_pcl:
                    # Rays in world coordinates
                    rays_o, rays_d = ret['rays_o'], ret['rays_d']
                    mask = rendered['mask_volume'] > 0.999
                    # Pts in world coordinates
                    pts = torch.addcmul(rays_o[mask], rays_d[mask], rendered['depth_volume'][mask].unsqueeze(-1))
                    pts_color = rendered['rgb_volume'][mask]
                    cam_pcl.append(pts)
                    cam_pcl_color.append(pts_color)
                
                def to_img(tensor) -> np.ndarray:
                    return tensor.reshape([cam.intr.H, cam.intr.W, -1]).data.cpu().numpy()

                #-----------------------------------------------
                # GT
                if not args.no_gt:
                    ground_truth = scene_dataloader.get_image_and_gts(scene.id, cam.id, frame_ind)
                    rgb_gt = (to_img(ground_truth['image_rgb'])*255).clip(0, 255).astype(np.uint8)
                    cur_frame_dict['rgb_gt'] = rgb_gt

                #-----------------------------------------------
                # RGB
                if 'rgb_volume' in rendered.keys():
                    rgb_volume = (to_img(rendered['rgb_volume']) * 255).clip(0, 255).astype(np.uint8)
                else:
                    rgb_volume = np.zeros([cam.intr.H, cam.intr.W, 3], dtype=np.uint8)
                cur_frame_dict['rgb_volume'] = rgb_volume
                
                #-----------------------------------------------
                # Mask and depth
                if not args.with_distant_depth:
                    # Since distant depth is usally messy and in-accurate
                    mask_volume = to_img(main_rendered_in_scene['mask_volume'])
                    depth_volume = to_img(main_rendered_in_scene['depth_volume'])
                else:
                    mask_volume = to_img(rendered['mask_volume'])
                    depth_volume = to_img(rendered['depth_volume'])
                depth_volume = mask_volume * np.clip(depth_volume/depth_max, 0, 1) + (1-mask_volume) * 1
                depth_volume = color_depth(depth_volume.squeeze(-1), scale=1, cmap='turbo')    # turbo_r, viridis, rainbow
                cur_frame_dict['depth_volume'] = depth_volume
                
                #-----------------------------------------------
                # Normals
                if with_normal:
                    if 'normals_volume' in cur_frame_dict:
                        assert 'normals_volume' in rendered
                        if not args.with_distant_normal:
                            normals_volume = to_img(main_rendered_in_scene['normals_volume'])
                        else:
                            normals_volume = to_img(rendered['normals_volume'])
                        normals_volume = ((normals_volume/2+0.5)*255).clip(0,255).astype(np.uint8)
                        cur_frame_dict['normals_volume'] = normals_volume

                #-----------------------------------------------
                # Static & Dynamic
                if with_static_dynamic:
                    for k in ['mask_static', 'mask_dynamic']:
                        assert k in main_rendered
                        cur_frame_dict[k] = (to_img(main_rendered[k].view(-1,1).expand(-1,3)) * 255).clip(0,255).astype(np.uint8)
                    for k in ['rgb_static', 'rgb_dynamic']:
                        assert k in main_rendered
                        cur_frame_dict[k] = (to_img(main_rendered[k]) * 255).clip(0,255).astype(np.uint8)
                    if with_normal:
                        for k in ['normals_static', 'normals_dynamic']:
                            assert k in main_rendered
                            cur_frame_dict[k] = (to_img(main_rendered[k]/2+0.5) * 255).clip(0,255).astype(np.uint8)

                #-----------------------------------------------
                # Flow
                if with_flow:
                    for k in ['flow_fwd', 'flow_fwd_pred_bwd', 'flow_bwd', 'flow_bwd_pred_fwd']:
                        if k in cur_frame_dict: 
                            assert k in main_rendered # main_rendered_in_scene
                            _im_flow = to_img(scene_flow_to_rgb(main_rendered[k], flow_max_radius=0.5))
                            _im_flow = (_im_flow * 255).clip(0, 255).astype(np.uint8)
                            cur_frame_dict[k] = _im_flow

                #---- NOTE: Ablation study on w/ and w/o distant-view model (run different exps)
                # debug_name = f'crdv_143481_withdistant_{frame_ind}'
                # os.makedirs(f"./dev_test/paper/{debug_name}", exist_ok=True)
                # rgb_street_in_total = to_img(ret['rendered_per_obj_in_scene']['street']['rgb_volume'])
                # alpha_street_in_total = to_img(ret['rendered_per_obj_in_scene']['street']['mask_volume'])
                # rgba_street_in_total = np.concatenate([rgb_street_in_total, alpha_street_in_total], -1)
                # imageio.imwrite(f'./dev_test/paper/{debug_name}/rgba_street_in_total.png', rgba_street_in_total)
                
                # rgb_sky_in_total = to_img(ret['rendered']['rgb_volume_non_occupied'])
                # alpha_sky_in_total = to_img(1-ret['rendered']['mask_volume'])
                # rgba_sky_in_total = np.concatenate([rgb_sky_in_total, alpha_sky_in_total], -1)
                # imageio.imwrite(f'./dev_test/paper/{debug_name}/rgb_sky_in_total.png', rgba_sky_in_total)
                
                # imageio.imwrite(f'./dev_test/paper/{debug_name}/depth.png', depth_volume)
                # imageio.imwrite(f'./dev_test/paper/{debug_name}/normal.png', normals_volume)
                
                # rgb_distant_in_total = to_img(ret['rendered_per_obj_in_scene']['distant']['rgb_volume'])
                # alpha_distant_in_total = to_img(ret['rendered_per_obj_in_scene']['distant']['mask_volume'])
                # rgba_distant_in_total = np.concatenate([rgb_distant_in_total, alpha_distant_in_total], -1)
                # imageio.imwrite(f'./dev_test/paper/{debug_name}/rgba_distant_in_total.png', rgba_distant_in_total)
                
                # imageio.imwrite(f'./dev_test/paper/{debug_name}/rgb_gt.png', rgb_gt)
                
                # # rgb_distant_seperate = to_img(ret['rendered_per_obj']['distant']['rgb_volume'])
                # # alpha_distant_seperate = to_img(ret['rendered_per_obj']['distant']['mask_volume'])
                # # rgba_distant_seperate = np.concatenate([rgb_distant_seperate, alpha_distant_seperate], -1)
                # # imageio.imwrite('./dev_test/paper/rgba_distant_seperate.png', rgba_distant_seperate)

                for k, v in cur_frame_dict.items():
                    all_gather[cam.id][k].append(v)
                    if args.save_perframe_camera:
                        obs_dir = os.path.join(vid_raw_root, cam_id)
                        cond_mkdir(obs_dir)
                        k_dir = os.path.join(obs_dir, k)
                        cond_mkdir(k_dir)
                        imageio.imwrite(os.path.join(k_dir, f"{frame_ind:08d}.jpg"), v)
        
        if not args.no_cam and args.gather_cam_pcl:
            pcl_filepath = os.path.join(vid_root, f"{name}_cam_pcl_ds={args.downscale}.ply")
            cam_pcl = torch.cat(cam_pcl, 0).view(-1,3).data.cpu().numpy()
            cam_pcl_color = (torch.cat(cam_pcl_color, 0).view(-1,3).data*255.).clamp_(0., 255.).to(dtype=torch.uint8).cpu().numpy()
            export_pcl_ply(cam_pcl, cam_pcl_color, filepath=pcl_filepath)

    #--------- Seperate video
    if not args.no_cam and args.save_seperate_keys:
        for cam_id, obs_dict in all_gather.items():
            for k, frames in obs_dict.items():
                write_video(os.path.join(vid_root, f"{name}_{cam_id}_{k}.mp4"), frames)

    #--------- Lidar render collection
    if args.render_lidar:
        write_video(os.path.join(vid_root, f"{name}_{args.lidar_model}_{args.lidar_id}_pcl_{args.lidar_vis_rgb_choice}.mp4"), np.array(pcl_imgs))

    #--------- 1 row collection
    if not args.no_cam:
        # keys_1l = ([] if args.no_gt else ['rgb_gt']) + ['rgb_volume', 'depth_volume', 'normals_volume'] + (['mesh'] if should_collect_mesh_imgs else [])
        keys_1l = collate_keys
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
        
        # Optionally append rescaled LiDAR pcl img
        if args.render_lidar:
            *_, H, W, _ = frames_per_obs_1l_all[0].shape
            H = int(H_lidar_vis*W/(W_lidar_vis*len(plt_lidar.renderers)))
            frames_pcl = (np.array([cpu_resize(im, (H, W)) for im in pcl_imgs])*255).clip(0, 255).astype(np.uint8)
            frames_per_obs_1l_all.append(frames_pcl)
        if len(frames_per_obs_1l_all) > 1:
            # NOTE: Only when different cameras has the same width
            frames_per_obs_1l_all = np.concatenate(frames_per_obs_1l_all, axis=1)
            write_video(os.path.join(vid_root, f"{name}_1l_all.mp4"), frames_per_obs_1l_all)

        #--------- All cams horizontal concat (from left to right, in the order specified by `cam_id_list`)
        # keys_1c = ([] if args.no_gt else ['rgb_gt']) + ['rgb_volume', 'normals_volume', 'depth_volume'] + (['mesh'] if should_collect_mesh_imgs else [])
        keys_1c = collate_keys
        all_keys = []
        for k in keys_1c:
            all_frames_per_cam_this_k = [np.array(all_gather[cam_id][k]) for cam_id in cam_id_list]
            all_frames_per_cam_this_k = pad_images_to_same_size(all_frames_per_cam_this_k, batched=True, padding='top_left') 
            all_frames_this_k = np.concatenate(all_frames_per_cam_this_k, axis=2)
            all_keys.append(all_frames_this_k)
        all_keys = np.concatenate(all_keys, axis=1)
        write_video(os.path.join(vid_root, f"{name}_1c_all.mp4"), all_keys)

def make_parser():
    bc = BaseConfig()
    bc.parser.add_argument("--load_pt", type=str, default=None, help="Typically unnecessary as the final or latest ckpt is loaded automatically. \n"\
        "Only specify the ckpt file path if indeed a non-final or non-latest ckpt needs to be loaded.")

    #---- General render configs
    bc.parser.add_argument("--no_output", action='store_true', help="If set, will skip saving any output videos.")
    bc.parser.add_argument("--rayschunk", type=int, default=65536, help="Chunkify the rendering process.")
    bc.parser.add_argument("--progress", action='store_true', help="If set, shows per frame progress.")
    bc.parser.add_argument("--fps", type=int, default=24)
    bc.parser.add_argument("--quality", type=int, default=None, help="Sets the quality for imageio.mimwrite (range: 0-10; 10 is the highest; default is 5).")
    bc.parser.add_argument("--dirname", type=str, default='videos', help="Sets the output directory to /path/to/exp_dir/${dirname}`.")
    bc.parser.add_argument("--outbase", type=str, default=datetime.now().strftime("%Y_%m_%d_%H_%M_%S"), help="Sets the basename of the output file (without extension).")

    # NeuS specific hacks
    bc.parser.add_argument("--forward_inv_s", type=str, default=None, help="Bypasses the inv_s parameter for NeuS during rendering.")

    #---- Replay / NVS reference configs
    bc.parser.add_argument("--num_frames", type=int, default=None)
    bc.parser.add_argument('--start_frame', type=int, default=0)
    bc.parser.add_argument('--stop_frame', type=int, default=None)
    
    #---- NVS configs
    bc.parser.add_argument("--nvs_path", type=str, default=None, 
                           help="Optionally, render on a novel path instead of the original one.\n"
                           "Setting this will turn the rendering into [NVS] mode instead of default [replay] mode.\n"\
                           "Typical options: [spherical_spiral, small_circle, interpolation, street_view].")
    bc.parser.add_argument("--nvs_num_frames", type=int, default=None, help="Sets the number of frames for NVS.")
    bc.parser.add_argument("--nvs_param", type=str, default=None)
    bc.parser.add_argument("--nvs_node_id", type=str, default=None, 
                           help="Specifies the node id to apply NVS motion to.\n"
                           "e.g., 'ego_car', 'camera_FRONT', 'camera', etc.")
    bc.parser.add_argument("--nvs_verbose", action='store_true', help="If set, a visualization of the camera trajectory will pop up.")

    #---- Camera Sim Argument
    bc.parser.add_argument("--no_cam", action='store_true', help="If set, skip all camera rendering")
    bc.parser.add_argument("--cam_id", type=str, default=None, help="If set, uses a specific camera; otherwise, uses all available cameras.")
    bc.parser.add_argument("--downscale", type=float, default=1.0, help="Sets the side length downscale for rendering and output.")
    bc.parser.add_argument("--save_perframe_camera", action='store_true', help="If set, saves raw per frame camera renderings.")
    bc.parser.add_argument("--save_seperate_keys", action='store_true', help="If set, saves raw per key camera renderings.")
    bc.parser.add_argument("--gather_cam_pcl", action='store_true', 
                           help="If set, will produce camera point clouds (colored point clouds projected using camera-rendered depth).")
    bc.parser.add_argument("--no_gt", action='store_true', 
                           help="If set, loading of the ground truth image is skipped.\n"\
                           "Useful when only rendering a pretrained model and ground truth is not needed.")
    bc.parser.add_argument("--no_sky", action='store_true', help="If set, the sky model will not be rendered.")
    bc.parser.add_argument("--only_cr", action='store_true', 
                           help="If set, only close-range is rendered, excluding distant-view and sky.\n"\
                           "NOTE: For LiDAR simulation, only_cr is always true.")
    bc.parser.add_argument("--with_distant_depth", action='store_true', help="If set, uses joint depth of cr+dv for depth visualization, otherwise uses cr.")
    bc.parser.add_argument("--with_distant_normal", action='store_true', help="If set, uses joint depth of cr+dv for normal visualization (only if dv can output normal), otherwise uses cr.")
    bc.parser.add_argument("--fast_render", action='store_true', help='Enables experimental fast volume rendering with extreme parameters.')
    # Ultra-wide-angle cameras
    bc.parser.add_argument("--no_distortion", action='store_true', help="If set, any camera distortion process will be discarded.")
    bc.parser.add_argument("--zoom_focal_scale", type=float, default=None, help="Optionally sets a zoom in/out factor.")
    bc.parser.add_argument("--ultra_wide_angle", action='store_true', 
                           help="If set, a distortion parameter that mimics the behavior of typical ultra-wide-angle cameras will be used.")

    bc.parser.add_argument("--cam_ref", type=str, default='camera_FRONT', help="Reference camera for visulization of LiDAR, mesh, etc.")

    #---- LiDAR Sim Argument
    bc.parser.add_argument("--render_lidar", action='store_true', help='If set, will also render LiDAR simulations.')
    bc.parser.add_argument("--save_perframe_lidar", action='store_true', help="If set, will save raw per frame LiDAR point cloud.")
    bc.parser.add_argument("--lidar_model", type=str, default="original_reren", help='Specifies the LiDAR model.')
    bc.parser.add_argument("--lidar_id", type=str, default="", help="Specifies the LiDAR name.")
    bc.parser.add_argument("--lidar_far", type=float, default=120.0)
    bc.parser.add_argument("--lidar_forward_inv_s", type=str, default="64000.0", help="Bypasses the inv_s parameter for NeuS during LiDAR rendering.")
    #---- LiDAR Sim's visulization argument
    bc.parser.add_argument("--lidar_vis_vmin", type=float, default=-2., 
                           help="Sets the minimum value for colorizing LiDAR pcl when 'lidar_vis_rgb_choice' is set to 'height' by default.")
    bc.parser.add_argument("--lidar_vis_vmax", type=float, default=9., 
                           help="Sets the maximum value for colorizing LiDAR pcl when 'lidar_vis_rgb_choice' is set to 'height' by default.")
    bc.parser.add_argument("--lidar_vis_width", type=int, default=640, help="Sets the width of the LiDAR visualization viewport.")
    bc.parser.add_argument("--lidar_vis_rgb_choice", type=str, default='height', 
                           help="Determines how to colorize LiDAR point clouds.\n"
                           "Supported options: ['height', 'grey', 'appearance'].")
    bc.parser.add_argument("--lidar_vis_radius", type=float, default=2.0, help="Sets the radius of LiDAR points in visualization.")
    bc.parser.add_argument("--lidar_vis_verbose", action='store_true', help="If set, a visualization window will pop up.")

    #---- Mesh render argument
    bc.parser.add_argument("--render_mesh", type=str, default=None, help="Optionally specify the file path of a mesh to be visualized in the camera viewport.")
    bc.parser.add_argument("--render_mesh_transform", type=str, default='identity', help="Specifies the type of transform for the input mesh.")
    bc.parser.add_argument("--render_mesh_verbose", action='store_true', help="If set, a visualization window for the mesh in the camera viewport will pop up.")

    return bc

if __name__ == "__main__":
    bc = make_parser()
    main_function(bc.parse())