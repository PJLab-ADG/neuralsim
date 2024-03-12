"""
@file   manipulate.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Manipulate / edit everything!
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
import random
import imageio
import functools
import numpy as np
from tqdm import tqdm
from datetime import datetime
from skimage.transform import resize
from scipy.spatial.transform import Rotation as R

import torch

from nr3d_lib.fmt import log
from nr3d_lib.config import load_config
from nr3d_lib.checkpoint import sorted_ckpts
from nr3d_lib.plot import get_n_ind_colors, color_depth, gallery
from nr3d_lib.utils import import_str, check_to_torch, IDListedDict, cond_mkdir, pad_images_to_same_size

from nr3d_lib.models.attributes import *

from app.resources import Scene, load_scenes_and_assets
from app.models.asset_base import AssetModelMixin
from app.resources.observers import Lidar, Camera
from app.renderers import BufferComposeRenderer

from dataio.scene_dataset import SceneDataset
from dataio.data_loader import SceneDataLoader

from code_multi.tools.utils import draw_box

def main_function(args, device=torch.device('cuda')):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    #---------------------------------------------
    #--------------     Load     -----------------
    #---------------------------------------------
    scene_bank, asset_bank, *_ = load_scenes_and_assets(**args, device=device)
    asset_bank.eval()
    
    fg_class_names = args.fg_class_name.split(',')
    
    # Fallsback to regular Mat4x4 for convenience
    # with torch.no_grad():
    #     for scene in scene_bank:
    #         for o in scene.all_nodes:
    #             for seg in o.attr_segments:
    #                 if 'transform' in seg.subattr:
    #                     seg.subattr['transform'] = TransformMat4x4(seg.subattr['transform'].mat_4x4())
            
    #         for o in scene.get_cameras(False):
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
    if args.no_gt:
        args.training.dataloader.tags = {}
    args.training.dataloader.preload = False
    args.training.val_dataloader.preload = False
    scene_dataloader = SceneDataLoader(scene_bank, dataset_impl, config=args.training.val_dataloader, device=device)
    scene_dataloader.set_camera_downscale(args.downscale)

    #---------------------------------------------
    #------------     Renderer     ---------------
    #---------------------------------------------
    renderer = BufferComposeRenderer(args.renderer)
    renderer.populate(asset_bank)
    renderer.eval()
    asset_bank.eval()
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

    cam_id_list = scene_dataloader.cam_id_list if args.cam_id is None else [args.cam_id]
    cam_ref_id = cam_id_list[0] if len(cam_id_list) == 1 else args.cam_ref

    expname = os.path.split(args.exp_dir.rstrip("/"))[-1]
    name = f"{expname[0:64]}_{args.outbase}_ds={args.downscale}" + ("" if args.forward_inv_s is None else f"_s={int(args.forward_inv_s)}") + f"_manipulate={args.mode}"

    #---------------------------------------------
    #-----------     Manipulate     --------------
    #---------------------------------------------
    scene: Scene = scene_bank[0]
    
    if args.num_frames is not None:
        if not args.fix_gt:
            args.stop_frame = min(args.start_frame + args.num_frames, len(scene))
            num_frames = args.stop_frame - args.start_frame
            print(f"=> args.stop_frame is set to {args.stop_frame}")
        else:
            num_frames = args.num_frames
    else:
        if args.stop_frame is None or args.stop_frame == -1:
            args.stop_frame = len(scene)
        else:
            args.stop_frame = min(args.stop_frame, len(scene))
        num_frames = max(args.stop_frame - args.start_frame, 1)
    # NOTE: froze scene at the start_frame from the very beginning
    scene.slice_at(args.start_frame)
    
    # TODO: Refactor into a class.
    if args.mode == 'rotation' or args.mode == 'replay_rotation':
        def op_scene(i, fi, scene: Scene):
            if 'replay' in args.mode:
                scene.slice_at(fi)
            cur_frame_angle = i*2*np.pi/args.fps * args.get('n_rots', 0.5)
            c = np.math.cos(cur_frame_angle)
            s = np.math.sin(cur_frame_angle)
            rot = np.array([
                [ c, -s,  0],
                [ s,  c,  0],
                [ 0,  0,  1]])
            rot = check_to_torch(rot, ref=scene)
            for obj in scene.drawables:
                if (obj.class_name in fg_class_names) and obj.i_valid and len(obj) > 0:
                    if 'replay' in args.mode:
                        obj.slice_at(fi)
                    else:
                        obj.slice_at(args.start_frame)
                    obj.world_transform.tensor[:3, :3] = rot @ obj.world_transform.tensor[:3, :3]
        def op_cam(i, fi, cam: Camera):
            pass

    elif args.mode == 'translation' or args.mode == 'replay_translation':
        def op_scene(i, fi, scene: Scene):
            if 'replay' in args.mode:
                scene.slice_at(fi)
            fps = float(args.fps)
            if i // fps == 0:
                trans = torch.tensor([-(i%fps)*2./fps,0,0], device=device)
            elif i // fps == 1:
                trans = torch.tensor([-2.,-(i%fps)*2./fps,0], device=device)
            elif i // fps == 2:
                trans = torch.tensor([-2.+(i%fps)*2./fps,-2,0], device=device)
            else:
                trans = torch.tensor([0,-2.+(i%fps)*2./fps,0], device=device)

            for obj in scene.drawables:
                if (obj.class_name in fg_class_names) and obj.i_valid and len(obj) > 0:
                    if 'replay' in args.mode:
                        obj.slice_at(fi)
                    else:
                        obj.slice_at(args.start_frame)
                    obj.world_transform.tensor[:3, 3] = obj.world_transform.tensor[:3, 3] + trans
        def op_cam(i, fi, cam: Camera):
            pass

    elif args.mode == 'scale' or args.mode == 'replay_scale':
        scales = {obj.id: obj.scale.vec_3().clone() for obj in scene.drawables}
        def op_scene(i, fi, scene: Scene):
            if 'replay' in args.mode:
                scene.slice_at(fi)
            for j, obj in enumerate(scene.drawables):
                if (obj.class_name in fg_class_names) and obj.i_valid and len(obj) > 0:
                    l = (i % args.fps) / float(args.fps)
                    if 'replay' in args.mode:
                        obj.slice_at(fi)
                    else:
                        obj.slice_at(args.start_frame)
                    if j % 2 == 0:
                        obj.scale.value()[0:2] = scales[obj.id][0:2] * (1.5-0.5*l)
                    else:
                        obj.scale.value()[0:2] = scales[obj.id][0:2] * (0.5+0.5*l)
        def op_cam(i, fi, cam: Camera):
            pass

    elif args.mode == 'random_rotate' or args.mode == 'replay_random_rotate':        
        fps = float(args.fps)
        d_rots = {obj.id: random.choice([np.random.uniform(-1.25*np.pi/fps, -0.75*np.pi/fps), np.random.uniform(1.25*np.pi/fps, 0.75*np.pi/fps)]) for obj in scene.drawables}
        def op_scene(i, fi, scene: Scene):
            if 'replay' in args.mode:
                scene.slice_at(fi)
            for j, obj in enumerate(scene.drawables):
                if 'replay' in args.mode:
                    obj.slice_at(fi)
                else:
                    obj.slice_at(args.start_frame)
                if (obj.class_name in fg_class_names) and obj.i_valid and len(obj) > 0:
                    rot = torch.tensor(R.from_rotvec(np.array([0,0, i * d_rots[obj.id]])).as_matrix(), dtype=torch.float, device=device) @ obj.world_transform.rotation()
                    obj.world_transform = TransformRT(rot=RotationMat3x3(rot), trans=Translation(obj.world_transform.translation())).to(device)
        def op_cam(i, fi, cam: Camera):
            pass

    elif args.mode == 'random' or args.mode == 'replay_random':        
        fps = float(args.fps)
        d_rots = {obj.id: random.choice([np.random.uniform(-1.25*np.pi/fps, -0.75*np.pi/fps), np.random.uniform(1.25*np.pi/fps, 0.75*np.pi/fps)]) for obj in scene.drawables}
        d_trans = {obj.id: np.random.uniform(-2./fps, 2./fps) for obj in scene.drawables}
        def op_scene(i, fi, scene: Scene):
            if 'replay' in args.mode:
                scene.slice_at(fi)
            for j, obj in enumerate(scene.drawables):
                if 'replay' in args.mode:
                    obj.slice_at(fi)
                else:
                    obj.slice_at(args.start_frame)
                if (obj.class_name in fg_class_names) and obj.i_valid and len(obj) > 0:
                    rot = torch.tensor(R.from_rotvec(np.array([0,0, i * d_rots[obj.id]])).as_matrix(), dtype=torch.float, device=device) @ obj.world_transform.rotation()
                    if (i // fps) % 4 == 0:
                        trans = torch.tensor(np.array([-(i%fps)*d_trans[obj.id], 0., 0.]), dtype=torch.float, device=device) + obj.world_transform.translation()
                    elif (i // fps) % 4 == 1:
                        trans = torch.tensor(np.array([-fps*d_trans[obj.id], -(i%fps)*d_trans[obj.id], 0.]), dtype=torch.float, device=device) + obj.world_transform.translation()
                    elif (i // fps) % 4 == 2:
                        trans = torch.tensor(np.array([(i%fps-fps)*d_trans[obj.id], -fps*d_trans[obj.id], 0.]), dtype=torch.float, device=device) + obj.world_transform.translation()
                    elif (i // fps) % 4 == 3:
                        trans = torch.tensor(np.array([0., (i%fps-fps)*d_trans[obj.id], 0.]), dtype=torch.float, device=device) + obj.world_transform.translation()
                    obj.world_transform = TransformRT(rot=RotationMat3x3(rot), trans=Translation(trans)).to(device)
        def op_cam(i, fi, cam: Camera):
            pass

    elif args.mode == 'thanos': # NOTE: Wipe out random half of objects
        from random import randint
        fg_obj_list = [o.id for o in scene.get_drawable_groups_by_class_name_list(fg_class_names, only_valid=False)]
        new_obj_list = IDListedDict()
        for obj in scene.drawables:
            if (obj.id not in fg_obj_list) or (randint(0,1) == 1):
                new_obj_list.append(obj)
        scene.drawables = new_obj_list
        def op_scene(i, fi, scene: Scene):
            if 'replay' in args.mode:
                scene.slice_at(fi)
            for obj in scene.drawables:
                obj.slice_at(fi)
        def op_cam(i, fi, cam: Camera):
            pass
            
    elif args.mode == 'self_trans' or args.mode == 'self_trans_fix_obj':
        scene.slice_at(args.start_frame)
        def op_scene(i, fi, scene: Scene):
            for obj in scene.drawables:
                if 'fix_obj' in args.mode:
                    obj.slice_at(args.start_frame)
                else:
                    obj.slice_at(fi)
        def op_cam(i, fi, cam: Camera):
            translation = torch.tensor([2.5/args.fps,0,0], device=device)
            transform = TransformRT(rot=cam.world_transform.rotation(), trans=cam.world_transform.translation() + translation)
            cam.world_transform = transform

    elif args.mode == 'self_rotate' or args.mode == 'self_rotate_fix_obj':
        scene.slice_at(args.start_frame)
        def op_scene(i, fi, scene: Scene):
            for obj in scene.drawables:
                if 'fix_obj' in args.mode:
                    obj.slice_at(args.start_frame)
                else:
                    obj.slice_at(fi)
        def op_cam(i, fi, cam: Camera):
            rot = torch.from_numpy(R.from_rotvec(np.array([0.,0.,2.0*np.pi/num_frames])).as_matrix()).float().to(device)
            transform = TransformRT(rot=rot @ cam.world_transform.rotation(), trans=cam.world_transform.translation())
            cam.world_transform = transform

    elif args.mode == 'self_fly' or args.mode == 'self_fly_fix_obj':
        scene.slice_at(args.start_frame)
        def op_scene(i, fi, scene: Scene):
            for obj in scene.drawables:
                if 'fix_obj' in args.mode:
                    obj.slice_at(args.start_frame)
                else:
                    obj.slice_at(fi)

        def op_cam(i, fi, cam: Camera):
            translation = torch.tensor([0,0,5.0/num_frames], device=device)
            transform = TransformRT(rot=cam.world_transform.rotation(), trans=cam.world_transform.translation() + translation)
            cam.world_transform = transform

    elif args.mode == 'self_zoom_out' or args.mode == 'self_zoom_out_fix_obj':
        name += f'_zoom_focal_scale={args.zoom_focal_scale}'
        scene.slice_at(args.start_frame)
        focals0 = {}
        for o in scene.get_cameras(False):
            focals0[o.id] = o.intr.focal()
        def op_scene(i, fi, scene: Scene):
            for obj in scene.drawables:
                if 'fix_obj' in args.mode:
                    obj.slice_at(args.start_frame)
                else:
                    obj.slice_at(fi)
        def op_cam(i, fi, cam: Camera):
            # from 2^2 to 2^-2
            # cam.intr.set_focal( 2**(-(i/float(num_frames))*4+2) * f0)
            # from 2^x to 2^-x
            if args.no_distortion:
                cam.intr.subattr.distortion.tensor.zero_()
            elif args.ultra_wide_angle:
                # A dummy distortion param for 3.0 zoom_focal_scale's ultra wide-angle
                cam.intr.subattr.distortion.tensor[..., 0] = -0.129
                cam.intr.subattr.distortion.tensor[..., 1] = 0.0108
                cam.intr.subattr.distortion.tensor[..., 4] = -0.00033
            focal_scale_log2 = np.log2(args.zoom_focal_scale)
            r = 2**(-(i/float(num_frames))*focal_scale_log2*2+focal_scale_log2)
            fx0, fy0 = focals0[cam.id]
            cam.intr.set_focal(r * fx0, r * fy0)

    elif args.mode == 'edit_shape':
        pass

    elif args.mode == 'edit_appearance':
        pass

    elif args.mode == 'edit_alter':
        scene.slice_at(args.start_frame)
        
        dimension_in_neus = [1.8, 1.2, 0.75]
        scale_tensor = check_to_torch(dimension_in_neus, device=device, dtype=torch.float)
        
        # rot0 = np.diag(1./np.array(dimension_in_neus))
        # rot0 = np.diag(np.array(dimension_in_neus))
        rot1 = R.from_rotvec(np.array([1., 0., 0.]) * -np.pi/15.).as_matrix()
        rot2 = R.from_rotvec(np.array([0., 1., 0.]) * np.deg2rad(4.5)).as_matrix()
        rot4 = np.array([
            [-1, 0, 0],
            [0, 0, -1],
            [0, -1, 0]
        ])
        neus_to_waymo = np.eye(4)
        neus_to_waymo[:3, :3] = rot4 @ rot2 @ rot1
        trans = np.eye(4)
        trans[:3, 3] = np.array([0.2, 0.05, 0.])
        neus_to_waymo = neus_to_waymo @ trans
        neus_to_waymo = check_to_torch(neus_to_waymo, device=device, dtype=torch.float)
        
        veh_list = scene.get_drawable_groups_by_class_name_list(fg_class_names, only_valid=False)
        # o1 = [v for v in veh_list if 'rDSH' in v.id][0]
        o1 = [v for v in veh_list if 'vadq' in v.id][0]
        
        alter_resume_dir = args.alter_resume_dir
        cfg = load_config(os.path.join(alter_resume_dir, 'config.yaml'))
        alter_model_cfg = cfg['assetbank_cfg'][args.alter_class_name]
        alter_model: AssetModelMixin = import_str(alter_model_cfg['model_class'])(**alter_model_cfg['model_params'], device=device)
        alter_model.asset_init_config(**alter_model_cfg['asset_params'])
        alter_model.asset_populate(scene=scene, obj=o1, config=alter_model.populate_cfg, device=device)
        alter_ckpt_file = sorted_ckpts(os.path.join(alter_resume_dir, 'ckpts'))[-1]
        alter_state_dict = torch.load(alter_ckpt_file)
        alter_model.load_state_dict(alter_state_dict['asset_bank']['model'])
        
        asset_bank.add_module('altered', alter_model)
        
        alter_model.space.aabb /= (scale_tensor/2.8)
        o1.model = alter_model
        
        def op_scene(i, fi, scene: Scene):
            scene.slice_at(fi)
            o1.world_transform = TransformMat4x4(o1.world_transform.mat_4x4() @ neus_to_waymo, dtype=torch.float, device=device)
        def op_cam(i, fi, cam: Camera):
            pass
    else:
        raise ValueError(f"Unknown mode={args.mode}")

    #---------------------------------------------
    #--------------     Plot     -----------------
    #---------------------------------------------
    vid_root = os.path.join(args.exp_dir, args.dirname)
    cond_mkdir(vid_root)
    if args.save_raw:
        vid_raw_root = os.path.join(vid_root, name)
        cond_mkdir(vid_raw_root)
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

    collate_keys = ([] if args.no_gt else ['rgb_gt']) + ['rgb_volume', 'depth_volume', 'normals_volume', \
        'ins_seg_mask_buffer', 'bgrgb_volume', 'bgdepth_volume', 'bg_normals_volume']
    all_gather = dict()
    # for k in collate_keys:
    #     all_gather[k] = {cam_id: [] for cam_id in cam_id_list}
    for cam_id in cam_id_list:
        all_gather[cam_id] = dict({k: [] for k in collate_keys})

    with torch.no_grad():
    # for scene in scene_bank:
        bg_obj = scene.get_drawable_groups_by_class_name(args.bg_class_name)[0]
        bg_obj_id = bg_obj.id
        bg_model = bg_obj.model
        if args.forward_inv_s is not None:
            bg_model.ray_query_cfg.forward_inv_s = args.forward_inv_s
        if args.render_lidar:
            assert cam_ref_id is not None and cam_ref_id in scene.observers, \
                f"A valid frontal reference camera is required.\nCurrent camera list={scene.get_observer_groups_by_class_name('Camera', False)}"
            
            cam0: Camera = scene.observers[cam_ref_id]
            
            
            
            # Create new lidar to be simulated and make it a child of cam0
            lidar = Lidar('sim_lidar', lidar_model=args.lidar_model, lidar_name=args.lidar_id, near=0.3, far=120.0, scene=scene).to(device=device)
            scene.add_node(lidar, parent=cam0)
            
            cam0.intr.set_downscale(args.downscale)
            if not args.no_cam:
                W_lidar_vis = min(cam0.intr.W * 4, args.lidar_vis_width) # 4 is the number of column when joint render
            else:
                W_lidar_vis = args.lidar_vis_width
            H_lidar_vis = W_lidar_vis*9//16
            
            from mayavi import mlab
            mlab.options.offscreen = True
            fig = mlab.figure(bgcolor=(0, 0, 0), size=(W_lidar_vis, H_lidar_vis))
            pcl_imgs1 = []
            pcl_imgs2 = []

            def render_pcl(scene):
                with torch.no_grad():
                    cam0: Camera = scene.observers[cam_ref_id]
                    # lidar.transform = cam0.world_transform
                    # lidar.world_transform = lidar.transform
                    asset_bank.rendering_before_per_view(renderer=renderer, observer=cam0, scene_id=scene.id)
                    lidar_rays_o, lidar_rays_d, lidar_rays_ts = lidar.get_all_rays(return_ts=True)
                    # render_fn = functools.partial(renderer.render_rays, scene=scene, with_rgb=False, with_normal=False, near=lidar.near, far=lidar.far)
                    # ret_lidar = batchify_query(render_fn, lidar_rays_o, lidar_rays_d, chunk=args.rayschunk, show_progress=args.progress)
                    ret_lidar = renderer.render(
                        scene, rays=(lidar_rays_o,lidar_rays_d,lidar_rays_ts), near=lidar.near, far=lidar.far, 
                        with_rgb=False, with_normal=False, rayschunk=args.rayschunk, show_progress=args.progress)                    
                    # ret_lidar = renderer.render(lidar, scene, with_rgb=False, with_normal=False)
                    lidar_rays_acc = ret_lidar['rendered']['mask_volume']
                    lidar_rays_depth = ret_lidar['rendered']['depth_volume']
                    valid = lidar_rays_acc > 0.95
                    lidar_pts = lidar_rays_o[valid] + lidar_rays_d[valid] * lidar_rays_depth[valid].unsqueeze(-1)
                    # NOTE: Convert to a common coordinate system (OpenCV pinhole camera in this case)
                    lidar_pts = lidar.world_transform(lidar_pts, inv=True).data.cpu().numpy()
                    # lidar_mask = lidar_rays_acc[valid].data.cpu().numpy()
                    # lidar_depth = lidar_mask * np.clip(lidar_rays_depth[valid].data.cpu().numpy() / depth_max, 0, 1) + (1-lidar_mask) * 1

                    mlab.clf()
                    # fig = mlab.figure(bgcolor=(0, 0, 0), size=(W_lidar, H_lidar))
                    # mlab.points3d(lidar_pts[...,0], lidar_pts[...,1], lidar_pts[...,2], lidar_pts[...,2], mode="point", colormap='rainbow', vmin=-2., vmax=9., figure=fig)
                    mlab.points3d(lidar_pts[...,0], lidar_pts[...,1], lidar_pts[...,2], -lidar_pts[...,1], mode="point", colormap='rainbow', vmin=-2., vmax=9., figure=fig)

                    # Top view
                    mlab.view(focalpoint=np.array([0., 0., 15.]), azimuth=90.0, elevation=-90.0, distance=100.0, roll=-90.0)
                    # # Front slope view
                    # mlab.view(focalpoint=np.array([0., 0., 50.]), azimuth=-90.0, elevation=176.0, distance=70.0, roll=179.0)
                    # mlab.show()
                    fig.scene._lift()
                    im1 = mlab.screenshot(figure=fig, mode='rgb', antialiased=True)
                    # # Top view
                    # mlab.view(focalpoint=np.array([0., 0., 15.]), azimuth=90.0, elevation=-90.0, distance=100.0, roll=-90.0)
                    # Front slope view
                    mlab.view(focalpoint=np.array([0., 0., 50.]), azimuth=-90.0, elevation=176.0, distance=70.0, roll=179.0)
                    fig.scene._lift()
                    im2 = mlab.screenshot(figure=fig, mode='rgb', antialiased=True)
                    return im1, im2

        # for frame_ind in tqdm(range(len(scene)), "rendering frames..."):
        #     if frame_ind < args.start_frame or frame_ind > args.stop_frame:
        #         continue
        log.info(f"Start [manipulate], mode={args.mode}, ds={args.downscale}, in {args.exp_dir}")
        for frame_ind in tqdm(range(args.start_frame, args.start_frame + num_frames, 1), "rendering frames..."):
            #----------------------------------------------------------
            #---------    apply manipulation on scene    --------------
            #----------------------------------------------------------
            op_scene(frame_ind-args.start_frame, frame_ind, scene)
            
            for cam_id in scene_dataloader.cam_id_list:
                #----------------------------------------------------------
                #-----    apply manipulation on particular camera    ------
                #----------------------------------------------------------
                op_cam(frame_ind-args.start_frame, frame_ind, scene.observers[cam_id])
            
            if args.render_lidar:
                im1, im2 = render_pcl(scene)
                pcl_imgs1.append(im1)
                pcl_imgs2.append(im2)
            
            if args.no_cam:
                continue
            
            # scene.slice_at(frame_ind)
            for cam_id in cam_id_list:
                cam: Camera = scene.observers[cam_id]
                # TODO: Move set_camera_downscale into the dataset? But wouldn't this create two sets of cams?
                #       The key point is that after each slice_at, the downscale parameter returns to its original position; is this a behavior we do not want?
                cam.intr.set_downscale(args.downscale)
                #-----------------------------------------------
                ret = renderer.render(scene, observer=cam, render_per_obj_individual=True, show_progress=args.progress, with_env=not args.no_sky)
                rendered = ret['rendered']
                #-----------------------------------------------

                cur_frame_dict = dict({k: [] for k in collate_keys})
                # TODO: Try refactoring this with imagegrid.
                # galleries = []
                def to_img(tensor):
                    return tensor.reshape([cam.intr.H, cam.intr.W, -1]).data.cpu().numpy()

                if not args.no_gt:
                    if args.fix_gt:
                        ground_truth = scene_dataloader.get_image_and_gts(scene.id, cam.id, args.start_frame)
                    else:
                        ground_truth = scene_dataloader.get_image_and_gts(scene.id, cam.id, frame_ind)
                    rgb_gt = np.clip((to_img(ground_truth['image_rgb'])*255).astype(np.uint8),0,255)
                    # galleries.append(rgb_gt)
                    # all_gather[cam.id]['rgb_gt'].append(rgb_gt)
                    cur_frame_dict['rgb_gt'] = rgb_gt
                
                mask_volume = to_img(rendered['mask_volume'])
                depth_volume = to_img(rendered['depth_volume'])
                depth_volume = mask_volume * np.clip(depth_volume/depth_max, 0, 1) + (1-mask_volume) * 1
                depth_volume = color_depth(depth_volume.squeeze(-1), scale=1, cmap='turbo')    # turbo_r, viridis, rainbow
                
                rgb_volume = np.clip((to_img(rendered['rgb_volume'])*255).astype(np.uint8),0,255)
                if args.draw_box:
                    drawables = scene.get_drawables(True)
                    drawables = cam.filter_drawable_groups(drawables)
                    for obj in drawables:
                        draw_box_fn(rgb_volume, obj, cam, inplace=True, nolabel=args.draw_box_no_label)
                
                # galleries.extend([rgb_volume, depth_volume])
                # all_gather[cam.id]['rgb_volume'].append(rgb_volume)
                # all_gather[cam.id]['depth_volume'].append(depth_volume)
                cur_frame_dict['rgb_volume'] = rgb_volume
                cur_frame_dict['depth_volume'] = depth_volume
                
                # normals
                if 'normals_volume' in rendered:
                    normals_volume = np.clip((to_img(rendered['normals_volume']/2+0.5)*255).astype(np.uint8),0,255)
                    # galleries.append(normals_volume)
                    # all_gather[cam.id]['normals_volume'].append(normals_volume)
                    cur_frame_dict['normals_volume'] = normals_volume
                
                # instance segmentation
                ins_seg_mask_buffer = np.take(instance_cmap, axis=0, indices=to_img(ret['ins_seg_mask_buffer'])[...,0]).astype(np.uint8)
                # galleries.append(ins_seg_mask_buffer)
                # all_gather[cam.id]['ins_seg_mask_buffer'].append(ins_seg_mask_buffer)
                cur_frame_dict['ins_seg_mask_buffer'] = ins_seg_mask_buffer
                # # class segmentation
                # plt.imshow(to_img(ret['class_seg_mask_buffer']))

                # bg
                bg_rendered = ret['rendered_per_obj'][bg_obj_id]
                bgmask_volume = to_img(bg_rendered['mask_volume'])
                bgdepth_volume = to_img((bg_rendered['depth_volume']))
                bgdepth_volume = bgmask_volume * np.clip(bgdepth_volume/depth_max, 0, 1) + (1-bgmask_volume) * 1
                bgdepth_volume = color_depth(bgdepth_volume.squeeze(-1), scale=1, cmap='turbo')    # turbo_r, viridis, rainbow
                bgrgb_volume = np.clip((to_img(bg_rendered['rgb_volume'])*255).astype(np.uint8),0,255)
                # galleries.extend([bgrgb_volume, bgdepth_volume])
                # all_gather[cam.id]['bgrgb_volume'].append(bgrgb_volume)
                # all_gather[cam.id]['bgdepth_volume'].append(bgdepth_volume)
                cur_frame_dict['bgrgb_volume'] = bgrgb_volume
                cur_frame_dict['bgdepth_volume'] = bgdepth_volume
                
                
                # normals
                if 'normals_volume' in bg_rendered:
                    bg_normals_volume = np.clip((to_img(bg_rendered['normals_volume']/2+0.5)*255).astype(np.uint8),0,255)
                    # galleries.append(bg_normals_volume)
                    # all_gather[cam.id]['bg_normals_volume'].append(bg_normals_volume)
                    cur_frame_dict['bg_normals_volume'] = bg_normals_volume

                # frame = gallery(galleries, nrows=2)
                # frame_simple = gallery(([rgb_gt] if not args.no_gt else []) +  [rgb_volume, depth_volume, ins_seg_mask_buffer], nrows=2)
                # plt.imshow(frame)
                # plt.show()
                # imageio.imwrite(os.path.join(args.exp_dir, 'test.png'), frame)
                # imageio.imwrite(os.path.join(args.exp_dir, 'test_simple.png'), frame_simple)

                for k, v in cur_frame_dict.items():
                    all_gather[cam.id][k].append(v)

                    # if args.save_raw:
                    #     obs_dir = os.path.join(vid_raw_root, cam_id)
                    #     cond_mkdir(obs_dir)
                    #     k_dir = os.path.join(obs_dir, k)
                    #     cond_mkdir(k_dir)
                    #     imageio.imwrite(os.path.join(k_dir, f"{frame_ind:08d}.png"), v)
    
    
    # for cam_id, obs_dict in all_gather.items():
    #     obs_dir = os.path.join(vid_raw_root, cam_id)
    #     cond_mkdir(obs_dir)
    #     for k, v in obs_dict.items():
    #         if len(v) > 0:
    #             k_dir = os.path.join(obs_dir, k)
    #             cond_mkdir(k_dir)
    #             for ind, im in enumerate(v):
    #                 if args.save_raw:
    #                     imageio.imwrite(os.path.join(k_dir, f"{ind:08d}.png"), im)
    
    #--------- Seperate video
    if not args.no_cam and args.save_seperate_keys:
        for cam_id, obs_dict in all_gather.items():
            for k, frames in obs_dict.items():
                write_video(os.path.join(vid_root, f"{name}_{cam_id}_{k}.mp4"), frames)
    
    #--------- 2 rows collection
    if not args.no_cam:
        frames_per_obs_all = []
        for cam_id, obs_dict in all_gather.items():
            frames_per_obs = []
            for kvs in zip(*(obs_dict.values())):
                if args.no_gt:
                    kvs = list(kvs)
                    kvs.insert(0, np.zeros_like(kvs[0]))
                frames_per_obs.append(gallery(np.stack(kvs, 0), nrows=2))
            if not args.only_all:
                write_video(os.path.join(vid_root, f"{name}_{cam_id}.mp4"), frames_per_obs)
            frames_per_obs_all.append(np.array(frames_per_obs))
        if len(frames_per_obs_all) > 1 or args.only_all:
            # NOTE: only for waymo: different cameras has the same width
            write_video(os.path.join(vid_root, f"{name}_all.mp4"), np.concatenate(frames_per_obs_all, axis=1))
    
    if args.render_lidar:
        # if not args.only_all: # NOTE: always render seperate pcl
        write_video(os.path.join(vid_root, f"{name}_{args.lidar_model}_{args.lidar_id}_pcl1.mp4"), np.array(pcl_imgs1))
        write_video(os.path.join(vid_root, f"{name}_{args.lidar_model}_{args.lidar_id}_pcl2.mp4"), np.array(pcl_imgs2))
        
        if not args.no_cam:
            *_, H, W, _ = frames_per_obs_all[0].shape
            H = int(H_lidar_vis*W/W_lidar_vis)
            frames_pcl_1 = np.clip((np.array([resize(im, (H, W)) for im in pcl_imgs1])*255), 0, 255).astype(np.uint8)
            frames_pcl_2 = np.clip((np.array([resize(im, (H, W)) for im in pcl_imgs2])*255), 0, 255).astype(np.uint8)
        else:
            frames_per_obs_all = []
            frames_pcl_1 = np.array(pcl_imgs1)
            frames_pcl_2 = np.array(pcl_imgs2)

        frames_per_obs_all.append( frames_pcl_1 )
        frames_per_obs_all.append( frames_pcl_2 )
        write_video(os.path.join(vid_root, f"{name}_{args.lidar_model}_{args.lidar_id}_all_with_pcl.mp4"), np.concatenate(frames_per_obs_all, axis=1))

    if not args.no_cam:
        #--------- All cams vertical concat
        keys_1l = ([] if args.no_gt else ['rgb_gt']) + ['rgb_volume', 'bgrgb_volume', 'depth_volume', 'bgdepth_volume', 'ins_seg_mask_buffer', 'normals_volume', 'bg_normals_volume']
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

        #--------- All cams horizontal concat (from left to right, in the order specified by `cam_id_list`)
        keys_1c = ([] if args.no_gt else ['rgb_gt']) + ['rgb_volume', 'bgrgb_volume', 'normals_volume', 'bg_normals_volume', 'depth_volume', 'bgdepth_volume', 'ins_seg_mask_buffer']
        all_keys = []
        for k in keys_1c:
            all_frames_per_cam_this_k = [np.array(all_gather[cam_id][k]) for cam_id in cam_id_list]
            all_frames_per_cam_this_k = pad_images_to_same_size(all_frames_per_cam_this_k, batched=True, padding='top_left') 
            all_frames_this_k = np.concatenate(all_frames_per_cam_this_k, axis=2)
            all_keys.append(all_frames_this_k)
        all_keys = np.concatenate(all_keys, axis=1)
        write_video(os.path.join(vid_root, f"{name}_1c_all.mp4"), all_keys)

def make_parser():
    from nr3d_lib.config import BaseConfig
    bc = BaseConfig()
    bc.parser.add_argument("--fg_class_name", type=str, default='Vehicle,Pedestrian', help="The class_name of the foreground object.")
    bc.parser.add_argument("--bg_class_name", type=str, default='Street', help="The class_name of the background object.")
    
    bc.parser.add_argument("--load_pt", type=str, default=None, help="Typically unnecessary as the final or latest ckpt is loaded automatically. \n"\
        "Only specify the ckpt file path if indeed a non-final or non-latest ckpt needs to be loaded.")
    bc.parser.add_argument("--downscale", type=float, default=1.0, help="Sets the side length downscale for rendering and output.")
    bc.parser.add_argument("--progress", action='store_true', help="If set, shows per frame progress.")
    bc.parser.add_argument("--rayschunk", type=int, default=8192)
    # NeuS specific hacks
    bc.parser.add_argument("--forward_inv_s", type=float, default=None)
    
    bc.parser.add_argument("--no_cam", action='store_true', help='skip all camera rendering')
    bc.parser.add_argument("--no_gt", action='store_true')
    bc.parser.add_argument("--no_sky", action='store_true')
    bc.parser.add_argument("--fix_gt", action='store_true')
    bc.parser.add_argument("--fix_objects", action='store_true')
    bc.parser.add_argument("--save_raw", action='store_true')
    bc.parser.add_argument("--save_seperate_keys", action='store_true')
    bc.parser.add_argument("--draw_box", action="store_true")
    bc.parser.add_argument("--draw_box_no_label", action="store_true")

    bc.parser.add_argument("--fps", type=int, default=24)
    bc.parser.add_argument("--quality", type=int, default=None, help="Sets the quality for imageio.mimwrite (range: 0-10; 10 is the highest; default is 5).")
    bc.parser.add_argument("--dirname", type=str, default='vid_manipulate', help="Sets the output directory to /path/to/exp_dir/${dirname}`.")
    bc.parser.add_argument("--outbase", type=str, default=datetime.now().strftime("%Y_%m_%d_%H_%M_%S"), help="Sets the basename of the output file (without extension).")
    bc.parser.add_argument("--only_all", action='store_true', help='render only one total video')
    
    bc.parser.add_argument("--cam_ref", type=str, default='camera_FRONT', help="Reference camera for visulization of LiDAR, mesh, etc.")
    
    # LiDAR Sim Arguments
    bc.parser.add_argument("--render_lidar", action='store_true', help='render lidar pointclouds')
    bc.parser.add_argument("--lidar_model", type=str, default="dummy", help='lidar model.')
    bc.parser.add_argument("--lidar_id", type=str, default="", help="Specifies the lidar name")
    bc.parser.add_argument("--lidar_vis_vmin", type=float, default=-2.)
    bc.parser.add_argument("--lidar_vis_vmax", type=float, default=9.)
    bc.parser.add_argument("--lidar_vis_width", type=int, default=1920, help="Width of lidar visualization image.")
    
    # Manipulation Arguments
    bc.parser.add_argument("--mode", type=str, default='random', help="")
    bc.parser.add_argument("--no_distortion", action='store_true')
    bc.parser.add_argument("--zoom_focal_scale", type=float, default=None)
    bc.parser.add_argument("--ultra_wide_angle", action='store_true')
    bc.parser.add_argument("--alter_resume_dir", type=str, default=None)
    bc.parser.add_argument("--alter_class_name", type=str)
    
    bc.parser.add_argument("--cam_id", type=str, default=None, help="If set, uses a specific camera; otherwise, uses all available cameras.")
    bc.parser.add_argument('--start_frame', type=int, default=25)
    bc.parser.add_argument("--num_frames", type=int, default=None)
    bc.parser.add_argument('--stop_frame', type=int, default=125)

    return bc

if __name__ == "__main__":
    bc = make_parser()
    main_function(bc.parse())