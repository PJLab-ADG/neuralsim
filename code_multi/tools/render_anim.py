"""
@file   render_anim.py
@author Nianchen Deng, Shanghai AI Lab
@brief  Render animation for a multi-object scene.
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
import torch
import imageio
import functools
import numpy as np
from tqdm import tqdm
from datetime import datetime
from skimage.transform import resize
from collections import defaultdict
from operator import itemgetter

from nr3d_lib.utils import import_str, cond_mkdir
from nr3d_lib.plot import get_n_ind_colors, color_depth, gallery
from nr3d_lib.config import ConfigDict
from nr3d_lib.graphics.utils import PSNR
from nr3d_lib.models.attributes import *
from nr3d_lib.fmt import log

from app.anim import create_anim
from app.renderers import BufferComposeRenderer
from app.visible_grid import VisibleGrid
from app.resources import load_scenes_and_assets, Scene
from app.resources.observers import Lidar, Camera
from dataio.scene_dataset import SceneDataset
from dataio.data_loader import SceneDataLoader
from code_multi.tools.utils import draw_box

DEBUG_LIDAR = False


def write_video(uri, frames, fps, quality, **kwargs):
    if len(frames) > 1:
        if ".mp4" not in uri:
            uri = f"{uri}.mp4"
        imageio.mimwrite(uri, frames, fps=fps, quality=quality, **kwargs)
        print(f"Video saved to {uri}")
    else:
        if ".mp4" in uri:
            uri = f"{os.path.splitext(uri)[0]}.png"
        imageio.imwrite(uri, frames[0], **kwargs)
        print(f"Image saved to {uri}")


def main(args: ConfigDict, device=torch.device('cuda')):
    scene_bank, *_ = load_scenes_and_assets(**args, device=device)

    # ---------------------------------------------
    # ------     Scene Bank Dataset     -----------
    # ---------------------------------------------
    dataset_impl: SceneDataset = import_str(args.dataset_cfg.target)(args.dataset_cfg.param)
    if args.no_gt:
        args.training.dataloader.tags = {}
    args.training.dataloader.preload = False
    scene_dataloader = SceneDataLoader(scene_bank, dataset_impl, config=args.training.val_dataloader,
                                    device=device)
    scene_dataloader.set_camera_downscale(args.downscale)

    # ---------------------------------------------
    # ------------       Anim       ---------------
    # ---------------------------------------------
    anim = create_anim(args.anim, scene_bank[0]) if args.anim else None

    # ---------------------------------------------
    # ------------     Renderer     ---------------
    # ---------------------------------------------
    renderer = BufferComposeRenderer(args.renderer).eval()
    renderer.eval()
    renderer.config.rayschunk = args.rayschunk
    renderer.config.with_normal = True

    scene = scene_bank[0]
    bg_obj = scene.get_drawable_groups_by_class_name(args.bg_class_name)[0]
    bg_model = bg_obj.model
    if args.forward_inv_s is not None:
        bg_model.ray_query_cfg.forward_inv_s = args.forward_inv_s

    ################################

    if args.no_coarse_sampling:
        bg_model.ray_query_cfg.sample_cfg.should_sample_coarse = False
    if args.visible_grid:
        visible_grid = VisibleGrid.load(args.visible_grid, bg_model.space)
        visible_grid.build_accel()
        # visible_grid.postprocess()
        bg_model.accel = visible_grid.accel

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

    # ---------------------------------------------
    # --------------     Plot     -----------------
    # ---------------------------------------------
    expname = os.path.split(args.exp_dir.rstrip("/"))[-1]
    name = f"{args.outbase}_ds={args.downscale}"
    if args.anim:
        name = f"{args.outbase}_{os.path.splitext(os.path.split(args.anim)[-1])[0]}_ds={args.downscale}"
    vid_root = os.path.join(args.exp_dir, args.dirname)
    cond_mkdir(vid_root)
    if args.saveraw:
        vid_raw_root = os.path.join(vid_root, name)
        cond_mkdir(vid_raw_root)

    if args.num_frames is not None:
        num_frames = args.num_frames
        args.stop_frame = args.start_frame + num_frames
        print(f"=> args.stop_frame is set to {args.stop_frame}")
    else:
        if args.stop_frame is None:
            args.stop_frame = max(
                len(scene),
                (len(anim.clip_range) + anim.start_at_scene_frame) if anim else 0) - 1
        num_frames = max(args.stop_frame - args.start_frame, 1)

    instance_id_map = scene.get_drawable_instance_ind_map()
    instance_cmap = np.array(get_n_ind_colors(len(instance_id_map)))

    classname_map = scene.get_drawable_class_ind_map()
    class_cmap = np.array(get_n_ind_colors(len(classname_map)))

    draw_box_fn = functools.partial(
        draw_box, thickness=(2 if args.downscale == 1 else 1),
        fontscale=(1. if args.downscale == 1 else 0.5),
        inplace=True, use_class_cmap=False,
        instance_id_map=instance_id_map, instance_cmap=instance_cmap,
        classname_map=classname_map, class_cmap=class_cmap)
    draw_box_ids = args.draw_anno
    draw_cmap_inds = args.anno_color

    def draw_anno_boxes(img):
        if not draw_box_ids:
            return
        for obj in scene.get_drawables(True):
            draw = draw_box_ids == ["all"] or \
                (i := next((i for i, id in enumerate(draw_box_ids) if obj.id.endswith(id)), -1)) >= 0
            if draw and obj.id in instance_id_map.keys() and obj.class_name in classname_map.keys():
                draw_box_fn(img, obj, observer, oind=draw_cmap_inds and draw_cmap_inds[i])

    all_gather = defaultdict(lambda: defaultdict(list))
    if not args.no_gt and not args.gt_only:
        all_psnr = defaultdict(list)

    with torch.no_grad():
        if args.render_lidar:
            assert cam_ref_id is not None and cam_ref_id in scene.observers, \
                f"A valid frontal reference camera is required.\nCurrent camera list={scene.get_observer_groups_by_class_name('Camera', False)}"
            
            from mayavi import mlab
            from code_multi.tools.utils import plot_box3d
            plot_box3d_fn = functools.partial(plot_box3d, use_class_cmap=False,
                                              instance_id_map=instance_id_map,
                                              instance_cmap=instance_cmap,
                                              classname_map=classname_map,
                                              class_cmap=class_cmap)

            def plot_anno_boxes3d(visible_ids, relative_to_node):
                if not draw_box_ids:
                    return
                for obj in scene.get_drawables(True):
                    if (visible_ids and obj.id not in visible_ids) or \
                            obj.id not in instance_id_map.keys() or \
                            obj.class_name not in classname_map.keys():
                        continue
                    if draw_box_ids != ["all"] and \
                            (i := next((i for i, id in enumerate(draw_box_ids) if obj.id.endswith(id)), -1)) < 0:
                        continue
                    plot_box3d_fn(obj, relative_to_node, oind=draw_cmap_inds and draw_cmap_inds[i])


            scene.slice_at(args.start_frame)
            cam0: Camera = scene.observers[cam_ref_id]
            if args.lidar_model == 'original' or args.lidar_model == 'original_reren':
                lidar = scene.observers[args.lidar_id]
            else:
                # Create new lidar to be simulated and make it a child of cam0
                lidar = Lidar('sim_lidar', lidar_model=args.lidar_model, lidar_name=args.lidar_id, near=0.3, far=120.0, scene=scene).to(device=device)
                scene.add_node(lidar, parent=cam0)

            W_lidar_vis = args.lidar_vis_width if args.no_cam else \
                min(cam0.intr.W // args.downcale * 4, args.lidar_vis_width)
            H_lidar_vis = W_lidar_vis * 9 // 16
            if not DEBUG_LIDAR:
                mlab.options.offscreen = True
                fig = mlab.figure(bgcolor=(0, 0, 0), size=(W_lidar_vis, H_lidar_vis))
            # # Set pcl viewer camera offset
            # # TODO: Replace all places in the current code where the Waymo coordinate system setting is hard-coded, with the more scientific form below.
            # # NOTE: up_vec in VTK  ==>  [+z] in VTK  ==>  [+z] in waymo
            # pcl_cam_up_vec = dataset.dataset_impl.up_vec
            # pcl_cam_forward_vec = dataset.dataset_impl.forward_vec
            # # pcl_cam_offset = dataset.dataset_impl.up_vec * 10.0 \
            # #     - dataset.dataset_impl.forward_vec * 5.0 \
            # #     + dataset.dataset_impl.right_vec * 5.0
            # pcl_cam_offset = dataset.dataset_impl.up_vec * 10.0
            # cx, cy, cz = pcl_cam_offset
            # pcl_cam_distance = np.linalg.norm(pcl_cam_offset)
            # pcl_cam_azimuth = np.arctan2(-cx, cz) / np.pi * 180
            # pcl_cam_elevation = np.arccos(-cy/pcl_cam_distance) / np.pi * 180
            pcl_imgs1 = []
            pcl_imgs2 = []

            def render_pcl(scene: Scene, frame_ind: int):
                nonlocal fig
                cam0: Camera = scene.observers[cam_ref_id]

                if args.lidar_model == 'original' or args.lidar_model == 'original_reren':
                    scene.slice_at(frame_ind)
                    ldr_gt = scene_dataloader.get_lidar_gts(scene.id, args.lidar_id, frame_ind, device=device)
                    ldr_rays_o, ldr_rays_d, ldr_z = itemgetter("rays_o", "rays_d", "ranges")(ldr_gt)
                    valid = ldr_z > 0
                    # Lidar pts in world
                    ldr_pts = lidar.world_transform(
                        ldr_rays_o[valid] + ldr_rays_d[valid] * ldr_z[valid][..., None])
                    visible_ids = None
                else:
                    (anim or scene).slice_at(frame_ind)
                    # lidar.transform = cam0.world_transform
                    # lidar.world_transform = lidar.transform
                    # TODO: We should use lidar here, not camera_0! We need to make rendering_before_per_view more compatible to accommodate lidar sensor parameters.
                    asset_bank.rendering_before_per_view(renderer=renderer, observer=cam0, scene_id=scene.id)
                    # Lidar rays in world
                    ldr_rays_o, ldr_rays_d, ldr_rays_ts = lidar.get_all_rays(return_ts=True)
                    # render_fn = functools.partial(renderer.render_rays, scene=scene, render_per_obj_individual=True,
                    #                               with_rgb=False, with_normal=False, far=depth_max)
                    # ret_lidar = batchify_query(render_fn, ldr_rays_o, ldr_rays_d,
                    #                            chunk=args.rayschunk, show_progress=args.progress)
                    ret_lidar = renderer.render(
                        scene, rays=(ldr_rays_o, ldr_rays_d, ldr_rays_ts), far=depth_max, 
                        with_rgb=False, with_normal=False, rayschunk=args.rayschunk, show_progress=args.progress)  
                    visible_ids = [
                        key for key, val in ret_lidar["rendered_per_obj"].items()
                        if val["mask_volume"].count_nonzero() > 0
                    ]
                    ldr_rays_acc = ret_lidar['rendered']['mask_volume']
                    ldr_rays_z = ret_lidar['rendered']['depth_volume']
                    valid = ldr_rays_acc > 0.95
                    # Lidar pts in world
                    ldr_pts = ldr_rays_o[valid] + ldr_rays_d[valid] * ldr_rays_z[valid][..., None]


                if args.lidar_vis_view:
                    lidar_vis_view = args.lidar_vis_view.split("@")
                    id = lidar_vis_view[0]
                    if len(lidar_vis_view) > 1:
                        azimuth, elevation = [float(val) for val in lidar_vis_view[1].split(",")]
                    else:
                        azimuth, elevation = 135.0, 60.0
                    node = next(node for node in scene.all_nodes if node.id.endswith(id))
                    ldr_pts_in_cam = cam0.world_transform(ldr_pts, inv=True).data.cpu().numpy()
                    ldr_pts = node.world_transform(ldr_pts, inv=True).data.cpu().numpy()
                    mlab.clf()
                    mlab.points3d(ldr_pts[..., 0], ldr_pts[..., 1], ldr_pts[..., 2],
                                  -ldr_pts_in_cam[..., 1], mode="point", colormap='rainbow',
                                  vmin=args.lidar_vis_vmin, vmax=args.lidar_vis_vmax, figure=fig)

                    plot_anno_boxes3d(visible_ids, node)
                    # Top view
                    mlab.view(focalpoint=np.array([0., 0., 0.]), azimuth=0.0, elevation=0.0,
                              distance=30.0)
                    fig.scene._lift()
                    im1 = mlab.screenshot(figure=fig, mode='rgb', antialiased=True)
                    # Front slope view
                    mlab.view(focalpoint=np.array([0., 0., 0.]), azimuth=azimuth,
                              elevation=elevation, distance=10.0)
                    fig.scene._lift()
                    im2 = mlab.screenshot(figure=fig, mode='rgb', antialiased=True)
                    return im1, im2
                else:
                    # NOTE: Convert to a common coordinate system (OpenCV pinhole camera in this case)
                    ldr_pts = cam0.world_transform(ldr_pts, inv=True).data.cpu().numpy()
                    mlab.clf()
                    mlab.points3d(ldr_pts[..., 0], ldr_pts[..., 1], ldr_pts[..., 2], -ldr_pts[..., 1],
                                mode="point", colormap='rainbow', vmin=args.lidar_vis_vmin, vmax=args.lidar_vis_vmax,
                                figure=fig)

                    plot_anno_boxes3d(visible_ids, cam0)

                    # Top view
                    mlab.view(focalpoint=np.array([0., 0., 15.]), azimuth=90.0, elevation=-90.0,
                              distance=100.0, roll=-90.0)
                    fig.scene._lift()
                    im1 = mlab.screenshot(figure=fig, mode='rgb', antialiased=True)
                    # Front slope view
                    mlab.view(focalpoint=np.array([0., 0., 50.]), azimuth=-90.0, elevation=176.0,
                              distance=70.0, roll=179.0)
                    fig.scene._lift()
                    im2 = mlab.screenshot(figure=fig, mode='rgb', antialiased=True)
                    return im1, im2

        log.info(f"Start [render_anim], ds={args.downscale} in {args.exp_dir}")
        for frame_ind in tqdm(range(args.start_frame, args.stop_frame, 1), "rendering frames..."):
            if args.render_lidar:
                im1, im2 = render_pcl(scene, frame_ind)
                if not DEBUG_LIDAR:
                    pcl_imgs1.append(im1)
                    pcl_imgs2.append(im2)
                if args.saveraw:
                    k_dir = f"{vid_raw_root}/{args.lidar_model}_{args.lidar_id}/pcl1"
                    os.makedirs(k_dir, exist_ok=True)
                    imageio.imwrite(os.path.join(k_dir, f"{frame_ind:08d}.png"), im1)
                    k_dir = f"{vid_raw_root}/{args.lidar_model}_{args.lidar_id}/pcl2"
                    os.makedirs(k_dir, exist_ok=True)
                    imageio.imwrite(os.path.join(k_dir, f"{frame_ind:08d}.png"), im2)

            if args.no_cam:
                continue

            (anim or scene).slice_at(frame_ind)
            for cam_id in cam_id_list:
                observer: Camera = scene.observers[cam_id]
                observer.intr.set_downscale(args.downscale)

                def to_img(tensor: torch.Tensor, rgb=False):
                    if len(tensor.shape) == 1:
                        tensor = tensor.reshape(observer.intr.H, observer.intr.W)
                    elif tensor.shape[0] != observer.intr.H:
                        tensor = tensor.reshape(observer.intr.H, observer.intr.W, -1)
                    img = np.ascontiguousarray(tensor.cpu().numpy())
                    if rgb:
                        img = np.clip((img * 255.).astype(np.uint8), 0, 255)
                    return img

                def to_depth_img(mask_volume, depth_volume):
                    mask = to_img(mask_volume)
                    depth = to_img(depth_volume)
                    depth = mask * np.clip(depth / depth_max, 0, 1) + 1 - mask
                    return color_depth(depth, scale=1, cmap='turbo')    # turbo_r, viridis, rainbow

                def to_normal_img(normals_volume):
                    return to_img(normals_volume / 2. + .5, True)
                
                cur_frame_dict = defaultdict(list)

                # Rendered
                if not args.gt_only:
                    ret = renderer.render(scene, observer=observer, render_per_obj_individual=True, show_progress=args.progress)
                    renderered = ret['rendered']
                    
                    # ---- All objects ----
                    cur_frame_dict['rgb_volume'] = to_img(renderered['rgb_volume'], True)
                    cur_frame_dict['depth_volume'] = to_depth_img(renderered["mask_volume"],
                                                                  renderered["depth_volume"])
                    cur_frame_dict['ins_seg_mask_buffer'] = np.take(
                        instance_cmap, axis=0, indices=to_img(ret['ins_seg_mask_buffer'])).astype(np.uint8)
                    if 'normals_volume' in renderered:
                        cur_frame_dict['normals_volume'] = to_normal_img(renderered['normals_volume'])
                    draw_anno_boxes(cur_frame_dict['rgb_volume'])

                    # ---- Background only ----
                    bg_rendered = ret['rendered_per_obj'][bg_obj.id]
                    cur_frame_dict['bgrgb_volume'] = to_img(bg_rendered['rgb_volume'], True)
                    cur_frame_dict['bgdepth_volume'] = to_depth_img(bg_rendered["mask_volume"],
                                                                    bg_rendered["depth_volume"])
                    if 'normals_volume' in bg_rendered:
                        cur_frame_dict['bg_normals_volume'] = to_normal_img(bg_rendered['normals_volume'])

                # Gound truth
                if not args.no_gt:
                    ground_truth = scene_dataloader.get_image_and_gts(scene.id, observer.id, frame_ind)
                    cur_frame_dict['gt_rgb'] = to_img(ground_truth['image_rgb'], True)
                    scene.slice_at(frame_ind)
                    observer.intr.set_downscale(args.downscale)
                    draw_anno_boxes(cur_frame_dict['gt_rgb'])
                    if not args.gt_only:
                        all_psnr[cam_id].append(
                            PSNR(renderered['rgb_volume'],
                                 ground_truth['image_rgb'].to(device).view_as(renderered['rgb_volume']))
                        )

                for k, v in cur_frame_dict.items():
                    all_gather[observer.id][k].append(v)
                    if args.saveraw:
                        k_dir = os.path.join(vid_raw_root, cam_id, k)
                        os.makedirs(k_dir, exist_ok=True)
                        imageio.imwrite(os.path.join(k_dir, f"{frame_ind:08d}.png"), v)

    # --------- Seperate video
    if not args.no_cam and args.save_seperate_keys:
        for cam_id, obs_dict in all_gather.items():
            for k, frames in obs_dict.items():
                write_video(os.path.join(vid_root, f"{name}_{cam_id}_{k}.mp4"), frames, args.fps,
                            args.quality)

    # 2 rows
    if not args.no_cam and not args.gt_only:
        frames_per_obs_all = []
        for cam_id, obs_dict in all_gather.items():
            frames_per_obs = []
            for kvs in zip(*(obs_dict.values())):
                frames_per_obs.append(gallery(np.stack(kvs, 0), nrows=2))
            write_video(os.path.join(vid_root, f"{name}_{cam_id}.mp4"),
                        frames_per_obs, args.fps, args.quality)
            frames_per_obs_all.append(np.array(frames_per_obs))

    if args.render_lidar and not DEBUG_LIDAR:
        if not args.no_cam:
            *_, H, W, _ = frames_per_obs_all[0].shape
            H = int(H_lidar_vis * W / W_lidar_vis)
            frames_pcl_1 = np.clip((np.array([resize(im, (H, W))
                                   for im in pcl_imgs1]) * 255), 0, 255).astype(np.uint8)
            frames_pcl_2 = np.clip((np.array([resize(im, (H, W))
                                   for im in pcl_imgs2]) * 255), 0, 255).astype(np.uint8)
        else:
            frames_per_obs_all = []
            frames_pcl_1 = np.array(pcl_imgs1)
            frames_pcl_2 = np.array(pcl_imgs2)

        write_video(os.path.join(vid_root, f"{name}_{args.lidar_model}_{args.lidar_id}_pcl1.mp4"),
                    frames_pcl_1, args.fps, args.quality)
        write_video(os.path.join(vid_root, f"{name}_{args.lidar_model}_{args.lidar_id}_pcl2.mp4"),
                    frames_pcl_2, args.fps, args.quality)

        frames_per_obs_all.append(frames_pcl_1)
        frames_per_obs_all.append(frames_pcl_2)
        write_video(os.path.join(vid_root, f"{name}_{args.lidar_model}_{args.lidar_id}_all_with_pcl.mp4"),
                    np.concatenate(frames_per_obs_all, axis=1), args.fps, args.quality)

    # 1 row
    if not args.no_cam and not args.gt_only:
        keys_1l = [
            'gt_rgb', 'rgb_volume', 'bgrgb_volume', 'depth_volume', 'bgdepth_volume',
            'ins_seg_mask_buffer', 'normals_volume', 'bg_normals_volume'
        ]
        frames_per_obs_1l_all = []
        for cam_id, obs_dict in all_gather.items():
            frames_per_obs_1l = []
            new_obs_dict = dict()
            for k in keys_1l:
                if k in obs_dict:
                    new_obs_dict[k] = obs_dict[k]
            for kvs in zip(*(new_obs_dict.values())):
                frames_per_obs_1l.append(np.concatenate(kvs, axis=1))
            write_video(os.path.join(vid_root, f"{name}_{cam_id}_1l.mp4"),
                        frames_per_obs_1l, args.fps, args.quality)
            frames_per_obs_1l_all.append(np.array(frames_per_obs_1l))

        if not args.no_gt:
            psnr_f = os.path.join(vid_root, f'{name}_psnr.txt')
            with open(psnr_f, 'w') as f:
                for cam_id, vals in all_psnr.items():
                    f.write(f"{cam_id}: \n")
                    f.writelines([f"{v:.4f}\n" for v in vals])
                    f.write("\n")
            print(f"PSNR saved to {psnr_f}")


if __name__ == "__main__":
    from nr3d_lib.config import BaseConfig
    bc = BaseConfig()
    bc.parser.add_argument("--bg_class_name", type=str, default='Street', help="The class_name of the background object.")
    bc.parser.add_argument("--progress", action='store_true', help="If set, shows per frame progress.")
    bc.parser.add_argument("--no_gt", action='store_true')
    bc.parser.add_argument("--gt_only", action='store_true')
    bc.parser.add_argument("--saveraw", action='store_true')
    bc.parser.add_argument("--load_pt", type=str, default=None, help="Typically unnecessary as the final or latest ckpt is loaded automatically. \n"\
        "Only specify the ckpt file path if indeed a non-final or non-latest ckpt needs to be loaded.")
    bc.parser.add_argument("--downscale", type=float, default=1.0, help="Sets the side length downscale for rendering and output.")
    bc.parser.add_argument("--fps", type=int, default=24)
    bc.parser.add_argument("--quality", type=int, default=None, help="Sets the quality for imageio.mimwrite (range: 0-10; 10 is the highest; default is 5).")
    bc.parser.add_argument(
        "--outbase", type=str, default=datetime.now().strftime("%Y_%m_%d_%H_%M_%S"), help="Sets the basename of the output file (without extension).")
    bc.parser.add_argument("--rayschunk", type=int, default=4096)
    bc.parser.add_argument("--dirname", type=str, default="videos", help="Sets the output directory to /path/to/exp_dir/${dirname}`.")
    
    bc.parser.add_argument("--no_cam", action='store_true', help='skip all camera rendering')
    bc.parser.add_argument("--render_lidar", action='store_true',
                           help='render lidar pointclouds')
    bc.parser.add_argument("--lidar_model", type=str, default="dummy", help='lidar model.')
    bc.parser.add_argument("--lidar_id", type=str, default="", help="Specifies the lidar name")
    bc.parser.add_argument("--lidar_vis_vmin", type=float, default=-2.)
    bc.parser.add_argument("--lidar_vis_vmax", type=float, default=9.)
    bc.parser.add_argument("--lidar_vis_width", type=int, default=1920,
                           help="Width of lidar visualization image.")
    bc.parser.add_argument("--lidar_vis_view", type=str)
    bc.parser.add_argument("--cam_id", type=str, default=None,
                           help="If set, uses a specific camera; otherwise, uses all available cameras.")
    bc.parser.add_argument("--cam_ref", type=str, default='camera_FRONT', help="Reference camera for visulization of LiDAR, mesh, etc.")
    bc.parser.add_argument('--start_frame', type=int, default=0)
    bc.parser.add_argument("--num_frames", type=int, default=None)
    bc.parser.add_argument('--stop_frame', type=int, default=None)
    bc.parser.add_argument("--anim", type=str)
    bc.parser.add_argument("--visible_grid", type=str)
    bc.parser.add_argument("--no_coarse_sampling", action='store_true')
    bc.parser.add_argument("--forward_inv_s", type=float, default=None)
    bc.parser.add_argument("--save_seperate_keys", action='store_true')
    bc.parser.add_argument("--draw_anno", type=str, nargs="+",
                           help="List of object ids to draw annotation boxes, or \"all\" to draw for all objects")
    bc.parser.add_argument("--anno_color", type=int, nargs="+",
                           help="List of color indices to draw annotation boxes, use the same colors as segmentation mask if not specified")
    main(bc.parse())
