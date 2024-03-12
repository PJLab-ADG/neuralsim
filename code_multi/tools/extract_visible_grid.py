"""
@file   extract_visible_grid.py
@author Nianchen Deng, Shanghai AI Lab
@brief  Extract a visible grid for background from sequences of observers in scene.
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
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
from operator import itemgetter
from tqdm import trange
from collections import defaultdict

from nr3d_lib.fmt import log
from nr3d_lib.config import ConfigDict
from nr3d_lib.models.attributes import *
from nr3d_lib.models.spatial import AABBSpace, ForestBlockSpace
from nr3d_lib.plot import create_camera_frustum_o3d

from app.resources import Scene, load_scenes_and_assets
from app.resources.observers import Lidar, Camera
from app.visible_grid import VisibleGrid
from app.renderers import BufferComposeRenderer
from app.visualizer.utils import initialize_visualizer, create_voxels_geometry
from app.visualizer.visualize_visible_grid import visualize_visible_grid


DEBUG_LIDAR = False
O3D_VISUALIZE = False


def main(args: ConfigDict, device: torch.device = torch.device('cuda')):
    scene_bank, asset_bank, *_ = load_scenes_and_assets(**args, class_name_list=[args.class_name])
    scene = scene_bank[0]
    bg_node = scene.get_drawable_groups_by_class_name(args.class_name)[0]
    space = bg_node.model.space

    args.stop_frame = args.start_frame + args.num_frames if args.num_frames else \
        max(args.stop_frame or len(scene), args.start_frame + 1)
    frame_ind = args.start_frame - 1

    # ---------------------------------------------
    # ------------     Renderer     ---------------
    # ---------------------------------------------
    renderer = BufferComposeRenderer(args.renderer)
    renderer.populate(asset_bank)
    renderer.eval()
    renderer.config.rayschunk = args.rayschunk
    renderer.config.with_normal = False
    renderer.config.with_rgb = O3D_VISUALIZE
    renderer.config.with_env = False
    renderer.config.forward_inv_s = args.forward_inv_s
    depth_max = args.depth_max or renderer.config.far
    for scene in scene_bank:
        for obs in scene.get_observers(False):
            obs.near = renderer.config.near
            obs.far = renderer.config.far

    # Infer a proper voxel size if not specified
    if not args.voxel_size:
        scene.frozen_at_global_frame(0)
        gap_between_ray = 0.
        for observer in scene.get_observer_groups_by_class_name("Camera"):
            observer.intr.set_downscale(args.downscale)
            gap_between_ray = max(gap_between_ray, (1. / observer.intr.focal()).norm().item())
        gap_between_ray_at_far_plane = gap_between_ray * renderer.config.far
        args.voxel_size = gap_between_ray_at_far_plane * 2.
        log.info(f"Infer voxel size: gap_between_ray={gap_between_ray}, far={renderer.config.far}, "
                 f"prefer_voxel_size={args.voxel_size}")
        scene.unfrozen()

    visible_grid = VisibleGrid(space, prefer_voxel_size=args.voxel_size)
    output_path = os.path.join(args.exp_dir, f'visible_grid_depth={visible_grid.octree_depth}.pt')

    @torch.no_grad()
    def next_frame():
        nonlocal frame_ind
        if frame_ind == args.stop_frame - 1:
            return False
        frame_ind += 1
        if args.render_lidar:
            scene.slice_at(args.start_frame)
            lidar = Lidar('test_lidar', lidar_model='livox horizon', near=0.3, far=120.0).to(device)
            cam0: Camera = scene.get_observer_groups_by_class_name("Camera")[0]
            scene.add_node(lidar, parent=cam0)
            cam0.intr.set_downscale(args.downscale)
            W_lidar_vis = min(cam0.intr.W * 4, 1920)  # 4 is the number of column when joint render
            H_lidar_vis = W_lidar_vis * 9 // 16
            from mayavi import mlab
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

            def render_pcl(scene: Scene):
                nonlocal fig
                with torch.no_grad():
                    cam0: Camera = scene.get_observer_groups_by_class_name("Camera")[0]
                    # lidar.transform = cam0.world_transform
                    # lidar.world_transform = lidar.transform
                    # TODO: Lidar should be used here, not camera_0! We need to make rendering_before_per_view more compatible, accommodating lidar sensor parameters.
                    asset_bank.rendering_before_per_view(renderer=renderer, observer=cam0, scene_id=scene.id)
                    lidar_rays_o, lidar_rays_d, lidar_rays_ts = lidar.get_all_rays(return_ts=True)
                    ret_lidar = renderer.render(
                        scene, rays=(lidar_rays_o,lidar_rays_d,lidar_rays_ts), far=depth_max, 
                        rayschunk=args.rayschunk, with_rgb=False, with_normal=False)
                    lidar_rays_acc = ret_lidar['rendered']['mask_volume']
                    lidar_rays_depth = ret_lidar['rendered']['depth_volume']
                    valid = lidar_rays_acc > 0.95
                    lidar_pts = lidar_rays_o[valid] + lidar_rays_d[valid] * \
                        lidar_rays_depth[valid].unsqueeze(-1)
                    # NOTE: Convert to a common coordinate system (OpenCV pinhole camera in this case)
                    lidar_pts = lidar.world_transform(lidar_pts, inv=True).data.cpu().numpy()
                    # lidar_mask = lidar_rays_acc[valid].data.cpu().numpy()
                    # lidar_depth = lidar_mask * np.clip(lidar_rays_depth[valid].data.cpu().numpy() / depth_max, 0, 1) + (1-lidar_mask) * 1

                    if not DEBUG_LIDAR:
                        mlab.clf()
                    else:
                        fig = mlab.figure(bgcolor=(0, 0, 0), size=(W_lidar_vis, H_lidar_vis))
                    # fig = mlab.figure(bgcolor=(0, 0, 0), size=(W_lidar, H_lidar))
                    # mlab.points3d(lidar_pts[...,0], lidar_pts[...,1], lidar_pts[...,2], lidar_pts[...,2], mode="point", colormap='rainbow', vmin=-2., vmax=9., figure=fig)
                    mlab.points3d(lidar_pts[..., 0], lidar_pts[..., 1], lidar_pts[..., 2], -
                                  lidar_pts[..., 1], mode="point", colormap='rainbow', vmin=-2., vmax=9., figure=fig)

                    # Top view
                    mlab.view(focalpoint=np.array([0., 0., 15.]), azimuth=90.0,
                              elevation=-90.0, distance=100.0, roll=-90.0)
                    # # Front slope view
                    # mlab.view(focalpoint=np.array([0., 0., 50.]), azimuth=-90.0, elevation=176.0, distance=70.0, roll=179.0)
                    # mlab.show()
                    if not DEBUG_LIDAR:
                        fig.scene._lift()
                        im1 = mlab.screenshot(figure=fig, mode='rgb', antialiased=True)
                        # # Top view
                        # mlab.view(focalpoint=np.array([0., 0., 15.]), azimuth=90.0, elevation=-90.0, distance=100.0, roll=-90.0)
                        # Front slope view
                        mlab.view(focalpoint=np.array(
                            [0., 0., 50.]), azimuth=-90.0, elevation=176.0, distance=70.0, roll=179.0)
                        fig.scene._lift()
                        im2 = mlab.screenshot(figure=fig, mode='rgb', antialiased=True)
                        return im1, im2
                    else:
                        mlab.show()
                        return None, None

        scene.slice_at(frame_ind)

        if args.render_lidar:
            im1, im2 = render_pcl(scene)
            if not DEBUG_LIDAR:
                pcl_imgs1.append(im1)
                pcl_imgs2.append(im2)

        frame_data = defaultdict(list)
        for observer in scene.get_observer_groups_by_class_name("Camera"):
            observer: Camera
            observer.intr.set_downscale(args.downscale)
            # ret: Dict[str, torch.Tensor] = renderer.render(observer, scene=scene,
            #                                               show_progress=args.progress,
            #                                               return_buffer=True)
            # rays_o, rays_d, vol_buf = itemgetter("rays_o", "rays_d", "volume_buffer")(ret)
            # rays_i, pack_infos, packed_t, packed_vw = itemgetter(
            #     "rays_inds_hit", "pack_infos_hit", "t", "vw_normalized")(vol_buf)
            # for offset in range(args.rayschunk, pack_infos.shape[0], args.rayschunk):
            #     pack_infos[offset:offset + args.rayschunk, 0] += pack_infos[offset - 1].sum()
            # index_selector = (packed_vw > 0.1).nonzero()[:, 0]
            # packed_t = packed_t[index_selector]
            # index_for_rays_i = torch.searchsorted(pack_infos[:, 0].contiguous(),
            #                                       index_selector, right=True) - 1
            # packed_rays_i = index_for_rays_i
            # pts = rays_o[packed_rays_i] + rays_d[packed_rays_i] * packed_t[:, None]
            # frame_data["pts"].append(pts)
            # if "rgb" in vol_buf:
            #     frame_data["rgb"].append(vol_buf["rgb"][index_selector])
            #-----------------------------------------
            asset_bank.rendering_before_per_view(renderer=renderer, observer=observer, scene_id=scene.id)
            drawables = observer.filter_drawable_groups(scene.get_drawables())
            rays_o, rays_d, rays_ts = observer.get_all_rays(return_ts=True)
            for i in trange(0, rays_o.shape[0], args.rayschunk, disable=not args.progress):
                index = slice(i, i + args.rayschunk)
                chunk_rays_o = rays_o[index]
                chunk_rays_d = rays_d[index]
                chunk_rays_ts = rays_ts[index]
                chunk_ret = renderer.forward(chunk_rays_o, chunk_rays_d, chunk_rays_ts, 
                                             scene=scene, drawables=drawables, 
                                             near=observer.near, far=observer.far, return_buffer=True)
                pack_infos, packed_t, packed_vw = itemgetter(
                    "pack_infos_hit", "t", "vw_normalized")(chunk_ret["volume_buffer"])
                packed_rgb = chunk_ret["volume_buffer"].get("rgb")
                del chunk_ret
                
                index_selector = (packed_vw > 0.1).nonzero()[:, 0]
                packed_t = packed_t[index_selector]
                index_for_rays_i = torch.searchsorted(pack_infos[:, 0].contiguous(),
                                                      index_selector, right=True) - 1
                packed_rays_i = index_for_rays_i
                pts = chunk_rays_o[packed_rays_i] + chunk_rays_d[packed_rays_i] * packed_t[:, None]
                frame_data["pts"].append(pts)
                if packed_rgb is not None:
                    frame_data["rgb"].append(packed_rgb[index_selector])

            #-----------------------------------------

        frame_data = {key: torch.cat(value, 0) for key, value in frame_data.items()}

        frame_voxels_in_block = visible_grid.reduce_points_and_add(frame_data["pts"])[0]

        if O3D_VISUALIZE:
            for observer in scene.get_observer_groups_by_class_name("Camera", False):
                for geo_name in geos[observer.id]:
                    visualizer.widget3d.scene.show_geometry(geo_name, observer.i_valid)
                    visualizer.widget3d.scene.set_geometry_transform(
                        geo_name, observer.world_transform.mat_4x4().cpu().numpy())
            geo_pcl = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(frame_data["pts"].cpu().numpy()))
            geo_pcl.colors = o3d.utility.Vector3dVector(frame_data["rgb"].cpu().numpy())
            add_geometry(scene.root, f"pcl_{frame_ind}", geo_pcl, unlit_mat)

            if isinstance(space, ForestBlockSpace):
                for blidx, voxel_indices in frame_voxels_in_block.items():
                    voxel_aabb = visible_grid.get_voxel_aabb_in_world(voxel_indices, blidx)
                    geo_voxels = create_voxels_geometry(*voxel_aabb, [0.3, 0.7, 1.0], mesh=True)
                    visualizer.widget3d.scene.add_geometry(
                        f"{args.class_name}.voxels_{blidx}_{frame_ind}", geo_voxels, lit_mat)

        del frame_data
        torch.cuda.empty_cache()
        return True

    if O3D_VISUALIZE:
        visualizer = initialize_visualizer("Extract visible grid", has_sidebar=True)
        gui_next_btn = gui.Button("Next")
        gui_next_btn.set_on_clicked(lambda: next_frame())
        visualizer.sidebar.add_child(gui_next_btn)
        line_mat = o3d.visualization.rendering.MaterialRecord()
        line_mat.line_width = 1.5
        line_mat.shader = "unlitLine"
        unlit_mat = o3d.visualization.rendering.MaterialRecord()
        unlit_mat.point_size = 1.5
        unlit_mat.shader = "defaultUnlit"
        lit_mat = o3d.visualization.rendering.MaterialRecord()
        lit_mat.shader = "defaultLit"

        geos = defaultdict(dict)

        def add_geometry(node, name: str, geometry, material):
            visualizer.widget3d.scene.add_geometry(name, geometry, material)
            if node.i_valid:
                visualizer.widget3d.scene.set_geometry_transform(
                    name, node.world_transform.mat_4x4().cpu().numpy())
            else:
                visualizer.widget3d.scene.show_geometry(name, False)
            geos[node.id][name] = geometry

        scene.frozen_at_global_frame(0)
        for camera in scene.get_observer_groups_by_class_name("Camera", False):
            geometry_camera = create_camera_frustum_o3d(
                img_wh=(camera.intr.W, camera.intr.H),
                intr=camera.intr.mat_4x4().cpu().numpy(),
                c2w=np.eye(4), frustum_length=camera.far, color=[0, 0, 1])
            add_geometry(camera, f"{camera.id}.frustum", geometry_camera, line_mat)
        scene.unfrozen()

        if isinstance(space, AABBSpace):
            num_blocks = 1
        elif isinstance(space, ForestBlockSpace):
            num_blocks = space.block_ks.shape[0]
        
        for i in range(num_blocks):
            grid_aabb = visible_grid.get_grid_aabb_in_world(i)
            grid_aabb = [item.cpu().numpy() for item in grid_aabb]
            geo_grid_aabb = o3d.geometry.AxisAlignedBoundingBox(*grid_aabb)
            geo_grid_aabb.color = [0.0, 1.0, 0.5]
            visualizer.widget3d.scene.add_geometry(f"{args.class_name}.grid_{i}", geo_grid_aabb, line_mat)

        bg_aabb = [item.cpu().numpy() for item in space.aabb]
        geo_bg_aabb = o3d.geometry.AxisAlignedBoundingBox(*bg_aabb)
        geo_bg_aabb.color = [1.0, 0.5, 0.0]
        visualizer.widget3d.scene.add_geometry(f"{args.class_name}.aabb", geo_bg_aabb, line_mat)
        visualizer.widget3d.setup_camera(60.0, geo_bg_aabb, geo_bg_aabb.get_center())

        visualizer.app.run()
    else:
        for _ in trange(args.stop_frame - args.start_frame):
            next_frame()

        visible_grid.reduce_voxels()
        visible_grid.save(output_path)

        visualize_visible_grid(visible_grid, "Extract visible grid")


if __name__ == "__main__":
    from nr3d_lib.config import BaseConfig

    bc = BaseConfig()
    bc.parser.add_argument("--class_name", type=str, default='Street', help="The class_name of the object you want to operate with.")
    bc.parser.add_argument("--progress", action='store_true', help="If set, shows per frame progress.")
    bc.parser.add_argument("--load_pt", type=str, default=None, help="Typically unnecessary as the final or latest ckpt is loaded automatically. \n"\
        "Only specify the ckpt file path if indeed a non-final or non-latest ckpt needs to be loaded.")
    bc.parser.add_argument("--downscale", type=float, default=1.0, help="Sets the side length downscale for rendering and output.")
    bc.parser.add_argument("--rayschunk", type=int, default=4096)
    bc.parser.add_argument("--forward_inv_s", type=float, default=64000)
    bc.parser.add_argument("--voxel_size", type=float)
    bc.parser.add_argument("--render_lidar", action='store_true', help='render lidar pointclouds')

    bc.parser.add_argument('--start_frame', type=int, default=0)
    bc.parser.add_argument("--num_frames", type=int)
    bc.parser.add_argument('--stop_frame', type=int)
    main(bc.parse())
