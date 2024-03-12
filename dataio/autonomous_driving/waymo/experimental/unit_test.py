from typing import Literal
import numpy as np
from scipy.spatial.transform import Rotation as R

from nr3d_lib.config import ConfigDict

from dataio.scene_dataset import SceneDataset
from dataio.autonomous_driving.waymo.waymo_dataset import WaymoDataset

if __name__ == "__main__":
    def make_unit_test_dataloader(
        device, 
        preload=False, 
        use_ts_interp=False, 
        with_mono_cues=False):
        from nr3d_lib.utils import import_str
        from nr3d_lib.config import ConfigDict
        from app.resources import create_scene_bank
        from dataio.data_loader import SceneDataLoader
        from app.resources.asset_bank import AssetBank
        dataset_cfg = ConfigDict(
            target='dataio.autonomous_driving.WaymoDataset', 
            param=ConfigDict(
                root='/data1/waymo/processed', 
                rgb_dirname="images", 
                lidar_dirname="lidars", 
                # lidar_dirname="lidars_ds=4", 
                mask_dirname="masks", 
            )
        )
        scenebank_cfg = ConfigDict(
            # scenarios=['segment-7670103006580549715_360_000_380_000_with_camera_labels, 15'], 
            # scenarios=['segment-16646360389507147817_3320_000_3340_000_with_camera_labels'], 
            # scenarios=['segment-10061305430875486848_1080_000_1100_000_with_camera_labels, 0, 163'], 
            scenarios=['segment-9385013624094020582_2547_650_2567_650_with_camera_labels'], 
            observer_cfgs=ConfigDict(
                Camera=ConfigDict(
                    list=['camera_FRONT', 'camera_FRONT_LEFT', 'camera_FRONT_RIGHT', 'camera_SIDE_LEFT', 'camera_SIDE_RIGHT']
                ), 
                RaysLidar=ConfigDict(
                    list=['lidar_TOP', 'lidar_FRONT', 'lidar_REAR', 'lidar_SIDE_LEFT', 'lidar_SIDE_RIGHT']
                    # list=['lidar_TOP']
                    # list=['lidar_FRONT', 'lidar_REAR', 'lidar_SIDE_LEFT', 'lidar_SIDE_RIGHT']
                ), 
            ), 
            object_cfgs=ConfigDict(
                Vehicle=ConfigDict(
                    dynamic_only=True
                ),
                Pedestrian=ConfigDict(
                    dynamic_only=True
                )
            ), 
            no_objects=False, 
            align_orientation=False, 
            aabb_extend=120., 
            consider_distortion=True, 
            scene_graph_has_ego_car=True,
            how_to_account_for_cam_timestamp_diff='use_ts_interp' \
                if use_ts_interp else 'correct_extrinsics',
            use_ts_interp=use_ts_interp, 
        )
        assetbank_cfg = ConfigDict({
            'Vehicle': {'model_class': 'app.models.base.AD_DummyBox', 'model_params': {}}, 
            'Pedestrian': {'model_class': 'app.models.base.AD_DummyBox', 'model_params': {}}, 
            'Street': {'model_class': 'app.models.base.DummyBox', 'model_params': {}}, 
            # 'Distant': {}
        })
        
        dataset_impl: SceneDataset = import_str(dataset_cfg.target)(dataset_cfg.param)
        scene_bank, scene_bank_meta = create_scene_bank(
            dataset=dataset_impl, device=device, 
            scenebank_root=None,
            scenebank_cfg=scenebank_cfg, 
            drawable_class_names=assetbank_cfg.keys(), 
            misc_node_class_names=['node', 'EgoVehicle', 'EgoDrone'], 
        )
        scene = scene_bank[0]
        cams = scene.get_observer_groups_by_class_name('Camera')
        # scene.debug_vis_multi_frame(40)
        
        scene_loader_config = ConfigDict(
            preload=preload, 
            tags=ConfigDict(
                camera=ConfigDict(
                    downscale=1, 
                    list=scenebank_cfg.observer_cfgs.Camera.list
                ), 
                lidar=ConfigDict(
                    list=scenebank_cfg.observer_cfgs.RaysLidar.list, 
                    multi_lidar_merge=True, 
                    filter_kwargs=ConfigDict(
                        filter_valid=True, 
                        # filter_in_cams=True, 
                        filter_out_objs=False
                    )
                )
            )
        )
        if with_mono_cues:
            scene_loader_config['tags']['image_mono_depth'] = {}
            scene_loader_config['tags']['image_mono_normals'] = {}
        scene_dataloader = SceneDataLoader(scene_bank, dataset_impl, device=device, config=scene_loader_config)
        asset_bank = AssetBank(assetbank_cfg)
        asset_bank.create_asset_bank(scene_bank, load_assets_into_scene=True, device=device)
        scene.load_assets(asset_bank)
        return scene_bank, scene_dataloader

    def test_scenegraph():
        import torch
        from icecream import ic
        device = torch.device('cuda')
        scene_bank, scene_dataloader = make_unit_test_dataloader(device, preload=False)
        scene = scene_bank[0]
        # scene.debug_vis_scene_graph(120, arrow_length=0.5, font_scale=1.0)
        scene.debug_vis_scene_graph(120)

    def test_scenario():
        import torch
        from icecream import ic
        device = torch.device('cuda')
        scene_bank, scene_bank_loader = make_unit_test_dataloader(device, preload=False)
        scene = scene_bank[0]
        scene.frozen_at_full_global_frame()
        # mesh_file = '/data2/lidar_only/exp1_lidaronly_5lidar_filterobj_dynamic_ext80.0_7.5k_all_wli=0.03/seg7670103/meshes/seg7670103_exp1_wli0.03_res=0.1.ply'
        mesh_file = "logs/streetsurf/seg100613.withmask_withlidar_ddp/meshes/segment-10061305430875486848_1080_000_1100_000_with_camera_labels#Street#street_level=0.0_res=0.2.ply"
        scene.debug_vis_anim(
            scene_dataloader=scene_bank_loader, 
            plot_image=True, camera_length=8., 
            plot_lidar=True, lidar_pts_downsample=2, 
            mesh_file=mesh_file, 
        )

    def test_lidar():
        import torch
        from nr3d_lib.plot import vis_lidar_vedo
        from app.resources.observers.lidars import MultiRaysLidarBundle
        device = torch.device('cuda')
        scene_bank, scene_bank_loader = make_unit_test_dataloader(device, preload=False)
        scene = scene_bank[0]
        
        frame_ind = 41
        lidar_gts = scene_bank_loader.get_merged_lidar_gts(scene.id, frame_ind, device=device, filter_if_configured=True)
        scene.slice_at(frame_ind)
        lidars = scene.all_nodes_by_class_name['RaysLidar'][scene_bank_loader.lidar_id_list]
        lidars = MultiRaysLidarBundle(lidars)
        l2w = lidars.world_transform[lidar_gts['li']]
        pts_local = torch.addcmul(lidar_gts['rays_o'], lidar_gts['rays_d'], lidar_gts['ranges'].unsqueeze(-1))
        pts = l2w.forward(pts_local)
        pts = pts.data.cpu().numpy()
        # vis_lidar_o3d(pts)
        vis_lidar_vedo(pts)

    def test_distort():
        import torch
        import imageio
        from nr3d_lib.utils import check_to_torch, cond_mkdir
        from kornia.geometry.calibration.undistort import undistort_image
        device = torch.device('cuda')
        config = ConfigDict(
            root='/data1/waymo/processed/', 
            rgb_dirname='images', 
        )
        dataset = WaymoDataset(config)
        
        scene_id = 'segment-7670103006580549715_360_000_380_000_with_camera_labels'
        cam_id = 'camera_FRONT'
        # frame_ind = 83
        frame_ind = 159
        scenario = dataset.get_scenario(scene_id)
        odict = scenario['observers']['ego_car']['children'][cam_id]
        K = check_to_torch(odict['data']['intr'][frame_ind], device=device, dtype=torch.float)
        dist = check_to_torch(odict['data']['distortion'][frame_ind], device=device, dtype=torch.float)
        rgb = dataset.get_image(scene_id, cam_id, frame_ind)
        # [H, W, C] -> [C, H, W]
        rgb_tensor = check_to_torch(rgb, device=device, dtype=torch.float).movedim(-1, 0)
        rgb0_tensor = undistort_image(rgb_tensor, K, dist)
        # {C, H, W} -> [H, W, C]
        rgb0 = rgb0_tensor.movedim(0, -1).contiguous().data.cpu().numpy()
        cond_mkdir('./dev_test/test_distortion')
        imageio.imwrite(f'./dev_test/test_distortion/frame_{frame_ind}.png', rgb)
        imageio.imwrite(f'./dev_test/test_distortion/frame_{frame_ind}_undist.png', rgb0)

    def test_mask():
        import torch
        import kornia
        device = torch.device('cuda')
        scene_bank, scene_bank_loader = make_unit_test_dataloader(device, preload=False)
        scene = scene_bank[0]
        
        mask = scene_bank_loader.get_image_occupancy_mask(scene.id, 'camera_FRONT', 45, device=device)
        gt = scene_bank_loader.get_image_and_metas(scene.id, 'camera_FRONT', 45, device=device)['image_rgb']
        
        from nr3d_lib.utils import img_to_torch_and_downscale
        mask = img_to_torch_and_downscale(mask, downscale=2)
        gt = img_to_torch_and_downscale(gt, downscale=2)
        
        import matplotlib.pyplot as plt
        mask_erode = 20 # 10 pixels of erosion
        mask_new = kornia.morphology.erosion(mask[None,None].float(), torch.ones([mask_erode,mask_erode], device=mask.device))[0,0].bool()
        mask = mask.data.cpu().numpy()
        mask_new = mask_new.data.cpu().numpy()
        im = np.zeros_like(mask, dtype=float)
        im[mask_new] += 0.4
        im[mask] += 0.6
        plt.imshow(im, vmin=0., vmax=1.)
        plt.show()
    
    def test_use_timestamp_interp():
        import torch
        from torch.utils.benchmark import Timer
        device = torch.device('cuda')
        
        #---- timestamp interpolation
        scene_bank, scene_bank_loader = make_unit_test_dataloader(device, preload=False, use_ts_interp=True)
        scene = scene_bank[0]
        cams = scene.get_cameras()
        for ci, c in enumerate(cams):
            ts = c.frame_global_ts[40]
            scene.interp_at(ts.item())
            print(c.id, ts.item(), scene.all_nodes['ego_car'].world_transform.translation().tolist())
        print(Timer(stmt="scene.interp_at(cams[0].frame_global_ts[40])", globals={'scene': scene, 'cams':cams}).blocked_autorange())

        #---- frame indices slicing
        scene_bank, scene_bank_loader = make_unit_test_dataloader(device, preload=False, use_ts_interp=False)
        scene = scene_bank[0]
        scene.slice_at(40)
        print("slice_at", scene.all_nodes['ego_car'].world_transform.translation().tolist()) # The same with camera_FRONT
        print(Timer(stmt="scene.slice_at(40)", globals={'scene': scene}).blocked_autorange())

    def test_depth_completion():
        import torch
        import numpy as np
        import matplotlib.pyplot as plt

        from torch.utils.benchmark import Timer
        from nr3d_lib.maths import depth_fill_in_fast, depth_fill_in_fast_pytorch
        from app.resources.observers.lidars import MultiRaysLidarBundle
        
        device = torch.device('cuda:1')
        scene_bank, scene_bank_loader = make_unit_test_dataloader(device, preload=False, use_ts_interp=False)
        scene = scene_bank[0]
        
        cam_id = 'camera_FRONT'
        frame_ind = 45
        
        mask = scene_bank_loader.get_image_occupancy_mask(scene.id, cam_id, frame_ind, device=device)
        gt_rgb = scene_bank_loader.get_image_and_metas(scene.id, cam_id, frame_ind, device=device)['image_rgb']
        
        from nr3d_lib.utils import img_to_torch_and_downscale
        mask = img_to_torch_and_downscale(mask, downscale=2)
        gt_rgb = img_to_torch_and_downscale(gt_rgb, downscale=2)
        
        lidar_gts = scene_bank_loader.get_merged_lidar_gts(scene.id, frame_ind, device=device, filter_if_configured=True)
        scene.slice_at(frame_ind)
        lidars = scene.all_nodes_by_class_name['RaysLidar'][scene_bank_loader.lidar_id_list]
        lidars = MultiRaysLidarBundle(lidars)
        l2w = lidars.world_transform[lidar_gts['li']]
        pts_local = torch.addcmul(lidar_gts['rays_o'], lidar_gts['rays_d'], lidar_gts['ranges'].unsqueeze(-1))
        pts = l2w.forward(pts_local)
        
        cam = scene.observers[cam_id]
        cam.intr.set_downscale(2)
        mask, n, u, v, d = cam.project_pts_in_image(pts)
        
        sparse_depth = torch.zeros(gt_rgb.shape[:-1], device=device, dtype=torch.float)
        sparse_depth[v,u] = d
        *_, H, W = sparse_depth.shape
        
        dense_depth = depth_fill_in_fast_pytorch(sparse_depth, max_depth=sparse_depth.max())
        
        sparse_depth_cpu = sparse_depth.data.cpu().numpy()
        dense_depth_cpu = depth_fill_in_fast(sparse_depth_cpu.copy(), max_depth=sparse_depth_cpu.max())
        
        plt.subplot(2,2,1)
        plt.imshow(sparse_depth.data.cpu().numpy())
        plt.subplot(2,2,2)
        plt.imshow(dense_depth.data.cpu().numpy())
        plt.subplot(2,2,3)
        plt.imshow(sparse_depth_cpu)
        plt.subplot(2,2,4)
        plt.imshow(dense_depth_cpu)
        plt.show()
        
        # GPU/pytorch: 9.06 ms
        print(Timer(stmt="depth_fill_in_fast_pytorch(sparse_depth, max_depth=sparse_depth.max())", 
                    globals={'depth_fill_in_fast_pytorch':depth_fill_in_fast_pytorch, 'sparse_depth':sparse_depth}).blocked_autorange())

        # CPU/cv2: 10.71 ms
        print(Timer(stmt="depth_fill_in_fast(sparse_depth_cpu.copy(), max_depth=sparse_depth_cpu.max())", 
                    globals={'depth_fill_in_fast':depth_fill_in_fast, 'sparse_depth_cpu':sparse_depth_cpu}).blocked_autorange())

        import kornia
        from nr3d_lib.maths.depth_completion_pytorch import DIAMOND_KERNEL_5, FULL_KERNEL_5, FULL_KERNEL_7

        def get_step_by_step_completion(
            depth_map: torch.Tensor, 
            max_depth: float = None, eps: float = 0.1, 
            custom_kernel=DIAMOND_KERNEL_5, 
            # extrapolate=False, 
            blur_type='bilateral', 
            engine: Literal['convolution', 'unfold'] = 'convolution'
            ):
        
            completion_steps = {}

            if max_depth is None:
                max_depth = depth_map.max()
            
            shape0 = list(depth_map.shape)
            *_, H, W = depth_map.shape
            depth_map = depth_map.view(-1,1,H,W)
            completion_steps['0.input'] = depth_map[0,0].data.cpu().numpy()

            #---- Invert
            depth_map = torch.where(depth_map > eps, max_depth - depth_map, depth_map)
            completion_steps['1.invert'] = depth_map[0,0].data.cpu().numpy()
            
            #---- Dilate
            depth_map = kornia.morphology.dilation(depth_map, custom_kernel.to(depth_map), engine=engine)
            completion_steps['2.dilate'] = depth_map[0,0].data.cpu().numpy()
            
            #---- Hole closing
            depth_map = kornia.morphology.closing(depth_map, FULL_KERNEL_5.to(depth_map), engine=engine)
            completion_steps['3.hole_closing'] = depth_map[0,0].data.cpu().numpy()
            
            #---- Fill empty spaces with dilated values
            dilated = kornia.morphology.dilation(depth_map, FULL_KERNEL_7.to(depth_map), engine=engine)
            depth_map = torch.where(depth_map < eps, dilated, depth_map)
            completion_steps['4.fill_empty'] = depth_map[0,0].data.cpu().numpy()
            
            #---- Median blur
            depth_map = kornia.filters.median_blur(depth_map, 5)
            completion_steps['5.median_blur'] = depth_map[0,0].data.cpu().numpy()
            
            #---- Bilateral or Gaussian blur
            if blur_type == 'bilateral':
                # Bilateral blur
                depth_map = kornia.filters.bilateral_blur(depth_map, 5, 1.5, (2.0, 2.0))
            elif blur_type == 'gaussian':
                # Gaussian blur
                blurred = kornia.filters.gaussian_blur2d(depth_map, (5,5), (0,0))
                depth_map = torch.where(depth_map > eps, blurred, depth_map)
                # valid_pixels_inds = (depth_map > eps).nonzero(as_tuple=True)
                # depth_map[valid_pixels_inds] = blurred[valid_pixels_inds]
            completion_steps[f'6.{blur_type}_blur'] = depth_map[0,0].data.cpu().numpy()
            
            #---- Invert
            depth_map = torch.where(depth_map > eps, max_depth - depth_map, depth_map)
            completion_steps[f'7.invert'] = depth_map[0,0].data.cpu().numpy()

            return completion_steps
        
        completion_steps = get_step_by_step_completion(sparse_depth)
        
        for i, (k,v) in enumerate(completion_steps.items()):
            plt.subplot(2,4,i+1)
            plt.title(k)
            plt.imshow(v)
        plt.show()
    
    def test_mono_depth():
        import cv2
        import torch
        import imageio
        from tqdm import tqdm, trange
        import matplotlib.pyplot as plt
        from nr3d_lib.plot.plot_basic import color_depth
        
        device = torch.device('cpu')
        scene_bank, scene_bank_loader = make_unit_test_dataloader(
            device, preload=False, use_ts_interp=False, with_mono_cues=True)
        scene = scene_bank[0]
        cam_id = 'camera_FRONT_LEFT'
        
        lst_img_and_depth_color = []
        alpha_depth = 0.8
        for frame_ind in trange(len(scene), desc='Making video...'):
            img = scene_bank_loader.get_image_and_metas(scene.id, cam_id, frame_ind, device=device)['image_rgb'].data.cpu().numpy()
            
            mono_depth = scene_bank_loader.get_image_mono_depth(scene.id, cam_id, frame_ind, device=device).data.squeeze().cpu().numpy()
            mono_depth_color = color_depth(mono_depth, out='float,0,1')
            # mono_normals = scene_bank_loader.get_image_mono_normals(scene.id, cam_id, frame_ind, device=device).data.cpu().numpy()
            
            mono_depth_grayscale = ((mono_depth / mono_depth.max())*255).clip(0,255).astype(np.uint8)
            mono_depth_edges = cv2.Canny(mono_depth_grayscale, threshold1=5, threshold2=10)
            mono_depth_edges_dilated = cv2.dilate(mono_depth_edges, np.ones((10,10),np.uint8), iterations = 1)

            img_and_depth_color = cv2.addWeighted(img, 1 - alpha_depth, mono_depth_color, alpha_depth, 0)
            img_and_depth_color[mono_depth_edges_dilated!=0] = [1.0,0.,0.]
            
            lst_img_and_depth_color.append((img_and_depth_color*255).clip(0,255).astype(np.uint8))
        
        imageio.mimwrite(f"{scene.id[0:15]}_{cam_id}_rgb_mono_depth.mp4", lst_img_and_depth_color)

    def test_perceptual_loss():
        import torch
        

    # test_scenegraph()
    # test_scenario()
    # test_lidar()
    # test_mask()
    # test_distort()
    # test_use_timestamp_interp()
    # test_depth_completion()
    test_mono_depth()