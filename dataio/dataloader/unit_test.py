"""
NOTE:
(Coding conventions)
- Avoid the use of "scene.frozen_at" or any other operations related to the scene graph or nodes, or use them strictly within a "no_grad()" context.
    - The reason for this is that the scene graph may involve propagation of pose gradients across nodes, 
        and we do not expect the dataloader to provide any gradients. 
        These gradients should only be present in the forward process of the trainer.
    - In particular, calculations of camera rays should not be performed here, as they may require pose gradients. 
        Instead, this code outputs a "selects" dictionary that includes only the sampled UV index as input
        for the "cam.get_selected_rays" function in the trainer's forward method.
    - LiDAR's merging and filter_in_cams require the scene graph, but should be performed within a "no_grad()" context.
"""

from dataio.dataloader import *

if __name__ == "__main__":
    import torch
    def unit_test(device=torch.device('cuda')):
        import numpy as np
        from icecream import ic
        from torch.utils.data.dataloader import DataLoader
        from nr3d_lib.utils import import_str
        from nr3d_lib.config import load_config, ConfigDict
        from app.resources import create_scene_bank, AssetBank
        from dataio.dataset_io import DatasetIO
        
        dataset_cfg = ConfigDict(
            target='dataio.autonomous_driving.WaymoDataset', 
            param=ConfigDict(
                root='/data1/waymo/processed', 
                rgb_dirname="images", 
                lidar_dirname="lidars"
            )
        )
                
        scenebank_cfg = ConfigDict(
            scenarios=['segment-7670103006580549715_360_000_380_000_with_camera_labels, 15'], 
            observer_cfgs=ConfigDict(
                Camera=ConfigDict(
                    list=['camera_FRONT', 'camera_FRONT_LEFT', 'camera_FRONT_RIGHT', 
                          'camera_SIDE_LEFT', 'camera_SIDE_RIGHT']
                ), 
                RaysLidar=ConfigDict(
                    list=['lidar_TOP', 'lidar_FRONT', 'lidar_REAR', 
                          'lidar_SIDE_LEFT', 'lidar_SIDE_RIGHT']
                ), 
            ), 
            object_cfgs=ConfigDict(
                Vehicle=ConfigDict(
                    dynamic_only=False
                ),
                Pedestrian=ConfigDict(
                    dynamic_only=False
                )
            ), 
            no_objects=False, 
            align_orientation=False, 
            aabb_extend=120., 
            consider_distortion=True
        )
        
        dataset_impl: DatasetIO = import_str(dataset_cfg.target)(dataset_cfg.param)

        scene_bank, _ = create_scene_bank(
            dataset=dataset_impl, device=device, 
            scenebank_cfg=scenebank_cfg, 
            drawable_class_names=['Vehicle', 'Pedestrian', 'Street', 'Distant'],
            misc_node_class_names=['node', 'EgoVehicle', 'EgoDrone'], 
        )
        
        scene = scene_bank[0]
        
        scene_dataloader = SceneDataLoader(
            scene_bank, dataset_impl, device=device, 
            config=ConfigDict(
                preload=False, 
                tags=ConfigDict(
                    camera=ConfigDict(
                        downscale=2, 
                        list=scenebank_cfg.observer_cfgs.Camera.list
                    ), 
                    lidar=ConfigDict(
                        list=scenebank_cfg.observer_cfgs.RaysLidar.list, 
                        multi_lidar_merge=True, 
                        filter_kwargs=ConfigDict(
                            filter_valid=True, 
                            filter_in_cams=False, 
                            filter_out_objs=False
                        ), 
                        downsample_cfg=ConfigDict(
                            lidar_TOP=4
                        )
                    )
                )
            ))

        cam0 = scene.observers[scene_dataloader.cam_id_list[0]]
        lidar0 = scene.observers[scene_dataloader.lidar_id_list[0]]
        
        multi_cam_weight = np.random.rand(len(scene_dataloader.cam_id_list))
        multi_cam_weight /= multi_cam_weight.sum()
        pixel_dataset = PixelDataset(
            scene_dataloader, 
            equal_mode='ray_batch', num_rays=4096, 
            camera_sample_mode='weighted', multi_cam_weight=multi_cam_weight, 
            frame_sample_mode='error_map', 
            pixel_sample_mode='error_map', error_map_res=(128,128))
        
        pixel_dataloader = DataLoader(pixel_dataset, batch_size=None, sampler=pixel_dataset.get_random_sampler())
        for sample, ground_truth in pixel_dataloader:
            print(sample, ground_truth)
            break
        
        pixel_dataloader = DataLoader(pixel_dataset, batch_size=7, sampler=pixel_dataset.get_random_sampler())
        for sample, ground_truth in pixel_dataloader:
            print(sample, ground_truth)
            break
        
        lidar_dataset = LidarDataset(
            scene_dataloader, 
            equal_mode= 'ray_batch', num_rays=8192, 
            frame_sample_mode = 'uniform', 
            lidar_sample_mode = 'merged_weighted', 
            multi_lidar_weight = [0.5, 0.1, 0.1, 0.1], 
        )
        # lidar_dataset.sample_merged(scene.id, 83, 8192)
        # lidar_dataset.sample_merged(scene.id, 91, 8192)
        lidar_dataset.sample_merged(scene.id, 132, 8192)
        # lidar_dataset.sample_merged(scene.id, 135, 8192)
        # lidar_dataset.sample_merged(scene.id, 186, 8192)
        # lidar_dataset.sample_merged(scene.id, 180, 8192)
        
    unit_test()