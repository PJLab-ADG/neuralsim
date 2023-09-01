"""
A tutorial on loading Ground Truth LiDAR pointclouds into world coordinates,
and compare it with a mesh in world coordinates (e.g. extracted mesh with `--to_world`).

Refering to https://github.com/PJLab-ADG/neuralsim/issues/11
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

import vedo # pip install vedo
import numpy as np
from tqdm import tqdm

import torch

from nr3d_lib.utils import import_str
from nr3d_lib.checkpoint import sorted_ckpts
from nr3d_lib.config import ConfigDict, BaseConfig

from app.resources import create_scene_bank, load_scene_bank, AssetBank

from dataio.dataset_io import DatasetIO
from dataio.dataloader import SceneDataLoader

device = torch.device('cuda')

def create_scene_loader(with_dynamic_objects=False, waymo_processed_root: str = '/data1/waymo/processed'):
    if not with_dynamic_objects:
        # No objects when loading scene_bank from dataset
        no_objects = True 
        
        # No Vehicle, Pedestrians when loading scene_bank from dataset
        drawable_class_names=['Street', 'Distant', 'Sky', 'ImageEmbeddings', 'LearnableParams']
        
        # Filter out dynamic objects when loading lidar GT
        filter_out_objs = True
        filter_out_obj_dynamic_only = True
    else:
        no_objects = False
        drawable_class_names=['Street', 'Vehicle', 'Pedestrians', 'Distant', 'Sky', 'ImageEmbeddings', 'LearnableParams']
        filter_out_objs = False

    #--------------------------------------------
    #---------- Create the scene_bank
    #--------------------------------------------
    scenebank_cfg = ConfigDict(
        scenarios=['segment-10061305430875486848_1080_000_1100_000_with_camera_labels, 0, 163'], 
        observer_cfgs=ConfigDict(
            Camera=ConfigDict(
                list=['camera_FRONT', 'camera_FRONT_LEFT', 'camera_FRONT_RIGHT', 'camera_SIDE_LEFT', 'camera_SIDE_RIGHT']
            ), 
            RaysLidar=ConfigDict(
                list=['lidar_TOP', 'lidar_FRONT', 'lidar_REAR', 'lidar_SIDE_LEFT', 'lidar_SIDE_RIGHT']
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
        no_objects=no_objects,
        align_orientation=True, 
        consider_distortion=True, 
        joint_camlidar=True,
        joint_camlidar_equivalent_extr=True, 
    )

    dataset_cfg = ConfigDict(
        target='dataio.autonomous_driving.WaymoDataset', 
        param=ConfigDict(
            root=waymo_processed_root, 
            rgb_dirname="images", 
            lidar_dirname="lidars", 
            mask_dirname="masks", 
        )
    )
    dataset_impl: DatasetIO = import_str(dataset_cfg.target)(dataset_cfg.param)

    scene_bank, scene_bank_meta = create_scene_bank(
        dataset=dataset_impl, device=device, 
        scenebank_root=None,
        scenebank_cfg=scenebank_cfg, 
        drawable_class_names=drawable_class_names, 
        misc_node_class_names=['node', 'EgoVehicle', 'EgoDrone'], 
    )

    #--------------------------------------------
    #---------- Create Dataloader
    #--------------------------------------------
    scene_dataloader = SceneDataLoader(
        scene_bank, dataset_impl, device=device, 
        config=ConfigDict(
            preload=False, # Not caching 
            tags=ConfigDict(
                # camera is needed for filter_in_cams
                camera=ConfigDict(
                    downscale=1, 
                    list=scenebank_cfg.observer_cfgs.Camera.list
                ), 
                lidar=ConfigDict(
                    list=scenebank_cfg.observer_cfgs.RaysLidar.list, 
                    multi_lidar_merge=True, 
                    filter_kwargs=ConfigDict(
                        filter_in_cams=True, 
                        filter_out_objs=filter_out_objs,
                        filter_out_obj_dynamic_only=filter_out_obj_dynamic_only,
                    )
                )
            )
    ))
    
    return scene_bank, scene_dataloader

def load_scene_from_pretrained(resume_dir: str):
    """
    Load scene_bank and dataloader from pretrained models.
    This will also load the refined poses into the scene's nodes' attributes.
    """
    bc = BaseConfig()
    args = bc.parse([f'--resume_dir={resume_dir}'])
    exp_dir = args.exp_dir
    
    #---- Load scene_bank
    scene_bank_trainval, _ = load_scene_bank(os.path.join(exp_dir, 'scenarios'), device=device)
    
    #---- Load checkpoint
    ckpt_file = sorted_ckpts(os.path.join(args.exp_dir, 'ckpts'))[-1]
    print("=> Use ckpt:" + str(ckpt_file))
    state_dict = torch.load(ckpt_file, map_location=device)
    
    #---- Load assetbank
    asset_bank = AssetBank(args.assetbank_cfg)
    asset_bank.create_asset_bank(scene_bank_trainval, load=state_dict['asset_bank'], device=device)
    
    #---- Load assets into scene
    for scene in scene_bank_trainval:
        scene.load_assets(asset_bank)
    # !!! Only call preprocess_per_train_step when all assets are ready & loaded !
    asset_bank.preprocess_per_train_step(args.training.num_iters) # NOTE: Finished training.
    
    #---- Create dataloader
    dataset_impl = import_str(args.dataset_cfg.target)(args.dataset_cfg.param)
    scene_dataloader = SceneDataLoader(scene_bank_trainval, dataset_impl=dataset_impl, config=args.training.val_dataloader, device=device)
    scene_dataloader.set_camera_downscale(1.)
    
    return scene_bank_trainval, scene_dataloader

def vis_pcl_mesh_vedo(pts_in_world: np.ndarray, mesh_in_world: str, colormap: str = 'rainbow'):
    # Just using z val to colorize
    pcl_val = pts_in_world[:, 2] 
    pts_c = (vedo.color_map(pcl_val, colormap) * 255.).clip(0,255).astype(np.uint8)
    pts_c = np.concatenate([pts_c, np.full_like(pts_c[:,:1], 255)], axis=-1) # RGBA is ~50x faster
    
    #
    pts_actor = vedo.Points(pts_in_world, c=pts_c, r=2)
    mesh = vedo.Mesh(mesh_in_world)
    mesh.color('gray7')
    
    vedo.show(pts_actor, mesh, axes=1)

@torch.no_grad()
def main_function():
    frame_ind = 60 # Freeze the scene at the frame you want
    lidar_id = "lidar_TOP" # The lidar id you want
    mesh_path = "logs/streetsurf/seg100613.withmask_withlidar_ddp/meshes/segment-10061305430875486848_1080_000_1100_000_with_camera_labels#Street#street_level=0.0_res=0.1.ply"
    assert os.path.exists(mesh_path), f"Not exist: {mesh_path}"
    
    #---- NOTE: Create a new scene_bank and loader from processed files
    scene_bank, scene_dataloader = create_scene_loader(
        with_dynamic_objects=False, waymo_processed_root='/data1/waymo/processed')
    
    #---- NOTE: Load the scene_bank and loader from pretrained
    #           This will also load the refined poeses
    # scene_bank, scene_dataloader = load_scene_from_pretrained(
    #     resume_dir="logs/streetsurf/seg100613.withmask_withlidar_ddp")
    
    scene = scene_bank[0]

    # scene.debug_vis_anim(
    #     scene_dataloader=scene_dataloader, 
    #     #  plot_image=True, camera_length=8., 
    #     plot_lidar=True, lidar_pts_downsample=2, 
    #     mesh_file=mesh_path
    # )

    #---- Get GT at the given frame
    # NOTE: This is the same with directly reading preprocessed lidar file,
    #       execpt for filtering
    lidar_gts = scene_dataloader.get_lidar_gts(
        scene.id, lidar_id, frame_ind, device=device, filter_if_configured=True)
    pts_local = torch.addcmul(lidar_gts['rays_o'], lidar_gts['rays_d'], lidar_gts['ranges'].unsqueeze(-1))

    #---- Freeze the scene 
    # NOTE: This will also update the scene_graph. 
    #       So all the nodes, including the sensors and ego_car / ego_drone is also frozen at the given frame,
    #       and each nodes' world_transform is automatically broadcasted from root to leaves.
    scene.frozen_at(frame_ind)

    #---- Get the sensor/observer
    lidar = scene.observers[lidar_id]

    #---- Transform to world
    pts = lidar.world_transform.forward(pts_local)

    #---- Visualize
    vis_pcl_mesh_vedo(pts.data.cpu().numpy(), mesh_path)
    
    #---- To visualize multiple frames of pcl together
    # pts_list = []
    # for fi in tqdm(range(len(scene)), 'Extracting lidar gt...'):
    #     lidar_gts = scene_dataloader.get_lidar_gts(
    #         scene.id, lidar.id, fi, device=device, filter_if_configured=True)
    #     scene.frozen_at(fi)
    #     pts_local = torch.addcmul(lidar_gts['rays_o'], lidar_gts['rays_d'], lidar_gts['ranges'].unsqueeze(-1))
    #     pts = lidar.world_transform.forward(pts_local)
    #     pts_list.append(pts.data.cpu().numpy())
    # pts_list = np.concatenate(pts_list, axis=0)
    # vis_pcl_mesh_vedo(pts_list, mesh_path)

if __name__ == "__main__":
    main_function()