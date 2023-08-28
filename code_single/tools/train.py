"""
@file   train.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Reconstruct a single static object / street / building / room etc. scene.
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
import sys
import time
import numpy as np
from tqdm import tqdm
from typing import Dict, Literal, List, Tuple

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from nr3d_lib.fmt import log
from nr3d_lib.logger import Logger
from nr3d_lib.plot import color_depth
from nr3d_lib.checkpoint import CheckpointIO
from nr3d_lib.config import ConfigDict, save_config
from nr3d_lib.distributed import get_local_rank, init_env, is_master, get_rank, get_world_size
from nr3d_lib.utils import IDListedDict, collate_nested_dict, import_str, backup_project, nested_dict_items, pad_images_to_same_size, wait_for_pid, zip_dict, zip_two_nested_dict

from nr3d_lib.render.utils import PSNR
from nr3d_lib.models.utils import get_scheduler, calc_grad_norm

from app.models.base import AssetAssignment, AssetModelMixin
from app.renderers import SingleVolumeRenderer
from app.resources import Scene, AssetBank, create_scene_bank, load_scene_bank
from app.resources.observers import Camera, MultiCamBundle, Lidar, RaysLidar, MultiRaysLidarBundle

from dataio.dataset_io import DatasetIO
from dataio.dataloader import SceneDataLoader, ImageDataset, ImagePatchDataset, LidarDataset, PixelDataset, JointFramePixelDataset

# from torch.autograd.anomaly_mode import set_detect_anomaly
# set_detect_anomaly(True)
class Trainer(nn.Module):
    def __init__(
        self, 
        config, 
        renderer: SingleVolumeRenderer, 
        asset_bank: AssetBank,
        scene_bank: List[Scene], 
        dataset: SceneDataLoader, 
        pixel_dataset: PixelDataset = None, 
        image_dataset: ImageDataset = None, 
        lidar_dataset: LidarDataset = None,
        i_log: int = -1, i_val: int = -1, 
        device_ids=[0]) -> None:
        super().__init__()
        self.config = config
        self.renderer = renderer
        self.scene_bank = scene_bank
        self.asset_bank = asset_bank
        self.dataset = dataset
        self.pixel_dataset = pixel_dataset
        self.image_dataset = image_dataset
        self.lidar_dataset = lidar_dataset
        self.device = device_ids[0]
        self.i_log = i_log
        self.i_val = i_val
        loss_cfg = config.losses

        self.renderer.train()
        self.asset_bank.train()

        self.pixel_step_require_vw_in_total = False
        self.image_patch_step_require_vw_in_total = False
        self.lidar_step_require_vw_in_total = False

        #------------ Configure losses
        drawable_class_names = list(asset_bank.class_name_configs.keys())
        
        self.rgb_loss = None
        if 'rgb' in loss_cfg:
            from app.loss import PhotometricLoss
            self.rgb_loss = PhotometricLoss(**loss_cfg['rgb']) # Avaiable in mode: [pixel, image_patch]

        self.occupancy_mask_loss = None # Avaiable in mode: [pixel, image_patch]
        if 'occupancy_mask' in loss_cfg:
            from app.loss import MaskOccupancyLoss
            self.occupancy_mask_loss = MaskOccupancyLoss(**loss_cfg['occupancy_mask'])

        self.depth_loss = None # Avaiable in mode: [pixel, image_patch]
        if 'depth' in loss_cfg:
            self.depth_loss = ...

        self.lidar_loss = None # Avaiable in mode: [lidar]
        if 'lidar' in loss_cfg:
            from app.loss import LidarLoss
            self.lidar_loss = LidarLoss(**loss_cfg['lidar'])

        self.mono_depth_loss = None # Avaiable in mode: [image_patch]
        if 'mono_depth' in loss_cfg:
            from app.loss import MonoSSIDepthLoss
            self.mono_depth_loss = MonoSSIDepthLoss(**loss_cfg['mono_depth'], debug_val_every=self.i_val)
            if self.mono_depth_loss.require_render_per_obj:
                self.image_patch_step_require_vw_in_total = True

        self.mono_normals_loss = None # Avaiable in mode: [pixel (optional), image_patch]
        if 'mono_normals' in loss_cfg:
            from app.loss import MonoNormalLoss
            self.mono_normals_loss = MonoNormalLoss(**loss_cfg['mono_normals'], debug_val_every=self.i_val)
            if self.mono_normals_loss.require_render_per_obj:
                self.image_patch_step_require_vw_in_total = True
                if self.mono_normals_loss.apply_in_pixel_train_step:
                    self.pixel_step_require_vw_in_total = True

        self.road_normals_loss = None
        if 'road_normals' in loss_cfg:
            from app.loss import RoadNormalLoss
            self.road_normals_loss = RoadNormalLoss(**loss_cfg['road_normals'])
            if self.road_normals_loss.require_render_per_obj:
                self.image_patch_step_require_vw_in_total = True
                if self.road_normals_loss.apply_in_pixel_train_step:
                    self.pixel_step_require_vw_in_total = True

        self.eikonal_loss = None # Regularization. Avaiable in mode: [pixel, image_patch, lidar]
        if 'eikonal' in loss_cfg:
            from app.loss import EikonalLoss
            self.eikonal_loss = EikonalLoss(**loss_cfg['eikonal'], drawable_class_names=drawable_class_names, log_every=self.i_log)
        
        self.mask_entropy_loss = None # Regularization. Avaiable in mode: [pixel, image_patch]; not with lidar, since lidar may only concerns fg
        if 'mask_entropy' in loss_cfg:
            from app.loss import MaskEntropyRegLoss
            self.mask_entropy_loss = MaskEntropyRegLoss(**loss_cfg['mask_entropy'], drawable_class_names=drawable_class_names)
        
        self.sparsity_loss = None # Regularization. Avaiable in mode: [pixel, image_patch, lidar]
        if 'sparsity' in loss_cfg:
            from app.loss import SparsityLoss
            self.sparsity_loss = SparsityLoss(**loss_cfg['sparsity'], drawable_class_names=drawable_class_names)
        
        self.clearance_loss = None # Regularization. Avaiable in mode: [pixel, image_patch, lidar]
        if 'clearance' in loss_cfg:
            from app.loss import ClearanceLoss
            self.clearance_loss = ClearanceLoss(**loss_cfg['clearance'], drawable_class_names=drawable_class_names)

        self.sdf_curvature_reg_loss = None # Regularization. Available in mode: [pixel, image_patch, lidar]
        if 'sdf_curvature_reg' in loss_cfg:
            from app.loss import SDFCurvatureRegLoss
            self.sdf_curvature_reg_loss = SDFCurvatureRegLoss(**loss_cfg['sdf_curvature_reg'], drawable_class_names=drawable_class_names)

        self.weight_reg_loss = None # Regularization. Avaiable in mode: [pixel, image_patch, lidar]
        if 'weight_reg' in loss_cfg:
            from app.loss import WeightRegLoss
            self.weight_reg_loss = WeightRegLoss(**loss_cfg['weight_reg'], drawable_class_names=drawable_class_names)

        self.color_net_reg_loss = None # Regularization. Available in mode: [pixel, image_patch]
        if 'color_net_reg' in loss_cfg:
            from app.loss import ColorLipshitzRegLoss
            self.color_net_reg_loss = ColorLipshitzRegLoss(**loss_cfg['color_net_reg'], drawable_class_names=drawable_class_names)

        if self.eikonal_loss is not None:
            self.renderer._config_train.with_normal = True
        if self.clearance_loss is not None:
            self.renderer._config_train.with_near_sdf = True
        
        #------------ DEBUG
        self.debug_grad = False
        self.debug_grad_threshold = 3.0
        self.debug_ret = False
    
    def initialize(self, logger=None) -> bool:
        log.info("=> Start initialize prepcess...")
        for class_name, model_id_map in self.asset_bank.class_name_infos.items():
            for model_id, scene_obj_id_list in model_id_map.items():
                model: AssetModelMixin = self.asset_bank[model_id]
                if model.assigned_to == AssetAssignment.OBJECT:
                    scene_id, obj_id = scene_obj_id_list[0]
                    scene = self.scene_bank[scene_id]
                    obj = scene.all_nodes[obj_id]
                    model.initialize(scene=scene, obj=obj, config=model.initialize_cfg, logger=logger, log_prefix=model_id)
                elif model.assigned_to == AssetAssignment.SCENE:
                    scene_id, _ = scene_obj_id_list[0]
                    scene = self.scene_bank[scene_id]
                    model.initialize(scene=scene, obj=None, config=model.initialize_cfg, logger=logger, log_prefix=model_id)
                elif model.assigned_to == AssetAssignment.MULTI_OBJ_ONE_SCENE:
                    scene_id, _ = scene_obj_id_list[0]
                    model.initialize(scene=scene, obj=None, config=model.initialize_cfg, logger=logger, log_prefix=model_id)
                elif model.assigned_to == AssetAssignment.MULTI_OBJ:
                    model.initialize(scene=None, obj=None, config=model.initialize_cfg, logger=logger, log_prefix=model_id)
                elif model.assigned_to == AssetAssignment.MULTI_SCENE:
                    pass
                elif model.assigned_to == AssetAssignment.MISC:
                    model.initialize(scene=None, obj=None, config=model.initialize_cfg, logger=logger, log_prefix=model_id)
        log.info("=> Done initialize prepcess.")
        return True

    def preprocess_per_train_step(self, cur_it: int, logger: Logger = None):
        self.asset_bank.preprocess_per_train_step(cur_it, logger=logger)
        if self.pixel_dataset is not None:
            self.pixel_dataset.cur_it = cur_it
        if self.image_dataset is not None:
            self.image_dataset.cur_it = cur_it
        if self.lidar_dataset is not None:
            self.lidar_dataset.cur_it = cur_it

    def forward(
        self, mode: Literal['pixel', 'image_patch', 'lidar'], 
        sample: dict, ground_truth: dict, it: int, logger: Logger=None):
        """
        In order to utilize various types of supervision, we have defined three different training steps based on the type of dataloader:

        1. `train_step_pixel` 
            - Loads `PixelDataset` / `JointFramePixelDataset` and samples unorganized pixels.
                - `JointFramePixelDataset` samples both frame indices and pixel positions simultaneously, \
                    similar to the implementation of [Instant-NGP](https://github.com/NVlabs/instant-ngp). \
                    Each sampled pixel comes from different frames and pixel positions.
                - `PixelDataset` first samples the frame indice, then samples a set of pixels within this single frame.
            - Mainly for photometric reconstruction loss. Monocular normals supervision can also be applied here.
        2. `train_step_lidar`
            - Loads `LidarDataset` and samples unorganized LiDAR beams.
            - It can sample beams from a single LiDAR or simultaneously from multiple LiDARs, controlled by `lidar_sample_mode`. \
                For more details, please refer to [dataio/dataloader.py](dataio/dataloader.py) :: `LidarDataset`.
            - Mainly for LiDAR sparse depth loss.
        3. `train_step_image_patch`
            - Loads `ImagePatchDataset` and samples down-sampled image patches, which are organized patch pixels compared to `train_step_pixel` .
            - One major application is that mono depth loss requires regular image patches to estimate the scale and shift of the inferred monocular depths. 
            - Also applicable for other losses that require regular images, such as perceptual loss, GAN loss, etc.
        """
        
        self.asset_bank.train()
        self.renderer.train()
        
        if mode == 'pixel':
            ret, losses = self.train_step_pixel(sample, ground_truth, it, logger=logger)
        elif mode == 'image_patch':
            ret, losses = self.train_step_image_patch(sample, ground_truth, it, logger=logger)
        elif mode == 'lidar':
            ret, losses = self.train_step_lidar(sample, ground_truth, it, logger=logger)
        else:
            raise RuntimeError(f"Invalid mode={mode}")
        
        if self.debug_ret:
            for *k, v in nested_dict_items(ret):
                if not isinstance(v, torch.Tensor):
                    continue
                if torch.isnan(v).any():
                    log.error("NAN found in return: " + '.'.join(k))
                elif torch.isinf(v).any():
                    log.error("INF found in return: " + '.'.join(k))

            for *k, v in nested_dict_items(losses):
                if not isinstance(v, torch.Tensor):
                    continue
                if torch.isnan(v).any():
                    log.error("NAN found in loss: " + '.'.join(k))
                elif torch.isinf(v).any():
                    log.error("INF found in loss: " + '.'.join(k))

        if self.debug_grad:
            scene = self.scene_bank[sample['scene_id'][0]]
            debug_obj = scene.all_nodes_by_class_name[scene.main_class_name][0]
            debug_obj_model = debug_obj.model
            debug_obj_model_id = debug_obj_model.id
            
            #---- NOTE: Retain grad in case of need.
            debug_pg = dict(debug_obj_model.named_parameters())
            for *k, v in nested_dict_items(ret):
                if hasattr(v, 'grad_fn') and v.grad_fn is not None:
                    v.retain_grad()
            
            for k, v in losses.items():
                if hasattr(v, 'grad_fn') and v.grad_fn is not None:
                    v.retain_grad()
            
            #---- NOTE: Check per return w.r.t. per parameter
            for *k, v in nested_dict_items(ret):
                if hasattr(v, 'grad_fn') and v.grad_fn is not None:
                    debug_grad = torch.autograd.grad(v.mean(), debug_pg.values(), retain_graph=True, allow_unused=True)
                    debug_pg_grad = {pn:pg  for pn,pg in zip(debug_pg.keys(), debug_grad)}
                    for pn, pg in debug_pg_grad.items():
                        if pg is None:
                            continue
                        if torch.isnan(pg).any():
                            log.error(f"NAN! d({k})_d({pn})")
                        elif torch.isinf(pg).any():
                            log.error(f"INF! d({k})_d({pn})")
                    del debug_grad, debug_pg_grad
            
            # loss_grads = dict()
            #---- NOTE: Check per loss w.r.t per parameter
            for k, v in losses.items():
                if hasattr(v, 'grad_fn') and v.grad_fn is not None:
                    debug_grad = torch.autograd.grad(v, debug_pg.values(), retain_graph=True, allow_unused=True)
                    debug_pg_grad = {pn:pg  for pn,pg in zip(debug_pg.keys(), debug_grad)}
                    for pn, pg in debug_pg_grad.items():
                        if pg is None:
                            continue
                        if torch.isnan(pg).any():
                            log.error(f"NAN! d({k})_d({pn})")
                        elif torch.isinf(pg).any():
                            log.error(f"INF! d({k})_d({pn})")
                    
                    # loss_grads[k] = np.sqrt((np.array([dg.data.norm().item() for dg in debug_grad if dg is not None] + [0])**2).sum())

                    # if loss_grads[k] > self.debug_grad_threshold:
                    #     log.error(f"!!! grad_norm = {loss_grads[k]} > {self.debug_grad_threshold}: on losses[{k}] w.r.t. asset_bank[{debug_obj_model_id}]")
                    #     for *_k, _v in nested_dict_items(ret):
                    #         if hasattr(_v, 'grad_fn') and _v.grad_fn is not None:
                    #             _tmp_grad = torch.autograd.grad(_v.mean(), debug_obj_model.parameters(), retain_graph=True, allow_unused=True)
                    #             _tmp_grad_norm = np.sqrt((np.array([dg.data.norm().item() for dg in _tmp_grad if dg is not None] + [0])**2).sum())
                    #             log.error('wrt. net', '.'.join(_k), _tmp_grad_norm)
                    #             del _tmp_grad, _tmp_grad_norm
                    #     del _k, _v
                    del debug_grad, debug_pg_grad

        return ret, losses

    def train_step_pixel(self, sample, ground_truth, it, logger: Logger=None):
        """ Execute a single training step for pixel data on one frame.

        Args:
            sample (dict): 'sample' from dataio.dataloader.PixelDatasetBase.
            ground_truth (dict): 'ground_truth' from dataio.dataloader.PixelDatasetBase.
            it (int): Current iteration.
            logger (Logger, optional): The logger. Defaults to None.

        Returns:
            Tuple[dict, dict]: 'ret', 'losses'; two nested dictionaries containing raw returns and calculated losses.
        """
        
        sample, ground_truth = next(zip_two_nested_dict(sample, ground_truth)) # NOTE: bs=1 in this case
        losses = dict()
        
        #----------------------- Get scene
        scene_id: str =  sample['scene_id']
        scene: Scene = self.scene_bank[scene_id]

        #----------------------- Get cam & frame
        cam_id, frame_i = sample['cam_id'], sample['frame_id']
        scene.frozen_at(frame_i)
        if isinstance(cam_id, (List, Tuple)):
            cams = [scene.observers[cid] for cid in cam_id]
            cam = MultiCamBundle(cams)
        else:
            cam: Camera = scene.observers[cam_id]
        cam.intr.set_downscale(ground_truth['rgb_downscale'])
        
        #----------------------- Get rays
        rays_o, rays_d = cam.get_selected_rays(**sample['selects'])
        
        #-----------------------------------------------
        ret = self.renderer.render(
            scene, 
            rays=(rays_o, rays_d, sample['rays_xy'], sample['rays_fi']), 
            observer=cam, only_cr=(it < self.config.get('enable_dv_after', 0)), 
            return_buffer=True, return_details=True, render_per_obj=False, 
            render_per_obj_in_total=self.pixel_step_require_vw_in_total)
        #-----------------------------------------------
        
        obj = scene.get_drawable_groups_by_class_name(scene.main_class_name)[0]
        obj_raw_ret = ret['raw_per_obj_model'][obj.id]
        model = obj.model

        #-----------------------------------------------
        #-----------  Uniform sampling
        # Used by eikonal loss and sparsity loss
        uniform_samples = None
        if (num_uniform_samples:=self.config.get('uniform_sample', 0)) > 0:
            uniform_samples = model.uniform_sample(num_uniform_samples)

        #-----------------------------------------------
        #-----------  Calc losses
        if self.rgb_loss is not None:
            ret_losses_rgb, err_map_rgb = self.rgb_loss(scene, ret, sample, ground_truth, it=it)
            if self.config.get('ignore_errmap_not_occupied', True) and 'rgb_mask' in ground_truth:
                err_map_rgb = err_map_rgb * ground_truth['rgb_mask'].float().view(*err_map_rgb.shape)
            losses.update(ret_losses_rgb)
        
        if self.occupancy_mask_loss is not None:
            ret_losses_mask, err_map_mask = self.occupancy_mask_loss(scene, ret, sample, ground_truth, it=it)
            losses.update(ret_losses_mask)
        else:
            err_map_mask = 0
        
        if self.mask_entropy_loss is not None:
            losses.update(self.mask_entropy_loss.forward_code_single(scene, ret, rays_o.shape[:-1], it=it))
        
        if (self.pixel_dataset.imp_samplers is not None) and (it >= self.pixel_dataset.respect_errormap_after):
            imp_sampler = self.pixel_dataset.imp_samplers[scene_id][cam_id]
            selects: dict = sample['selects'].copy() # NOTE: Not deepcopy
            selects.setdefault('i', frame_i)
            imp_sampler.step_error_map(**selects, val=(err_map_rgb + err_map_mask))

        if self.mono_normals_loss is not None and self.mono_normals_loss.apply_in_pixel_train_step:
            losses.update(self.mono_normals_loss(
                scene, cam, ret, sample, ground_truth, it=it, 
                mode='pixel', logger=logger))

        if self.road_normals_loss is not None and self.road_normals_loss.apply_in_pixel_train_step:
            losses.update(self.road_normals_loss(
                scene, cam, ret, sample, ground_truth, it=it))

        if self.eikonal_loss is not None:
            losses.update(self.eikonal_loss.forward_code_single(
                obj, obj_raw_ret, uniform_samples, sample, ground_truth, it=it, 
                mode='pixel', logger=logger))
        
        if self.sparsity_loss is not None:
            losses.update(self.sparsity_loss.forward_code_single(
                obj, obj_raw_ret, uniform_samples, sample, ground_truth, it=it))
        
        if self.clearance_loss is not None:
            losses.update(self.clearance_loss.forward_code_single(
                obj, obj_raw_ret, uniform_samples, sample, ground_truth, it=it))
        
        if self.sdf_curvature_reg_loss is not None:
            losses.update(self.sdf_curvature_reg_loss.forward_code_single(
                obj, obj_raw_ret, uniform_samples, sample, ground_truth, it=it))
        
        if self.color_net_reg_loss is not None:
            losses.update(self.color_net_reg_loss.forward_code_single(
                obj, obj_raw_ret, uniform_samples, sample, ground_truth, it=it))
        
        if self.weight_reg_loss is not None:
            losses.update(self.weight_reg_loss(scene, ret, sample, ground_truth, it=it))

        return ret, losses

    def train_step_image_patch(self, sample: dict, ground_truth: dict, it: int, logger: Logger=None):
        """ Execute a single training step for image patch data on one frame.

        Args:
            sample (dict): 'sample' from dataio.dataloader.ImagePatchDataset.
            ground_truth (dict): 'ground_truth' from dataio.dataloader.ImagePatchDataset.
            it (int): Current iteration.
            logger (Logger, optional): The logger. Defaults to None.

        Returns:
            Tuple[dict, dict]: 'ret', 'losses'; two nested dictionaries containing raw returns and calculated losses.
        """
        
        sample, ground_truth = next(zip_two_nested_dict(sample, ground_truth)) # NOTE: bs=1 in this case
        losses = dict()
        
        #----------------------- Get scene
        scene_id: str =  sample['scene_id']
        scene: Scene = self.scene_bank[scene_id]

        #----------------------- Get cam & frame
        cam_id, frame_i = sample['cam_id'], sample['frame_id']
        scene.frozen_at(frame_i)
        if isinstance(cam_id, (List, Tuple)):
            cams = [scene.observers[cid] for cid in cam_id]
            cam = MultiCamBundle(cams)
        else:
            cam: Camera = scene.observers[cam_id]
        cam.intr.set_downscale(ground_truth['rgb_downscale'])

        #----------------------- Get rays
        rays_o, rays_d = cam.get_selected_rays(**sample['selects'])
        H, W, _ = rays_o.shape

        #-----------------------------------------------
        ret = self.renderer.render(
            scene, 
            rays=(rays_o, rays_d, sample['rays_xy'], sample['rays_fi']), 
            observer=cam, only_cr=(it < self.config.get('enable_dv_after', 0)), 
            return_buffer=True, return_details=True, 
            render_per_obj=False, 
            render_per_obj_in_total=self.image_patch_step_require_vw_in_total)
        #-----------------------------------------------

        obj = scene.get_drawable_groups_by_class_name(scene.main_class_name)[0]
        obj_raw_ret = ret['raw_per_obj_model'][obj.id]
        model = obj.model

        #-----------------------------------------------
        #-----------  Uniform sampling
        # Used by eikonal loss and sparsity loss
        uniform_samples = None
        if (num_uniform_samples:=self.config.get('uniform_sample', 0)) > 0:
            uniform_samples = model.uniform_sample(num_uniform_samples)

        #-----------------------------------------------
        #-----------  Calc losses
        if self.rgb_loss is not None:
            ret_losses_rgb, err = self.rgb_loss(scene, ret, sample, ground_truth, it=it)
            losses.update(ret_losses_rgb)

        if self.occupancy_mask_loss is not None:
            ret_losses_mask, err_map_mask = self.occupancy_mask_loss(scene, ret, sample, ground_truth, it=it)
            losses.update(ret_losses_mask)

        if self.road_normals_loss is not None:
            losses.update(self.road_normals_loss(
                scene, cam, ret, sample, ground_truth, it=it))

        if self.mono_normals_loss is not None:
            losses.update(self.mono_normals_loss(
                scene, cam, ret, sample, ground_truth, it=it, 
                mode='image_patch', logger=logger))

        if self.mono_depth_loss is not None:
            losses.update(self.mono_depth_loss(
                scene, ret, sample, ground_truth, it=it, far=cam.far,  
                mode='image_patch', logger=logger))

        if self.mask_entropy_loss is not None:
            losses.update(self.mask_entropy_loss.forward_code_single(
                scene, ret, (H*W,), it=it))
        
        if self.eikonal_loss is not None:
            losses.update(self.eikonal_loss.forward_code_single(
                obj, obj_raw_ret, uniform_samples, sample, ground_truth, it=it, 
                mode='image_patch', logger=logger))
        
        if self.sparsity_loss is not None:
            losses.update(self.sparsity_loss.forward_code_single(
                obj, obj_raw_ret, uniform_samples, sample, ground_truth, it=it))
        
        if self.clearance_loss is not None:
            losses.update(self.clearance_loss.forward_code_single(
                obj, obj_raw_ret, uniform_samples, sample, ground_truth, it=it))
        
        if self.sdf_curvature_reg_loss is not None:
            losses.update(self.sdf_curvature_reg_loss.forward_code_single(
                obj, obj_raw_ret, uniform_samples, sample, ground_truth, it=it))
        
        if self.color_net_reg_loss is not None:
            losses.update(self.color_net_reg_loss.forward_code_single(
                obj, obj_raw_ret, uniform_samples, sample, ground_truth, it=it))
        
        if self.weight_reg_loss is not None:
            losses.update(self.weight_reg_loss(scene, ret, sample, ground_truth, it=it))

        return ret, losses

    def train_step_lidar(self, sample: dict, ground_truth: dict, it: int, logger: Logger=None) -> Tuple[dict, dict]:
        """ Execute a single training step for LiDAR data on one frame.

        Args:
            sample (dict): 'sample' from dataio.dataloader.LidarDataset.
            ground_truth (dict): 'ground_truth' from dataio.dataloader.LidarDataset.
            it (int): Current iteration.
            logger (Logger, optional): The logger. Defaults to None.

        Returns:
            Tuple[dict, dict]: 'ret', 'losses'; two nested dictionaries containing raw returns and calculated losses.
        """
        
        sample, ground_truth = next(zip_two_nested_dict(sample, ground_truth)) # NOTE: bs=1 in this case
        losses = dict()
        
        #----------------------- Get scene
        scene_id: str =  sample['scene_id']
        scene: Scene = self.scene_bank[scene_id]
        
        #----------------------- Get lidar & frame
        lidar_id, frame_i = sample['lidar_id'], sample['frame_id']
        scene.frozen_at(frame_i)
        if isinstance(lidar_id, (List, Tuple)):
            lidars = [scene.observers[lid] for lid in lidar_id]
            lidar = MultiRaysLidarBundle(lidars)
        else:
            lidar: RaysLidar = scene.observers[lidar_id]
        
        #----------------------- Get rays
        rays_o, rays_d = lidar.get_selected_rays(**sample['selects'])
        
        #-----------------------------------------------
        ret = self.renderer.render(
            scene, 
            rays=(rays_o, rays_d), observer=lidar, 
            only_cr=(it < self.config.get('enable_dv_after', 0)), 
            with_rgb=False, with_normal=(self.eikonal_loss is not None), 
            return_buffer=True, return_details=True, render_per_obj=False, 
            render_per_obj_in_total=self.lidar_step_require_vw_in_total)
        #-----------------------------------------------
        
        obj = scene.get_drawable_groups_by_class_name(scene.main_class_name)[0]
        obj_raw_ret = ret['raw_per_obj_model'][obj.id]
        model = obj.model
        
        #-----------------------------------------------
        #-----------  Uniform sampling
        # Used by eikonal loss and sparsity loss
        uniform_samples = None
        if (num_uniform_samples:=self.config.get('uniform_sample', 0)) > 0:
            uniform_samples = model.uniform_sample(num_uniform_samples)

        #-----------------------------------------------
        #-----------  Calc losses
        assert self.lidar_loss is not None, "Need to config lidar_loss when forwarding lidar data"
        losses.update(self.lidar_loss(
            scene, ret, sample, ground_truth, it=it, far=lidar.far, logger=logger))
        
        if self.eikonal_loss is not None:
            losses.update(self.eikonal_loss.forward_code_single(
                obj, obj_raw_ret, uniform_samples, sample, ground_truth, it=it, 
                mode='lidar', logger=logger))
        
        if self.sparsity_loss is not None:
            losses.update(self.sparsity_loss.forward_code_single(
                obj, obj_raw_ret, uniform_samples, sample, ground_truth, it=it))
        
        if self.clearance_loss is not None:
            losses.update(self.clearance_loss.forward_code_single(
                obj, obj_raw_ret, uniform_samples, sample, ground_truth, it=it))
        
        if self.sdf_curvature_reg_loss is not None:
            losses.update(self.sdf_curvature_reg_loss.forward_code_single(
                obj, obj_raw_ret, uniform_samples, sample, ground_truth, it=it))
        
        if self.weight_reg_loss is not None:
            losses.update(self.weight_reg_loss(scene, ret, sample, ground_truth, it=it))

        return ret, losses

    @torch.no_grad()
    def _validate_single_cam(
        self, scene_id: str, cam_id: str, frame_id: int, 
        ground_truth: dict, logger: Logger, it: int, 
        should_log_img=True, log_prefix: str='') -> dict:
        """ Validate one camera at one frame.

        Args:
            scene_id (str): The given scene_id to validate.
            cam_id (str): The given cam_id to validate.
            frame_id (int): The given frame_id to validate.
            ground_truth (dict): The corresponding groud truth dict.
            logger (Logger): Logger
            it (int): The current iteration
            should_log_img (bool, optional): 
                If true, will log this single camera validations to logger. 
                Defaults to True.
            log_prefix (str, optional): Logger prefix. Defaults to ''.

        Returns:
            dict: A collection of single camera validation images
        """
        
        log_collect = {}
        def add_log_img_entry(key: str, img: torch.Tensor):
            log_collect[key] = img
            if should_log_img:
                logger.add_imgs(log_prefix, key, img, it=it)
        
        self.asset_bank.eval()
        self.renderer.eval()
        
        scene: Scene = self.scene_bank[scene_id]
        scene.frozen_at(frame_id)
        cam: Camera = scene.observers[cam_id]
        cam.intr.set_downscale(ground_truth['rgb_downscale'])

        objs = scene.get_drawable_groups_by_class_name(scene.main_class_name).to_list()
        distant_objs = scene.get_drawable_groups_by_class_name('Distant').to_list()
        
        #-----------------------------------------------
        ret = self.renderer.render(
            scene, observer=cam, 
            only_cr=(it < self.config.get('enable_dv_after', 0)), 
            render_per_obj=(len(objs) + len(distant_objs) > 1), 
            render_per_obj_in_total=(len(objs) + len(distant_objs) > 1))
        rendered = ret['rendered']
        rendered_per_obj = ret.get("rendered_per_obj", {})
        rendered_per_obj_in_total = ret.get("rendered_per_obj_in_total", {})
        #-----------------------------------------------

        def to_img(tensor: torch.Tensor):
            return tensor.reshape([cam.intr.H, cam.intr.W, -1]).data.cpu().movedim(-1,0) # [C,H,W]

        def log_imgs(log_suffix: str, render_dict: Dict[str, torch.Tensor]):
            if 'mask_volume' in render_dict and 'depth_volume' in render_dict:
                mask_volume = to_img(render_dict['mask_volume'].unsqueeze(-1))
                depth_volume = to_img(render_dict['depth_volume'].unsqueeze(-1))
                depth_max = depth_volume.max().clamp(1e-10, self.renderer.config.far or cam.far)
                depth_volume = mask_volume * (depth_volume/depth_max) + (1-mask_volume) * 1
                depth_volume = color_depth(depth_volume.squeeze(-3).data.cpu().numpy(), scale=1, cmap='turbo', out='float,0,1')    # turbo_r, viridis, rainbow
                depth_volume = torch.tensor(depth_volume, dtype=torch.float).movedim(-1,0) # [C,H,W]
                add_log_img_entry(f"pred_mask_volume{log_suffix}", mask_volume)
                add_log_img_entry(f"pred_depth_volume{log_suffix}", depth_volume)
            if 'rgb_volume' in render_dict:
                add_log_img_entry(f"pred_rgb_volume{log_suffix}", to_img(render_dict['rgb_volume']))
                add_log_img_entry(f"pred_rgb_volume{log_suffix}", to_img(render_dict['rgb_volume']))
            if 'normals_volume' in render_dict:
                add_log_img_entry(f"pred_normals_volume{log_suffix}", to_img(render_dict['normals_volume']/2+0.5))
        
        log_imgs('', rendered)
        for obj_id, obj_rendered in rendered_per_obj.items():
            log_imgs(f".{obj_id}.seperate", obj_rendered)
        for obj_id, obj_rendered in rendered_per_obj_in_total.items():
            log_imgs(f".{obj_id}.in_total", obj_rendered)

        if 'rgb' in ground_truth:
            add_log_img_entry("gt_rgb", to_img(ground_truth['rgb'].float()))
        
        for gt_mask_key in ('rgb_mask', 'rgb_dynamic_mask', 'rgb_human_mask', 'rgb_ignore_mask', 'rgb_road_mask'):
            if gt_mask_key in ground_truth:
                add_log_img_entry(f"gt_{gt_mask_key}", to_img(ground_truth[gt_mask_key].unsqueeze(-1)/ground_truth[gt_mask_key].max().float()))

        if 'rgb_mono_depth' in ground_truth:
            add_log_img_entry("gt_rgb_mono_depth", to_img(ground_truth['rgb_mono_depth'].unsqueeze(-1)/ground_truth['rgb_mono_depth'].max().float()))
        
        if 'rgb_mono_normals' in ground_truth:
            add_log_img_entry("gt_rgb_mono_normals", to_img(ground_truth['rgb_mono_normals']/2+0.5))

        if 'ray_intersections' in ret:
            add_log_img_entry("ray_intersections/samples_cnt", to_img(ret['ray_intersections']['samples_cnt'].unsqueeze(-1).float() / ret['ray_intersections']['samples_cnt'].max().clamp(1e-5)))

        #---- Validate errormap (if any)
        if self.pixel_dataset is not None and self.pixel_dataset.imp_samplers is not None:
            imp_sampler = self.pixel_dataset.imp_samplers[scene_id][cam_id]
            err_map = imp_sampler.error_map[frame_id]
            add_log_img_entry("error_map", err_map.unsqueeze(0) / err_map.max().clamp_min(1e-5))

        #---- Scalar validation (NOTE: NOT eval! Since only one frame is used!!)
        if self.rgb_loss is not None:
            #--------- Calc psnr
            # Full PSNR
            eval_rgb_pred = rendered['rgb_volume'].view(cam.intr.H, cam.intr.W, -1)
            eval_rgb_gt = ground_truth['rgb'].to(self.device).view(cam.intr.H, cam.intr.W, -1)
            psnr = PSNR(eval_rgb_pred, eval_rgb_gt)
            logger.add(log_prefix, f"psnr.full", psnr, it)
            
            if 'rgb_mask' in ground_truth:
                # Foreground PSNR
                eval_rgb_mask_gt = ground_truth['rgb_mask'].to(self.device).view(cam.intr.H, cam.intr.W, 1)
                fg_eval_pred = rendered['rgb_volume_occupied'].view(cam.intr.H, cam.intr.W, -1)
                fg_eval_gt = eval_rgb_gt * eval_rgb_mask_gt.view(cam.intr.H, cam.intr.W, 1)
                fg_psnr = PSNR(fg_eval_pred, fg_eval_gt, eval_rgb_mask_gt).item()
                logger.add(log_prefix, f"psnr.fg", fg_psnr, it)
                
                # Background PSNR
                non_occupied_mask = ~eval_rgb_mask_gt
                # Decide bg_eval_gt
                if (non_occupied_rgb_gt_mode:=self.rgb_loss.non_occupied_rgb_gt) is None:
                    bg_eval_gt = eval_rgb_gt * non_occupied_mask.view(cam.intr.H, cam.intr.W, 1)
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
                    bg_eval_pred = eval_rgb_pred * non_occupied_mask.view(cam.intr.H, cam.intr.W, 1)
                bg_psnr = PSNR(bg_eval_pred, bg_eval_gt, non_occupied_mask).item()
                logger.add(log_prefix, f"psnr.bg", bg_psnr, it)

        #---- Special collections
        # (col) gt_rgb + pred_rgb
        if 'pred_rgb_volume' in log_collect and 'gt_rgb' in log_collect:
            add_log_img_entry(
                "coll_rgb_gt_pred", 
                torch.cat([log_collect['gt_rgb'], log_collect['pred_rgb_volume']], dim=1))
            # (col) gt_rgb + pred_rgb + cr_pred_rgb + dv_pred_rgb
            if len(objs) == 1 and len(distant_objs) == 1:
                add_log_img_entry(
                    "coll_rgb_gt_pred_cr_dv", 
                    torch.cat([
                        log_collect['gt_rgb'], log_collect['pred_rgb_volume'], 
                        log_collect[f'pred_rgb_volume.{objs[0].id}.in_total'], 
                        log_collect[f'pred_rgb_volume.{distant_objs[0].id}.in_total'], 
                    ], dim=1))
        # (col) gt_mask + pred_mask
        if 'pred_mask_volume' in log_collect and 'gt_rgb_mask' in log_collect:
            add_log_img_entry(
                "coll_mask_gt_pred", 
                torch.cat([log_collect['gt_rgb_mask'], log_collect['pred_mask_volume']], dim=1))
        # (col) mono_normals + pred_normals
        if 'pred_normals_volume' in log_collect and 'gt_rgb_mono_normals' in log_collect:
            add_log_img_entry(
                "coll_normals_gt_pred", 
                torch.cat([log_collect['gt_rgb_mono_normals'], log_collect['pred_normals_volume']], dim=1))

        return log_collect

    @torch.no_grad()
    def validate_cameras(self, sample: dict, ground_truth: dict, logger: Logger, it: int):
        """ Validate one camera or multiple cameras at one frame.
        If validating on multiple cameras (i.e. `sample` and `groud_truth` contains multiple cameras), 
            will automatically make paddings and concat multiple cameras' validations

        Args:
            sample (dict): `sample` from dataio.dataloader.ImageDataset
            ground_truth (dict): `ground_truth` from dataio.dataloader.ImageDataset
            logger (Logger): The logger
            it (int): Current iteration
        """
        
        sample, ground_truth = next(zip_two_nested_dict(sample, ground_truth)) # NOTE: bs=1 in this case
        scene_id = sample['scene_id']
        cam_id = sample['cam_id']
        frame_id = sample['frame_id']
        
        #---- Camera-wise validation
        if isinstance(cam_id, list):
            log_collect_all_cams = {}
            for ci, gt in enumerate(zip_dict(ground_truth)):
                log_collect_all_cams[cam_id[ci]] = self._validate_single_cam(
                    scene_id, cam_id[ci], frame_id, ground_truth=gt, logger=logger, it=it, 
                    should_log_img=False, log_prefix=f"validate_camera.{scene_id}/{cam_id[ci]}")
            #---- Collate results from different cameras and log them
            log_collated_all_cams = collate_nested_dict(list(log_collect_all_cams.values()), stack=False)
            for key, imgs in log_collated_all_cams.items():
                logger.add_imgs(f"validate_camera.{scene_id}", key, torch.cat(pad_images_to_same_size(imgs, padding='top_left'), dim=2), it=it)
        else:
            self._validate_single_cam(
                scene_id, cam_id, frame_id, ground_truth=ground_truth, logger=logger, it=it, 
                should_log_img=True, log_prefix=f"validate_camera.{scene_id}/{cam_id}")
        
        #---- Model-wise validation (if any)
        for class_name, model_id_map in self.asset_bank.class_name_infos.items():
            for model_id, scene_obj_id_list in model_id_map.items():
                model: AssetModelMixin = self.asset_bank[model_id]
                if model.assigned_to == AssetAssignment.OBJECT:
                    scene_id, obj_id = scene_obj_id_list[0]
                    scene = self.scene_bank[scene_id]
                    obj = scene.all_nodes[obj_id]
                    model.val(scene=scene, obj=obj, it=it, logger=logger, log_prefix=f"validate_camera.{scene_id}.{class_name}.{obj.id}")
                elif model.assigned_to == AssetAssignment.SCENE:
                    scene_id, _ = scene_obj_id_list[0]
                    scene = self.scene_bank[scene_id]
                    model.val(scene=scene, obj=None, it=it, logger=logger, log_prefix=f"validate_camera.{scene_id}.{class_name}")
                elif model.assigned_to == AssetAssignment.MULTI_OBJ_ONE_SCENE:
                    scene_id, _ = scene_obj_id_list[0]
                    model.val(scene=scene, obj=None, it=it, logger=logger, log_prefix=f"validate_camera.{scene_id}.{class_name}")
                elif model.assigned_to == AssetAssignment.MULTI_OBJ:
                    model.val(scene=None, obj=None, it=it, logger=logger, log_prefix=f"validate_camera.{class_name}")
                elif model.assigned_to == AssetAssignment.MULTI_SCENE:
                    pass
                elif model.assigned_to == AssetAssignment.MISC:
                    model.val(scene=None, obj=None, it=it, logger=logger, log_prefix=f"validate_camera.{class_name}")

def main_function(args: ConfigDict):
    exp_dir = args.exp_dir
    os.makedirs(exp_dir, exist_ok=True)
    
    if args.get('wait_for', None) is not None:
        log.info(f"=> Cur pid={os.getpid()}, waiting for pid={args.wait_for}. \nexp_dir: {exp_dir}")
        wait_for_pid(int(args.wait_for))

    seed = args.get('seed', 42)
    init_env(args, seed=seed)

    #----------------------------
    #-------- Shortcuts ---------
    rank = get_rank()
    local_rank = get_local_rank()
    world_size = get_world_size()
    device = torch.device('cuda', local_rank)
    
    i_backup = int(args.training.i_backup // world_size) if args.training.i_backup > 0 else -1
    i_val = int(args.training.i_val // world_size) if args.training.i_val > 0 else -1
    i_save = int(args.training.i_save) if args.training.i_save > 0 else -1
    if 'i_log' in args.training:
        i_log = max(int(args.training.i_log // world_size), 1) if args.training.i_log > 0 else -1
    else:
        i_log = 1
    # TODO
    # if 'i_val_benchmark' in args.training:
    #     i_val_benchmark = int(args.training.i_val_benchmark // world_size) if args.training.i_val_benchmark > 0 else -1
    log_grad = args.training.get('log_grad', False)
    log_param = args.training.get('log_param', False)

    # Logger
    logger = Logger(
        root=exp_dir,
        img_root=os.path.join(exp_dir, 'imgs'),
        monitoring=args.training.get('monitoring', 'tensorboard'),
        monitoring_dir=os.path.join(exp_dir, 'events'),
        rank=rank, is_master=is_master(), multi_process_logging=(world_size > 1))

    log.info(f"=> Experiments dir: {exp_dir}")

    if is_master():
        # Backup codes
        backup_project(
            os.path.join(exp_dir, 'backup'), "./", 
            ["app", "code_multi", "code_single", "dataio", "nr3d_lib"], 
            [".py", ".h", ".cpp", ".cuh", ".cu", ".sh"]
        )

        # Save configs
        save_config(args, os.path.join(exp_dir, 'config.yaml'))
    
    #---------------------------------------------
    #-----------     Scene Bank     --------------
    #---------------------------------------------
    dataset_impl: DatasetIO = import_str(args.dataset_cfg.target)(args.dataset_cfg.param)
    asset_bank = AssetBank(args.assetbank_cfg)
    
    scene_bank: IDListedDict[Scene] = IDListedDict()
    scenebank_root = os.path.join(args.exp_dir, 'scenarios')
    if is_master():
        log.info("=> Creating scene_bank...")
        scene_bank, _ = create_scene_bank(
            dataset=dataset_impl, device=device, 
            scenebank_cfg=args.scenebank_cfg, 
            drawable_class_names=asset_bank.class_name_configs.keys(),
            misc_node_class_names=asset_bank.misc_node_class_names, 
            scenebank_root=scenebank_root
        )
        log.warning("=> Done creating scene_bank.")
    if world_size > 1:
        dist.barrier()
    if not is_master():
        scene_bank, _ = load_scene_bank(scenebank_root, device=device)

    #---------------------------------------------
    #------------     Renderer     ---------------
    #---------------------------------------------
    renderer = SingleVolumeRenderer(args.renderer)
    renderer.train()
    for scene in scene_bank:
        # NOTE: When training, set all observer's near&far to a larger value
        for obs in scene.observers.values():
            obs.near = renderer.config.near # NOTE: Modify scene_bank observers in advance
            obs.far = renderer.config.far

    #---------------------------------------------
    #-----------     Asset Bank     --------------
    #---------------------------------------------
    asset_bank.create_asset_bank(scene_bank, optim_cfg=args.training.optim, device=device)
    asset_bank.to(device)
    # log.info(asset_bank)
    if is_master():
        model_txt = os.path.join(exp_dir, 'model.txt')
        with open(model_txt, 'w') as f:
            f.write(repr(asset_bank))
        log.info(f"=> Model structure saved to {model_txt}")

    #---------------------------------------------
    #---     Load assets to scene objects     ----
    #---------------------------------------------
    for scene in scene_bank:
        scene.load_assets(asset_bank)
    renderer.populate(asset_bank)

    #---------------------------------------------
    #------------     Optimizer     --------------
    #---------------------------------------------
    optimizer = optim.Adam(asset_bank.param_groups)
    asset_bank.configure_clip_grad_group(scene_bank, args.training.get('clip_grad_val', None))
    
    #---------------------------------------------
    #-------------     Dataset     ---------------
    #---------------------------------------------
    log.info(f"=> Start loading data, for exp: {exp_dir}")
    scene_dataloader_train = SceneDataLoader(scene_bank, dataset_impl, config=args.training.dataloader, device=device, is_master=is_master())
    scene_dataloader_val = SceneDataLoader(scene_bank, dataset_impl, config=args.training.val_dataloader, device=device, is_master=is_master())
    log.info(f"=> Done loading data.")
    
    def cycle(dataloader):
        epoch_idx = 0
        while True:
            for (s,g) in dataloader:
                yield s, g
            epoch_idx += 1
            if args.ddp:
                dataloader.dataset.sampler.set_epoch(epoch_idx)
    
    sampler_kwargs = {'scene_sample_mode': 'weighted_by_len', 
                     'ddp': args.ddp, 'seed': seed, 'drop_last': False}
    if params:=args.training.dataloader.get('pixel_dataset', None):
        params = params.copy()
        joint = params.pop('joint', False)
        if not joint:
            pixel_dataset = PixelDataset(scene_dataloader_train, **params, **sampler_kwargs)
        else:
            pixel_dataset = JointFramePixelDataset(scene_dataloader_train, **params, **sampler_kwargs)
        pixel_dataloader_cyc = cycle(pixel_dataset.get_dataloader())
    else:
        pixel_dataset = None
        pixel_dataloader_cyc = None
    
    if params:=args.training.dataloader.get('lidar_dataset', None):
        lidar_dataset = LidarDataset(scene_dataloader_train, **params, **sampler_kwargs)
        lidar_dataloader_cyc = cycle(lidar_dataset.get_dataloader())
    else:
        lidar_dataset = None
        lidar_dataloader_cyc = None
    
    if params:=args.training.dataloader.get('image_patch_dataset', None):
        image_patch_dataset = ImagePatchDataset(scene_dataloader_train, **params, **sampler_kwargs)
        image_patch_dataloader_cyc = cycle(image_patch_dataset.get_dataloader())
    else:
        image_patch_dataset = None
        image_patch_dataloader_cyc = None
    
    # if params:=args.training.val_dataloader.get('lidar_dataset', None):
    #     lidar_val_dataset = LidarDataset(scene_dataloader_val, **params)
    #     lidar_val_dataloader_cyc = cycle(lidar_val_dataset.get_dataloader())
    if params:=args.training.val_dataloader.get('image_dataset', None):
        image_val_dataset = ImageDataset(scene_dataloader_val, **params, **sampler_kwargs)
        image_val_dataloader_cyc = cycle(image_val_dataset.get_dataloader())
    
    #---------------------------------------------
    #----------     Checkpoints     --------------
    #---------------------------------------------
    # Checkpoints
    checkpoint_io = CheckpointIO(checkpoint_dir=os.path.join(exp_dir, 'ckpts'), allow_mkdir=is_master())
    if world_size > 1:
        dist.barrier()

    # Register modules to checkpoint
    checkpoint_io.register_modules(
        asset_bank=asset_bank,
        optimizer=optimizer,
    )
    
    # Load checkpoints
    load_dict = checkpoint_io.load_file(
        args.training.get('ckpt_file', None), # Bypass ckpt file
        ignore_keys=args.training.ckpt_ignore_keys,
        only_use_keys=args.training.ckpt_only_use_keys,
        map_location=device)
    
    logger.load_stats('stats.p') # Used for plotting
    it = load_dict.get('global_step', 0)
    epoch_idx = load_dict.get('epoch_idx', 0)
    
    #---------------------------------------------
    #---------     Training tools     ------------
    #---------------------------------------------
    # Build scheduler
    scheduler = get_scheduler(args.training.scheduler, optimizer, last_epoch=it-1)
    trainer = Trainer(
        config=args.training, renderer=renderer, asset_bank=asset_bank, scene_bank=scene_bank, 
        dataset=scene_dataloader_train, pixel_dataset=pixel_dataset, device_ids=args.device_ids, 
        i_log=i_log, i_val=i_val)
    trainer_module = trainer

    # Initialize (e.g. pretrain)
    if (it==0) and is_master():
        just_done_initialize = trainer.initialize(logger=logger)
        if just_done_initialize:
            checkpoint_io.save(filename='0.pt', global_step=it, epoch_idx=epoch_idx)

    if args.ddp:
        # NOTE: When there are no intersecting rays with distant-view models, \
        #       their parameters will not be used and hence raise an error.
        #       For now, we set find_unused_parameters to True
        trainer = DDP(trainer, device_ids=args.device_ids, output_device=local_rank, find_unused_parameters=True)
        trainer_module = trainer.module

    # Default to [True]], since grad scaler is harmless even not using half precision.
    enable_grad_scaler = args.training.get("enable_grad_scaler", True)
    scaler_pixel = GradScaler(init_scale=128.0, enabled=enable_grad_scaler)
    scaler_lidar = GradScaler(init_scale=128.0, enabled=enable_grad_scaler)
    scaler_image_patch = GradScaler(init_scale=128.0, enabled=enable_grad_scaler)

    t0 = time.time()
    log.info(f"=> Start [train], it={it}, lr={optimizer.param_groups[0]['lr']}, in {exp_dir}")
    end = (it >= args.training.num_iters)
    # total_start = time.time()
    # iter_timestamps = []

    with tqdm(range(args.training.num_iters), disable=not is_master()) as pbar:
        if is_master():
            pbar.update(it)
        # @profile
        def train_step():
            nonlocal it, epoch_idx, t0, end
            int_it = int(it // world_size)
            local_it = it + rank
            trainer_module.preprocess_per_train_step(local_it, logger=logger)

            #----------------------------------------------------------------------------
            #----------------------------     Validate     ------------------------------
            #----------------------------------------------------------------------------
            if i_val > 0 and int_it % i_val == 0:
                renderer.eval()
                asset_bank.eval()
                with torch.no_grad():
                    sample, ground_truth = next(image_val_dataloader_cyc)
                    trainer_module.validate_cameras(sample, ground_truth, logger, local_it)
                    del sample, ground_truth

            if it >= args.training.num_iters:
                end = True
                return

            #----------------------------------------------------------------------------
            #-----------------------------     Train     --------------------------------
            #----------------------------------------------------------------------------
            asset_bank.train()
            renderer.train()
            start_time = time.time()
            if i_log > 0 and int_it % i_log == 0:
                for pg in optimizer.param_groups:
                    logger.add('learning rates', pg['name'], pg['lr'], local_it)
            
            loss_total = 0.
            #----------------------------------------------------------------------------
            #-------------------------     Train - pixel     ----------------------------
            #----------------------------------------------------------------------------
            if pixel_dataloader_cyc is not None:
                optimizer.zero_grad()
                sample, ground_truth = next(pixel_dataloader_cyc)
                scene = scene_bank[sample['scene_id'][0]]
                
                #----------------------------------------------------------------------------
                ret, losses = trainer('pixel', sample, ground_truth, local_it, logger=logger)
                #----------------------------------------------------------------------------
                
                losses['total'] = sum([v for v in losses.values()])
                
                # losses['total'].backward()
                scaler_pixel.scale(losses['total']).backward()
                scaler_pixel.unscale_(optimizer) # Unscale param's grad in-place (for normal grad clipping and debugging)
                asset_bank.apply_clip_grad()
                grad_norms = calc_grad_norm(**asset_bank) if log_grad else {}
                
                # optimizer.step()
                scaler_pixel.step(optimizer)
                scaler_pixel.update()
                scheduler.step(it)  # NOTE: important! when world_size is not 1
                
                loss_total += losses['total'].item()
            
                #----------------------------------------------------------------------------
                #----------------------------     Logging     -------------------------------
                with torch.no_grad():
                    if i_log > 0 and int_it % i_log == 0:
                        #---------------------
                        # Log training related
                        for k, v in losses.items():
                            logger.add(f"train_step_pixel.losses", k, v.item(), local_it)
                        
                        for k, v in grad_norms.items():
                            logger.add(f"train_step_pixel.grad_norm", k, v, local_it)
                        
                        if log_param:
                            logger.add_nested_dict(f"train_step_pixel.dbg_main_grad", d=asset_bank.get_scene_main_model(scene).stat_param(with_grad=True), it=local_it)
                        
                        # Log raw buffer
                        logger.add_nested_dict(f"train_step_pixel.scene={scene.id}", d=ret['volume_buffer'], it=local_it)
                        # Log per object
                        for obj_id, raw_ret in ret['raw_per_obj_model'].items():
                            logger.add_nested_dict(f"train_step_pixel.obj={obj_id}", d=raw_ret, it=local_it)

                del scene, losses, ret, sample, ground_truth

            #----------------------------------------------------------------------------
            #-------------------------     Train - lidar     ----------------------------
            #----------------------------------------------------------------------------
            if lidar_dataloader_cyc is not None:
                optimizer.zero_grad()
                sample, ground_truth = next(lidar_dataloader_cyc)
                scene = scene_bank[sample['scene_id'][0]]

                #----------------------------------------------------------------------------
                ret, losses = trainer('lidar', sample, ground_truth, local_it, logger=logger)
                #----------------------------------------------------------------------------
                
                losses['total'] = sum([v for v in losses.values()])
                
                # losses['total'].backward()
                scaler_lidar.scale(losses['total']).backward()
                scaler_lidar.unscale_(optimizer) # Unscale param's grad in-place (for normal grad clipping and debugging)
                asset_bank.apply_clip_grad()
                grad_norms = calc_grad_norm(**asset_bank) if log_grad else {}
                
                # optimizer.step()
                scaler_lidar.step(optimizer)
                scaler_lidar.update()
                scheduler.step(it)  # NOTE: Important! when world_size is not 1
                
                loss_total += losses['total'].item()
                
                #----------------------------------------------------------------------------
                #----------------------------     Logging     -------------------------------
                with torch.no_grad():
                    if i_log > 0 and int_it % i_log == 0:
                        #---------------------
                        # Log training related
                        for k, v in losses.items():
                            logger.add("train_step_lidar.losses", k, v.item(), local_it)
                        
                        for k, v in grad_norms.items():
                            logger.add("train_step_lidar.grad_norm", k, v, local_it)

                        if log_param:
                            logger.add_nested_dict(f"train_step_lidar.dbg_main_grad", d=asset_bank.get_scene_main_model(scene).stat_param(with_grad=True), it=local_it)

                        # Log raw buffer
                        logger.add_nested_dict(f"train_step_lidar.scene={scene.id}", d=ret['volume_buffer'], it=local_it)
                        # Log per object
                        for obj_id, raw_ret in ret['raw_per_obj_model'].items():
                            logger.add_nested_dict(f"train_step_lidar.obj={obj_id}", d=raw_ret, it=local_it)
                del scene, losses, ret, sample, ground_truth

            #----------------------------------------------------------------------------
            #----------------------     Train - image_patch     -------------------------
            #----------------------------------------------------------------------------
            if image_patch_dataloader_cyc is not None:
                optimizer.zero_grad()
                sample, ground_truth = next(image_patch_dataloader_cyc)
                scene = scene_bank[sample['scene_id'][0]]
                
                #----------------------------------------------------------------------------
                ret, losses = trainer('image_patch', sample, ground_truth, local_it, logger=logger)
                #----------------------------------------------------------------------------
                
                losses['total'] = sum([v for v in losses.values()])
                
                # losses['total'].backward()
                scaler_image_patch.scale(losses['total']).backward()
                scaler_image_patch.unscale_(optimizer) # Unscale param's grad in-place (for normal grad clipping and debugging)
                asset_bank.apply_clip_grad()
                grad_norms = calc_grad_norm(**asset_bank) if log_grad else {}
                
                # optimizer.step()
                scaler_image_patch.step(optimizer)
                scaler_image_patch.update()
                scheduler.step(it)  # NOTE: important! when world_size is not 1
                
                loss_total += losses['total'].item()
                
                #----------------------------------------------------------------------------
                #----------------------------     Logging     -------------------------------
                with torch.no_grad():
                    if i_log > 0 and int_it % i_log == 0:
                        #---------------------
                        # Log training related
                        for k, v in losses.items():
                            logger.add("train_step_image_patch.losses", k, v.item(), local_it)
                        
                        for k, v in grad_norms.items():
                            logger.add("train_step_image_patch.grad_norm", k, v, local_it)
                        
                        if log_param:
                            logger.add_nested_dict(f"train_step_image_patch.dbg_main_grad", d=asset_bank.get_scene_main_model(scene).stat_param(with_grad=True), it=local_it)
                        
                        # Log raw buffer
                        logger.add_nested_dict(f"train_step_image_patch.scene={scene.id}", d=ret['volume_buffer'], it=local_it)
                        # Log per object
                        for obj_id, raw_ret in ret['raw_per_obj_model'].items():
                            logger.add_nested_dict(f"train_step_image_patch.obj={obj_id}", d=raw_ret, it=local_it)
                del scene, losses, ret, sample, ground_truth

            #----------------------------------------------------------------------------
            #----------------------     End of one iteration     ------------------------
            #----------------------------------------------------------------------------
            asset_bank.postprocess_per_train_step(it, logger=logger)
            end_time = time.time()
            log.debug(f"=> One iteration time is {(end_time - start_time):.2f}")

            it += world_size
            if is_master():
                pbar.update(world_size)

            #----------------------------------------------------------------------------
            #----------------------------     Saving     --------------------------------
            #----------------------------------------------------------------------------
            if i_save > 0 and time.time() - t0 > i_save:
                if is_master():
                    checkpoint_io.save(filename='latest.pt', global_step=it, epoch_idx=epoch_idx)
                # This will be used for plotting
                logger.save_stats('stats.p')
                t0 = time.time()

            if is_master():
                # pbar.set_postfix(lr=optimizer.param_groups[0]['lr'], loss_total=losses['total'].item(), loss_img=losses['loss_rgb'].item())
                pbar.set_postfix(lr=optimizer.param_groups[0]['lr'], loss_total=loss_total)
                if i_backup > 0 and int_it % i_backup == 0 and it > 0:
                    checkpoint_io.save(filename=f'{it:08d}.pt', global_step=it, epoch_idx=epoch_idx)
            
        while it <= args.training.num_iters and not end:
            try:
                # iter_timestamps.append(f"{time.time() - total_start:.3f}\n")
                train_step()
            except KeyboardInterrupt:
                if is_master():
                    checkpoint_io.save(filename='latest.pt', global_step=it, epoch_idx=epoch_idx)
                logger.save_stats('stats.p')
                sys.exit()
            except Exception as e:
                print(f"Error occurred in: {exp_dir}")
                raise e

    if is_master():
        checkpoint_io.save(filename=f'final_{it:08d}.pt', global_step=it, epoch_idx=epoch_idx)
        logger.save_stats('stats.p')
        log.info("Everything done.")

    # with open("./dev_tools/demo_3090_speed.txt", 'w') as f:
    #     f.writelines(iter_timestamps)

def make_parser():
    from nr3d_lib.config import BaseConfig
    bc = BaseConfig()
    return bc

if __name__ == "__main__":
    bc = make_parser()
    main_function(bc.parse(print_config=False))