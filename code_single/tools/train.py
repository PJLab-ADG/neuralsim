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

import time
from tqdm import tqdm
from copy import deepcopy
from numbers import Number
from typing import Dict, Literal, List, Tuple, Union

import torch
import torch.nn as nn
from torch import optim
import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.autograd.anomaly_mode import set_detect_anomaly
from torch.nn.parallel import DistributedDataParallel as DDP

from nr3d_lib.fmt import log
from nr3d_lib.logger import Logger
from nr3d_lib.plot import color_depth, scene_flow_to_rgb
from nr3d_lib.checkpoint import CheckpointIO
from nr3d_lib.config import ConfigDict, save_config
from nr3d_lib.distributed import get_local_rank, init_env, is_master, get_rank, get_world_size
from nr3d_lib.utils import IDListedDict, collate_nested_dict, import_str, backup_project, is_scalar, \
    nested_dict_items, pad_images_to_same_size, wait_for_pid, zip_dict, zip_two_nested_dict
from nr3d_lib.profile import profile, Profiler

from nr3d_lib.graphics.utils import PSNR
from nr3d_lib.models.utils import calc_grad_norm
from nr3d_lib.models.importance import ErrorMap, ImpSampler

from app.models.asset_base import AssetAssignment, AssetModelMixin
from app.renderers import SingleVolumeRenderer
from app.resources import Scene, AssetBank, create_scene_bank, load_scene_bank
from app.resources.observers import Camera, MultiCamBundle, RaysLidar, MultiRaysLidarBundle

from dataio.scene_dataset import SceneDataset
from dataio.data_loader import SceneDataLoader, ImageDataset, ImagePatchDataset, LidarDataset, \
    PixelDataset, JointFramePixelDataset


class Trainer(nn.Module):
    def __init__(
        self, 
        config, 
        renderer: SingleVolumeRenderer, 
        asset_bank: AssetBank,
        scene_bank: List[Scene], 
        scene_loader: SceneDataLoader, 
        
        train_pixel_dataset: Union[PixelDataset, JointFramePixelDataset] = None, 
        train_image_dataset: ImageDataset = None, 
        train_lidar_dataset: LidarDataset = None, 
        train_image_patch_dataset: ImagePatchDataset = None, 
        
        val_image_dataset: ImageDataset = None, 
        
        i_log: int = -1, i_val: int = -1, 
        device_ids=[0]) -> None:

        super().__init__()
        self.device = torch.device(f'cuda:{device_ids[0]}')
        
        self.config = config
        self.renderer = renderer
        self.scene_bank = scene_bank
        self.asset_bank = asset_bank
        
        self.scene_loader = scene_loader
        self.train_pixel_dataset = train_pixel_dataset
        self.train_image_dataset = train_image_dataset
        self.train_lidar_dataset = train_lidar_dataset
        self.train_image_patch_dataset = train_image_patch_dataset
        self.val_image_dataset = val_image_dataset
        self.train_datasets = [self.train_pixel_dataset, self.train_image_dataset, self.train_lidar_dataset, self.train_image_patch_dataset]
        self.val_datasets = [self.val_image_dataset]
        
        self.i_log = i_log
        self.i_val = i_val

        self.renderer.train()
        self.asset_bank.train()
        
        #------------ Initialize error map samplers
        self.imp_samplers: Dict[str, Dict[str,ImpSampler]] = None
        self.error_map_enable_after: int = 0
        self.error_map_ignore_not_occupied: bool = True
        self.error_map_on_classnames: List[str] = []
        
        error_map_cfg: dict = self.config.get('error_map', None)
        if error_map_cfg is not None:
            assert 'error_map_hw' in error_map_cfg, "Please specify `error_map_hw`"
            error_map_cfg = deepcopy(error_map_cfg)
            
            self.error_map_enable_after: int = error_map_cfg.pop('enable_after', 0)
            self.error_map_ignore_not_occupied: bool = error_map_cfg.pop('ignore_not_occupied', True)
            self.error_map_on_classnames: List[str] = error_map_cfg.pop('on_classnames', [])
            frac_uniform: float = error_map_cfg.pop('frac_uniform', 0.5)
            frac_rgb_err: float = error_map_cfg.pop('frac_rgb_err', 0.5)
            frac_mask_err: float = error_map_cfg.pop('frac_mask_err', 0)
            frac_on_classnames: float = error_map_cfg.pop('frac_on_classnames', 0)
            
            imp_samplers = nn.ModuleDict()
            for scene_id, scene in self.scene_bank.items():
                imp_samplers[scene_id] = nn.ModuleDict()
                for cam_id, cam in scene.get_cameras(only_valid=False).items():
                    n_images = len(cam)
                    error_maps = {}
                    # Basic RGB error map
                    error_maps['rgb'] = (ErrorMap(n_images, **error_map_cfg, dtype=torch.float, device=self.device), 
                                         frac_rgb_err)
                    # Optional error maps that focus on class_name(s) specified in error_map_cfg['on_classnames']
                    if (frac_on_classnames > 0) and len(self.error_map_on_classnames) > 0:
                        error_maps['focus_on'] = (ErrorMap(n_images, max_pdf=1.0, **error_map_cfg, dtype=torch.float, device=self.device), 
                                                  frac_on_classnames)
                    # Optional error maps on mask errors
                    if (frac_mask_err > 0) and (self.occupancy_mask_loss is not None):
                        error_maps['mask'] = (ErrorMap(n_images, max_pdf=1.0, **error_map_cfg, dtype=torch.float, device=self.device), 
                                              frac_mask_err)
                    sampler = ImpSampler(error_maps, frac_uniform=frac_uniform)
                    imp_samplers[scene_id][cam_id] = sampler
            self.imp_samplers = imp_samplers
            self.train_pixel_dataset.set_imp_samplers(self.imp_samplers, enable_after=self.error_map_enable_after)

        # self.pixel_step_require_vw_in_total = False
        # self.image_patch_step_require_vw_in_total = False
        # self.lidar_step_require_vw_in_total = False

        #------------ Configure losses
        loss_cfg = self.config.losses
        drawable_class_names = list(self.asset_bank.class_name_configs.keys())
        
        self.rgb_loss = None # Avaiable in mode: [pixel, image_patch]
        if 'rgb' in loss_cfg:
            from app.loss import PhotometricLoss
            self.rgb_loss = PhotometricLoss(**loss_cfg['rgb'])

        self.rgb_s3im_loss = None # Avaiable in mode: [pixel, image_patch]
        if 'rgb_s3im' in loss_cfg:
            from app.loss import S3IMLoss
            self.rgb_s3im_loss = S3IMLoss(**loss_cfg['rgb_s3im'])

        self.rgb_perceptual_loss = None # Avaiable in mode: [image_patch]
        if 'rgb_perceptual' in loss_cfg:
            from app.loss import PerceptualLoss
            self.rgb_perceptual_loss = PerceptualLoss(**loss_cfg['rgb_perceptual'], device=self.device)

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
            from app.loss import MonoDepthLoss
            self.mono_depth_loss = MonoDepthLoss(**loss_cfg['mono_depth'], debug_val_every=self.i_val)
            # if self.mono_depth_loss.requires_render_per_class:
            #     self.image_patch_step_require_vw_in_total = True

        self.mono_normals_loss = None # Avaiable in mode: [pixel (optional), image_patch]
        if 'mono_normals' in loss_cfg:
            from app.loss import MonoNormalLoss
            self.mono_normals_loss = MonoNormalLoss(**loss_cfg['mono_normals'], debug_val_every=self.i_val)
            # if self.mono_normals_loss.requires_render_per_class:
            #     self.image_patch_step_require_vw_in_total = True
            #     if self.mono_normals_loss.apply_in_pixel_train_step:
            #         self.pixel_step_require_vw_in_total = True

        self.road_normals_loss = None
        if 'road_normals' in loss_cfg:
            from app.loss import RoadNormalLoss
            self.road_normals_loss = RoadNormalLoss(**loss_cfg['road_normals'])
            # if self.road_normals_loss.requires_render_per_class:
            #     self.image_patch_step_require_vw_in_total = True
            #     if self.road_normals_loss.apply_in_pixel_train_step:
            #         self.pixel_step_require_vw_in_total = True

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

        self.color_net_reg_loss = None # Regularization. Available in mode: [pixel, image_patch]
        if 'color_net_reg' in loss_cfg:
            from app.loss import ColorLipshitzRegLoss
            self.color_net_reg_loss = ColorLipshitzRegLoss(**loss_cfg['color_net_reg'], drawable_class_names=drawable_class_names)

        self.flow_cycle_loss = None # Regularization. Available in mode: [pixel, image_patch, lidar]
        if 'flow' in loss_cfg:
            from app.loss import FlowLoss
            self.flow_cycle_loss = FlowLoss(**loss_cfg['flow'], drawable_class_names=drawable_class_names)

        self.weight_reg_loss = None # Regularization. Avaiable in mode: [pixel, image_patch, lidar]
        if 'weight_reg' in loss_cfg:
            from app.loss import WeightRegLoss
            self.weight_reg_loss = WeightRegLoss(**loss_cfg['weight_reg'], drawable_class_names=drawable_class_names)

        if self.eikonal_loss is not None:
            self.renderer._config_train.with_normal = True
        if self.clearance_loss is not None:
            self.renderer._config_train.with_near_sdf = True

        #------------ DEBUG
        self.debug_grad = self.config.get('debug_grad', False)
        self.debug_grad_detect_anomaly = self.config.get('debug_grad_detect_anomaly', False)
        self.debug_grad_threshold = self.config.get('debug_grad_threshold', 3.0)
        self.debug_ret = self.config.get('debug_ret', False)
        self.debug_raise = False
        
        if self.debug_grad_detect_anomaly:
            set_detect_anomaly(True)
    
    def training_initialize(self, logger=None) -> bool:
        log.info("=> Start initialize prepcess...")
        for class_name, model_id_map in self.asset_bank.class_name_infos.items():
            for model_id, scene_obj_id_list in model_id_map.items():
                log.info(f"=> Initializing model: {model_id}")
                model: AssetModelMixin = self.asset_bank[model_id]
                if model.assigned_to == AssetAssignment.OBJECT:
                    scene_id, obj_id = scene_obj_id_list[0]
                    scene = self.scene_bank[scene_id]
                    obj = scene.all_nodes[obj_id]
                    model.asset_training_initialize(scene=scene, obj=obj, config=model.initialize_cfg, logger=logger, log_prefix=model_id)
                elif model.assigned_to == AssetAssignment.SCENE:
                    scene_id, _ = scene_obj_id_list[0]
                    scene = self.scene_bank[scene_id]
                    model.asset_training_initialize(scene=scene, obj=None, config=model.initialize_cfg, logger=logger, log_prefix=model_id)
                elif model.assigned_to == AssetAssignment.MULTI_OBJ_ONE_SCENE:
                    scene_id, _ = scene_obj_id_list[0]
                    scene = self.scene_bank[scene_id]
                    model.asset_training_initialize(scene=scene, obj=None, config=model.initialize_cfg, logger=logger, log_prefix=model_id)
                elif model.assigned_to == AssetAssignment.MULTI_OBJ:
                    model.asset_training_initialize(scene=None, obj=None, config=model.initialize_cfg, logger=logger, log_prefix=model_id)
                elif model.assigned_to == AssetAssignment.MULTI_SCENE:
                    pass
                elif model.assigned_to == AssetAssignment.MISC:
                    model.asset_training_initialize(scene=None, obj=None, config=model.initialize_cfg, logger=logger, log_prefix=model_id)
        log.info("=> Done initialize prepcess.")
        return True

    @profile
    def training_before_per_step(self, cur_it: int, logger: Logger = None):
        self.asset_bank.training_before_per_step(cur_it, logger=logger)
        for dataset in self.train_datasets:
            if dataset is not None:
                dataset.cur_it = cur_it
        for dataset in self.val_datasets:
            if dataset is not None:
                dataset.cur_it = cur_it

    @profile
    def training_after_per_step(self, cur_it: int, logger: Logger = None):
        self.asset_bank.training_after_per_step(cur_it, logger=logger)

    @profile
    def forward(
        self, mode: Literal['view', 'pixel', 'image_patch', 'lidar'], 
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
        
        if mode == 'view': # view
            ret, losses = self.train_step_view(sample, ground_truth, it, logger=logger)
        elif mode == 'pixel': # rays, random
            ret, losses = self.train_step_pixel(sample, ground_truth, it, logger=logger)
        elif mode == 'image_patch': # rays of image patch
            ret, losses = self.train_step_image_patch(sample, ground_truth, it, logger=logger)
        elif mode == 'lidar': # rays of LiDAR beams
            ret, losses = self.train_step_lidar(sample, ground_truth, it, logger=logger)
        else:
            raise RuntimeError(f"Invalid mode={mode}")
        
        if self.debug_ret:
            #----------------------------------------------------------------------------
            #-------------------   debug return
            for *k, v in nested_dict_items(ret):
                if not isinstance(v, torch.Tensor):
                    continue
                if torch.isnan(v).any():
                    err_msg = "NAN found in return: " + '.'.join(k)
                    log.error(err_msg)
                    assert not self.debug_raise, err_msg
                elif torch.isinf(v).any():
                    err_msg = "INF found in return: " + '.'.join(k)
                    log.error(err_msg)
                    assert not self.debug_raise, err_msg

            for *k, v in nested_dict_items(losses):
                if not isinstance(v, torch.Tensor):
                    continue
                if torch.isnan(v).any():
                    err_msg = "NAN found in loss: " + '.'.join(k)
                    log.error(err_msg)
                    assert not self.debug_raise, err_msg
                elif torch.isinf(v).any():
                    err_msg = "INF found in loss: " + '.'.join(k)
                    log.error(err_msg)
                    assert not self.debug_raise, err_msg

        if self.debug_grad:
            scene = self.scene_bank[sample['scene_id'][0]]
            #----------------------------------------------------------------------------
            #-------------------   debug grad
            self.debug_obj = scene.all_nodes_by_class_name[scene.main_class_name][0]
            debug_obj_model = self.debug_obj.model
            debug_obj_model_id = debug_obj_model.id
            debug_pg = {k:v for k,v in debug_obj_model.named_parameters() if v.requires_grad}
            #---- NOTE: Retain grad in case of need.
            for *k, v in nested_dict_items(ret):
                if hasattr(v, 'grad_fn') and v.grad_fn is not None:
                    v.retain_grad()
            
            for k, v in losses.items():
                if hasattr(v, 'grad_fn') and v.grad_fn is not None:
                    v.retain_grad()
            
            #---- NOTE: Check per return w.r.t. per parameter
            for *k, v in nested_dict_items(ret):
                if hasattr(v, 'grad_fn') and v.grad_fn is not None:
                    try:
                        debug_grad = torch.autograd.grad(v.mean(), debug_pg.values(), retain_graph=True, allow_unused=True)
                        debug_pg_grad = {pn:pg for pn,pg in zip(debug_pg.keys(), debug_grad)}
                        for pn, pg in debug_pg_grad.items():
                            if pg is None:
                                continue
                            if torch.isnan(pg).any():
                                err_msg = f"NAN! d({k})_d({pn})"
                                log.error(err_msg)
                                assert not self.debug_raise, err_msg
                            elif torch.isinf(pg).any():
                                err_msg = f"INF! d({k})_d({pn})"
                                log.error(err_msg)
                                assert not self.debug_raise, err_msg
                        del debug_grad, debug_pg_grad
                    except Exception as e:
                        msg = f"d({'.'.join(k)})_dpg: " + repr(e)
                        log.error(msg)
                        assert not self.debug_raise, err_msg

            # loss_grads = dict()
            #---- NOTE: Check per loss w.r.t per parameter
            for k, v in losses.items():
                if hasattr(v, 'grad_fn') and v.grad_fn is not None:
                    try:
                        debug_grad = torch.autograd.grad(v, debug_pg.values(), retain_graph=True, allow_unused=True)
                        debug_pg_grad = {pn:pg  for pn,pg in zip(debug_pg.keys(), debug_grad)}
                        for pn, pg in debug_pg_grad.items():
                            if pg is None:
                                continue
                            if torch.isnan(pg).any():
                                err_msg = f"NAN! d({k})_d({pn})"
                                log.error(err_msg)
                                assert not self.debug_raise, err_msg
                            elif torch.isinf(pg).any():
                                err_msg = f"INF! d({k})_d({pn})"
                                log.error(err_msg)
                                assert not self.debug_raise, err_msg
                    except Exception as e:
                        msg = f"d({k})_d(pg): " + repr(e)
                        log.error(msg)
                        assert not self.debug_raise, err_msg
                    
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

    def train_step_view(self, sample: dict, ground_truth: dict, it: int, logger: Logger=None):
        sample, ground_truth = next(zip_two_nested_dict(sample, ground_truth)) # NOTE: bs=1 in this case
        losses = dict()

        #----------------------- Get scene
        scene_id: str =  sample['scene_id']
        scene: Scene = self.scene_bank[scene_id]
        
        #----------------------- Get cam & frame
        cam_id, cam_fi = sample['cam_id'], sample['cam_fi']
        cam = scene.observers[cam_id]
        
        assert not isinstance(cam_id, (list, tuple)) and is_scalar(cam_fi), \
            "train step view only supports single camera and single time frame slicing"
        
        if scene.use_ts_interp:
            # NOTE: Rolling shutter effect is ignored.
            scene.interp_at(cam.frame_global_ts[cam_fi])
        else:
            scene.slice_at(cam_fi)

        cam.intr.set_downscale(ground_truth['image_downscale'])
        W, H = ground_truth['image_wh'].tolist()
        #-----------------------------------------------
        ret = self.renderer.render(
            scene, 
            observer=cam, 
            render_per_class_in_scene=True)
        #-----------------------------------------------
        
        obj = scene.get_drawable_groups_by_class_name(scene.main_class_name)[0]
        model = obj.model
        
        #-----------------------------------------------
        #-----------  Uniform sampling
        # Used by eikonal loss and sparsity loss
        uniform_samples = {}
        with profile("Uniform sampling"):
            for _, obj_raw_ret in ret['raw_per_obj_model'].items():
                if obj_raw_ret['volume_buffer']['type'] == 'empty':
                    continue # Skip not rendered models
                class_name = obj_raw_ret['class_name']
                num = self.config.get('uniform_sample', {}).get(class_name, 0)
                if num == 0:
                    continue
                model_id = obj_raw_ret['model_id']
                model = scene.asset_bank[model_id]
                uniform_samples[obj.class_name] = model.sample_pts_uniform(num)
        
        #-----------------------------------------------
        #-----------  Calc losses
        with profile("Calculate losss"):
            if self.rgb_loss is not None:
                ret_losses_rgb, _ = self.rgb_loss(scene, ret, sample, ground_truth, it=it)
                losses.update(ret_losses_rgb)
            
            if self.rgb_perceptual_loss is not None:
                losses.update(self.rgb_perceptual_loss(
                    scene, ret, sample, ground_truth, it=it, 
                    mode='view', logger=logger))
            
            if self.occupancy_mask_loss is not None:
                ret_losses_mask, _ = self.occupancy_mask_loss(scene, ret, sample, ground_truth, it=it)
                losses.update(ret_losses_mask)
            
            if self.mono_normals_loss is not None:
                losses.update(self.mono_normals_loss(
                    scene, cam, ret, sample, ground_truth, it=it, 
                    mode='view', logger=logger))
            
            if self.mono_depth_loss is not None:
                losses.update(self.mono_depth_loss(
                    scene, ret, sample, ground_truth, it=it, far=cam.far,  
                    mode='view', logger=logger))
            
            if self.mask_entropy_loss is not None:
                losses.update(self.mask_entropy_loss.forward_code_single(
                    scene, ret, (H*W,), it=it))
            
            if self.eikonal_loss is not None:
                losses.update(self.eikonal_loss(
                    scene, ret, uniform_samples, sample, ground_truth, it=it, 
                    mode='view', logger=logger))

            if self.sparsity_loss is not None:
                losses.update(self.sparsity_loss(
                    scene, ret, uniform_samples, sample, ground_truth, it=it))
            
            if self.clearance_loss is not None:
                losses.update(self.clearance_loss(
                    scene, ret, uniform_samples, sample, ground_truth, it=it))

            if self.flow_cycle_loss is not None:
                losses.update(self.flow_cycle_loss(
                    scene, ret, uniform_samples, sample, ground_truth, it=it))

            if self.weight_reg_loss is not None:
                losses.update(self.weight_reg_loss(scene, ret, sample, ground_truth, it=it))
            
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
        cam_id, cam_fi = sample['cam_id'], sample['cam_fi']
        cam = scene.observers[cam_id]
        
        if isinstance(cam_id, (list, tuple)):
            rays_ts = [c.get_timestamps(fi=sample['rays_fidx'], pix=sample['rays_pix']) for c in cam]
            rays_ts = torch.stack(rays_ts, dim=0).take_along_dim(sample['rays_sel'].unsqueeze(0), dim=0).squeeze(0)
        else:
            rays_ts = cam.get_timestamps(fi=sample['rays_fidx'], pix=sample['rays_pix'])
        
        if scene.use_ts_interp:
            scene.interp_at(rays_ts)
        else:
            scene.slice_at(sample['rays_fidx'])

        if isinstance(cam_id, (list, tuple)):
            cam = MultiCamBundle(cam, sample['rays_sel'])
        
        #----------------------- Get rays
        cam.intr.set_downscale(ground_truth['image_downscale'])
        rays_o, rays_d = cam.get_selected_rays(xy=sample['rays_pix'])
        
        #-----------------------------------------------
        ret = self.renderer.render(
            scene, 
            rays=(rays_o, rays_d, rays_ts, sample['rays_pix']), 
            observer=cam, only_cr=(it < self.config.get('enable_dv_after', 0)), 
            return_buffer=True, return_details=True, 
            render_per_class_in_scene=True)
        #-----------------------------------------------
        
        obj = scene.get_drawable_groups_by_class_name(scene.main_class_name)[0]
        obj_raw_ret = ret['raw_per_obj_model'][obj.id]
        model = obj.model

        #-----------------------------------------------
        #-----------  Uniform sampling
        # Used by eikonal loss and sparsity loss
        uniform_samples = {}
        with profile("Uniform sampling"):
            for _, obj_raw_ret in ret['raw_per_obj_model'].items():
                if obj_raw_ret['volume_buffer']['type'] == 'empty':
                    continue # Skip not rendered models
                class_name = obj_raw_ret['class_name']
                num = self.config.get('uniform_sample', {}).get(class_name, 0)
                if num == 0:
                    continue
                model_id = obj_raw_ret['model_id']
                model = scene.asset_bank[model_id]
                uniform_samples[class_name] = model.sample_pts_uniform(num)

        #-----------------------------------------------
        #-----------  Calc losses
        with profile("Calculate losss"):
            if self.rgb_loss is not None:
                ret_losses_rgb, err_map_rgb = self.rgb_loss(scene, ret, sample, ground_truth, it=it)
                if self.error_map_ignore_not_occupied and 'image_occupancy_mask' in ground_truth:
                    err_map_rgb = err_map_rgb * ground_truth['image_occupancy_mask'].float().view(*err_map_rgb.shape)
                losses.update(ret_losses_rgb)
            else:
                err_map_rgb = 0
            
            if self.rgb_s3im_loss is not None:
                losses.update(self.rgb_s3im_loss(scene, ret, sample, ground_truth, it=it))
            
            if self.occupancy_mask_loss is not None:
                ret_losses_mask, err_map_mask = self.occupancy_mask_loss(scene, ret, sample, ground_truth, it=it)
                losses.update(ret_losses_mask)
            else:
                err_map_mask = 0
            
            if self.mask_entropy_loss is not None:
                losses.update(self.mask_entropy_loss.forward_code_single(
                    scene, ret, rays_o.shape[:-1], it=it))

            if self.mono_normals_loss is not None and self.mono_normals_loss.apply_in_pixel_train_step:
                losses.update(self.mono_normals_loss(
                    scene, cam, ret, sample, ground_truth, it=it, 
                    mode='pixel', logger=logger))

            if self.road_normals_loss is not None and self.road_normals_loss.apply_in_pixel_train_step:
                losses.update(self.road_normals_loss(
                    scene, cam, ret, sample, ground_truth, it=it))

            if self.eikonal_loss is not None:
                losses.update(self.eikonal_loss(
                    scene, ret, uniform_samples, sample, ground_truth, it=it, 
                    mode='pixel', logger=logger))

            if self.sparsity_loss is not None:
                losses.update(self.sparsity_loss(
                    scene, ret, uniform_samples, sample, ground_truth, it=it))

            if self.clearance_loss is not None:
                losses.update(self.clearance_loss(
                    scene, ret, uniform_samples, sample, ground_truth, it=it))
            
            if self.sdf_curvature_reg_loss is not None:
                losses.update(self.sdf_curvature_reg_loss(
                    scene, ret, uniform_samples, sample, ground_truth, it=it))
            
            if self.color_net_reg_loss is not None:
                losses.update(self.color_net_reg_loss(
                    scene, ret, uniform_samples, sample, ground_truth, it=it))

            if self.flow_cycle_loss is not None:
                losses.update(self.flow_cycle_loss(
                    scene, ret, uniform_samples, sample, ground_truth, it=it))

            if self.weight_reg_loss is not None:
                losses.update(self.weight_reg_loss(scene, ret, sample, ground_truth, it=it))

        #-----------------------------------------------
        #-----------  Update pixel error map
        if (self.imp_samplers is not None) and (it >= self.error_map_enable_after):
            assert isinstance(cam_id, str), f"Only supports single cam_id, but got {cam_id}"
            
            imp_sampler = self.imp_samplers[scene_id][cam_id]
            
            imp_sampler.error_maps['rgb'].step_error_map(
                i=sample['rays_fidx'], xy=sample['rays_pix'], val=err_map_rgb)
            
            if 'mask' in imp_sampler.error_maps.keys():
                imp_sampler.error_maps['mask'].step_error_map(
                    i=sample['rays_fidx'], xy=sample['rays_pix'], val=err_map_mask)
            
            # # In order to focus on difficult or less-observed areas
            # for class_name, cls_rendered in ret['rendered_per_class_in_scene'].items():
            #     if class_name in self.error_map_on_classnames:
            #         imp_sampler.error_maps['focus_on'].step_error_map(
            #             i=sample['rays_fidx'], xy=sample['rays_pix'], val=cls_rendered['mask_volume'].data)

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
        cam_id, cam_fi = sample['cam_id'], sample['cam_fi']
        cam = scene.observers[cam_id]
        
        if isinstance(cam_id, (list, tuple)):
            rays_ts = [c.get_timestamps(fi=sample['rays_fidx'], pix=sample['rays_pix']) for c in cam]
            rays_ts = torch.stack(rays_ts, dim=0).take_along_dim(sample['rays_sel'].unsqueeze(0), dim=0).squeeze(0)
        else:
            rays_ts = cam.get_timestamps(fi=sample['rays_fidx'], pix=sample['rays_pix'])
        
        if scene.use_ts_interp:
            scene.interp_at(rays_ts)
        else:
            scene.slice_at(sample['rays_fidx'])

        if isinstance(cam_id, (list, tuple)):
            cam = MultiCamBundle(cam, sample['rays_sel'])

        #----------------------- Get rays
        cam.intr.set_downscale(ground_truth['image_downscale'])
        rays_o, rays_d = cam.get_selected_rays(xy=sample['rays_pix'])
        H, W, _ = rays_o.shape

        #-----------------------------------------------
        ret = self.renderer.render(
            scene, 
            rays=(rays_o, rays_d, rays_ts, sample['rays_pix']), 
            observer=cam, only_cr=(it < self.config.get('enable_dv_after', 0)), 
            return_buffer=True, return_details=True, 
            render_per_class_in_scene=True)
        #-----------------------------------------------

        obj = scene.get_drawable_groups_by_class_name(scene.main_class_name)[0]
        obj_raw_ret = ret['raw_per_obj_model'][obj.id]
        model = obj.model

        #-----------------------------------------------
        #-----------  Debug
        if logger is not None:
            with torch.no_grad():
                if (self.i_log > 0) and (it % self.i_log == 0):
                    if 'dbg_infos' in sample:
                        logger.add_nested_dict(f"train_step_{'image_patch'}.{scene.id}", '', sample['dbg_infos'], it)
                    # logger.add(f"train_step_{'image_patch'}.{scene.id}", "patch_scale", sample['patch_scale'], it)
                if (self.i_val > 0) and (it % self.i_val == 0):
                    logger.add_imgs(f"train_step_{'image_patch'}.{scene.id}", "gt_rgb", ground_truth['image_rgb'].float().data.cpu().movedim(-1,0), it)
                    logger.add_imgs(f"train_step_{'image_patch'}.{scene.id}", "pred_rgb_volume", ret['rendered']['rgb_volume'].data.cpu().movedim(-1,0), it)

        #-----------------------------------------------
        #-----------  Uniform sampling
        # Used by eikonal loss and sparsity loss
        uniform_samples = {}
        with profile("Uniform sampling"):
            for _, obj_raw_ret in ret['raw_per_obj_model'].items():
                if obj_raw_ret['volume_buffer']['type'] == 'empty':
                    continue # Skip not rendered models
                class_name = obj_raw_ret['class_name']
                num = self.config.get('uniform_sample', {}).get(class_name, 0)
                if num == 0:
                    continue
                model_id = obj_raw_ret['model_id']
                model = scene.asset_bank[model_id]
                uniform_samples[class_name] = model.sample_pts_uniform(num)

        #-----------------------------------------------
        #-----------  Calc losses
        with profile("Calculate losss"):
            if self.rgb_loss is not None:
                ret_losses_rgb, err_map_rgb = self.rgb_loss(scene, ret, sample, ground_truth, it=it)
                losses.update(ret_losses_rgb)

            if self.rgb_s3im_loss is not None:
                losses.update(self.rgb_s3im_loss(scene, ret, sample, ground_truth, it=it))

            if self.rgb_perceptual_loss is not None:
                losses.update(self.rgb_perceptual_loss(
                    scene, ret, sample, ground_truth, it=it, 
                    mode='image_patch', logger=logger))

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
                losses.update(self.eikonal_loss(
                    scene, ret, uniform_samples, sample, ground_truth, it=it, 
                    mode='image_patch', logger=logger))

            if self.sparsity_loss is not None:
                losses.update(self.sparsity_loss(
                    scene, ret, uniform_samples, sample, ground_truth, it=it))

            if self.clearance_loss is not None:
                losses.update(self.clearance_loss(
                    scene, ret, uniform_samples, sample, ground_truth, it=it))
            
            if self.sdf_curvature_reg_loss is not None:
                losses.update(self.sdf_curvature_reg_loss(
                    scene, ret, uniform_samples, sample, ground_truth, it=it))
            
            if self.color_net_reg_loss is not None:
                losses.update(self.color_net_reg_loss(
                    scene, ret, uniform_samples, sample, ground_truth, it=it))

            if self.flow_cycle_loss is not None:
                losses.update(self.flow_cycle_loss(
                    scene, ret, uniform_samples, sample, ground_truth, it=it))

            if self.weight_reg_loss is not None:
                losses.update(self.weight_reg_loss(scene, ret, sample, ground_truth, it=it))

        return ret, losses

    def train_step_scene_nvs(self, sample: dict, ground_truth: dict, it: int, logger: Logger=None) -> Tuple[dict, dict]:
        pass

    def train_step_asset_nvs(self, sample: dict, ground_truth: dict, it: int, logger: Logger=None) -> Tuple[dict, dict]:
        pass

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
        lidar_id, lidar_fi = sample['lidar_id'], sample['lidar_fi']
        lidar = scene.observers[lidar_id]
        
        if isinstance(lidar_id, (list, tuple)):
            rays_ts = [l.get_timestamps(fi=sample['rays_fidx']) for l in lidar]
            rays_ts = torch.stack(rays_ts, dim=0).take_along_dim(sample['rays_sel'].unsqueeze(0), dim=0).squeeze(0)
        else:
            rays_ts = lidar.get_timestamps(fi=sample['rays_fidx'])
        
        if scene.use_ts_interp:
            scene.interp_at(rays_ts)
        else:
            scene.slice_at(sample['rays_fidx'])
        
        if isinstance(lidar_id, (list, tuple)):
            lidar = MultiRaysLidarBundle(lidar, sample['rays_sel'])

        rays_o, rays_d = lidar.get_selected_rays(rays_o=sample['rays_o'], rays_d=sample['rays_d'])
        
        #-----------------------------------------------
        ret = self.renderer.render(
            scene, 
            rays=(rays_o, rays_d, rays_ts), observer=lidar, 
            only_cr=(it < self.config.get('enable_dv_after', 0)), 
            with_rgb=False, with_normal=(self.eikonal_loss is not None), 
            return_buffer=True, return_details=True, 
            render_per_class_in_scene=True)
        #-----------------------------------------------
        
        obj = scene.get_drawable_groups_by_class_name(scene.main_class_name)[0]
        obj_raw_ret = ret['raw_per_obj_model'][obj.id]
        model = obj.model
        
        #-----------------------------------------------
        #-----------  Uniform sampling
        # Used by eikonal loss and sparsity loss
        uniform_samples = {}
        with profile("Uniform sampling"):
            for _, obj_raw_ret in ret['raw_per_obj_model'].items():
                if obj_raw_ret['volume_buffer']['type'] == 'empty':
                    continue # Skip not rendered models
                class_name = obj_raw_ret['class_name']
                num = self.config.get('uniform_sample', {}).get(class_name, 0)
                if num == 0:
                    continue
                model_id = obj_raw_ret['model_id']
                model = scene.asset_bank[model_id]
                uniform_samples[class_name] = model.sample_pts_uniform(num)

        #-----------------------------------------------
        #-----------  Calc losses
        with profile("Calculate losss"):
            assert self.lidar_loss is not None, "Need to config lidar_loss when forwarding lidar data"
            losses.update(self.lidar_loss(
                scene, ret, sample, ground_truth, it=it, far=lidar.far, logger=logger))
            
            if self.eikonal_loss is not None:
                losses.update(self.eikonal_loss(
                    scene, ret, uniform_samples, sample, ground_truth, it=it, 
                    mode='lidar', logger=logger))
            
            if self.sparsity_loss is not None:
                losses.update(self.sparsity_loss(
                    scene, ret, uniform_samples, sample, ground_truth, it=it))
            
            if self.clearance_loss is not None:
                losses.update(self.clearance_loss(
                    scene, ret, uniform_samples, sample, ground_truth, it=it))
            
            if self.sdf_curvature_reg_loss is not None:
                losses.update(self.sdf_curvature_reg_loss(
                    scene, ret, uniform_samples, sample, ground_truth, it=it))

            if self.flow_cycle_loss is not None:
                losses.update(self.flow_cycle_loss(
                    scene, ret, uniform_samples, sample, ground_truth, it=it))

            if self.weight_reg_loss is not None:
                losses.update(self.weight_reg_loss(scene, ret, sample, ground_truth, it=it))

        return ret, losses

    @torch.no_grad()
    def _validate_single_cam(
        self, scene_id: str, cam_id: str, cam_fi: int, 
        ground_truth: dict, logger: Logger, it: int, 
        should_log_img=True, log_prefix: str='') -> dict:
        """ Validate one camera at one frame.

        Args:
            scene_id (str): The given scene_id to validate.
            cam_id (str): The given cam_id to validate.
            cam_fi (int): The given cam_fi to validate.
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
        cam: Camera = scene.observers[cam_id]
        if scene.use_ts_interp:
            cam_ts = cam.frame_global_ts[cam_fi] # TODO: For now, ignore rolling shutter effect
            scene.interp_at(cam_ts)
        else:
            scene.slice_at(cam_fi)
        cam.intr.set_downscale(ground_truth['image_downscale'])

        objs = scene.get_drawable_groups_by_class_name(scene.main_class_name).to_list()
        distant_objs = scene.get_drawable_groups_by_class_name('Distant').to_list()
        
        #-----------------------------------------------
        ret = self.renderer.render(
            scene, observer=cam, 
            only_cr=(it < self.config.get('enable_dv_after', 0)), 
            render_per_obj_individual=(len(objs) + len(distant_objs) > 1), 
            render_per_obj_in_scene=(len(objs) + len(distant_objs) > 1))
        rendered = ret['rendered']
        rendered_per_obj = ret.get("rendered_per_obj", {})
        rendered_per_obj_in_scene = ret.get("rendered_per_obj_in_scene", {})
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
            for k in ['flow_fwd', 'flow_bwd']:
                if k in render_dict:
                    _im_flow = to_img(scene_flow_to_rgb(render_dict[k], flow_max_radius=0.5))
                    add_log_img_entry(f"pred_{k}{log_suffix}", _im_flow)

        log_imgs('', rendered)
        for obj_id, obj_rendered in rendered_per_obj.items():
            log_imgs(f".{obj_id}.seperate", obj_rendered)
        for obj_id, obj_rendered in rendered_per_obj_in_scene.items():
            log_imgs(f".{obj_id}.in_scene", obj_rendered)

        if 'image_rgb' in ground_truth:
            add_log_img_entry("gt_rgb", to_img(ground_truth['image_rgb'].float()))

        for gt_mask_key in ('image_occupancy_mask', 'image_dynamic_mask', 'image_human_mask', 'image_ignore_mask', 'image_road_mask'):
            if gt_mask_key in ground_truth:
                add_log_img_entry(f"gt_{gt_mask_key}", to_img(ground_truth[gt_mask_key].unsqueeze(-1)/ground_truth[gt_mask_key].max().float()))

        if 'image_mono_depth' in ground_truth:
            add_log_img_entry("gt_image_mono_depth", to_img(ground_truth['image_mono_depth'].unsqueeze(-1)/ground_truth['image_mono_depth'].max().float()))
        
        if 'image_mono_normals' in ground_truth:
            add_log_img_entry("gt_image_mono_normals", to_img(ground_truth['image_mono_normals']/2+0.5))

        if 'ray_intersections' in ret:
            add_log_img_entry("ray_intersections/samples_cnt", to_img(ret['ray_intersections']['samples_cnt'].unsqueeze(-1).float() / ret['ray_intersections']['samples_cnt'].max().clamp(1e-5)))

        #---- Validate errormap (if any)
        if self.imp_samplers is not None:
            imp_sampler = self.imp_samplers[scene_id][cam_id]
            for name, err_map in imp_sampler.error_maps.items():
                err_map = err_map.get_normalized_error_map(cam_fi)
                add_log_img_entry(f"error_map.{name}", err_map.unsqueeze(0))

        #---- Scalar validation (NOTE: NOT eval! Since only one frame is used!!)
        if self.rgb_loss is not None:
            #--------- Calc psnr
            # Full PSNR
            eval_rgb_pred = rendered['rgb_volume'].view(cam.intr.H, cam.intr.W, -1)
            eval_rgb_gt = ground_truth['image_rgb'].to(self.device).view(cam.intr.H, cam.intr.W, -1)
            psnr = PSNR(eval_rgb_pred, eval_rgb_gt).item()
            logger.add(log_prefix, f"psnr.full", psnr, it)
            
            if 'image_occupancy_mask' in ground_truth:
                # Foreground PSNR
                eval_rgb_mask_gt = ground_truth['image_occupancy_mask'].to(self.device).view(cam.intr.H, cam.intr.W, 1)
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
                        log_collect[f'pred_rgb_volume.{objs[0].id}.in_scene'], 
                        log_collect[f'pred_rgb_volume.{distant_objs[0].id}.in_scene'], 
                    ], dim=1))
        # (col) gt_mask + pred_mask
        if 'pred_mask_volume' in log_collect and 'gt_rgb_mask' in log_collect:
            add_log_img_entry(
                "coll_mask_gt_pred", 
                torch.cat([log_collect['gt_rgb_mask'], log_collect['pred_mask_volume']], dim=1))
        # (col) mono_normals + pred_normals
        if 'pred_normals_volume' in log_collect and 'gt_image_mono_normals' in log_collect:
            add_log_img_entry(
                "coll_normals_gt_pred", 
                torch.cat([log_collect['gt_image_mono_normals'], log_collect['pred_normals_volume']], dim=1))

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
        cam_fi = sample['cam_fi']
        
        #---- Camera-wise validation
        if isinstance(cam_id, (list, tuple)):
            log_collect_all_cams = {}
            for ci, gt in enumerate(zip_dict(ground_truth)):
                log_collect_all_cams[cam_id[ci]] = self._validate_single_cam(
                    scene_id, cam_id[ci], cam_fi, ground_truth=gt, logger=logger, it=it, 
                    should_log_img=False, log_prefix=f"validate_camera.{scene_id}/{cam_id[ci]}")
            #---- Collate results from different cameras and log them
            log_collated_all_cams = collate_nested_dict(list(log_collect_all_cams.values()), stack=False)
            for key, imgs in log_collated_all_cams.items():
                logger.add_imgs(f"validate_camera.{scene_id}", key, torch.cat(pad_images_to_same_size(imgs, padding='top_left'), dim=2), it=it)
        else:
            self._validate_single_cam(
                scene_id, cam_id, cam_fi, ground_truth=ground_truth, logger=logger, it=it, 
                should_log_img=True, log_prefix=f"validate_camera.{scene_id}/{cam_id}")
        
        #---- Model-wise validation (if any)
        for class_name, model_id_map in self.asset_bank.class_name_infos.items():
            for model_id, scene_obj_id_list in model_id_map.items():
                model: AssetModelMixin = self.asset_bank[model_id]
                if model.assigned_to == AssetAssignment.OBJECT:
                    scene_id, obj_id = scene_obj_id_list[0]
                    scene = self.scene_bank[scene_id]
                    obj = scene.all_nodes[obj_id]
                    model.asset_val(scene=scene, obj=obj, it=it, logger=logger, log_prefix=f"validate_camera.{scene_id}.{class_name}.{obj.id}")
                elif model.assigned_to == AssetAssignment.SCENE:
                    scene_id, _ = scene_obj_id_list[0]
                    scene = self.scene_bank[scene_id]
                    model.asset_val(scene=scene, obj=None, it=it, logger=logger, log_prefix=f"validate_camera.{scene_id}.{class_name}")
                elif model.assigned_to == AssetAssignment.MULTI_OBJ_ONE_SCENE:
                    scene_id, _ = scene_obj_id_list[0]
                    model.asset_val(scene=scene, obj=None, it=it, logger=logger, log_prefix=f"validate_camera.{scene_id}.{class_name}")
                elif model.assigned_to == AssetAssignment.MULTI_OBJ:
                    model.asset_val(scene=None, obj=None, it=it, logger=logger, log_prefix=f"validate_camera.{class_name}")
                elif model.assigned_to == AssetAssignment.MULTI_SCENE:
                    pass
                elif model.assigned_to == AssetAssignment.MISC:
                    model.asset_val(scene=None, obj=None, it=it, logger=logger, log_prefix=f"validate_camera.{class_name}")

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
    log_details = args.training.get('log_details', True)
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
    dataset_impl: SceneDataset = import_str(args.dataset_cfg.target)(args.dataset_cfg.param)
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
    asset_bank.create_asset_bank(scene_bank, do_training_setup=True, device=device)
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

    # NOTE: Optionally, run training with render_parallel (multi-GPU rendering and buffer merging, similar to DataParallel)
    if (train_parallel_devices:=args.get('train_parallel_devices', None)) is not None:
        renderer.make_train_parallel(train_parallel_devices)
    
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
                if hasattr(dataloader.dataset, 'update_weights'):
                    dataloader.dataset.update_weights()
    
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
        val_image_dataset = ImageDataset(scene_dataloader_val, **params, **sampler_kwargs)
        val_image_dataloader_cyc = cycle(val_image_dataset.get_dataloader())
    
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
        **{f'optimizer_{n}':o for n,o in asset_bank.named_optimzers()}
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
    #-------------     Trainer     ---------------
    #---------------------------------------------
    trainer = Trainer(
        config=args.training, 
        renderer=renderer, 
        asset_bank=asset_bank, 
        scene_bank=scene_bank, 
        scene_loader=scene_dataloader_train, 
        train_pixel_dataset=pixel_dataset, 
        train_image_dataset=None, 
        train_image_patch_dataset=image_patch_dataset, 
        train_lidar_dataset=lidar_dataset, 
        val_image_dataset=val_image_dataset, 
        device_ids=args.device_ids, 
        i_log=i_log, i_val=i_val)
    trainer_module = trainer
    
    # Update the first learning rate after restore ckpt / it=0.
    asset_bank.training_update_lr(it)

    # Initialize (e.g. pretrain)
    if (it==0) and is_master():
        updated = trainer_module.training_initialize(logger=logger)
        if updated:
            checkpoint_io.save(filename='0.pt', global_step=it, epoch_idx=epoch_idx)

    if args.ddp:
        # NOTE: When there are no intersecting rays with distant-view models, \
        #       their parameters will not be used and hence raise an error.
        #       For now, we set find_unused_parameters to True
        trainer = DDP(trainer, device_ids=args.device_ids, output_device=local_rank, find_unused_parameters=True)
        trainer_module = trainer.module

    # Default to [True], since grad scaler is harmless even not using half precision.
    enable_grad_scaler = args.training.get("enable_grad_scaler", True)
    scaler_pixel = GradScaler(init_scale=128.0, enabled=enable_grad_scaler)
    scaler_lidar = GradScaler(init_scale=128.0, enabled=enable_grad_scaler)
    scaler_image_patch = GradScaler(init_scale=128.0, enabled=enable_grad_scaler)
    
    #--------------------------------------------------
    # NOTE: Debug Scene and SceneDataLoader before training
    if args.get('debug_scene', False):
        scene = scene_bank[0]
        scene.debug_vis_anim(
            scene_dataloader=scene_dataloader_train, 
            plot_image='camera' in scene_dataloader_train.config.tags, camera_length=2.0, 
            plot_lidar='lidar' in scene_dataloader_train.config.tags, lidar_pts_downsample=2, 
            # mesh_file=mesh_path, 
        )

    trainer_module.training_before_per_step(it, logger=logger)
    
    t0 = time.time()
    log.info(f"=> Start [train], it={it}, parallel={train_parallel_devices}, in {exp_dir}")
    end = (it >= args.training.num_iters)
    # total_start = time.time()
    # iter_timestamps = []

    if args.get('profile_iters', None):
        def profile_done(profiler: Profiler):
            print(profiler.get_result().get_statistic("device_duration").get_report(
                sort_by="device_duration"))
            exit()
        Profiler(warmup_frames=10, record_frames=args.profile_iters, then=profile_done).enable()

    with tqdm(range(args.training.num_iters), disable=not is_master()) as pbar:
        if is_master():
            pbar.update(it)
        @profile
        def train_step():
            nonlocal it, epoch_idx, t0, end
            int_it = int(it // world_size)
            local_it = it + rank # NOTE: it += it + world_size
            
            asset_bank.training_update_lr(local_it)
            trainer_module.training_before_per_step(local_it, logger=logger)

            #----------------------------------------------------------------------------
            #----------------------------     Validate     ------------------------------
            #----------------------------------------------------------------------------
            if i_val > 0 and int_it % i_val == 0:
                renderer.eval()
                asset_bank.eval()
                with torch.no_grad():
                    sample, ground_truth = next(val_image_dataloader_cyc)
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
                for n, o in asset_bank.named_optimzers():
                    for pg in o.param_groups:
                        logger.add('learning rates', n + '.' + pg['name'], pg['lr'], local_it)
            
            loss_total = 0.
            #----------------------------------------------------------------------------
            #-------------------------     Train - pixel     ----------------------------
            #----------------------------------------------------------------------------
            if pixel_dataloader_cyc is not None:
                with profile("Train on pixel dataloader"):
                    with profile("Get next batch"):
                        sample, ground_truth = next(pixel_dataloader_cyc)
                        scene = scene_bank[sample['scene_id'][0]]
                    
                    #----------------------------------------------------------------------------
                    ret, losses = trainer('pixel', sample, ground_truth, local_it, logger=logger)
                    #----------------------------------------------------------------------------
                    
                    with profile("Backward"):
                        losses['total'] = sum([v for v in losses.values()])
                        scaler_pixel.scale(losses['total']).backward()
                        for n, o in asset_bank.named_optimzers(only_used=enable_grad_scaler):
                            scaler_pixel.unscale_(o)
                        asset_bank.training_clip_grad()

                    with profile("Step"):
                        for n, o in asset_bank.named_optimzers(only_used=enable_grad_scaler):
                            scaler_pixel.step(o) # Step optimizer via scaler
                        scaler_pixel.update()
                    
                    loss_total += losses['total'].item()
                
                    #----------------------------------------------------------------------------
                    #----------------------------     Logging     -------------------------------
                    with torch.no_grad():
                        if i_log > 0 and int_it % i_log == 0:
                            #---------------------
                            # Log training related
                            for k, v in losses.items():
                                logger.add(f"train_step_pixel.losses", k, v.item(), local_it)
                            
                            if log_grad:
                                grad_norms = calc_grad_norm(**asset_bank)
                                for k, v in grad_norms.items():
                                    logger.add(f"train_step_pixel.grad_norm", k, v, local_it)
                            
                            if log_param:
                                logger.add_nested_dict(f"train_step_pixel.dbg_main_grad", d=asset_bank.get_scene_main_model(scene).stat_param(with_grad=True), it=local_it)
                            
                            if log_details:
                                # Log raw buffer
                                logger.add_nested_dict(f"train_step_pixel.scene={scene.id}", d=ret['volume_buffer'], it=local_it)
                                # Log per object
                                for obj_id, raw_ret in ret['raw_per_obj_model'].items():
                                    logger.add_nested_dict(f"train_step_pixel.obj={obj_id}", d=raw_ret, it=local_it)

                    for n, o in asset_bank.named_optimzers():
                        o.zero_grad(set_to_none=True)
                    del scene, losses, ret, sample, ground_truth

            #----------------------------------------------------------------------------
            #-------------------------     Train - lidar     ----------------------------
            #----------------------------------------------------------------------------
            if lidar_dataloader_cyc is not None:
                with profile("Train on Lidar dataloader"):
                    with profile("Get next batch"):
                        sample, ground_truth = next(lidar_dataloader_cyc)
                        scene = scene_bank[sample['scene_id'][0]]

                    #----------------------------------------------------------------------------
                    ret, losses = trainer('lidar', sample, ground_truth, local_it, logger=logger)
                    #----------------------------------------------------------------------------
                    
                    with profile("Backward"):
                        losses['total'] = sum([v for v in losses.values()])
                        scaler_lidar.scale(losses['total']).backward()
                        for n, o in asset_bank.named_optimzers(only_used=enable_grad_scaler):
                            scaler_lidar.unscale_(o)
                        asset_bank.training_clip_grad()
                    
                    with profile("Step"):
                        for n, o in asset_bank.named_optimzers(only_used=enable_grad_scaler):
                            scaler_lidar.step(o) # Step optimizer via scaler
                        scaler_lidar.update()
                    
                    loss_total += losses['total'].item()
                    
                    #----------------------------------------------------------------------------
                    #----------------------------     Logging     -------------------------------
                    with torch.no_grad():
                        if i_log > 0 and int_it % i_log == 0:
                            #---------------------
                            # Log training related
                            for k, v in losses.items():
                                logger.add("train_step_lidar.losses", k, v.item(), local_it)
                            
                            if log_grad:
                                grad_norms = calc_grad_norm(**asset_bank)
                                for k, v in grad_norms.items():
                                    logger.add("train_step_lidar.grad_norm", k, v, local_it)

                            if log_param:
                                logger.add_nested_dict(f"train_step_lidar.dbg_main_grad", d=asset_bank.get_scene_main_model(scene).stat_param(with_grad=True), it=local_it)

                            if log_details:
                                # Log raw buffer
                                logger.add_nested_dict(f"train_step_lidar.scene={scene.id}", d=ret['volume_buffer'], it=local_it)
                                # Log per object
                                for obj_id, raw_ret in ret['raw_per_obj_model'].items():
                                    logger.add_nested_dict(f"train_step_lidar.obj={obj_id}", d=raw_ret, it=local_it)
                    
                    for n, o in asset_bank.named_optimzers():
                        o.zero_grad(set_to_none=True)
                    del scene, losses, ret, sample, ground_truth

            #----------------------------------------------------------------------------
            #----------------------     Train - image_patch     -------------------------
            #----------------------------------------------------------------------------
            if image_patch_dataloader_cyc is not None:
                with profile("Train on Imagepatch dataloader"):
                    with profile("Get next batch"):
                        sample, ground_truth = next(image_patch_dataloader_cyc)
                        scene = scene_bank[sample['scene_id'][0]]
                    
                    #----------------------------------------------------------------------------
                    ret, losses = trainer('image_patch', sample, ground_truth, local_it, logger=logger)
                    #----------------------------------------------------------------------------
                    
                    with profile("Backward"):
                        losses['total'] = sum([v for v in losses.values()])
                        scaler_image_patch.scale(losses['total']).backward()
                        for n, o in asset_bank.named_optimzers(only_used=enable_grad_scaler):
                            scaler_image_patch.unscale_(o)
                        asset_bank.training_clip_grad()
                    
                    with profile("Step"):
                        for n, o in asset_bank.named_optimzers(only_used=enable_grad_scaler):
                            scaler_image_patch.step(o) # Step optimizer via scaler
                        scaler_image_patch.update()
                    
                    loss_total += losses['total'].item()
                    
                    #----------------------------------------------------------------------------
                    #----------------------------     Logging     -------------------------------
                    with torch.no_grad():
                        if i_log > 0 and int_it % i_log == 0:
                            #---------------------
                            # Log training related
                            for k, v in losses.items():
                                logger.add("train_step_image_patch.losses", k, v.item(), local_it)
                            
                            if log_grad:
                                grad_norms = calc_grad_norm(**asset_bank)
                                for k, v in grad_norms.items():
                                    logger.add("train_step_image_patch.grad_norm", k, v, local_it)
                            
                            if log_param:
                                logger.add_nested_dict(f"train_step_image_patch.dbg_main_grad", d=asset_bank.get_scene_main_model(scene).stat_param(with_grad=True), it=local_it)
                            
                            if log_details:
                                # Log raw buffer
                                logger.add_nested_dict(f"train_step_image_patch.scene={scene.id}", d=ret['volume_buffer'], it=local_it)
                                # Log per object
                                for obj_id, raw_ret in ret['raw_per_obj_model'].items():
                                    logger.add_nested_dict(f"train_step_image_patch.obj={obj_id}", d=raw_ret, it=local_it)
                    
                    for n, o in asset_bank.named_optimzers():
                        o.zero_grad(set_to_none=True)
                    del scene, losses, ret, sample, ground_truth

            #----------------------------------------------------------------------------
            #----------------------     End of one iteration     ------------------------
            #----------------------------------------------------------------------------
            trainer_module.training_after_per_step(local_it, logger=logger)
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
                pbar.set_postfix(loss_total=loss_total)
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
                print(f"Error occurred in exp: {exp_dir}")
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
    bc.parser.add_argument("--profile_iters", type=int, default=None)
    return bc

if __name__ == "__main__":
    bc = make_parser()
    main_function(bc.parse(print_config=False))