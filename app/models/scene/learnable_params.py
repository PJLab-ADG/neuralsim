"""
@file   learnable_params.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Learnable scene params (ego_motion, camera intr, extr, etc.)
"""

__all__ = [
    'LearnableParams', 
    'LearnableSceneParams'
]

import numpy as np
from copy import deepcopy
from numbers import Number
from typing import Union, List

import torch
import torch.nn as nn

from nr3d_lib.logger import Logger, log
from nr3d_lib.config import ConfigDict
from nr3d_lib.models.attributes import *

from app.resources import Scene, SceneNode
from nr3d_lib.models.utils import get_optimizer, get_scheduler
from app.models.asset_base import AssetAssignment, AssetModelMixin

class LearnableParams(AssetModelMixin, nn.ModuleDict):
    """
    This class stores and learns scene parameters and scene node attributes, if configured to do so.
    It handles parameters such as `node.transform`, which includes ego motions or camera extrinsics,
    and `cam.intr`, which includes camera intrinsics, among others.
    In essence, functions related to pose refinement and self-calibration are implemented here.
    """
    assigned_to = AssetAssignment.SCENE
    def __init__(
        self, 
        
        refine_ego_motion: dict = None, 
        refine_other_motion: dict = None, 
        refine_camera_intr: dict = None, 
        refine_camera_extr: dict = None, 
        refine_sensor_ts: dict = None, 
        
        enable_after: int = 0, 
        alpha_lr_rotation: float = 0.05, # Common factor multiplied to rotation's learning rate.
        device=None
        ) -> None:
        super().__init__()
        
        #---- Optional refinement: Ego motion
        self.refine_ego_motion = refine_ego_motion
        
        #---- Optional refinement: Other object motion
        self.refine_other_motion = refine_other_motion

        #---- Optional refinement: Camera intrinsics & extrinsics (self-calibration)
        self.refine_camera_intr = refine_camera_intr
        self.refine_camera_extr = refine_camera_extr

        #---- Optional refinement: Sensor timestamps (self-calibration)
        self.refine_sensor_ts = refine_sensor_ts

        #---- Common
        # Common weight for rotation repr when refining transformations (e.g. ego_motion, other_motion, camera_extr)
        self.alpha_lr_rotation = alpha_lr_rotation
        self.enable_after = enable_after
        self.set_device = device
        self.is_enabled = False

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def asset_populate(
        self, 
        scene: Scene = None, obj: SceneNode = None, config: dict = None, 
        device=None, **kwargs):
        
        device = device or self.set_device
        self.set_device = device
        
        #--------------------------------------
        #------   Refine ego motion    --------
        #--------------------------------------
        self.scene = scene
        if self.refine_ego_motion:
            nn_ego_motion_dict = nn.ModuleDict()
            
            if 'node_id' in self.refine_ego_motion:
                ego_node_list = [scene.all_nodes[self.refine_ego_motion['node_id']]]
            elif 'class_name' in self.refine_ego_motion:
                ego_node_list = scene.all_nodes_by_class_name[self.refine_ego_motion['class_name']]
            else:
                raise RuntimeError(f"Please specify at least one of (node_id,class_name)")
            
            for ego_node in ego_node_list:
                old_transforms = ego_node.frame_data.subattr['transform']
                prefix = old_transforms.prefix
                new_transforms = TransformRT(
                    rot=RotationQuaternionRefinedAdd(
                        attr0=RotationQuaternion.from_mat_3x3(old_transforms.rotation()),
                        delta=Vector_4(torch.zeros([*prefix,4]), learnable=True)
                    ),
                    trans=TranslationRefinedAdd(
                        attr0=Translation(old_transforms.translation()),
                        delta=Vector_3(torch.zeros([*prefix,3]), learnable=True)
                    )
                )
                # NOTE: Will be enabled later
                # ego_node.frame_data.subattr['transform'] = new_transforms
                nn_ego_motion_dict[ego_node.id] = new_transforms
            
            self.add_module('ego_motion', nn_ego_motion_dict)

        #--------------------------------------
        #-----   Refine other motion    -------
        #--------------------------------------
        if self.refine_other_motion:
            nn_other_motion_dict = nn.ModuleDict()
            
            assert ('class_name' in self.refine_other_motion) and len(self.refine_other_motion['class_name']) > 0, \
                "Please specify `class_name` for `refine_other_motion`"
            
            for class_name in self.refine_other_motion['class_name']:
                for node in scene.get_drawable_groups_by_class_name(class_name, False):
                    old_transforms = node.frame_data.subattr['transform']
                    prefix = old_transforms.prefix
                    new_transforms = TransformRT(
                        rot=RotationQuaternionRefinedAdd(
                            attr0=RotationQuaternion.from_mat_3x3(old_transforms.rotation()),
                            delta=Vector_4(torch.zeros([*prefix,4]), learnable=True)
                        ),
                        trans=TranslationRefinedAdd(
                            attr0=Translation(old_transforms.translation()),
                            delta=Vector_3(torch.zeros([*prefix,3]), learnable=True)
                        )
                    )
                    # NOTE: Will be enabled later
                    # node.frame_data.subattr['transform'] = new_transforms
                    nn_other_motion_dict[node.id] = new_transforms
            
            self.add_module("other_motion", nn_other_motion_dict)

        #--------------------------------------
        #------   Refine camera intr    -------
        #--------------------------------------
        if self.refine_camera_intr:
            pass
        
        #--------------------------------------
        #------   Refine camera intr    -------
        #--------------------------------------
        if self.refine_camera_extr:
            pass
        
        #--------------------------------------
        #-------   Refine sensor ts    --------
        #--------------------------------------
        if self.refine_sensor_ts and not scene.use_ts_interp:
            log.warn(
                "Ignoring `refine_sensor_ts`: `refine_sensor_ts` is set to True, "\
                "but `scene.use_ts_interp` is False which means the scene does not use timestamps. ")
        
        if self.refine_sensor_ts and scene.use_ts_interp:
            nn_sensor_ts_dict = nn.ModuleDict()
            
            # [Optional] Only estimate a holistic offset (assuming the intervals of original timestamps are accurate)
            learn_holistic_offset = self.refine_sensor_ts.get('learn_holistic_offset', False)
            
            if 'node_id' in self.refine_sensor_ts:
                sensor_node_list = [scene.all_nodes[self.refine_sensor_ts['node_id']]]
            elif 'class_name' in self.refine_sensor_ts:
                sensor_node_list = scene.all_nodes_by_class_name[self.refine_sensor_ts['class_name']]
            else:
                raise RuntimeError(f"Please specify at least one of (node_id,class_name)")

            for sensor_node in sensor_node_list:
                old_ts = sensor_node.frame_data.subattr['global_ts']
                #---- Opt1: Directly learnable
                # new_ts = Scalar(old_ts.value(), learnable=True)
                #---- Opt2: Learnable delta
                new_ts = ScalarRefinedAdd(
                    attr0=old_ts, 
                    delta=Scalar(torch.zeros(old_ts.prefix), learnable=True)\
                        if not learn_holistic_offset else Scalar(learnable=True)
                )
                # NOTE: Will be enabled later
                # ego_node.frame_data.subattr['global_ts'] = new_ts
                nn_sensor_ts_dict[sensor_node.id] = new_ts

            self.add_module('sensor_ts', nn_sensor_ts_dict)

    def enable(self, scene: Scene = None):
        """ Actually loads the learnable params into the scene nodes attr

        Args:
            scene (Scene, optional): An optional target scene to load the learnable params.
                If not provided, `self.scene` will be used. Defaults to None.
        """
        self.is_enabled = True
        scene = self.scene or scene
        if self.refine_ego_motion:
            for node_id, transform in self['ego_motion'].items():
                node = scene.all_nodes[node_id]
                # NOTE: Here, 'transform' of node.frame_data.subattr will be set to \
                #           the handle of the learnable 'transform' nn.Module here. 
                #       This allows the learnable transform to be used with gradients during the rendering process, 
                #       making the learnable param here part of the computation graph.
                node.frame_data.subattr['transform'] = transform

        if self.refine_other_motion:
            for node_id, transform in self['other_motion'].items():
                node = scene.all_nodes[node_id]
                node.frame_data.subattr['transform'] = transform

        if self.refine_camera_intr:
            pass
        
        if self.refine_camera_extr:
            pass

        if self.refine_sensor_ts:
            for node_id, ts in self['sensor_ts'].items():
                node = scene.all_nodes[node_id]
                node.frame_data.subattr['global_ts'] = ts

    def training_before_per_step(self, cur_it: int, logger: Logger = None):
        if cur_it >= self.enable_after and not self.is_enabled:
            self.enable()

    def training_setup(self, training_cfg: Union[Number, dict], name_prefix: str = ''):
        """
        Overwrites the default param group
        """
        prefix_ = name_prefix + ('.' if name_prefix and not name_prefix.endswith('.') else '')
        all_pg = []
        training_cfg = deepcopy(training_cfg)
        sched_kwargs = training_cfg.pop('scheduler', None)
        if self.refine_ego_motion:
            ego_motion_cfg = training_cfg.pop('ego_motion', None)
            assert isinstance(ego_motion_cfg, dict) and 'lr' in ego_motion_cfg, \
                "`ego_motion` should be in `training_cfg`, should be a dict with at least `lr` keys."
            alpha_lr_rotation = ego_motion_cfg.pop('alpha_lr_rotation', None) or self.alpha_lr_rotation
            
            ego_motion_rotation_pg = {'name': prefix_ + "ego_motion.rotation_group", 'params': [], **ego_motion_cfg}
            ego_motion_rotation_pg['lr'] = alpha_lr_rotation * ego_motion_rotation_pg['lr']
            ego_motion_other_pg = {'name': prefix_ + "ego_motion.translation_group", 'params': [], **ego_motion_cfg}
            for name, param in self['ego_motion'].named_parameters():
                if 'rot' in name: # 'trans' and 'rot'
                    ego_motion_rotation_pg['params'].append(param)
                else:
                    ego_motion_other_pg['params'].append(param)
            
            all_pg.append(ego_motion_rotation_pg)
            all_pg.append(ego_motion_other_pg)

        if self.refine_other_motion:
            other_motion_cfg = training_cfg.pop('other_motion', None)
            assert isinstance(other_motion_cfg, dict) and 'lr' in other_motion_cfg, \
                "`other_motion` should be in `training_cfg`, should be a dict with at least `lr` keys."
            alpha_lr_rotation = other_motion_cfg.pop('alpha_lr_rotation', None) or self.alpha_lr_rotation
            
            other_motion_rotation_pg = {'name': prefix_ + "other_motion.rotation_group", 'params': [], **other_motion_cfg}
            other_motion_rotation_pg['lr'] = alpha_lr_rotation * other_motion_rotation_pg['lr']
            other_motion_other_pg = {'name': prefix_ + "other_motion.translation_group", 'params': [], **other_motion_cfg}
            for name, param in self['other_motion'].named_parameters():
                if 'rot' in name: # 'trans' and 'rot'
                    other_motion_rotation_pg['params'].append(param)
                else:
                    other_motion_other_pg['params'].append(param)
            
            all_pg.append(other_motion_rotation_pg)
            all_pg.append(other_motion_other_pg)

        if self.refine_camera_intr:
            camera_intr_cfg = training_cfg.pop('camera_intr', None)
            assert isinstance(camera_intr_cfg, dict) and 'lr' in camera_intr_cfg, \
                "`camera_intr` should be in `training_cfg`, should be a dict with at least `lr` keys."
        
        if self.refine_camera_extr:
            camera_extr_cfg = training_cfg.pop('camera_extr', None)
            assert isinstance(camera_extr_cfg, dict) and 'lr' in camera_extr_cfg, \
                "`camera_extr` should be in `training_cfg`, should be a dict with at least `lr` keys."

        if self.refine_sensor_ts:
            sensor_ts_cfg = training_cfg.pop('sensor_ts', None)
            assert isinstance(sensor_ts_cfg, dict) and 'lr' in sensor_ts_cfg, \
                "`sensor_ts` should be in `training_cfg`, should be a dict with at least `lr` keys."
            pg = {'name': prefix_ + "sensor_ts", 'params': self['sensor_ts'].parameters(), **sensor_ts_cfg}
            all_pg.append(pg)

        self.optimizer = get_optimizer(all_pg, **training_cfg)
        self.scheduler = get_scheduler(self.optimizer, **sched_kwargs)

    @torch.no_grad()
    def asset_val(self, scene: Scene = None, obj: SceneNode = None, it: int = ..., logger: Logger = None, log_prefix: str=''):
        assert logger is not None
        # NOTE: python only import once. Importing for the 2nd time just gets a reference to the already imported module.
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from textwrap import wrap
        def safe_arccos(x):
            return np.arccos(np.clip(x.astype(np.float64),0,1))
        dim_str = ['x', 'y', 'z']
        
        scene = self.scene or scene
        if self.refine_ego_motion:
            for node_id, transform in self['ego_motion'].items():
                fig = plt.figure(figsize=(20.,8.))
                gs = gridspec.GridSpec(2, 3, wspace=0.4, hspace=0.2)
                #-----------------------------------
                #---- Ego_motion: translation ------
                trans = transform.subattr.trans
                transp = trans.vec_3().data.cpu().numpy()
                trans0 = trans.subattr.attr0.vec_3().data.cpu().numpy()
                dtrans = transp - trans0
                t = np.arange(len(trans0))
                for dim in range(3):
                    ax = fig.add_subplot(gs[0, dim])
                    ax.plot(t, trans0[:,dim], label=f"t{dim_str[dim]}0", color="g")
                    ax.plot(t, transp[:,dim], label=f"t{dim_str[dim]}'", color="r")
                    ax.legend(loc='upper left')
                    
                    ax_ = ax.twinx()
                    ax_.plot(t, dtrans[:,dim], 'x-', label=f"dt{dim_str[dim]}", color="b")
                    ax_.set_ylabel(f"delta {dim_str[dim]}")
                    ax_.legend(loc='upper right')
                    ax_.set_ylim([min(dtrans[:,dim].min(),-0.1), max(dtrans[:,dim].max(), 0.1)])
                
                #-----------------------------------
                #---- Ego_motion: translation ------
                rot = transform.subattr.rot
                rot0 = rot.subattr.attr0.mat_3x3().data.cpu().numpy()
                rotp = rot.mat_3x3().data.cpu().numpy()
                for dim in range(3):
                    ax = fig.add_subplot(gs[1, dim])
                    # the angle between dx/dy/dz vector with x/y/z axis
                    angle0 = np.rad2deg(safe_arccos(rot0[:,dim,dim]))
                    anglep = np.rad2deg(safe_arccos(rotp[:,dim,dim]))
                    # the angle between new dx/dy/dz vector with old ones
                    #   NOTE: This is not a complete description of rotation refinement, just for verbose 
                    dangle = np.rad2deg(safe_arccos(np.sum(rot0[:,:3,dim]*rotp[:,:3,dim], axis=-1)))
                    ax.plot(t, angle0, label=f"R{dim_str[dim]}0", color="g")
                    ax.plot(t, anglep, label=f"R{dim_str[dim]}'", color="r")
                    ax.legend(loc='upper left')
                    
                    ax_ = ax.twinx()
                    ax_.plot(t, dangle, 'x-', label=f"dR{dim_str[dim]}", color="b")
                    ax_.set_ylabel(f"delta degrees {dim_str[dim]}")
                    ax_.legend(loc='upper right')
                    ax_.set_ylim([min(dangle.min(),-1.), max(dangle.max(), 1.)])

                fig.suptitle(f"ego_motion @ {it} it" + "\nscene = " + '\n'.join(wrap(scene.id, 80)) + f"\n obs = {node_id}", size=16)
                gs.tight_layout(fig, rect=[0, 0, 1, 0.97])

                logger.add_figure(log_prefix, f"{node_id}/ego_motion", fig, it)

        if self.refine_sensor_ts:
            for node_id, ts in self['sensor_ts'].items():
                ts_0 = ts.subattr.attr0.value().data.cpu().numpy()
                ts_new = ts.value().data.cpu().numpy()
                dts = ts_new - ts_0
                t = np.arange(len(ts_new))
                fig = plt.figure(figsize=(20.,8.))
                ax = fig.add_subplot(1,1,1)
                ax.plot(t, ts_0, label="ts0", color="g")
                ax.plot(t, ts_new, label="ts'", color="r")
                ax.legend(loc='upper left')
                
                ax_ = ax.twinx()
                ax_.plot(t, dts, 'x-', label="dts", color="b")
                ax_.set_ylabel("delta ts")
                ax_.legend(loc='upper right')
                ax_.set_ylim([min(dts.min(), dts.max() - 0.001), max(dts.max(), dts.min() + 0.001)])
                
                fig.suptitle(f"sensor_ts @ {it} it" + "\nscene = " + '\n'.join(wrap(scene.id, 80)) + f"\n obs = {node_id}", size=16)
                fig.tight_layout()
                
                logger.add_figure(log_prefix, f"{node_id}/sensor_ts", fig, it)

    @classmethod
    def asset_compute_id(cls, scene: Scene = None, obj: SceneNode = None, class_name: str = None) -> str:
        return f"{cls.__name__}#{scene.id}"

class LearnableSceneParams(LearnableParams):
    pass # For compatibility