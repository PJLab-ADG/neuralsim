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

from nr3d_lib.logger import Logger
from nr3d_lib.config import ConfigDict
from nr3d_lib.models.attributes import *

from app.resources import Scene, SceneNode
from app.models.base import AssetAssignment, AssetModelMixin

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
        enable_after: int = 0, 
        refine_ego_motion: bool=False, ego_node_id: str=None, ego_class_name: str=None, 
        alpha_lr_rotation: float = 0.05, # Additional factor multiplied to rotation's learning rate.
        refine_camera_intr: bool=False, refine_camera_extr: bool=False, 
        device=torch.device('cuda')
        ) -> None:
        super().__init__()
        
        self.ego_node_id = ego_node_id
        self.ego_class_name = ego_class_name
        
        self.refine_ego_motion = refine_ego_motion
        self.alpha_lr_rotation = alpha_lr_rotation
        
        self.refine_camera_intr = refine_camera_intr
        self.refine_camera_extr = refine_camera_extr

        self.device = device
        self.enable_after = enable_after
        self.is_enabled = False

    def populate(
        self, scene: Scene = None, obj: SceneNode = None, config: ConfigDict = None, 
        dtype=torch.float, device=torch.device('cuda'), **kwargs):
        #--------------------------------------
        #------   Refine ego motion    --------
        #--------------------------------------
        self.scene = scene
        if self.refine_ego_motion:
            nn_ego_motion_dict = nn.ModuleDict()
            
            if self.ego_node_id is not None:
                ego_node_list = [scene.all_nodes[self.ego_node_id]]
            elif self.ego_class_name is not None:
                ego_node_list = scene.all_nodes_by_class_name[self.ego_class_name]
            else:
                raise RuntimeError(f"Invalid combination of arguments ego_node_id={self.ego_node_id}, ego_class_name={self.ego_class_name}")
            
            for ego_node in ego_node_list:
                attr_array = ego_node.attr_array
                original_transforms = attr_array.subattr['transform']
                prefix = original_transforms.prefix
                new_transforms = TransformRT(
                    rot=RotationQuaternionRefinedAdd(
                        attr0=RotationQuaternion.from_mat_3x3(original_transforms.rotation()),
                        delta=Vector_4(torch.zeros([*prefix,4]), learnable=True)
                    ),
                    trans=TranslationRefinedAdd(
                        attr0=Translation(original_transforms.translation()),
                        delta=Vector_3(torch.zeros([*prefix,3]), learnable=True)
                    )
                )
                # NOTE: Will be enabled later
                # attr_array.subattr['transform'] = new_transforms
                nn_ego_motion_dict[ego_node.id] = new_transforms
            
            self.add_module('ego_motion', nn_ego_motion_dict)

    def enable(self, scene: Scene = None):
        """ Actually loads the learnable params into the scene nodess attr

        Args:
            scene (Scene, optional): An optional target scene to load the learnable params.
                If not provided, `self.scene` will be used. Defaults to None.
        """
        self.is_enabled = True
        scene = self.scene or scene
        if self.refine_ego_motion:
            for node_id, transform in self['ego_motion'].items():
                node = scene.all_nodes[node_id]
                # NOTE: Here, the 'transform' attribute of the node will be set to the handle of the learnable 'transform' nn.Module here. 
                #       This allows the learnable transform to be used with gradients during the rendering process, 
                #       making the learnable param here part of the computation graph.
                node.attr_array.subattr['transform'] = transform

    def preprocess_per_train_step(self, cur_it: int, logger: Logger = None):
        if cur_it >= self.enable_after and not self.is_enabled:
            self.enable()

    def get_param_group(self, optim_cfg: dict, prefix: str = '') -> List[dict]:
        """
        Overwrites the default param group
        """
        prefix_ = prefix + ('.' if prefix and not prefix.endswith('.') else '')
        all_pg = []
        optim_cfg = deepcopy(optim_cfg)
        if self.refine_ego_motion:
            assert (ego_motion_cfg := optim_cfg.get('ego_motion', None)) is not None, "`ego_motion` should be in optim_cfg"
            assert isinstance(ego_motion_cfg, dict) and 'lr' in optim_cfg['ego_motion'], "`lr` should be in optim_cfg['ego_motion']"
            alpha_lr_rotation = ego_motion_cfg.pop('alpha_lr_rotation', None) or self.alpha_lr_rotation
            
            ego_motion_rotation_pg = {'name': prefix_ + "ego_motion.rotation_group", 'params': [], **ego_motion_cfg}
            ego_motion_rotation_pg['lr'] = alpha_lr_rotation * ego_motion_rotation_pg['lr']
            ego_motion_other_pg = {'name': prefix_ + "ego_motion.translation_group", 'params': [], **ego_motion_cfg}
            for name, param in self.named_parameters():
                if 'rot' in name: # 'trans' and 'rot'
                    ego_motion_rotation_pg['params'].append(param)
                else:
                    ego_motion_other_pg['params'].append(param)
            
            all_pg.append(ego_motion_rotation_pg)
            all_pg.append(ego_motion_other_pg)

        if self.refine_camera_intr:
            pass
        
        if self.refine_camera_extr:
            pass

        return all_pg

    @torch.no_grad()
    def val(self, scene: Scene = None, obj: SceneNode = None, it: int = ..., logger: Logger = None, log_prefix: str=''):
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

    @classmethod
    def compute_model_id(cls, scene: Scene = None, obj: SceneNode = None, class_name: str = None) -> str:
        return f"{cls.__name__}#{scene.id}"

class LearnableSceneParams(LearnableParams):
    pass # For compatibility