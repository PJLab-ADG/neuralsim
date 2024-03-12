"""
@file   asset_bank.py
@author Jianfei Guo, Shanghai AI Lab
@brief  AssetBank creation, loading, optimization and management.
"""

__all__ = [
    'AssetBank'
]

import os
import re
import numpy as np
from copy import deepcopy
from numbers import Number
from typing import Dict, List, Tuple, Union, Type

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils import model_zoo

from nr3d_lib.fmt import log
from nr3d_lib.logger import Logger
from nr3d_lib.config import ConfigDict
from nr3d_lib.utils import IDListedDict, import_str, is_scalar

from app.resources.scenes import Scene, SceneNode

class AssetBank(nn.ModuleDict):
    def __init__(self, config: dict) -> None:
        super().__init__()        
        self.config = deepcopy(config)
        self.misc_node_class_names = ['node', 'EgoVehicle', 'EgoDrone']
        
        self._optimzers: Dict[str, Optimizer] = {}
        
        #---- More collections of model_ids for convinient use
        # {class_name: [  [model_id, [[scene_id, obj_id], [...]]],   [...]  ]}
        self.class_name_infos: Dict[str, Dict[str, List[Tuple[str, str]]]] = {}
        # model_ids that could belong to certain scene or certain object of a certain scene.
        self.scene_model_ids: Dict[str, List[str]] = {} 
        # model_ids that don't belong to certain scene; i.e. shared across multiple scenes
        self.across_model_ids: List[str] = []

        """ Example on how to use `self.class_name_infos`: 
        >>> for class_name, model_id_map in self.asset_bank.class_name_infos.items():
        >>>     for model_id, scene_obj_id_list in model_id_map.items():
        >>>         pass # Do something
        """

    @property
    def class_name_configs(self) -> Dict[str, ConfigDict]:
        return self.config

    def named_optimzers(self, only_used=False):
        """
        NOTE:
        Q: Which situations require `only_used` to be True ?
        A: 
        - Preventing GradScaler complaining "No inf checks were recorded for this optimizer."\
            due to empty `optimizer_state["found_inf_per_device"]` when all .grad is None.
        NOTE: Performance (Pure check time):
        - 5.76 us for StreetSurf
        - 22.9 us for neuralsim
        """
        if only_used:
            for n, o in self._optimzers.items():
                if any(param.grad is not None for group in o.param_groups for param in group['params']):
                    yield n, o
        else:
            yield from self._optimzers.items()

    def optimzers(self, only_used=True):
        for n, o in self.named_optimzers(only_used=only_used):
            yield o

    def asset_compute_id(self, obj: SceneNode = None, scene: Scene = None, class_name: str = None) -> str:
        """
        Computes model_id using configured model_class in `class_name_configs`.
        """
        from app.models.asset_base import AssetModelMixin # To avoid circular import 
        if obj is not None:
            class_name = class_name or obj.class_name
            scene = scene or obj.scene
        if (cfg:=self.class_name_configs.get(class_name, None)) is not None:
            model_class: Type[AssetModelMixin] = import_str(cfg.model_class)
            model_id = model_class.asset_compute_id(scene=scene, obj=obj, class_name=class_name)
            return model_id
        else:
            raise RuntimeError(f"Can not find class_name={class_name} in assetbank.")

    def get_scene_main_model(self, scene: Scene):
        obj = scene.drawable_groups_by_class_name[scene.main_class_name][0]
        return obj.model

    def get_scene_distant_model(self, scene: Scene):
        obj = scene.drawable_groups_by_class_name['Distant'][0]
        return obj.model

    def get_scene_sky_model(self, scene: Scene):
        obj = scene.drawable_groups_by_class_name['Sky'][0]
        return obj.model

    def get_scene_related_model_ids(self, scene_id: Union[Scene, str]):
        model_ids = self.across_model_ids + self.scene_model_ids[scene_id if isinstance(scene_id, str) else scene_id.id]
        return model_ids

    def create_asset_bank(
        self, 
        scene_bank: IDListedDict[Scene], *, 
        load_state_dict: Union[str, dict] = None, 
        load_assets_into_scene: bool = True, 
        do_training_setup: bool = False, 
        class_name_list: List[str] = None, # Optionally,  specify a list of class_names
        device=None):
        
        from app.models.asset_base import AssetModelMixin, AssetAssignment # To avoid circular import
        self.scene_bank = scene_bank
        
        if class_name_list is not None:
            if not isinstance(class_name_list, list):
                class_name_list = [class_name_list]

        for class_name, cfg in self.class_name_configs.items():
            if class_name_list is not None and class_name not in class_name_list:
                continue
            
            model_class: Type[AssetModelMixin] = import_str(cfg.model_class)
            # NOTE: Default `assigned_to` can be overwritten by configs
            assigned_to_str: str = cfg.get('assigned_to', None) or model_class.assigned_to.name
            assigned_to = AssetAssignment[assigned_to_str.upper()]
            
            if assigned_to == AssetAssignment.OBJECT:
                for scene in scene_bank:
                    for obj in scene.all_nodes_by_class_name.get(class_name, []):
                        model_id = model_class.asset_compute_id(scene=scene, obj=obj, class_name=class_name)
                        model: AssetModelMixin = model_class(**cfg.model_params, device=device)
                        model.assigned_to = assigned_to
                        model.asset_init_config(**cfg.get('asset_params', {}))
                        model.asset_populate(scene=scene, obj=obj, config=model.populate_cfg, device=device)
                        model.to(device) # To make sure
                        self.add_module(model_id, model)
                        self.class_name_infos.setdefault(class_name, {})[model_id] = [(scene.id, obj.id),]
                        self.scene_model_ids.setdefault(scene.id, []).append(model_id)
                        if load_assets_into_scene:
                            obj.model = model
                            scene.add_node_to_drawable(obj)
                        if do_training_setup:
                            model.training_setup(model.training_cfg)
                            self._optimzers[model_id] = model.optimizer

            elif assigned_to == AssetAssignment.MULTI_OBJ: # The same with MULTI_OBJ_MULTI_SCENE
                obj_list = []
                for scene in scene_bank:
                    obj_list.extend(scene.all_nodes_by_class_name.get(class_name, []))
                if len(obj_list) == 0:
                    continue
                model_id = model_class.asset_compute_id(scene=None, obj=None, class_name=class_name)
                model: AssetModelMixin = model_class(**cfg.model_params, device=device)
                model.assigned_to = assigned_to
                model.asset_init_config(**cfg.get('asset_params', {}))
                model.asset_populate(scene=scene_bank.to_list(), obj=obj_list, config=model.populate_cfg, device=device)
                model.to(device) # To make sure
                self.add_module(model_id, model)
                self.class_name_infos.setdefault(class_name, {})[model_id] = [(obj.scene.id, obj.id) for obj in obj_list]
                self.across_model_ids.append(model_id)
                if load_assets_into_scene:
                    for obj in obj_list:
                        obj.model = model
                        scene.add_node_to_drawable(obj)
                if do_training_setup:
                    model.training_setup(model.training_cfg)
                    self._optimzers[model_id] = model.optimizer

            elif assigned_to == AssetAssignment.MULTI_OBJ_ONE_SCENE:
                for scene in scene_bank:
                    obj_list = scene.all_nodes_by_class_name.get(class_name, [])
                    if len(obj_list) == 0:
                        continue
                    model_id = model_class.asset_compute_id(scene=scene, obj=None, class_name=class_name)
                    model: AssetModelMixin = model_class(**cfg.model_params, device=device)
                    model.assigned_to = assigned_to
                    model.asset_init_config(**cfg.get('asset_params', {}))
                    model.asset_populate(scene=scene, obj=obj_list, config=model.populate_cfg, device=device)
                    model.to(device) # To make sure
                    self.add_module(model_id, model)
                    self.class_name_infos.setdefault(class_name, {})[model_id] = [(scene.id, obj.id) for obj in obj_list]
                    self.scene_model_ids.setdefault(scene.id, []).append(model_id)
                    if load_assets_into_scene:
                        for obj in obj_list:
                            obj.model = model
                            scene.add_node_to_drawable(obj)
                    if do_training_setup:
                        model.training_setup(model.training_cfg)
                        self._optimzers[model_id] = model.optimizer

            elif assigned_to == AssetAssignment.SCENE:
                for scene in scene_bank:
                    model_id = model_class.asset_compute_id(scene=scene, obj=None, class_name=class_name)
                    model: AssetModelMixin = model_class(**cfg.model_params, device=device)
                    model.assigned_to = assigned_to
                    model.asset_init_config(**cfg.get('asset_params', {}))
                    model.asset_populate(scene=scene, obj=None, config=model.populate_cfg, device=device)
                    model.to(device) # To make sure
                    self.add_module(model_id, model)
                    self.class_name_infos.setdefault(class_name, {})[model_id] = [(scene.id, None),]
                    self.scene_model_ids.setdefault(scene.id, []).append(model_id)
                    if load_assets_into_scene:
                        if class_name == 'LearnableParams':
                            scene.learnable_params = model
                            model.scene = scene
                        elif class_name == 'ImageEmbeddings':
                            scene.image_embeddings = model
                            model.scene = scene
                        else:
                            raise RuntimeError(f"Unsupported scene-level class_name={class_name}")
                    if do_training_setup:
                        model.training_setup(model.training_cfg)
                        self._optimzers[model_id] = model.optimizer

            elif assigned_to == AssetAssignment.MULTI_SCENE:
                # NOTE: Most of the functionlities should be covered by MULTI_OBJ_MULTI_SCENE ?
                pass
            
            elif assigned_to == AssetAssignment.MISC:
                # NOTE: For models that does not belong to scene / objects (e.g. belonging to renderers)
                model_id = model_class.asset_compute_id(scene=None, obj=None, class_name=class_name)
                model: AssetModelMixin = model_class(**cfg.model_params, device=device)
                model.assigned_to = assigned_to
                model.asset_init_config(**cfg.get('asset_params', {}))
                model.asset_populate(scene=None, obj=None, config=model.populate_cfg, device=device)
                model.to(device) # To make sure
                self.add_module(model_id, model)
                self.class_name_infos.setdefault(class_name, {})[model_id] = []
                self.across_model_ids.append(model_id)
                if do_training_setup:
                    model.training_setup(model.training_cfg)
                    self._optimzers[model_id] = model.optimizer

        if load_state_dict is not None:
            self.load_asset_bank(load_state_dict, strict=(class_name_list is None))

    # Overwrites
    def state_dict(self, destination=None, prefix: str='', keep_vars=False):
        assert (prefix == '') and (destination is None), "Do not support storing assetbank with prefix."
        destination = dict()
        for name, module in self.items():
            destination[name] = module.state_dict(keep_vars=keep_vars)
        return destination

    # Overwrites
    def load_state_dict(self, state_dict, strict: bool = True):
        for name, module in self.items():
            if name in state_dict:
                module.load_state_dict(state_dict[name], strict=strict)
            elif strict:
                raise RuntimeError(f"{name} not found in state_dict of asset_bank")

    def load_asset_bank(self, load: Union[str, dict], strict=True):
        if isinstance(load, str):
            load = torch.load(load)
        self.load_state_dict(load, strict=strict)

    def add_module(self, name: str, module: nn.Module) -> None:
        module.id = name
        return super().add_module(name, module)

    def model_setup(self, scene_id: str = None):
        """
        Operations that need to be executed only once throughout the entire rendering process, 
        as long as the network params remains untouched.
        """
        model_ids = list(self.keys()) if scene_id is None else self.get_scene_related_model_ids(scene_id)
        for model_id in model_ids:
            model = self[model_id]
            model.model_setup()

    def training_update_lr(self, cur_it: int, scene_id: str = None):
        model_ids = list(self.keys()) if scene_id is None else self.get_scene_related_model_ids(scene_id)
        for model_id in model_ids:
            model = self[model_id]
            model.training_update_lr(cur_it)

    def training_clip_grad(self, scene_id: str = None):
        model_ids = list(self.keys()) if scene_id is None else self.get_scene_related_model_ids(scene_id)
        for model_id in model_ids:
            model = self[model_id]
            model.training_clip_grad()

    def training_before_per_step(self, cur_it: int, logger: Logger=None, scene_id: str = None):
        """
        Operations that need to be executed before each training step (before `trainer.forward`).
        """
        model_ids = list(self.keys()) if scene_id is None else self.get_scene_related_model_ids(scene_id)
        for model_id in model_ids:
            model = self[model_id]
            model.training_before_per_step(cur_it, logger=logger)

    def training_after_per_step(self, cur_it: int, logger: Logger=None, scene_id: str = None):
        """
        Operations that need to be executed after each training step (after `optmizer.step`).
        """
        model_ids = list(self.keys()) if scene_id is None else self.get_scene_related_model_ids(scene_id)
        for model_id in model_ids:
            model = self[model_id]
            model.training_after_per_step(cur_it, logger=logger)

    def rendering_before_per_view(self, renderer, observer, per_frame_info: dict={}, scene_id: str = None):
        """
        Operations that need to be executed for every frame or view.
        """
        model_ids = list(self.keys()) if scene_id is None else self.get_scene_related_model_ids(scene_id)
        for model_id in model_ids:
            model = self[model_id]
            model.rendering_before_per_view(renderer, observer, per_frame_info=per_frame_info)
    
    def stat_param(self, with_grad=False, prefix: str='') -> Dict[str, float]:
        prefix_ = prefix + ('.' if prefix and not prefix.endswith('.') else '')
        for model_id, model in self.items():
            model.stat_param(with_grad=with_grad, prefix=prefix_ + model_id)


    # def load_from_neuralgen(self, class_name: str='Vehicle', resume_dir: str=None, load_pt: str=None, load_pt_save_dir: str=None, use_neuralgen_latents:bool=False, zero_init_latents=False):
    #     if is_url(load_pt):
    #         log.info(f'=> Load from neuralgen ckpt url:' + load_pt)
    #         state_dict = model_zoo.load_url(load_pt, model_dir=load_pt_save_dir, progress=True)
    #     else:
    #         if load_pt is None:
    #             # Automatically load 'final_xxx.pt' or 'latest.pt'
    #             ckpt_file = sorted_ckpts(os.path.join(resume_dir, 'ckpts'))[-1]
    #         else:
    #             ckpt_file = load_pt
    #         log.info("=> Load from neuralgen ckpt:" + str(ckpt_file))
    #         state_dict = torch.load(ckpt_file, map_location=self.device)

    #     fg_ids = state_dict["scene"][".".join(["fg_net", "_keys"])]
    #     fg_state_dict = dict()
        
    #     fg_state_dict["_models"] = state_dict["scene"][".".join(["fg_net", "_models"])]
    #     fg_model_id = self.get_shared_model_id(class_name)
    #     self[fg_model_id].load_state_dict(fg_state_dict, strict=False)
        
    #     if zero_init_latents:
    #         for l in self[fg_model_id]._latents.values():
    #             nn.init.zeros_(l.weight)

    #     if use_neuralgen_latents:
    #         fg_latents = state_dict["scene"][".".join(["fg_net", "_latents"])]
    #         for k in self[fg_model_id]._latents.keys():
    #             l = fg_latents[f"{k}.weight"]
    #             fg_len = len(l)
    #             self_len = len(self[fg_model_id]._latents[k].weight)
    #             inds = np.random.choice(fg_len, size=(self_len,), replace=(self_len > fg_len))
    #             self[fg_model_id]._latents[k].load_state_dict({"weight": l[inds]})

    # def load_external_categorical_assets(self, file: str):
    #     from omegaconf import OmegaConf
    #     cfg = ConfigDict(OmegaConf.to_container(OmegaConf.load(file), resolve=True))
    #     for class_name, class_cfg in cfg.class_name_cfgs.items():
    #         #---- Add new model cfg into the asset bank.
    #         self.class_name_configs[class_name] = class_cfg
    #         self.drawable_shared_map[class_name] = cfg.drawable_shared_map[class_name]

    #         obj_full_unique_ids = [f"{sid}#{oid}" for sid, oid in cfg.drawable_shared_map[class_name]]
    #         model = import_str(class_cfg.target)(
    #             key_list=obj_full_unique_ids, cfg=class_cfg.param).to(dtype=self.dtype, device=self.device)
    #         model.ray_query_cfg = class_cfg.ray_query_cfg
    #         model.initialize_cfg = cfg.get('initialize_cfg', {})
    #         model.preload_cfg = cfg.get('preload_cfg', None)
    #         model.dtype = self.dtype
    #         self.add_module(self.get_shared_model_id(class_name), model)
