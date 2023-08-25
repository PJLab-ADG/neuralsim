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
from torch.utils import model_zoo

from nr3d_lib.fmt import log
from nr3d_lib.logger import Logger
from nr3d_lib.config import ConfigDict
from nr3d_lib.utils import IDListedDict, import_str, is_scalar

from nr3d_lib.models.utils import get_param_group

from app.resources.scenes import Scene, SceneNode

class AssetBank(nn.ModuleDict):
    def __init__(self, config: ConfigDict) -> None:
        super().__init__()
        self._param_groups: List[dict] = []
        self._clip_grad_groups: List[dict] = []
        self.config = deepcopy(config)
        self.misc_node_class_names = ['node', 'EgoVehicle', 'EgoDrone']
        # {class_name: [  [model_id, [[scene_id, obj_id], [...]]],   [...]  ]}
        self.class_name_infos: Dict[str, Dict[str, List[Tuple[str, str]]]] = {}

        """ Example on how to use `self.class_name_infos`: 
        >>> for class_name, model_id_map in self.asset_bank.class_name_infos.items():
        >>>     for model_id, scene_obj_id_list in model_id_map.items():
        >>>         pass # Do something
        """

    @property
    def class_name_configs(self) -> Dict[str, ConfigDict]:
        return self.config

    @property
    def param_groups(self):
        assert len(self._param_groups) > 0, "Empty param group. \nPlease pass 'optim_cfg' when invoking 'load_asset_bank'"
        return self._param_groups

    def compute_model_id(self, obj: SceneNode = None, scene: Scene = None, class_name: str = None) -> str:
        """
        Computes model_id using configured model_class in `class_name_configs`.
        """
        from app.models.base import AssetModelMixin # To avoid circular import 
        if obj is not None:
            class_name = class_name or obj.class_name
            scene = scene or obj.scene
        if (cfg:=self.class_name_configs.get(class_name, None)) is not None:
            model_class: Type[AssetModelMixin] = import_str(cfg.model_class)
            model_id = model_class.compute_model_id(scene=scene, obj=obj, class_name=class_name)
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

    def create_asset_bank(
        self, 
        scene_bank: IDListedDict[Scene], *, 
        optim_cfg: Union[Number, dict] = None, 
        load: Union[str, dict] = None, 
        class_name_list: List[str] = None, # Optionally specify a list of class_names
        load_assets_into_scene=True, 
        device=torch.device("cuda")):
        
        from app.models.base import AssetModelMixin, AssetAssignment # To avoid circular import
        self.scene_bank = scene_bank
        self.device = device
        
        if class_name_list is not None:
            if not isinstance(class_name_list, list):
                class_name_list = [class_name_list]

        if optim_cfg is not None:
            if is_scalar(optim_cfg):
                default_optim_cfg = optim_cfg
                optim_cfg = {}
            elif isinstance(optim_cfg, dict):
                optim_cfg = optim_cfg.copy()
                default_optim_cfg = optim_cfg.pop('default')
            else:
                raise ValueError(f"Invalid type of optim_cfg={type(optim_cfg)}; the value is {optim_cfg}")

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
                        model_id = model_class.compute_model_id(scene=scene, obj=obj, class_name=class_name)
                        model: AssetModelMixin = model_class(**cfg.model_params, device=device)
                        model.assigned_to = assigned_to
                        model.init_asset_config(**cfg.get('asset_params', {}))
                        model.populate(scene=scene, obj=obj, config=model.populate_cfg, device=device)
                        self.add_module(model_id, model)
                        self.class_name_infos.setdefault(class_name, {})[model_id] = [(scene.id, obj.id),]
                        if load_assets_into_scene:
                            obj.model = model
                            scene.add_node_to_drawable(obj)
                        if optim_cfg is not None:
                            cls_optim_cfg = optim_cfg.get(class_name, default_optim_cfg)
                            self._param_groups.extend(model.get_param_group(cls_optim_cfg, prefix=model_id))

            elif assigned_to == AssetAssignment.MULTI_OBJ: # The same with MULTI_OBJ_MULTI_SCENE
                obj_list = []
                for scene in scene_bank:
                    obj_list.extend(scene.all_nodes_by_class_name.get(class_name, []))
                obj_full_unique_ids = [obj.full_unique_id for obj in obj_list]
                model_id = model_class.compute_model_id(scene=None, obj=None, class_name=class_name)
                model: AssetModelMixin = model_class(key_list=obj_full_unique_ids, **cfg.model_params, device=device)
                model.assigned_to = assigned_to
                model.init_asset_config(**cfg.get('asset_params', {}))
                model.populate(scene=None, obj=None, config=model.populate_cfg, device=device)
                self.add_module(model_id, model)
                self.class_name_infos.setdefault(class_name, {})[model_id] = [(obj.scene.id, obj.id) for obj in obj_list]
                if load_assets_into_scene:
                    for obj in obj_list:
                        obj.model = model
                        scene.add_node_to_drawable(obj)
                if optim_cfg is not None:
                    cls_optim_cfg = optim_cfg.get(class_name, default_optim_cfg)
                    self._param_groups.extend(model.get_param_group(cls_optim_cfg, prefix=model_id))

            elif assigned_to == AssetAssignment.MULTI_OBJ_ONE_SCENE:
                for scene in scene_bank:
                    obj_list = scene.all_nodes_by_class_name.get(class_name, [])
                    if len(obj_list) == 0:
                        continue
                    obj_full_unique_ids = [obj.full_unique_id for obj in obj_list]
                    model_id = model_class.compute_model_id(scene=scene, obj=None, class_name=class_name)
                    model: AssetModelMixin = model_class(key_list=obj_full_unique_ids, **cfg.model_params, device=device)
                    model.assigned_to = assigned_to
                    model.init_asset_config(**cfg.get('asset_params', {}))
                    model.populate(scene=scene, obj=None, config=model.populate_cfg, device=device)
                    self.add_module(model_id, model)
                    self.class_name_infos.setdefault(class_name, {})[model_id] = [(scene.id, obj.id) for obj in obj_list]
                    if load_assets_into_scene:
                        for obj in obj_list:
                            obj.model = model
                            scene.add_node_to_drawable(obj)
                    if optim_cfg is not None:
                        cls_optim_cfg = optim_cfg.get(class_name, default_optim_cfg)
                        self._param_groups.extend(model.get_param_group(cls_optim_cfg, prefix=model_id))

            elif assigned_to == AssetAssignment.SCENE:
                for scene in scene_bank:
                    model_id = model_class.compute_model_id(scene=scene, obj=None, class_name=class_name)
                    model: AssetModelMixin = model_class(**cfg.model_params, device=device)
                    model.assigned_to = assigned_to
                    model.init_asset_config(**cfg.get('asset_params', {}))
                    model.populate(scene=scene, obj=None, config=model.populate_cfg, device=device)
                    self.add_module(model_id, model)
                    self.class_name_infos.setdefault(class_name, {})[model_id] = [(scene.id, None),]
                    if load_assets_into_scene:
                        if class_name == 'LearnableParams':
                            scene.learnable_params = model
                            model.scene = scene
                        elif class_name == 'ImageEmbeddings':
                            scene.image_embeddings = model
                            model.scene = scene
                        else:
                            raise RuntimeError(f"Unsupported scene-level class_name={class_name}")
                    if optim_cfg is not None:
                        cls_optim_cfg = optim_cfg.get(class_name, default_optim_cfg)
                        self._param_groups.extend(model.get_param_group(cls_optim_cfg, prefix=model_id))

            elif assigned_to == AssetAssignment.MULTI_SCENE:
                # NOTE: Most of the functionlities should be supported with MULTI_OBJ_MULTI_SCENE ?
                pass
            
            elif assigned_to == AssetAssignment.MISC:
                # NOTE: For models that does not belong to scene / objects (e.g. belonging to renderers)
                model_id = model_class.compute_model_id(scene=None, obj=None, class_name=class_name)
                model: AssetModelMixin = model_class(**cfg.model_params, device=device)
                model.assigned_to = assigned_to
                model.init_asset_config(**cfg.get('asset_params', {}))
                model.populate(scene=None, obj=None, config=model.populate_cfg, device=device)
                self.add_module(model_id, model)
                self.class_name_infos.setdefault(class_name, {})[model_id] = []
                if optim_cfg is not None:
                    cls_optim_cfg = optim_cfg.get(class_name, default_optim_cfg)
                    self._param_groups.extend(model.get_param_group(cls_optim_cfg, prefix=model_id))

        if load is not None:
            self.load_asset_bank(load, strict=(class_name_list is None))
        
        self.to(device)

    def load_per_scene_models(self, scene: Scene):
        from app.models.base import AssetModelMixin, AssetAssignment # To avoid circular import
        
        class_name = 'LearnableParams'
        if class_name in self.class_name_configs.keys():
            cfg = self.class_name_configs[class_name]
            model_class: Type[AssetModelMixin] = import_str(cfg.model_class)
            model_id = model_class.compute_model_id(scene=scene, obj=None, class_name=class_name)
            scene.learnable_params = self[model_id]
            # NOTE: !!! Important !!! Might be initialized with scene_bank_trainval and needed to load into scene_bank_test
            scene.learnable_params.scene = scene
        
        class_name = 'ImageEmbeddings'
        if class_name in self.class_name_configs.keys():
            cfg = self.class_name_configs[class_name]
            model_class: Type[AssetModelMixin] = import_str(cfg.model_class)
            model_id = model_class.compute_model_id(scene=scene, obj=None, class_name=class_name)
            scene.image_embeddings = self[model_id]
            # NOTE: !!! Important !!! Might be initialized with scene_bank_trainval and needed to load into scene_bank_test
            scene.image_embeddings.scene = scene # !!! Important !!!

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

    def preprocess_model(self):
        """
        Operations that need to be executed only once throughout the entire rendering process, 
        as long as the network params remains untouched.
        """
        for model in self.values():
            model.preprocess_model()

    def preprocess_per_train_step(self, cur_it: int, logger: Logger=None):
        """
        Operations that need to be executed before each training step (before `trainer.forward`).
        """
        for model in self.values():
            model.preprocess_per_train_step(cur_it, logger=logger)

    def postprocess_per_train_step(self, cur_it: int, logger: Logger=None):
        """
        Operations that need to be executed after each training step (after `optmizer.step`).
        """
        for model in self.values():
            model.postprocess_per_train_step(cur_it, logger=logger)

    def preprocess_per_render_frame(self, renderer, observer, per_frame_info: dict={}):
        """
        Operations that need to be executed for every frame or view.
        """
        for model in self.values():
            model.preprocess_per_render_frame(renderer, observer, per_frame_info=per_frame_info)

    def configure_clip_grad_group(self, scene_bank, clip_grad_cfg: ConfigDict):
        clip_grad_groups = []
        if isinstance(clip_grad_cfg, Number):
            # for g in self._param_groups:
            #     clip_grad_groups.append({
            #         'name': g['name'], 
            #         'params': g['params'], 
            #         'clip_grad_val': float(clip_grad_cfg)
            #     })
            clip_grad_groups = [{'params':self.parameters(), 'clip_grad_val': float(clip_grad_cfg)}]
        elif isinstance(clip_grad_cfg, dict):
            clip_grad_cfg = clip_grad_cfg.deepcopy()
            
            if 'default' in clip_grad_cfg:
                clip_grad_val = clip_grad_cfg.pop('default')
                clip_grad_groups.append({'params':self.parameters(), 'clip_grad_val': clip_grad_val})
            
            for class_name, cfg in clip_grad_cfg.items():
                def _configure(model: nn.Module):
                    if isinstance(cfg, Number):
                        clip_grad_groups.append({'params':model.parameters(), 'clip_grad_val': clip_grad_val})
                    elif isinstance(cfg, dict):
                        if 'default' in cfg:
                            clip_grad_val = cfg.pop('default')
                            clip_grad_groups.append({'params':model.parameters(), 'clip_grad_val': clip_grad_val})
                        
                        for pnp, clip_grad_val in cfg.items():
                            plist = []
                            for pn, p in model.named_parameters():
                                if re.search('^'+pnp, pn):
                                    plist.append(p)
                            if len(plist) == 0:
                                log.warn(f"pattern '{pnp}' is not found when setting clip grad.")
                            else:
                                clip_grad_groups.append({'params': plist, 'clip_grad_val': clip_grad_val})
                
                if class_name in self.class_name_configs.keys():
                    for scene in scene_bank:
                        for obj in scene.get_drawable_groups_by_class_name(class_name, False):
                            model = obj.model
                            _configure(model)
                else:
                    raise RuntimeError(f"Invalid class_name={class_name}")

        self._clip_grad_groups = clip_grad_groups

    def apply_clip_grad(self):
        #---- Configured grad clip val
        if len(self._clip_grad_groups) > 0:
            for cg in self._clip_grad_groups:
                torch.nn.utils.clip_grad.clip_grad_value_(cg['params'], cg['clip_grad_val'])
        #---- Custom layer-wise grad clips
        for model in self.values():
            model.custom_grad_clip_step()
    
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
