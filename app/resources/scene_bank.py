"""
@file   scene_bank.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Create and load scene banks.
"""

__all__ = [
    'parse_scene_bank_cfg', 
    'get_dataset_scenario', 
    'get_scenario', 
    'load_scene', 
    'create_scene_bank', 
    'load_scene_bank', 
]

import os
import json
import pickle
import numpy as np
from glob import glob
from copy import deepcopy
from typing import Dict, List, Optional, Union, Tuple

import torch

from nr3d_lib.config import ConfigDict
from nr3d_lib.utils import IDListedDict, cond_mkdir

from app.resources.scenes import Scene
from dataio.scene_dataset import SceneDataset

def parse_scene_bank_cfg(scenario_cfg_str: str) -> Tuple[str, Union[int,float], Union[int,float]]:
    # scene_id, start, stop
    items = scenario_cfg_str.split(',')
    items = [it.strip(' ') for it in items]
    scene_id = items[0]
    start = None
    stop = None
    if len(items) > 1:
        start = int(items[1]) if str.isdecimal(items[1]) else float(items[1])
    if len(items) > 2:
        stop = int(items[2]) if str.isdecimal(items[2]) else float(items[2])
    return scene_id, start, stop

def filter_scene(dataset):
    """
    Filter scene according to certain rule (e.g. number of moving cars, etc.)
    """
    raise NotImplementedError

def get_dataset_scenario(
    dataset: SceneDataset, scene_id: str, *, 
    scenebank_cfg: dict) -> dict:
    
    scenebank_cfg = deepcopy(scenebank_cfg)
    scenarios = scenebank_cfg.pop('scenarios', None)
    for scenario_cfg_str in scenarios:
        _scene_id, start, stop = parse_scene_bank_cfg(scenario_cfg_str)
        if _scene_id == scene_id:
            scenebank_cfg.update(start=start, stop=stop)
            scenebank_cfg.update(scenebank_cfg.pop("on_load", {}))
            data_scenario = dataset.get_scenario(scene_id, **scenebank_cfg)
            return data_scenario
    raise RuntimeError(f"scene_id={scene_id} not in the given scenebank_cfg")

def get_scenario(
    dataset: SceneDataset, scene_id: str, *,
    scenebank_cfg = dict(), # [scenarios, observers, objects, **], 
    drawable_class_names: List[str] = [], 
    misc_node_class_names: List[str] = ['node', 'Ego', 'EgoVehicle', 'EgoDrone'],
    start=None, stop=None, 
    scenario_save_fpath: str = None, # (Optional) to save the created scenario
    ):
    """
    Main logic includes:
        - Validity check: Ensures all nodes are used, either as drawable, misc, or observer.
        - Adds "non-existent" nodes/objects: distant-view, sky, etc.
    """
    scenebank_cfg = deepcopy(scenebank_cfg)
    scenebank_cfg.pop("scenarios", None)
    scenebank_cfg.update(start=start, stop=stop)
    scenebank_cfg.update(scenebank_cfg.pop("on_load", {}))
    data_scenario = dataset.get_scenario(scene_id, **scenebank_cfg)
    app_scenario = dict(
        scene_id=data_scenario['scene_id'],
        metas=data_scenario['metas'],
        objects=dict(),
        observers=dict())

    #------------------------  Check objects
    def check_objects(oid: str, odict: dict, put_in: dict):
        class_name = odict['class_name']
        if not ((class_name in misc_node_class_names) or (class_name in drawable_class_names)):
            raise RuntimeError(f"Found un-used object node with class_name={class_name}")
        # Finally put in after all checks
        put_in[oid] = odict
        # Recursively check & add childrens
        if 'children' in odict: 
            old_children = odict.pop('children')
            odict['children'] = {}
            for cid, cdict in old_children.items():
                check_objects(cid, cdict, odict['children'])
    
    for oid, odict in data_scenario['objects'].items():
        check_objects(oid, odict, app_scenario['objects'])
    
    #------------------------  Check observers
    def check_observers(oid: str, odict: dict, put_in: dict):
        class_name = odict['class_name']
        if not (
            (class_name in misc_node_class_names) \
            or (class_name in scenebank_cfg.observer_cfgs.keys() and oid in scenebank_cfg.observer_cfgs[class_name].list)
            ):
            raise RuntimeError(f"Found un-used observer node with class_name={class_name}")
        # Finally put in after all checks
        put_in[oid] = odict
        # Recursively check & add childrens
        if 'children' in odict:
            old_children = odict.pop('children')
            odict['children'] = {}
            for cid, cdict in old_children.items():
                check_observers(cid, cdict, odict['children'])
    
    for oid, odict in data_scenario['observers'].items():
        check_observers(oid, odict, app_scenario['observers'])
    
    #------------ Add Sky, if needed
    if (class_name:='Sky') in drawable_class_names:
        str_id = f"{class_name.lower()}"
        obj_dict = dict(
            id=str_id,
            class_name=class_name,
        )
        app_scenario['objects'][str_id] = obj_dict

    #------------ Add NeRF++ background, if needed
    if (class_name:='Distant') in drawable_class_names:
        str_id = f"{class_name.lower()}"
        obj_dict = dict(
            id=str_id,
            class_name=class_name,
        )
        app_scenario['objects'][str_id] = obj_dict

    if scenario_save_fpath is not None:
        with open(scenario_save_fpath, 'wb') as fw:
            pickle.dump(app_scenario, fw)
        print(f"=> scenario file saved to {scenario_save_fpath}")
    
    return app_scenario

def load_scene(scenario_or_path: Union[str, Dict], device=None) -> Scene:
    scene = Scene(device=device)
    if isinstance(scenario_or_path, str):
        scene.load_from_scenario_file(scenario_or_path, device=device)
    elif isinstance(scenario_or_path, dict):
        scene.load_from_scenario(scenario_or_path, device=device)
    else:
        raise RuntimeError(f"Invalid type of input `scenario_or_path`={type(scenario_or_path)}")
    return scene

def create_scene_bank(
    dataset: SceneDataset, *, 
    scenebank_cfg = dict(), # [scenarios, observers, on_load]
    drawable_class_names: List[str] = [], 
    misc_node_class_names: List[str] = ['node', 'Ego', 'EgoVehicle', 'EgoDrone'], 
    scenebank_root: Optional[str] = None, # The directory to optionally save the created scene_bank
    device=None) -> Tuple[IDListedDict[Scene], dict]:
    
    scenario_list = []    
    scene_bank: IDListedDict[Scene] = IDListedDict()
    
    scenebank_cfg = deepcopy(scenebank_cfg)
    if (scenarios:=scenebank_cfg.pop('scenarios', None)) is None:
        scenarios = filter_scene(dataset)
    
    # Select from raw processed dataset scenarios
    for scenario_cfg in scenarios:
        scene_id, start, stop = parse_scene_bank_cfg(scenario_cfg)
        
        scenario = get_scenario(
            dataset, scene_id,  
            scenebank_cfg=scenebank_cfg, 
            drawable_class_names=drawable_class_names, 
            misc_node_class_names=misc_node_class_names, 
            start=start, stop=stop)
        scene = load_scene(scenario, device=device)
        scenario_list.append(scenario)
        scene_bank.append(scene)

    scenebank_meta = dict(
        scene_id_list=list(scene_bank.keys()),
    )
    
    if scenebank_root is not None:
        cond_mkdir(scenebank_root)
        for scenario in scenario_list:
            scenario_fpath = os.path.join(scenebank_root, f"{scene_id}.pt")
            with open(scenario_fpath, 'wb') as fw:
                pickle.dump(scenario, fw)
            print(f"=> scenario file saved to {scenario_fpath}")

        scene_bank_meta_fpath = os.path.join(scenebank_root, 'metadata.json')
        with open(scene_bank_meta_fpath, 'w') as f:
            json.dump(scenebank_meta, f)
        print(f"=> scene bank metadata saved to {scene_bank_meta_fpath}")
    return scene_bank, scenebank_meta

def load_scene_bank(scenebank_root: str, device=None) -> Tuple[IDListedDict[Scene], dict]:
    scene_bank: IDListedDict[Scene] = IDListedDict()
    
    with open(os.path.join(scenebank_root, 'metadata.json'), 'r') as f:
        scenebank_meta = json.load(f)
    
    scenario_file_pattern = os.path.join(scenebank_root, '*.pt')
    scenario_file_list = sorted(glob(scenario_file_pattern))
    assert len(scenario_file_list) > 0, f"Empty scenario: {scenario_file_pattern}"
    assert len(scenario_file_list)==len(scenebank_meta['scene_id_list']), "Mismatched scenario and metadata."
    for scenario_fpath in scenario_file_list:
        scene = load_scene(scenario_fpath, device=device)
        scene_bank.append(scene)
    
    return scene_bank, scenebank_meta
