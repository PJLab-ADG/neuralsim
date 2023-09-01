"""
@file   scenes.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Basic scene node structure
"""

__all__ = [
    'Scene', 
    'namedtuple_ind_id_obj', 
    'namedtuple_observer_infos'
]

import os
import pickle
import functools
import numpy as np
from typing import Callable, Literal, Union, List, Dict, NamedTuple, Tuple, Type

import torch
import torch.nn as nn

from nr3d_lib.models.attributes import *
from nr3d_lib.utils import IDListedDict, import_str
from nr3d_lib.models.spatial_accel.occgrid_accel import OccupancyGridAS

from app.resources import SceneNode
from app.resources.observers import OBSERVER_CLASS_NAMES, OBSERVER_TYPE, Camera, Lidar, RaysLidar

# namedtuple_ind_id_obj = namedtuple("namedtuple_ind_id_obj", "inds ids objs")
# namedtuple_ind_id_obj = NamedTuple("namedtuple_ind_id_obj", [('inds', List[int]), ('ids', List[str]), ('objs', List[SceneNode])])
class namedtuple_ind_id_obj(NamedTuple):
    inds: List[int]
    ids: List[str]
    objs: List[SceneNode]

class namedtuple_observer_infos(NamedTuple):
    tracks: torch.Tensor
    all_tracks: torch.Tensor
    all_frustum_pts: torch.Tensor

class Scene(object):
    def __init__(self, unique_id: str='WORLD', device=torch.device('cuda'), dtype=torch.float) -> None:
        self.device = device
        self.dtype = dtype
        
        self.id = unique_id
        
        #-------- Init scene graph basic structures
        self._init_graph()
        
        #-------- For scene with loaded sequence/log data
        # The frame ind(s) frozen at 
        self.i = None
        # Total data frames of this scene (if this scene has a `dataset`)
        self.n_frames: int = 0 
        # The offset between scene's frames and dataset's frames (if this scene has a `dataset`)
        self.data_frame_offset: int = 0
        
        #-------- Scene meta data
        self.metas: dict = {}
        
        #-------- Optional scene-level misc models
        # app/models_misc/learnable_params.py; Scene parameter (pose, intr, extr) refinement 
        self.learnable_params = None
        # app/models_misc/image_embeddings.py; Per-frame image embeddings, as in NeRF in the wild
        self.image_embeddings = None
        
        #-------- For code_single, the single object that we focus on. 
        # Load from scenario; different tasks could have different settings.
        #     e.g. "Street" for street views, "Main" for other
        self.main_class_name = 'Main'

    def _init_graph(self):
        #-------- Scene-Graph basic structures
        self.root = SceneNode(self.id, scene=self)
        # self.root.transform = None
        # self.root.world_transform = None

        self.drawables: IDListedDict[SceneNode] = IDListedDict()
        self.drawable_groups_by_class_name: Dict[str, IDListedDict[SceneNode]] = {}
        self.drawable_groups_by_model_id: Dict[str, IDListedDict[SceneNode]] = {}
        
        self.observers: IDListedDict[OBSERVER_TYPE] = IDListedDict()
        self.observer_groups_by_class_name: Dict[str, IDListedDict[OBSERVER_TYPE]] = {}
        
        self.all_nodes: IDListedDict[SceneNode] = IDListedDict([self.root])
        self.all_nodes_by_class_name: Dict[str, IDListedDict[SceneNode]] = {self.root.class_name: IDListedDict([self.root])}

    #------------------------------------------------------------------
    #----------------------  Loading, updating, reading
    # @profile
    def frozen_at(self, i: Union[int, torch.LongTensor]):
        """ Frozen at a single time frame or multiple time frames, and update the scene graph.
        This will first retrieve data at the given frame for each nodes' attributes, 
        and then update the scene graph from the root to leaves, 
        in which the `world_transform` of each node will also be calculated from root to leaf.

        Args:
            i (Union[int, torch.LongTensor]): The frame indice(s) to freeze the scene at
        """
        self.i = i
        # self.root.frozen_at(i)
        for n in self.all_nodes:
            n._frozen_at(i)
        
        # NOTE: Update the scene graph, from root to leaf.
        #       The `world_transform` of each node will also be calculated from root to leaf.
        self.root.update()
    
    def frozen_at_full(self):
        """
        Frozen at full indices, from start to end
        """
        self.frozen_at(torch.arange(len(self), device=self.device, dtype=torch.long))
    
    def unfrozen(self):
        """
        Un-freeze a scene; i.e. resets all its nodes to default in-active state
        """
        self.reset_attr()

    def reset_attr(self):
        """
        Resets all self nodes to default in-active state
        """
        # Reset frozen status
        del self.i
        self.i = None
        for n in self.all_nodes:
            n.reset_attr()

    def clear_graph(self):
        """
        Complemenly cleans this scene graph and makes self a empty scene
        """
        del self.root
        del self.drawables, self.drawable_groups_by_class_name, self.drawable_groups_by_model_id
        del self.observers, self.observer_groups_by_class_name
        del self.all_nodes, self.all_nodes_by_class_name
        self._init_graph()

    def load_from_scenario(self, scenario: dict, device=None):
        """
        Load one scene from the given scenario description dict.
        
        NOTE: DO NOT let any useful node to be not-used parent's child, 
              because all children and descendants of a not-used node will be ignored.
        TODO: Consider making this a classmethod.

        Args:
            scenario (dict): The given scenario description dict.
            device (torch.device, optional): The target torch.device to store this scene's nodes' attributes. 
                Defaults to None.
        """
        #---- Load scene meta
        # NOTE: Scene's id is changed when loading a scenario
        self.id: str = scenario['scene_id']
        self.data_frame_offset: int = scenario['metas'].get('data_frame_offset', 0)
        self.n_frames: int = scenario['metas']['n_frames']
        self.metas: dict = scenario['metas']
        
        # NOTE: For code_single, the single object that we focus on. 
        # e.g. "Street" for street views, "Obj" for single-object reconstruction, "Room" for in-door tasks
        self.main_class_name = scenario['metas'].get('main_class_name', None)
        
        #---- Process objects
        def load_objects(oid: str, odict: dict, parent:SceneNode=self.root):
            """
            Recursively load
            """
            o = SceneNode(oid, class_name=odict['class_name'], scene=self, device=device)
            # Load object attr segments
            if 'segments' in odict:
                o.set_n_total_frames(self.n_frames)
                for seg in odict['segments']:
                    data = seg['data']
                    d = {}
                    # TODO: Determine the type of each attribute during dataset preprocessing.
                    if 'transform' in data: d.update(transform=TransformMat4x4(data['transform'], dtype=torch.float, device=device))
                    if 'scale' in data: d.update(scale=Scale(data['scale'], dtype=torch.float, device=device))
                    if 'timestamp' in data: d.update(timestamp=Scalar(data['timestamp'], device=device))
                    if 'global_frame_ind' in data: d.update(global_frame_ind=Scalar(data['global_frame_ind'], device=device))
                    o.load_attr_segment(n_frames=seg['n_frames'], start_frame=seg.get('start_frame', 0), **d)
                o.finish_attr_segments()
            elif 'data' in odict:
                o.set_n_total_frames(self.n_frames)
                d = {}
                data = odict['data']
                if 'transform' in data: d.update(transform=TransformMat4x4(data['transform'], dtype=torch.float, device=device))
                if 'scale' in data: d.update(scale=Scale(data['scale'], dtype=torch.float, device=device))
                if 'timestamp' in data: d.update(timestamp=Scalar(data['timestamp'], device=device))
                if 'global_frame_ind' in data: d.update(global_frame_ind=Scalar(data['global_frame_ind'], device=device))
                o.load_attr_segment(n_frames=odict['n_frames'], start_frame=odict.get('start_frame', 0), **d)
                o.finish_attr_segments()
            self.add_node(o, parent=parent)
            # Recursively load node's childrens
            if 'children' in odict:
                for cid, cdict in odict['children'].items():
                    load_objects(cid, cdict, parent=o)
        for oid, odict in scenario['objects'].items():
            load_objects(oid, odict, parent=self.root)

        #---- Process observers
        def load_observers(oid: str, odict: dict, parent:SceneNode=self.root):
            """
            # Recursively load
            """
            if odict['class_name'] == 'Camera':
                o = Camera(oid, scene=self, device=device)
            elif odict['class_name'] == 'Lidar':
                o = Lidar(oid, scene=self, device=device)
            elif odict['class_name'] == 'RaysLidar':
                o = RaysLidar(oid, scene=self, device=device)
            else:
                o = SceneNode(oid, class_name=odict['class_name'], scene=self, device=device)
            if 'data' in odict:
                o.set_n_total_frames(self.n_frames)
                data = odict['data']
                d = {}
                if 'intr' in data:
                    hw = data['hw'].astype(np.float32)
                    intr_dict = dict(H=hw[:,0], W=hw[:,1], device=device)
                    camera_model: str = odict.get('camera_model', 'pinhole')
                    if camera_model.lower() == 'pinhole':
                        intr_dict['mat'] = CameraMatrix3x3(data['intr'][..., :3, :3], dtype=torch.float, device=device)
                        intr = PinholeCameraMatHW(**intr_dict)
                    elif camera_model.lower() == 'opencv':
                        distortion = data['distortion']
                        intr_dict['mat'] = CameraMatrix3x3(data['intr'][..., :3, :3], dtype=torch.float, device=device)
                        intr_dict['distortion'] = make_vector(distortion.shape[-1])(distortion, dtype=torch.float, device=device)
                        intr = OpenCVCameraMatHW(**intr_dict)
                    elif camera_model.lower() == 'fisheye':
                        distortion = data['distortion']
                        intr_dict['mat'] = CameraMatrix3x3(data['intr'][..., :3, :3], dtype=torch.float, device=device)
                        intr_dict['distortion'] = make_vector(distortion.shape[-1])(distortion, dtype=torch.float, device=device)
                        intr = FisheyeCameraMatHW(**intr_dict)
                    else:
                        raise RuntimeError(f"Invalid camera_model={camera_model}")
                    d.update(intr=intr)
                if 'transform' in data: d.update(transform=TransformMat4x4(data['transform'], dtype=torch.float, device=device))
                if 'timestamp' in data: d.update(timestamp=Scalar(data['timestamp'], device=device))
                if 'global_frame_ind' in data: d.update(global_frame_ind=Scalar(data['global_frame_ind'], device=device))
                if 'exposure' in data: d.update(exposure=Scalar(data['exposure'], device=device))
                o.load_attr_segment(n_frames=odict['n_frames'], start_frame=odict.get('start_frame', 0), **d)
                o.finish_attr_segments()
            self.add_node(o, parent=parent)
            # Recursively load node's childrens
            if 'children' in odict:
                for cid, cdict in odict['children'].items():
                    load_observers(cid, cdict, parent=o)
        for oid, odict in scenario['observers'].items():
            load_observers(oid, odict, parent=self.root)

    def load_from_scenario_file(self, scenario_file: str, device=None):
        """
        Load one scene from the given path string to the scenario description dict (pickle).
        TODO: Consider making this a classmethod.

        Args:
            scenario_file (str): The given path string to the scenario description dict (pickle). 
            device (torch.device, optional): The target torch.device to store this scene's nodes' attributes. 
                Defaults to None.
        """
        self.clear_graph()
        if not os.path.exists(scenario_file):
            raise RuntimeError(f"Not exist: {scenario_file}")
        with open(scenario_file, 'rb') as f:
            scenario = pickle.load(f)
            self.load_from_scenario(scenario, device=device)

    def load_from_nodes(self, nodes: List[SceneNode]):
        """
        Load one scene from the given list of SceneNode
        
        TODO: Consider making this a classmethod.
        """
        self.clear_graph()
        for o in nodes:
            self.add_node(o)

    def load_assets(self, asset_bank):
        """
        Load the corresponding asset models for each node in the this scene from the asset bank.
        """
        from app.models.base import AssetModelMixin # To avoid circular import
        self.asset_bank = asset_bank
        for class_name, obj_group in self.all_nodes_by_class_name.items():
            # NOTE: The same functionality with asset_bank.compute_model_id
            if (cfg:=asset_bank.class_name_configs.get(class_name, None)) is not None:
                model_class: Type[AssetModelMixin] = import_str(cfg.model_class)
                for o in obj_group:
                    model_id = model_class.compute_model_id(scene=self, obj=o, class_name=class_name)
                    o.model = self.asset_bank[model_id]
                    self.add_node_to_drawable(o)
        self.asset_bank.load_per_scene_models(self)

    def __len__(self):
        return self.n_frames

    #------------------------------------------------------------------
    #----------------------  Node management
    def add_node(self, node: SceneNode, parent: SceneNode=None):
        """
        Add one node to this scene. 
        If the node has non-empty `.model`, it will be added to the drawable list.
        If the node has a observer class_name, it will be added to the observer list.

        Args:
            node (SceneNode): The node to add.
            parent (SceneNode, optional): Optionally specify a parent for the given node. Defaults to None.
        """
        
        node.scene = self

        # self.all_nodes[node.id] = node
        self.all_nodes.append(node)
        
        if parent is None: parent = self.root
        parent.add_child(node)
        
        class_name = node.class_name
        if class_name not in self.all_nodes_by_class_name.keys():
            self.all_nodes_by_class_name[class_name] = IDListedDict()
        self.all_nodes_by_class_name[class_name][node.id] = node
        
        # Drawables
        if node.model is not None:
            self.add_node_to_drawable(node)
        # Observers
        if class_name in OBSERVER_CLASS_NAMES:
            self.add_node_to_observer(node)

    def add_node_to_drawable(self, node: SceneNode, model_id: str = None):
        """
        Add one node to the drawable list (nodes with non-empty `.model`).
        Will automatically group nodes by their `model_id`.

        Args:
            node (SceneNode): The given node.
            model_id (str, optional): Bypass `node.model.id`. Defaults to None.
        """
        assert node.model is not None, 'only nodes with `model` is drawable'
        self.drawables[node.id] = node
        
        class_name = node.class_name
        if class_name not in self.drawable_groups_by_class_name.keys():
            self.drawable_groups_by_class_name[class_name] = IDListedDict()
        self.drawable_groups_by_class_name[class_name][node.id] = node

        if model_id is None: model_id = node.model.id
        if model_id not in self.drawable_groups_by_model_id.keys():
            self.drawable_groups_by_model_id[model_id] = IDListedDict()
        self.drawable_groups_by_model_id[model_id][node.id] = node

    def add_node_to_observer(self, node: SceneNode):
        """
        Add one node to the observer list (nodes with observer class_name)
        Will automatically group nodes by their class_name.

        Args:
            node (SceneNode): The given node.
        """
        assert node.class_name in OBSERVER_CLASS_NAMES, f"only nodes of class_name in {OBSERVER_CLASS_NAMES} is an observer"
        self.observers[node.id] = node
        
        class_name = node.class_name
        if class_name not in self.observer_groups_by_class_name.keys():
            self.observer_groups_by_class_name[class_name] = IDListedDict()
        self.observer_groups_by_class_name[class_name][node.id] = node

    # @profile
    def get_drawables(self, only_valid=True) -> IDListedDict[SceneNode]:
        if only_valid:
            return IDListedDict([o for o in self.drawables if o.valid])
        else:
            return self.drawables
    # @profile
    def get_drawable_groups_by_class_name(self, class_name: str, only_valid=True) -> IDListedDict[SceneNode]:
        if class_name in self.drawable_groups_by_class_name.keys():
            if only_valid:
                return IDListedDict([o for o in self.drawable_groups_by_class_name[class_name] if o.valid])
            else:
                return self.drawable_groups_by_class_name[class_name]
        else:
            return IDListedDict()
    def get_drawable_groups_by_class_name_list(self, class_name_list: List[str], only_valid=True) -> IDListedDict[SceneNode]:
        if only_valid:
            node_list = [o for o in self.drawables if o.valid and o.class_name in class_name_list]
        else:
            node_list = [o for o in self.drawables if o.class_name in class_name_list]
        return IDListedDict(node_list)
    # @profile
    def get_drawable_groups_by_model_id(self, model_id: str, only_valid=True) -> IDListedDict[SceneNode]:
        if only_valid:
            return IDListedDict([o for o in self.drawable_groups_by_model_id[model_id] if o.valid])
        else:
            return self.drawable_groups_by_model_id[model_id]
    @staticmethod
    def group_drawables_by_class_name(drawables: List[SceneNode]) -> Dict[str, namedtuple_ind_id_obj]:
        def key_fn(o: SceneNode):
            return o.class_name
        return Scene.group_drawables_by_key_fn(drawables, key_fn)
    @staticmethod
    def group_drawables_by_model_id(drawables: List[SceneNode]) -> Dict[str, namedtuple_ind_id_obj]:
        def key_fn(o: SceneNode):
            return o.model.id
        return Scene.group_drawables_by_key_fn(drawables, key_fn)
    @staticmethod
    def group_drawables_by_key_fn(drawables: List[SceneNode], key_fn: Callable[[SceneNode],str]) -> Dict[str, namedtuple_ind_id_obj]:
        map_groups: Dict[str, namedtuple_ind_id_obj] = {}
        for ind, o in enumerate(drawables):
            key = key_fn(o)
            item = [ind, o.id, o]
            if not key in map_groups:
                map_groups[key] = [item]
            else:
                map_groups[key].append(item)
        for n, g in map_groups.items():
            map_groups[n] = namedtuple_ind_id_obj(*[list(x) for x in zip(*g)])
        return map_groups
    def get_drawable_class_ind_map(self):
        # TODO: This could be adapted into a more scientific mapping method, such as following COCO.
        all_classnames = self.drawable_groups_by_class_name.keys()
        return {cn: i for i, cn in enumerate(all_classnames)}
    def get_drawable_instance_ind_map(self):
        all_ids = self.drawables.keys()
        return {id:i for i, id in enumerate(all_ids)}

    # @profile
    def get_observers(self, only_valid=True) -> IDListedDict[SceneNode]:
        if only_valid:
            return IDListedDict([o for o in self.observers if o.valid])
        else:
            return self.observers
    # @profile
    def get_observer_groups_by_class_name(self, class_name: str, only_valid=True) -> IDListedDict[SceneNode]:
        if class_name in self.observer_groups_by_class_name.keys():
            if only_valid:
                return IDListedDict([o for o in self.observer_groups_by_class_name[class_name] if o.valid])
            else:
                return self.observer_groups_by_class_name[class_name]
        else:
            return IDListedDict()

    def get_cameras(self, only_valid=True) -> IDListedDict[Camera]:
        if only_valid:
            return IDListedDict([o for o in self.observer_groups_by_class_name['Camera'] if o.valid])
        else:
            return self.observer_groups_by_class_name['Camera']

    #------------------------------------------------------------------
    #----------------------  Render & ray query
    @staticmethod
    # @profile
    def convert_rays_in_nodes_list(
        rays_o: torch.Tensor, rays_d: torch.Tensor, node_list: List[SceneNode]
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Convert rays in world coords to node object local coords
        
        Args:
            rays_o (torch.Tensor): [..., 3], Ray origins in world coords
            rays_d (torch.Tensor): [..., 3], Ray directions in world coords
            node_list (List[SceneNode]): List of nodes

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                rays_o_o: [B_object, ..., 3], Ray origins in object local coords
                rays_d_o: [B_object, ..., 3], Ray directions in object local coords
        """
        rotations = [obj.world_transform.rotation() for obj in node_list]
        translations = [obj.world_transform.translation() for obj in node_list]
        scales = [obj.scale.ratio() for obj in node_list]
        # for obj in groups:
        #     rotations.append(obj.world_transform.rotation())
        #     translations.append(obj.world_transform.translation())
        #     scales.append(obj.scale.ratio())
        rotations_inv = torch.stack(rotations, 0).transpose(-1,-2).unsqueeze(-3)
        translations = torch.stack(translations, 0).unsqueeze(-2)
        scales = torch.stack(scales, 0).unsqueeze(-2)
        
        # rays_o_o = torch.einsum('...ij,...j->...i', rotations_inv, rays_o.unsqueeze(0)-translations)
        # rays_d_o = torch.einsum('...ij,...j->...i', rotations_inv, rays_d.unsqueeze(0))
        rays_o_o = (rotations_inv * (rays_o.unsqueeze(0)-translations).unsqueeze(-2)).sum(-1)
        rays_d_o = (rotations_inv * rays_d[None,...,None,:]).sum(-1)
        
        rays_o_o = rays_o_o / scales
        rays_d_o = rays_d_o / scales
        return rays_o_o, rays_d_o

    @staticmethod
    def convert_rays_in_node(
        rays_o: torch.Tensor, rays_d: torch.Tensor, node: SceneNode
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Convert rays in world coords to node object local coords

        Args:
            rays_o (torch.Tensor): [..., 3], Ray origins in world coords
            rays_d (torch.Tensor): [..., 3], Ray directions in world coords
            node (SceneNode): The given node

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                rays_o_o: [..., 3], Ray origins in object local coords
                rays_d_o: [..., 3], Ray directions in object local coords
        """
        rotations_inv = node.world_transform.rotation().transpose(-1,-2)
        translations = node.world_transform.translation()
        scales = node.scale.ratio()
        
        # [3, 3] @ [..., 1, 3]
        rays_o = (rotations_inv * (rays_o - translations).unsqueeze(-2)).sum(-1) / scales
        rays_d = (rotations_inv * rays_d.unsqueeze(-2)).sum(-1) / scales
        return rays_o, rays_d

    @torch.no_grad()
    def process_observer_infos(self, **kwargs) -> namedtuple_observer_infos:
        """
        Iterate through each frame of each camera, calculating information related to the camera trajectory.
        
        class_name: [default=]'Camera'
        far_clip: [default=]None
        all: [defalt=]True
        """
        
        # return namedtuple_observer_infos(
        #     tracks=torch.stack([
        #         torch.tensor([-1,-1,-1], dtype=torch.float, device=self.device),
        #         torch.tensor([1,1,1], dtype=torch.float, device=self.device)
        #         ], 0), 
        #     all_frustum_pts=torch.stack([
        #         torch.tensor([-1,-1,-1], dtype=torch.float, device=self.device),
        #         torch.tensor([1,1,1], dtype=torch.float, device=self.device)
        #         ], 0))

        # NOTE: Caching related
        # if hasattr(self.process_observer_infos, '_cache_kwargs'):
        #     # NOTE: Since frozing scenes costs, we cache the first returns,
        #     #       and use them if input is not changed
        #     changed = False
        #     cached_kwargs = self.process_observer_infos._cache_kwargs
        #     for k, v in kwargs.items():
        #         if k not in cached_kwargs.keys() or v != cached_kwargs[k]:
        #             changed = True
        #             break
        #     if not changed:
        #         return self.process_observer_infos._cache_ret
                
        class_name = kwargs.setdefault('class_name', 'Camera')
        far_clip = kwargs.setdefault('far_clip', None)
        
        assert self.i is None, f"Can not call process_camera_infos() in the middle of a frozen scene"
        cams = self.get_cameras(False)

        frustum_extend_pts = []
        cam_poses = []

        self.frozen_at_full()
        cam0_poses = cams[0].world_transform.mat_4x4()
        for cam in cams:
            cam_poses.append(cam.world_transform.mat_4x4())
            frustum_extend_pts.append(cam.get_view_frustum_pts(near=0., far=far_clip))
        
        frustum_extend_pts = torch.stack(frustum_extend_pts, 0)
        cam_poses = torch.stack(cam_poses, 0)
        all_cam_tracks = cam_poses[..., :3, 3]
        cam0_tracks = cam0_poses[..., :3, 3]
        self.reset_attr()
        ret = namedtuple_observer_infos(tracks=cam0_tracks, 
                                        all_tracks=all_cam_tracks, 
                                        all_frustum_pts=frustum_extend_pts)
        
        # NOTE: Caching related
        # self.process_observer_infos._cache_kwargs = deepcopy(kwargs)
        # self.process_observer_infos._cache_ret = deepcopy(ret)

        return ret

    #------------------------------------------------------------------
    #----------------------  Misc
    def _apply(self, fn):
        # NOTE: Already inplace
        for k in self.all_nodes.keys():
            self.all_nodes[k]._apply(fn)
        return self
    @functools.wraps(torch.Tensor.to)
    def to(self, *args, **kwargs):
        # NOTE: Already inplace
        for k in self.all_nodes.keys():
            self.all_nodes[k].to(*args, **kwargs)
        return self
    @functools.wraps(nn.Module.cuda)
    def cuda(self, device=None):
        return self._apply(lambda t: t.cuda(device))
    @functools.wraps(nn.Module.cpu)
    def cpu(self):
        return self._apply(lambda t: t.cpu())
    @functools.wraps(nn.Module.float)
    def float(self):
        return self._apply(lambda t: t.float() if t.is_floating_point() else t)
    @functools.wraps(nn.Module.double)
    def double(self):
        return self._apply(lambda t: t.double() if t.is_floating_point() else t)
    @functools.wraps(nn.Module.double)
    def half(self):
        return self._apply(lambda t: t.half() if t.is_floating_point() else t)

    #------------------------------------------------------
    #--------------- DEBUG Functionalities ----------------
    #------------------------------------------------------
    @torch.no_grad()
    def debug_vis_scene_graph(self, frame_ind=None, arrow_length: float = 4.0, font_scale: float = 0.5):
        from nr3d_lib.plot import create_camera_frustum_o3d
        import open3d as o3d
        import open3d.visualization.gui as gui
        
        app = gui.Application.instance
        app.initialize()
        w = app.create_window(f"Scene-graph: {self.id}", 1024, 768)
        widget3d = gui.SceneWidget()
        w.add_child(widget3d)
        widget3d.scene = o3d.visualization.rendering.Open3DScene(w.renderer)
        # red = o3d.visualization.rendering.MaterialRecord()
        # red.base_color = [1., 0., 0., 1.0]
        # red.shader = "defaultUnlit"
        # green = o3d.visualization.rendering.MaterialRecord()
        # green.base_color = [0., 1., 0., 1.0]
        # green.shader = "defaultUnlit"
        # blue = o3d.visualization.rendering.MaterialRecord()
        # blue.base_color = [0., 0., 1., 1.0]
        # blue.shader = "defaultUnlit"
        # black = o3d.visualization.rendering.MaterialRecord()
        # black.base_color = [0., 0., 0., 1.]
        # black.line_width = 10
        # black.shader = "defaultUnlit"

        line_mat = o3d.visualization.rendering.MaterialRecord()
        line_mat.line_width = 3
        line_mat.shader = "unlitLine"

        edge_line_mat = o3d.visualization.rendering.MaterialRecord()
        edge_line_mat.line_width = 10
        edge_line_mat.shader = "unlitLine"

        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        
        if frame_ind is not None:
            self.frozen_at(frame_ind)
        cam0: Camera = self.get_observer_groups_by_class_name('Camera', False)['camera_FRONT']
        all_drawables = self.get_drawables(True)
        filtered_drawables = IDListedDict(cam0.filter_drawable_groups(all_drawables))

        geometry_camera = None

        # things_to_draw = vis_camera_o3d(
        #     colored_camera_dicts=[
        #         {
        #             'intr': cam0.intr.mat_4x4().data.cpu().numpy(), 
        #             'c2w': cam0.world_transform.mat_4x4().data.cpu().numpy(),
        #             'img_wh': (cam0.intr.W, cam0.intr.H)
        #             }
        #         ], 
        #     cam_size=cam0.far, per_cam_axis=False, show=False)

        def draw_node(o:SceneNode, parent:SceneNode=None):
            """
            Recursively adds to things to draw
            """
            # nonlocal things_to_draw
            nonlocal geometry_camera
            if o.valid:
                #---- Coordinate frame
                coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=arrow_length)
                # coord_frame.compute_vertex_normals()
                coord_frame.transform(o.world_transform.mat_4x4().data.cpu().numpy())
                
                # things_to_draw.append(coord_frame)
                widget3d.scene.add_geometry(o.id, coord_frame, mat)
                
                if o.class_name in OBSERVER_CLASS_NAMES:
                    l = widget3d.add_3d_label(
                        o.world_transform(torch.tensor([0.,0.,arrow_length],
                                                       device=self.device, dtype=self.dtype)).data.cpu().numpy(), 
                        o.class_name + ' : ' + o.id)
                else:
                    l = widget3d.add_3d_label(
                        o.world_transform.translation().data.cpu().numpy(), 
                        o.class_name + ' : ' + o.id)
                l.scale = font_scale

                #---- Draw Camera frustum
                if o.class_name == 'Camera' and o.id == 'camera_FRONT':
                    geometry_camera = create_camera_frustum_o3d(
                        img_wh=(o.intr.W, o.intr.H), 
                        intr=o.intr.mat_4x4().data.cpu().numpy(), 
                        c2w=o.world_transform.mat_4x4().data.cpu().numpy(), 
                        frustum_length=o.far if o.far is not None else 120.0, color=[0,0,1])
                    widget3d.scene.add_geometry(f"{o.id}.frustum", geometry_camera, line_mat)
                    
                #---- Draw OBB, if needed
                # if o.model_BSPH is not None:
                if o.attr_array is not None and 'scale' in o.attr_array.subattr.keys():
                    # OBB
                    OBB = o3d.geometry.OrientedBoundingBox(
                        center=o.world_transform.translation().data.cpu().numpy(),
                        R=o.world_transform.rotation().data.cpu().numpy(),
                        extent=o.scale.ratio().data.cpu().numpy()
                    )
                    # OBB.compute_vertex_normals()
                    
                    if o.id in filtered_drawables.keys():
                        OBB.color = [1.0,0.0,0.0] # Drawables remained after frustum culling
                    else:
                        OBB.color = [0.0,0.0,0.0] # Drawables culled away
                    # things_to_draw.append(OBB)
                    widget3d.scene.add_geometry(f"{o.id}.box", OBB, line_mat)

                #---- Draw parent-child connection edge, if needed
                if parent is not None:
                    p_center = parent.world_transform.translation().data.cpu().numpy()
                    o_center = o.world_transform.translation().data.cpu().numpy()
                    if np.linalg.norm(p_center-o_center) > 0.01:
                        points = [p_center, o_center]
                        lines = [[0,1]]
                        colors = [[0.7,0.7,0.7]]
                        connection = o3d.geometry.LineSet()
                        connection.points = o3d.utility.Vector3dVector(points)
                        connection.lines = o3d.utility.Vector2iVector(lines)
                        connection.colors = o3d.utility.Vector3dVector(colors)
                        widget3d.scene.add_geometry(f"{parent.id}-{o.id}", connection, edge_line_mat)

            for child in o.children:
                draw_node(child, parent=o)
        
        for child in self.root.children:
            draw_node(child)
        
        # o3d.visualization.draw_geometries(things_to_draw)
        
        # (Optional) Try debug background space
        if len(lst:=self.get_drawable_groups_by_class_name('Street')) > 0:
            from nr3d_lib.models.spatial import ForestBlockSpace
            bg_obj = lst[0]
            bg_model = bg_obj.model
            if (space:=getattr(bg_model, 'space', None)) and isinstance((space:=bg_model.space), ForestBlockSpace):
                lineset_bg = space.debug_vis(show=False, draw_lines=True, draw_mesh=False)[0]
                widget3d.scene.add_geometry(f"background.space", lineset_bg, line_mat)
        
        bbox = widget3d.scene.bounding_box
        widget3d.setup_camera(60.0, bbox, bbox.get_center())
        app.run()
        
        if frame_ind is not None:
            self.unfrozen()

    @torch.no_grad()
    def debug_vis_multi_frame(
        self, 
        plot_details_at: int = 0, # The frame at watch to plot dynamic objects and other details
        plot_freq: int = 5 # In case things are too dense, only plot at every N frames. Set to 1 to plot all
        ):
        import vedo
        from nr3d_lib.plot import create_camera_frustum_vedo
        
        fi = torch.arange(len(self), device=self.device, dtype=torch.long)
        self.frozen_at(fi)
        
        if 'cam_id_list' in self.metas.keys():
            cam0 = self.observer_groups_by_class_name['Camera'][self.metas['cam_id_list'][0]]
        else:
            cam0 = self.observer_groups_by_class_name['Camera'][0]
        intrs = cam0.intr.mat_3x3().data.cpu().numpy()
        c2ws = cam0.world_transform.mat_4x4().data.cpu().numpy()
        WHs = cam0.intr.unscaled_wh().data.cpu().numpy().astype(int)
        
        #---- Plot all camera frustums
        cam_actors = []
        for i, (WH, intr, c2w) in enumerate(zip(WHs, intrs, c2ws)):
            if i == plot_details_at:
                color = 'red'
                lw = 4
            else:
                if plot_freq > 1 and (i % plot_freq < plot_freq):
                    continue
                color = 'k4'
                lw = 2
            cam = create_camera_frustum_vedo(WH, intr, c2w, 0.2, color, lw)
            cam_actors.append(cam)
        self.unfrozen()
        
        veh_boxes = []
        self.frozen_at(plot_details_at)
        for node in self.all_nodes_by_class_name['Vehicle']:
            if not node.valid:
                continue
            scale = node.scale.ratio().data.cpu().numpy()
            bound = np.stack([-scale/2.,scale/2.], axis=-1).reshape(6)
            box = vedo.Box(size=bound)
            box.apply_transform(node.world_transform.mat_4x4().data.cpu().numpy())
            length = 2.
            ax = vedo.Arrow([0,0,0], [length,0,0], c='r')
            ay = vedo.Arrow([0,0,0], [0,length,0], c='g')
            az = vedo.Arrow([0,0,0], [0,0,length], c='b')
            axes = vedo.Assembly(ax,ay,az)
            axes.lighting('off')
            axes.apply_transform(node.world_transform.mat_4x4().data.cpu().numpy())
            box = vedo.Assembly(box.wireframe(), axes)
            veh_boxes.append(box)
        self.unfrozen()
        actors = cam_actors + veh_boxes
        vedo.show(actors, axis=1)
    
    def debug_vis_anim(
        self, 
        scene_dataloader = None, # Needed for loading and plotting camera and lidar observations
        camera_length: float = 0.6, 
        plot_image = False, # Plot camera observations (images)
        plot_lidar = False, # Plot lidar observations (pointcloud)
        lidar_pts_downsample: int = 2, # In case lidar pts are too dense. Set to 1 to plot all
        mesh_file: str = None, # Filepath to the mesh
        extra_draw: List[Literal['main.occ_grid']] = [], 
    ):
        if plot_lidar or plot_image:
            assert scene_dataloader is not None, "Need scene_dataloader to plot lidar / camera data"
        
        import vedo
        from nr3d_lib.plot import create_camera_frustum_vedo, get_n_ind_pallete
                
        # Setup the scene       
        plt = vedo.Plotter(axes=1, interactive=False)
        lidar_pts = None
        pics = []
        fi_counter = 0

        #---- Plot input mesh
        surf = None
        if mesh_file is not None:
            surf = vedo.Mesh(mesh_file)
            # surf.lighting('ambient')
            surf.color('gray8')

        #---- Plot camera frustums
        self.frozen_at(0)
        cam_actors = {}
        colors = get_n_ind_pallete(len(self.observer_groups_by_class_name['Camera'].keys()))
        for ci, (cam_id, cam_node) in enumerate(self.observer_groups_by_class_name['Camera'].items()):
            WH = (cam_node.intr.W, cam_node.intr.H)
            intr = cam_node.intr.mat_3x3().data.cpu().numpy()
            # c2w = cam_node.world_transform.mat_4x4().data.cpu().numpy() # Set in animation
            cam = create_camera_frustum_vedo(WH, intr, np.eye(4), camera_length, color=colors[ci], lw=3)
            cam_actors[cam_id] = cam

        #---- Plot main object (i.e. the focused object in object-centric scenes, \
        #       room in indoor scenes, street in street-views)
        aabb = None # [[xmin,ymin,zmin], [xmax,ymax,zmax]]
        main_obj = self.all_nodes_by_class_name[self.main_class_name][0]
        main_box = None
        main_label = None
        main_occ_grid = None
        if main_obj.model is not None and main_obj.model.space is not None:
            aabb = (main_obj.scale.ratio() * main_obj.model.space.aabb).data.cpu().numpy()
        elif 'aabb' in self.metas:
            # A nice alternative aabb choice when models are not available
            aabb = self.metas['aabb']
        else:
            aabb = None
        if aabb is not None:
            bound = aabb.T.reshape(-1).tolist() # [xmin, xmax, ymin, ymax, zmin, zmax]
            corner = aabb[0]
            main_box = vedo.Box(size=bound)
            main_box.apply_transform(main_obj.world_transform.mat_4x4().data.cpu().numpy(), reset=True)       
            main_box.lw(4)
            main_box.wireframe()
            main_box.lighting('off')
            
            # Add box axis (local coordinate frame) on the corner
            ax_length = (aabb[1] - aabb[0]).min()
            ax = vedo.Arrow(corner, corner+np.array([ax_length,0,0]), c='r', s=0.2)
            ay = vedo.Arrow(corner, corner+np.array([0,ax_length,0]), c='g', s=0.2)
            az = vedo.Arrow(corner, corner+np.array([0,0,ax_length]), c='b', s=0.2)
            axes = vedo.Assembly(ax,ay,az)
            axes.apply_transform(main_obj.world_transform.mat_4x4().data.cpu().numpy())
            main_box = vedo.Assembly(main_box, axes)
            
            # Add box label
            label_str = main_obj.id[:6]
            corner_in_world = main_obj.world_transform(torch.tensor(corner, device=self.device)).data.cpu().numpy()
            main_label = vedo.Text3D(f"{main_obj.class_name}:{label_str}", pos=corner_in_world, s=1, c='black', literal=True)
            main_label.follow_camera(plt.camera) # Use plt's active camera

        if main_obj.model is not None and isinstance(getattr(main_obj.model, 'accel', None), OccupancyGridAS):
            accel = main_obj.model.accel
            main_occ_grid = accel.debug_vis(draw=False)
            main_occ_grid.apply_transform(main_obj.world_transform.mat_4x4().data.cpu().numpy(), reset=True)

        def handle_timer(event):
            nonlocal fi_counter, node_actors, node_labels, txt2d, lidar_pts, lidar_pts_downsample
            fi = fi_counter % len(self) # Replay again and agian ...
            self.frozen_at(fi)
            #---- Set camera pose
            for cam_id, cam_node in self.observer_groups_by_class_name['Camera'].items():
                cam_actors[cam_id].apply_transform(cam_node.world_transform.mat_4x4().data.cpu().numpy())
            
            #---- Plot current frame boxes
            if fi_counter > 0:
                plt.remove(*node_actors.values(), *node_labels)
                node_actors.clear()
                node_labels.clear()
            
            #---- Plot dynamic objects
            for oid, node in self.all_nodes_by_class_name.get('Vehicle', {}).items():
                if not node.valid:
                    continue
                
                # Add object boxes
                scale = node.scale.ratio().data.cpu().numpy()
                bound = np.stack([-scale/2.,scale/2.], axis=-1).reshape(6)
                box = vedo.Box(size=bound)
                box.apply_transform(node.world_transform.mat_4x4().data.cpu().numpy())
                
                # Add box label
                label_str = node.id[:6]
                label = vedo.Text3D(f"{node.class_name}:{label_str}", pos=box.points()[0], s=1, c='black', literal=True)
                label.follow_camera(plt.camera) # Use plt's active camera
                
                # Mark dynamic objects to be red boxes and labels; others will be black
                if 'dynamic_stats' in self.metas and oid in self.metas['dynamic_stats'][node.class_name]['is_dynamic']:
                    box.color('red6')
                    box.lw(4)
                    label.color('red6')
                else:
                    box.color('black')
                    box.lw(3)
                    label.color('black')
                box.wireframe() # Only plot edges of box 
                box.lighting('off')

                # Do not assemble label with box (will break follow_camera)
                node_labels.append(label)
                
                # Box local coornidate frame
                ax_length = 2.
                ax = vedo.Arrow([0,0,0], [ax_length,0,0], c='r')
                ay = vedo.Arrow([0,0,0], [0,ax_length,0], c='g')
                az = vedo.Arrow([0,0,0], [0,0,ax_length], c='b')
                axes = vedo.Assembly(ax,ay,az)
                # axes.lighting('off')
                axes.apply_transform(node.world_transform.mat_4x4().data.cpu().numpy())
                
                box = vedo.Assembly(box, axes)
                # box.lighting('off')
                node_actors[oid] = box

            #---- Plot lidar pointcloud
            if plot_lidar:
                lidar_pts_downsample = int(lidar_pts_downsample)
                if fi_counter > 0:
                    plt.remove(lidar_pts)
                lidar0 = self.all_nodes_by_class_name['RaysLidar'][scene_dataloader.lidar_id_list[0]]
                lidar_data = scene_dataloader.get_lidar_gts(self.id, lidar0.id, fi, filter_if_configured=False, device=self.device)
                pts = torch.addcmul(lidar_data['rays_o'][::lidar_pts_downsample], lidar_data['rays_d'][::lidar_pts_downsample], lidar_data['ranges'][::lidar_pts_downsample].unsqueeze(-1))
                pts_in_world = lidar0.world_transform.forward(pts).data.cpu().numpy()
                pts_c = (vedo.color_map(pts_in_world[:, 2], 'rainbow', vmin=-2., vmax=9.) * 255.).clip(0,255).astype(np.uint8)
                pts_c = np.concatenate([pts_c, np.full_like(pts_c[:,:1], 255)], axis=-1) # RGBA is ~50x faster
                lidar_pts = vedo.Points(pts_in_world, c=pts_c, r=2)
                lidar_pts.lighting('off')
            
            #---- Plot camera image
            if plot_image:
                if fi_counter > 0:
                    plt.remove(*pics)
                    pics.clear()
                bypass_downscale = 8.
                for ci, (cam_id, cam_node) in enumerate(self.observer_groups_by_class_name['Camera'].items()):
                    cam_node.intr.set_downscale(bypass_downscale)
                    intr = cam_node.intr.mat_3x3().data.cpu().numpy()
                    c2w = cam_node.world_transform.mat_4x4().data.cpu().numpy()
                    rgb = scene_dataloader.get_rgb(self.id, cam_id, fi, device=torch.device('cpu'), bypass_downscale=bypass_downscale)['rgb'].numpy()
                    rgb = (rgb * 255.).clip(0,255).astype(np.uint8)
                    H, W, _ = rgb.shape
                    hfov = np.rad2deg(np.arctan(W / 2. / intr[0, 0]) * 2.)
                    vfov = np.rad2deg(np.arctan(H / 2. / intr[1, 1]) * 2.)
                    half_w = camera_length * np.tan(np.deg2rad(hfov / 2.))
                    half_h = camera_length * np.tan(np.deg2rad(vfov / 2.))
                    scale = 2. * half_h / H
                    # Pinhole cams invert (reverse) image upside-down
                    rgb = np.flipud(rgb)
                    pic = vedo.Picture(rgb)
                    pic.alpha(0.8)
                    pic.scale(scale)
                    # For OpenCV pinhole camera coord
                    pic.pos(x=-half_w, y=-half_h, z=camera_length)
                    pic.apply_transform(c2w, reset=True)
                    pics.append(pic)
            
            txt2d.text(f"current frame = [{fi}/{len(self)}]", "top-right")
            txt2d.text(f"scene.id = {self.id}", "top-left")
            txt2d.text("..press q to quit", "bottom-right")
            plt.show(txt2d, main_box, main_label, main_occ_grid, 
                     lidar_pts, surf, *pics, *cam_actors.values(), *node_actors.values(), *node_labels, 
                     axes=0, resetcam=(fi_counter==0))
            fi_counter += 1

        timerevt = plt.add_callback('timer', handle_timer)
        is_playing = False
        timer_id = None
        button = None
        txt2d = vedo.CornerAnnotation()
        
        def button_fnc():
            nonlocal timer_id, button, is_playing
            if timer_id is not None:
                plt.timer_callback("destroy", timer_id)
            if not is_playing:
                timer_id = plt.timer_callback("create", dt=10 if plot_lidar else 100)
            button.switch()
            is_playing = not is_playing
        
        button = plt.add_button(button_fnc, states=["\u23F5 Play  ","\u23F8 Pause"], size=32)
        
        node_actors = {}
        node_labels = []
        for label in node_labels:
            label.follow_camera()
        plt.interactive()
        plt.close()

if __name__ == "__main__":
    pass