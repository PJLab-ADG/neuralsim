"""
@file   nodes.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Basic scene node structure
"""

__all__ = [
    'StandAloneSceneNode', 
    'SceneNode'
]

import functools
from typing import List, Union

import torch
import torch.nn as nn

from nr3d_lib.maths import inverse_transform_matrix
from nr3d_lib.utils import IDListedDict, check_to_torch, get_shape
from nr3d_lib.models.attributes import ObjectWithAttr, AttrNested, Valid, TransformMat4x4, Scale, Scalar

class StandAloneSceneNode(ObjectWithAttr):
    """
    Single controllable scene node
    """
    def __init__(
        self, 
        unique_id: str, class_name='node', scene=..., *, 
        dtype=torch.float, device=None):
        from app.models.asset_base import AssetModelMixin # To avoid circular import
        
        super().__init__(device=device, dtype=dtype)
        
        self.id = unique_id
        self.class_name = class_name

        # self.scene = ...
        # self.scene_id = ...

        #-------- Basic scene graph struture
        self.parent: SceneNode = None
        self.children: IDListedDict[SceneNode] = IDListedDict()
        
        self.scene = scene
        
        #-------- Local transform and world transform
        # Usage example:
        # - [from obj to world]: self.world_transform.forward(x * self.scale.value())
        # - [from world to obj]: self.world_transform.forward(x, inv=True) / self.scale.value()
        self.transform = TransformMat4x4(device=device) # [node-to-parent / node-in-parent]
        self.world_transform = TransformMat4x4(device=device) # [node-to-world / node-in-world]
        self.scale = Scale(device=device)
        
        self.i = None
        self.i_valid = True
        
        #-------- Drawable model
        self.model: AssetModelMixin = None
        # self.model_AABB = None
        # # NOTE: Model's ray-intersectable Oriented Bounding Box in world-space 
        # #       p.s. Usally a bit larger then objects' real size when it comes to volume rendering
        # self.model_OBB = None
        # NOTE: Model's ray-intersectable Bounding Sphere in world-space
        self.model_bounding_sphere: torch.Tensor = None # Model's bounding sphere

    @property
    def full_unique_id(self):
        return f"{self.scene.id}#{self.id}"
    
    # @profile
    def update(self):
        #------ Update node-to-world transformation
        if self.parent is None:
            self.world_transform = self.transform
        else:
            if self.parent.world_transform is not None:
                # NOTE: `world_tranform` is passed on from parent to child
                
                # self.world_transform = TransformMat4x4(torch.einsum("...ij,...jk->...ik", self.parent.world_transform.mat_4x4(), self.transform.mat_4x4()))
                # NOTE: Below fails if type of self.world_transform is not TransformMat4x4
                # self.world_transform.tensor = torch.einsum("...ij,...jk->...ik", self.parent.world_transform.mat_4x4(), self.transform.mat_4x4())
                # self.world_transform.tensor = self.parent.world_transform.mat_4x4() @ self.transform.mat_4x4()
                # self.world_transform.tensor = (self.parent.world_transform.mat_4x4().unsqueeze(-1) * self.transform.mat_4x4().unsqueeze(-3)).sum(-2)
                self.world_transform = TransformMat4x4((self.parent.world_transform.mat_4x4().unsqueeze(-1) * self.transform.mat_4x4().unsqueeze(-3)).sum(-2))
            else:
                self.world_transform = self.transform

            # NOTE: `valid` is passed on from parent to child
            self.i_valid = self.parent.i_valid & self.i_valid
            
        #------ Update Bounding Volume in world coordinates
        bounding_volume = None # [..., 6=3(center)+3(radius3d)]
        if self.model is not None:
            if hasattr(self.model, 'get_bounding_volume'):
                bounding_volume = self.model.get_bounding_volume()
            elif getattr(self.model, 'space', None) is not None:
                bounding_volume = self.model.space.get_bounding_volume()
        
        if bounding_volume is not None:
            center = self.world_transform.forward(bounding_volume[..., :3])
            # NOTE: scalar radius remains unchanged after world_transform
            radius = (self.scale.vec_3() * bounding_volume[..., 3:]).norm(dim=-1, keepdim=True)
            self.model_bounding_sphere = torch.cat([center, radius], dim=-1)
        
        #------ Recursively update childrens
        for child in self.children:
            child.update()

    def update_children(self):
        #------ Recursively update childrens
        for child in self.children:
            child.update()

    def reset_attr(self):
        # Reset attributes of current node and its desendants
        ObjectWithAttr._reset(self)
        if hasattr(self.model, 'reset'):
            self.model.reset()
        for child in self.children:
            child.reset_attr()

    def add_child(self, node):
        assert node.id not in self.children
        node.parent = self
        self.children[node.id] = node
        node.update() # Immediately update child after set `.parent`
    
    def __repr__(self) -> str:
        return f"{type(self).__name__}(" + \
            ",\n".join(
                [f"class_name={self.class_name}", f"id={self.id}", f"valid={self.i_valid}"] +
                [f"{n}={repr(a)}" for n,a in self.named_attrs()]
            ) + "\n)"

    def _slice_at(self, i: Union[int, torch.LongTensor]):
        pass
    
    def slice_at(self, i: Union[int, torch.LongTensor]):
        pass

    def _interp_at(self, ts: torch.Tensor):
        pass
    
    def interp_at(self, ts: torch.Tensor):
        pass

    def unfrozen(self):
        pass

    def _replicate_for_parallel(self, device) -> 'StandAloneSceneNode':
        replica = self.__new__(type(self))
        replica.__dict__ = self.__dict__.copy()
        replica.device = device
        return replica

    def pose_in_world_to_in_node(self, pose_in_world: torch.Tensor):
        """
        pose_in_world @ vec = vec_in_world
        node_to_world @ pose_in_node @ vec = vec_in_world
        -> pose_in_world = node_to_world @ pose_in_node
        -> pose_in_node = node_to_world.inv() @ pose_in_world
        """
        node_to_world = self.world_transform.mat_4x4()
        # ij,jk->ik NOTE: Don't use einsum or bmm, both of which introduce noticeable error
        pose_in_node = (inverse_transform_matrix(node_to_world).unsqueeze(-1) * pose_in_world.unsqueeze(-3)).sum(-2)
        return pose_in_node

    def pose_in_node_to_in_world(self, pose_in_node: torch.Tensor):
        node_to_world = self.world_transform.mat_4x4()
        pose_in_world = (node_to_world.unsqueeze(-1) * pose_in_node.unsqueeze(-3)).sum(-2)
        return pose_in_world

class SceneNode(StandAloneSceneNode):
    """
    Scene nodes with recorded sequence/log data
    """
    def __init__(
        self, 
        unique_id: str, class_name='node', scene=..., *, 
        dtype=torch.float, device=None):
        
        super().__init__(unique_id, class_name, scene, device=device, dtype=dtype)
        
        #-----------------------------------------------
        #---- Permanent & fixed storage, loaded from records / log
        #-----------------------------------------------
        # The full length of the overall scene.
        self.n_global_frames: int = -1
        
        # The full length of `frame_data` (the local data of current node.)
        #   - If `scene.use_ts_interp` is True: (using timestamp interpolation mode): 
        #       `n_frames` might be different across different nodes;
        #   - If `scene.use_ts_interp` is False: (using frame indexing mode), 
        #       `n_frames` are the same across all nodes (==len(scene)==self.n_global_frames).
        self.n_frames: int = -1
        
        # NOTE: Matrix-formed node data (might contain invalid data at invalid frames)
        # [n_frames, ...], attribute data of each frame
        self.frame_data: AttrNested = None
        
        # NOTE: See below `@property`s
        # self.frame_global_fi: torch.LongTensor = None
        # self.frame_global_ts: torch.Tensor = None
        # self.frame_valid_flags: torch.BoolTensor = None
        # self.valid_fi: torch.LongTensor = None
        # self.valid_global_fi: torch.LongTensor = None
        # self.valid_global_ts: torch.Tensor = None
        
        # NOTE: When training with timestamp interpolation, \
        #       to allow for validity checks that are robust to minor timestamp differences
        #       For now, this value is calculated load_from_odict() / fill_default_data()
        self.valid_ts_extend: float = None
        
        #-----------------------------------------------
        #---- Temporary data, change with every slice_at() or interp_at() call
        #-----------------------------------------------
        # The temporary frame indices frozen at / timestamps interpolated at
        self.i: Union[int, torch.Tensor] = None 
        # Whether the current `self.i` represents timestamps. If False, `i` represents the frame indices.
        self.i_is_timestamp: bool = None
        
        # NOTE: See below `@property`s
        # self.i_is_single = None 
        # self.i_prefix = None
        
        # A temporary, single boolean value indicating whether this node is valid in any of the frozen time (i.e. `self.i`)
        self.i_valid: bool = True
        # A temporary, vector of validity flags, marking the validity of the current node across the frozen time (i.e. `self.i`)
        self.i_valid_flags: torch.BoolTensor = None
        
        # NOTE: Below is all temporary storage of current slided fi / interpolated ts.
        #       The full node data is stored in `self.frame_data`
        # self.transform = TransformMat4x4(device=device) # [node-to-parent / node-in-parent]
        # self.world_transform = TransformMat4x4(device=device) # [node-to-world / node-in-world]
        # self.scale = Scale(device=device)

    @property
    def frame_valid_flags(self) -> torch.BoolTensor:
        """[n_frames], validity markers of ecah frame"""
        return self.frame_data.subattr.valid_flags.value()
    
    @property
    def frame_global_fi(self) -> torch.LongTensor:
        """[n_frames], *global* frame indices of each frame"""
        return self.frame_data.subattr.global_fi.value()
    
    @property
    def frame_global_ts(self) -> torch.Tensor:
        """[n_frames], *global* timestamps of each frame, in range [0,n_global_frames-1]"""
        return self.frame_data.subattr.global_ts.value()

    @property
    def valid_fi(self) -> torch.LongTensor:
        """[n_valid_frames], *local* frame indices of each *valid* frame, in range [0,n_frames-1]"""
        return self.frame_valid_flags.nonzero().long()[..., 0]
    
    @property
    def valid_global_fi(self) -> torch.LongTensor:
        """[n_valid_frames], *global* frame indices of each *valid* frame, in range [0,n_global_frames-1]"""
        return self.frame_global_fi[self.valid_fi]
    
    @property
    def valid_global_ts(self) -> torch.Tensor:
        """[n_valid_frames], *global* timestamp of each *valid* frame"""
        return self.frame_global_ts[self.valid_fi]

    @property
    def i_prefix(self):
        """The tensor shape of currently sliced frame indices / intperpolated timestamps"""
        return get_shape(self.i)

    @property
    def i_is_single(self):
        """Whether sliced at single frame index / interpolated at a single timestamp"""
        return len(self.i_prefix) == 0

    def __len__(self):
        return self.n_frames

    def _parse_attr_data(self, odict: dict, data: dict, device=None) -> dict:
        """ Parse one segment of data in `scenario.pt` of this node into Attr.
        This function can be overloaded to load custom-defined attributes.
        For now, this contains the basic `transform` and `scale` data loading for all nodes \
            to build a valid scene_graph.

        Args:
            odict (dict): The full node data dict. 
            data (dict): This given segment of data.
            device (Any, optional): The target device to load the data. Defaults to None.

        Returns:
            dict: Parsed Attr from the given data.
        """
        device = device or self.device
        parsed_attr_data = {}
        assert 'world_transform' not in data, "Please do not store `world_transform` directly in node's data."\
            "You should only store `transform`, the node's transform to its `parent`"\
            " (while `parent` can be any node, for example, the root world node. )"
        
        if 'transform' in data:
            parsed_attr_data.update(transform=TransformMat4x4(data['transform'], dtype=torch.float, device=device))
        if 'scale' in data:
            parsed_attr_data.update(scale=Scale(data['scale'], dtype=torch.float, device=device))
        return parsed_attr_data
    
    def load_from_odict(self, n_global_frames: int, odict: dict, device=None, default_global_ts: torch.Tensor = None):
        """ Load from the full node data dict of this node in `scenario.pt`, \
            and save them into `self.frame_data`.
        This function will invoke `_parse_attr_data` to do the actual data parsing, \
            in which will convert data segment(s) in `scenratio.pt` into Attr.
        It supports either a full length data, or multiple segments of data.

        Args:
            n_global_frames (int): Number of total frames of the whole scene.
            odict (dict): The full node data dict of this node in `scenario.pt`.
                For full-length data, an example odict:
                {
                    'id': str, 
                    'class_name': str, 
                    'n_frames': int, Number of valid frames of this node.
                    'data': {
                        'transform': np.ndarray of shape [n_frames, ...], 
                        'scale': np.ndarray of shape [n_frames, ...], 
                        ...
                    }
                }
                For multi-segments data, an example odict:
                {
                    'id': str, 
                    'class_name': str, 
                    'start_frame: int, 
                    'n_frames': int, Number of valid frames of this segment.
                    'segments': [ # A list of dicts
                        {
                            'transform': np.ndarray of shape [n_frames, ...], 
                            'scale': np.ndarray of shape [n_frames, ...], 
                            ...
                        }, 
                        {...}
                    ]
                }
            device (Any, optional): The target device to load the data. Defaults to None.
        """
        
        self.n_global_frames = n_global_frames
        
        device = device or self.device
        
        #-----------------------------------
        #---- Load and parse data from `scenario.pt`
        frame_data_dict = None
        if 'data' in odict:
            #-----------------------------------
            #---- Full length data (Valid across all frames)
            n_frames = odict['n_frames']
            
            # [*] Invoke `_parse_attr_data` to actually do data parsing (from `scenario.pt` to Attr)
            frame_data_dict = self._parse_attr_data(odict, odict['data'], device=device)
            
            # Every frame is valid since it's full-data
            frame_data_dict['valid_flags'] = Valid(torch.ones([n_frames], device=device, dtype=torch.bool))
            if (frame_global_fi := odict['data'].get('global_frame_inds', None)) is not None:
                frame_global_fi = check_to_torch(frame_global_fi, device=device, dtype=torch.long)
                frame_data_dict['global_fi'] = Scalar(frame_global_fi)
            if (frame_global_ts := odict['data'].get('global_timestamps', default_global_ts)) is not None:
                frame_global_ts = check_to_torch(frame_global_ts, device=device, dtype=torch.float)
                frame_data_dict['global_ts'] = Scalar(frame_global_ts)
                # Currently, a time span of three frame periods is used, which should be sufficient.
                self.valid_ts_extend = 3 * frame_global_ts.diff(dim=-1).mean().item()

        elif 'segments' in odict:
            #-----------------------------------
            #---- Multi-segment data (Sparse; partially valid and partially invalid across different frames)
            n_frames = odict.get('n_full_frames', n_global_frames)
            frame_data_dict = {}
            
            frame_valid_flags = torch.zeros([n_frames], device=device, dtype=torch.bool)
            frame_global_fi = torch.full([n_frames], -1, device=device, dtype=torch.long)
            frame_global_ts = torch.full([n_frames], -1, device=device, dtype=torch.float)
            
            valid_fi = []
            for seg in odict['segments']:
                local_fi = torch.arange(seg['start_frame'], seg['start_frame']+seg['n_frames'], dtype=torch.long, device=device)
                # Check whether this segment overlaps with existing segments
                if frame_valid_flags[local_fi].any():
                    raise RuntimeError(f"Invalid segment data as one of the segment overlaps with the other.")
                
                # Mark the frames this segment contains to be valid
                frame_valid_flags[local_fi] = True
                if 'global_frame_inds' in seg['data']:
                    frame_global_fi[local_fi] = seg_global_fi = check_to_torch(seg['data']['global_frame_inds'], dtype=torch.long, device=device)
                if 'global_timestamps' in seg['data']:
                    frame_global_ts[local_fi] = seg_global_ts = check_to_torch(seg['data']['global_timestamps'], dtype=torch.float, device=device)
                    if (self.valid_ts_extend is None) and len(seg_global_ts) > 1:
                        # Currently, a time span of three frame periods is used, which should be sufficient.
                        self.valid_ts_extend = 3 * seg_global_ts.diff(dim=-1).mean().item()
                
                # [*] Invoke `_parse_attr_data` to actually do data parsing (from `scenario.pt` to Attr)
                _parsed = self._parse_attr_data(odict, seg['data'], device=device)
                # Seperatly gather attr lists
                for k, v in _parsed.items():
                    frame_data_dict.setdefault(k, []).append(v)

                valid_fi.append(local_fi)
            
            valid_fi = torch.cat(valid_fi)
            
            # NOTE: Currently, single node have the same validness definition across different attrs
            for k, v in frame_data_dict.items():
                # Concat all segments
                val = type(v[0]).concat(v)
                # Make full-length array data
                frame_data_dict[k] = val.new([n_frames])
                # Put concated valid segment data into frames that are valid
                #   (i.e. Similar to pytorch's sparse tensor)
                frame_data_dict[k][valid_fi] = val
            
            frame_data_dict['valid_flags'] = Valid(frame_valid_flags)
            frame_data_dict['global_fi'] = Scalar(frame_global_fi)
            frame_data_dict['global_ts'] = Scalar(frame_global_ts)

        #-----------------------------------
        #---- Save data into self.frame_data
        if frame_data_dict is not None:
            self.n_frames = n_frames
            # For registered attrs but not set data, fill with full-length default
            for k, v in self.named_attrs():
                if k not in frame_data_dict:
                    frame_data_dict[k] = v.new((self.n_frames,))
            # Make and store frame_data
            frame_data = AttrNested(allow_new_attr=True, **frame_data_dict, device=self.device)
            object.__setattr__(self, 'frame_data', frame_data)
        elif self.frame_data is None:
            # For nodes with no data, fill with full-length default.
            # e.g. For Distant or Sky nodes, their odict usually holds no 'data' or 'segments'
            self.fill_default_data(n_global_frames, default_global_ts=default_global_ts)

    def fill_default_data(
        self, 
        n_global_frames: int, 
        default_global_ts: torch.Tensor = None, 
        device=None):
        """ 
        If no actual data is assigned to this node, it's still necessary to \
            fill the node with default data of the same full length \
            to ensure aligned behavior of timestamp freezing / interpolation \
            across multiple nodes.

        Args:
            n_global_frames (int): _description_
            default_global_ts (torch.Tensor, optional): _description_. Defaults to None.
            device (_type_, optional): _description_. Defaults to None.
        """
        self.n_frames = self.n_global_frames = n_global_frames
        # For nodes with no data, expand their default values in self._attrs (registered attrs in __init__()) to full length
        frame_data_dict = {k: v.new((self.n_frames,)) for k,v in self.named_attrs()}
        frame_data_dict['valid_flags'] = Valid(torch.ones([self.n_frames], device=device, dtype=torch.bool))
        frame_data_dict['global_fi'] = Scalar(torch.arange(self.n_frames, device=device, dtype=torch.long)) # Since n_frames is n_global_frames
        if default_global_ts is not None:
            frame_data_dict['global_ts'] = Scalar(default_global_ts.clone())
            # Currently, a time span of three frame periods is used, which should be sufficient.
            self.valid_ts_extend = 3 * default_global_ts.diff(dim=-1).mean().item()
        # Make and store frame_data
        frame_data = AttrNested(allow_new_attr=True, **frame_data_dict, device=self.device)
        object.__setattr__(self, 'frame_data', frame_data) 

    def _slice_at(self, i: Union[int, torch.LongTensor]):
        """
        Frozen at a certain slice of attr data for only this node.
        """
        self.i = i
        self.i_is_timestamp = False
        if self.n_frames > 0:
            self.i_valid_flags = self.frame_valid_flags[i]
            self.i_valid = self.i_valid_flags.any()
            {
                setattr(self, k, v[i]) 
                for k, v in self.frame_data.subattr.items() 
                if k not in ['valid_flags', 'global_ts', 'global_fi']
            }
        else:
            raise RuntimeError(f"{self.__class__.__name__}({self.id}): "\
                "For nodes without data, please use fill_default_data().\n"\
                "This is an experimental modification. If any issues arise, please reach out to Jianfei Guo.")
    
    def slice_at(self, i: Union[int, torch.LongTensor]):
        """
        Frozen at a certain slice of attr data for this node, and update all its decendants 
        """
        self._slice_at(i)
        self.update()

    def _interp_at(self, ts: torch.Tensor):
        """
        Interpolate at given timestamp `ts` for only this node.
        
        Similar to `_slice_at`, but performs interpolation at given novel timestamps `ts` \
            among valid keyframe timestamps (`self.frame_global_ts`). 
        Will interpolate all attrs in the (automatically) registered `named_attrs()`. 

        Args:
            ts (torch.Tensor): Timestamps at which to perform interpolation.
        """
        self.i = ts
        self.i_is_timestamp = True
        if self.n_frames > 0:
            valid_fi, valid_global_ts = self.valid_fi, self.valid_global_ts
            # NOTE: Experimental: assume all middle ts to be valid.
            self.i_valid_flags = (ts >= (valid_global_ts[0] - self.valid_ts_extend)) & (ts <= (valid_global_ts[-1] + self.valid_ts_extend))
            self.i_valid = self.i_valid_flags.any()  
            # NOTE: `valid_global_ts` could carry gradients (see learnable_params:refine_sensor_ts), \
            #       and this is where the gradient flows of timestamps are respected \
            #       i.e. Allows for self-calibration of differentiable timestamps
            { 
                setattr(self, k, v[valid_fi].interp1d(valid_global_ts, ts)) 
                for k, v in self.frame_data.subattr.items() 
                if k not in ['valid_flags', 'global_ts', 'global_fi']
            }
        else:
            raise RuntimeError(f"{self.__class__.__name__}({self.id}): "\
                "For nodes without data, please use fill_default_data().\n"\
                "This is an experimental modification. If any issues arise, please reach out to Jianfei Guo.")

    def interp_at(self, ts: torch.Tensor):
        """
        Interpolate at given timestamp `ts` for this note, and update all its decendants
        """
        ts = check_to_torch(ts, dtype=torch.float, device=self.device)
        self._interp_at(ts)
        self.update()

    def unfrozen(self):
        self.reset_attr()

    def reset_attr(self):
        """
        Un-frozen and reset the node and its descendants to default state.
        """
        self.i = None
        self.i_valid = True
        self.i_is_timestamp = None
        self.i_valid_flags = None
        super().reset_attr()

    def _apply(self, fn):
        if self.n_frames > 0:
            with torch.no_grad():
                # NOTE: Already in-place op
                self.frame_data._apply(fn)
        super()._apply(fn)
        return self
    
    @functools.wraps(nn.Module.to)
    def to(self, *args, **kwargs):
        if self.n_frames > 0:
            with torch.no_grad():
                # NOTE: Already in-place op
                self.frame_data.to(*args, **kwargs)
        super().to(*args, **kwargs)
        return self

    def _replicate_for_parallel(self, device) -> 'SceneNode':
        replica = self.__new__(type(self))
        replica.__dict__ = self.__dict__.copy() # NOTE: Must be shallow copy
        
        replica.device = device
        # NOTE: Below is either directly or in-directly stored in `self.frame_data`, 
        #       which will be replicated later and not now.
        # if self.frame_global_fi is not None:
        #     replica.frame_global_fi = self.frame_global_fi.to(device, copy=True)
        # if self.frame_global_ts is not None:
        #     replica.frame_global_ts = self.frame_global_ts.to(device, copy=True)
        # if self.frame_valid_flags is not None:
        #     replica.frame_valid_flags = self.frame_valid_flags.to(device, copy=True)
        # if self.valid_fi is not None:
        #     replica.valid_fi = self.valid_fi.to(device, copy=True)
        # if self.valid_global_fi is not None:
        #     replica.valid_global_fi = self.valid_global_fi.to(device, copy=True)
        # if self.valid_global_ts is not None:
        #     replica.valid_global_ts = self.valid_global_ts.to(device, copy=True)
        if self.i is not None and isinstance(self.i, torch.Tensor):
            replica.i = self.i.to(device, copy=True)
        if self.i_valid_flags is not None:
            replica.i_valid_flags = self.i_valid_flags.to(device, copy=True)
        
        # NOTE: Similar to pytorch's replicate(), \
        #       `_attrs` and `frame_data` is replicated when replicate_scene(), \
        #       done by _broadcast_coalesced_reshape(),  
        #       since they might require gradients.
        replica._attrs = {}
        replica.frame_data = None
        
        return replica