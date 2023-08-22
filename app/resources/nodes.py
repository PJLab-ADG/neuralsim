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

from nr3d_lib.utils import IDListedDict, get_shape
from nr3d_lib.models.attributes import ObjectWithAttr, AttrNested, AttrSegment, Valid, TransformMat4x4, Scale

class StandAloneSceneNode(ObjectWithAttr):
    """
    Single controllable scene node
    """
    def __init__(
        self, 
        unique_id: str, class_name='node', scene=..., *, 
        device=torch.device('cuda'), dtype=torch.float):
        from app.models.base import AssetModelMixin # To avoid circular import
        
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
        # - [from obj to world]: self.world_transform.forward(x * self.scale.ratio())
        # - [from world to obj]: self.world_transform.forward(x, inv=True) / self.scale.ratio()
        self.transform = TransformMat4x4(device=device) # [node-to-parent / node-in-parent]
        self.world_transform = TransformMat4x4(device=device) # [node-to-world / node-in-world]
        self.scale = Scale(device=device)
        
        self.valid = True
        
        #-------- Drawable model
        self.model: AssetModelMixin = None
        # self.model_AABB = None
        # # NOTE: Model's ray-intersectable Oriented Bounding Box in world-space 
        # #       p.s. Usally a bit larger then objects' real size when it comes to volume rendering
        # self.model_OBB = None
        # NOTE: Model's ray-intersectable Bounding Sphere in world-space
        self.model_BSPH: torch.Tensor = None
        # self.model_scale = None
    
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
            self.valid = self.parent.valid & self.valid
            
        #------ Update world-space OBB / Bounding Sphere
        if (self.model is not None) and (space:=getattr(self.model, 'space', None)):
            prefix = self.world_transform.prefix
            aabb = getattr(space, 'aabb', None)
            if aabb is not None:
                bounding_radius = (aabb * self.scale.ratio().unsqueeze(-2)).norm(dim=-1).max(dim=-1).values # Use maximum diagonal length as radius.
            else:
                bounding_radius = getattr(self.model.space, 'radius', 1.0)
        
            model_BSPH = torch.zeros([*prefix, 4], dtype=torch.float, device=self.device)
            model_BSPH[..., :3], model_BSPH[..., 3] = self.world_transform.translation(), bounding_radius
            self.model_BSPH = model_BSPH
        
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
        node.update() # Immediate update child after set `.parent`
    
    def __repr__(self) -> str:
        return f"{type(self).__name__}(" + \
            ",\n".join(
                [f"class_name={self.class_name}", f"id={self.id}", f"valid={self.valid}"] +
                [f"{n}={repr(a)}" for n,a in self.named_attrs()]
            ) + "\n)"

    def _frozen_at(self, i: Union[int, torch.LongTensor]):
        pass
    
    def frozen_at(self, i: Union[int, torch.LongTensor]):
        pass

    def unfrozen(self):
        pass

class SceneNode(StandAloneSceneNode):
    """
    Scene nodes with recorded sequence/log data
    """
    def __init__(
        self, 
        unique_id: str, class_name='node', scene=..., *, 
        device=torch.device('cuda'), dtype=torch.float):
        
        super().__init__(unique_id, class_name, scene, device=device, dtype=dtype)
                
        # Raw recorded list of attribute segments
        self.attr_segments: List[AttrSegment] = [] 
        
        # Matrix-formed attribute data (might contain invalid data at invalid frames)
        self.attr_array: AttrNested = None 
        
        # The length of `attr_array`
        self.n_total_frames: int = -1
        
        # The frame ind(s) frozen at 
        self.i: Union[int, torch.Tensor] = None 
        # Whether this node is valid in any of the frozen frames
        self.valid: bool = True
        # The detailed validness marker of each frozen frame
        self.valids: torch.Tensor = None
    
    def __len__(self):
        return self.n_total_frames
    
    @property
    def frozen_prefix(self):
        # The tensor shape of currently frozen frame_ind(s)
        return get_shape(self.i)
    
    def set_n_total_frames(self, length: int):
        """
        1st step of loading scene node sequence data
        """
        self.n_total_frames = length
    
    def load_attr_segment(self, n_frames, start_frame=0, **kwargs):
        """
        2nd step of loading scene node sequence data
        """
        stop_frame = start_frame + n_frames
        for existing_seg in self.attr_segments:
            assert not existing_seg.is_valid(slice(start_frame, stop_frame, 1)), \
                f"segment ({start_frame}:{stop_frame}) overlaps with existing segment ({existing_seg.start_frame}:{existing_seg.stop_frame})"
        seg = AttrSegment(**kwargs)
        seg.n_frames = n_frames
        seg.start_frame = start_frame
        seg.stop_frame = stop_frame
        self.attr_segments.append(seg)
    
    def finish_attr_segments(self):
        """
        3rd step (the last step) of loading scene node sequence data
        """
        # Make full matrix-formed validness marker
        valids = Valid(torch.zeros([self.n_total_frames], device=self.device, dtype=torch.bool))
        
        attr_dict = {}
        for seg in self.attr_segments:
            # Mark the frames this segment contains to be valid
            valids[seg.start_frame:seg.stop_frame] = True
            # Seperatly gather attr lists
            for k, v in seg.subattr.items():
                attr_dict.setdefault(k, []).append(v)
        
        # The frame ind(s) that are valid
        # NOTE: Currently, all attributes of one single node should share the same validness timestamps (self.valids)
        global_frame_inds = attr_dict.pop('global_frame_ind', None)
        if global_frame_inds is not None:
            global_frame_inds = torch.cat([v.tensor for v in global_frame_inds])
        else:
            global_frame_inds = Ellipsis

        for k, v in attr_dict.items():
            # Concat all segments
            val = type(v[0]).concat(v)
            # Make full matrix-formed data
            attr_dict[k] = val.new([self.n_total_frames])
            # Put concated valid segment data into frames that are valid
            attr_dict[k][global_frame_inds] = val

        # Make attr_array
        object.__setattr__(self, 'attr_array', AttrNested(allow_new_attr=True, **attr_dict, valids=valids, device=self.device))

    def _frozen_at(self, i: Union[int, torch.LongTensor]):
        """
        Frozen_at a certain slice of attr log data
        """
        self.i = i
        # NOTE: Only meaninful when it actually holds any attr array
        if self.n_total_frames > 0:
            {setattr(self, k, v[i]) for k, v in self.attr_array.subattr.items() if k != 'valids'}
            self.valids = valids = self.attr_array.subattr.valids[i].tensor
            self.valid = valids.any()
        else:
            # {setattr(self, k, v[i]) for k, v in self._attrs.items() if k != 'valids'}
            # {setattr(self, k, v.tile(self.frozen_prefix)) for k, v in self._attrs.items() if k != 'valids'}
            self.valids = torch.full(self.frozen_prefix, getattr(self, 'valid', True), dtype=torch.bool, device=self.device)
            self.valid = True
    
    def frozen_at(self, i: Union[int, torch.LongTensor]):
        """
        Frozen at a certain slice and update all its decendants 
        """
        self._frozen_at(i)
        self.update()

    def unfrozen(self):
        self.reset_attr()

    def reset_attr(self):
        """
        Un-frozen and reset the node and its descendants to default state.
        """
        self.i = None
        super().reset_attr()

    def _apply(self, fn):
        if self.n_total_frames > 0:
            with torch.no_grad():
                # NOTE: Already in-place op
                self.attr_array._apply(fn)
        super()._apply(fn)
        return self
    
    @functools.wraps(nn.Module.to)
    def to(self, *args, **kwargs):
        if self.n_total_frames > 0:
            with torch.no_grad():
                # NOTE: Already in-place op
                self.attr_array.to(*args, **kwargs)
        super().to(*args, **kwargs)
        return self
