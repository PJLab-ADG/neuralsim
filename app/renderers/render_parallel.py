"""
@file   render_parallel.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Built on top of DataParallel with special modification on scene replication and result collection
"""

__all__ = [
    'replicate_scene', 
    'replicate_scene_observer', 
    'gather_render_ret', 
    'render_parallel', 
    'render_parallel_with_replicas', 
]

from typing import Iterable, List, Tuple, Union
from itertools import islice, accumulate, chain

import torch
import torch.nn as nn
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import scatter, gather
from torch.nn.parallel.parallel_apply import parallel_apply

from app.resources import Scene
from app.resources.nodes import SceneNode
from app.models.asset_base import AssetAssignment

def replicate_scene(
    scene: Scene, asset_bank_replicas: List[nn.ModuleDict] = None, 
    devices: List[int] = ..., detach=False) -> List[Scene]:
    """
    Replicate a scene to multiple GPUs. 
    The scene can be either already frozen or unfrozen, \
        since both `node.frame_data` and `node._attrs` are also replicated.
    """
    #---- Replicate scene's asset_bank (only if not provided and actually needed) (82 ms)
    if asset_bank_replicas is None and scene.asset_bank is not None:
        asset_bank_replicas = replicate(scene.asset_bank, devices=devices, detach=detach)

    #---- Replicate scene's object structure (500 us)
    # NOTE: Shallow copy. `frame_data` and `_attrs` are replicated and re-distributed later.
    scene_replicas = [scene._replicate_for_parallel(device=device) for device in devices]
    all_nodes = list(scene.all_nodes.values())

    #---- Replicate `node.frame_data` (6~12 ms)
    node_frame_data_list = nn.ModuleList([o.frame_data for o in all_nodes])
    node_frame_data_list_copies = replicate(node_frame_data_list, devices=devices, detach=detach)

    #---- Replicate `node._attrs` (6 ms)
    node_attr_keys = [o._attrs.keys() for o in all_nodes]
    node_attr_values = [o._attrs.values() for o in all_nodes]
    lengths = [len(values) for values in node_attr_values]
    acc_lengths = list(accumulate(lengths))
    start_end_indices = list(zip([0] + acc_lengths[:-1], acc_lengths))
    node_attr_flat = nn.ModuleList(chain.from_iterable(node_attr_values))
    node_attr_flat_copies = replicate(node_attr_flat, devices=devices, detach=detach)
    
    #---- Put replicated assets, `node.frame_data`, `node._attrs` (800 us)
    for j, (array_copy, flat_attr_copy) in enumerate(zip(node_frame_data_list_copies, node_attr_flat_copies)):
        s = scene_replicas[j]
        #---- Put replicated `node.frame_data`
        for i, o in enumerate(s.all_nodes):
            object.__setattr__(o, 'frame_data', array_copy[i])
        #---- Load replicated asset_bank' assets (if needed)
        if asset_bank_replicas is not None:
            s.load_assets(asset_bank_replicas[j])
        # NOTE: Do not invoke scene.slice_at() / interp_at(); 
        #       We should directly use replicated `node_attr_flat_copies`)
        #---- Put replicated `node._attrs`
        for i, (o, (start, end)) in enumerate(zip(s.all_nodes, start_end_indices)):
            # if start != end: # NOTE: There's always some `_attrs` (at least `transform`, `world_transform`, `scale`)
            object.__setattr__(o, '_attrs', {k: v for k, v in zip(node_attr_keys[i], islice(flat_attr_copy, start, end))})

    return scene_replicas

def replicate_scene_observer(
    scene: Scene, observer: SceneNode = None, 
    asset_bank_replicas: List[nn.ModuleDict] = None, 
    devices: List[int] = ..., detach=False):
    scene_replicas = replicate_scene(scene, asset_bank_replicas, devices=devices, detach=detach) # 83 ms
    if observer is not None:
        observer_replicas = [s.observers[observer.id] for s in scene_replicas]
        if isinstance(observer.id, (Tuple, list)):
            observer_replicas = [o[0].make_bundle(o) for o in observer_replicas]
    else:
        observer_replicas = [None] * len(devices)
    return scene_replicas, observer_replicas

def gather_volume_buffer(volume_buffers: List[dict], ray_chunk_sizes: List[int], target_device) -> dict:
    buffer_types = [vb['type'] for vb in volume_buffers]
    first_non_empty = next((i for i, vbt in enumerate(buffer_types) if vbt != 'empty'), None)
    if first_non_empty is None: # All empty
        return {'type': 'empty'}
    assert not any(('pack_infos_collect' not in vb) and (vb['type'] == 'batched') for vb in volume_buffers), \
        "Can only gather volume_buffer processed by renderers with `pack_infos_collect`"
    
    #---- Merge buffer data
    all_keys = set(chain.from_iterable(vb.keys() for vb in volume_buffers))
    ret = {
        k: gather(
            [(vb[k] if vbt == 'packed' else vb[k].flatten(0,1)) for vb, vbt in zip(volume_buffers, buffer_types) if vbt != 'empty'], 
            target_device
        ) for k in all_keys \
        if k not in ['type', 'pack_infos_hit', 'pack_infos_collect', 'rays_inds_hit', 'rays_inds_collect', 'pidx_in_total', 'vw_in_total', 'rays_bidx_hit', 'num_per_hit']
    }
    # NOTE: `rays_bidx_hit` and `num_per_hit` could be differently defined in different workers; hence should not be gathered.
    
    #---- Merge pack_infos_hit / pack_infos_collect
    # For the total results, the keys are xxx_hit; for per-obj returns, the keys should be xxx_collect
    pkey, rkey = ('pack_infos_collect', 'rays_inds_collect') if 'pack_infos_collect' in volume_buffers[first_non_empty] else ('pack_infos_hit', 'rays_inds_hit')
    old_pack_infos = [vb[pkey].to(target_device) for vb, vbt in zip(volume_buffers, buffer_types) if vbt != 'empty'] # Skipped length
    total_num_acc = torch.cumsum(torch.stack([pk[-1].sum() for pk in old_pack_infos]), dim=0) # Skipped length
    pack_infos = []
    for j, pk in enumerate(old_pack_infos):
        new_pk = pk.clone()
        if j >= 1:
            new_pk[:, 0] += total_num_acc[j-1]
        pack_infos.append(new_pk)
    pack_infos = torch.cat(pack_infos, dim=0)
    ret[pkey] = pack_infos
    
    if 'vw_in_total' in all_keys:
        ret['vw_in_total'] = gather([vb['vw_in_total'] for vb, vbt in zip(volume_buffers, buffer_types) if vbt != 'empty'], target_device)
    
    if 'pidx_in_total' in all_keys:
        pidx_in_total = [vb['pidx_in_total'].to(target_device) for vb, vbt in zip(volume_buffers, buffer_types) if vbt != 'empty'] # Skipped length
        pidx_in_total = [p + (0 if j == 0 else total_num_acc[j-1]) for j, p in enumerate(pidx_in_total)] # Skipped length
        pidx_in_total = torch.cat(pidx_in_total)
        ret['pidx_in_total'] = pidx_in_total
    
    #---- Merge rays_inds_hit / rays_inds_collect
    rays_inds_offset = [0] + list(accumulate(ray_chunk_sizes))
    rays_inds = [(vb[rkey].to(target_device) + rays_inds_offset[j]) for j, (vb, vbt) in enumerate(zip(volume_buffers, buffer_types)) if vbt != 'empty']
    rays_inds = torch.cat(rays_inds, dim=0)
    ret[rkey] = rays_inds
    
    ret['type'] = 'packed'
    return ret

def gather_nested_dict_simple(batch: List[dict], target_device) -> dict:
    elem = batch[0]
    if isinstance(elem, dict):
        # Should traverse all elem dict in the batch in case some of them is empty dict
        all_elem_keys = list(set(chain.from_iterable(b.keys() for b in batch))) 
        # Check if k in d for not aligned keys
        return {k: gather_nested_dict_simple([d[k] for d in batch if k in d], target_device) for k in all_elem_keys}
    elif isinstance(elem, torch.Tensor):
        try:
            return gather(batch, target_device)
        except:
            return batch # Leave untouched (list)
    else:
        return batch # Leave untouched (list)

def gather_render_ret(rets: List[dict], ray_chunk_sizes: List[int], target_device):
    """
    Possible keys in rets[0]:
    - rendered, dict with easy-gatherable tensors
    - volume_buffer, dict, must be packed
      - buffer_type: str
      - rays_inds_hit: torch.Tensor
      - pack_infos_hit: torch.Tensor
      - t, opacity_alpha, rgb, nablas, vw, vw_normalized, ...: packed (or batched tensors)
    - raw_per_obj_model
      - volume_buffer, dict (packed, batched)
        - **regular volume buffer items
        - rays_inds_collect, converted volume buffer info (from packed/batched to packed)
        - pack_infos_collect, converted volume buffer info (from packed/batched to packed)
        - vw_in_total, packed tensor of the same shape with packed tensor data
        - pidx_in_total, packed tensor of the same shape with packed tensor data (but require index offset when gathering)
      - details, dict
      - rays_inds, torch.Tensor
      - ray_near, torch.Tensor
      - ray_far, torch.Tensor
      - num_rays, int
      - model_id: str
      - class_name: str
      - obj_id: str or list[str]
    """
    ret = {}
    if 'volume_buffer' in rets[0]:
        ret['volume_buffer'] = gather_volume_buffer([entry['volume_buffer'] for entry in rets], ray_chunk_sizes, target_device)
    
    if 'raw_per_obj_model' in rets[0]:
        all_keys = set(chain.from_iterable(entry['raw_per_obj_model'].keys() for entry in rets))
        # Per-model volume_buffer
        raw_per_obj_model = {
            k: {'volume_buffer': gather_volume_buffer(
                [entry['raw_per_obj_model'].get(k, {}).pop('volume_buffer', {'type': 'empty'}) for entry in rets], 
                ray_chunk_sizes, target_device)
            } for k in all_keys
        }
        
        for k in all_keys:
            # Per-model meta info
            valid_dicts = [entry['raw_per_obj_model'][k] for entry in rets if k in entry['raw_per_obj_model']]
            raw_per_obj_model[k]['class_name'] = valid_dicts[0]['class_name']
            raw_per_obj_model[k]['model_id'] = valid_dicts[0]['model_id']
            obj_ids = [vd['obj_id'] for vd in valid_dicts]
            if isinstance(obj_ids[0], list):
                obj_ids = chain.from_iterable(obj_ids)
            raw_per_obj_model[k]['obj_id'] = list(set(obj_ids))
            if 'num_rays' in valid_dicts[0].keys():
                raw_per_obj_model[k]['num_rays'] = sum([vd['num_rays'] for vd in valid_dicts])
            
            # Per-model details
            if 'details' in valid_dicts[0].keys():
                raw_per_obj_model[k]['details'] = gather_nested_dict_simple([vd['details'] for vd in valid_dicts], target_device)
        
        ret['raw_per_obj_model'] = raw_per_obj_model

    #---- Directly gatherable, without requiring extra efforts
    for key in ('ray_intersections', 'rendered', 'rendered_per_obj', 'class_seg_mask_buffer', 'ins_seg_mask_buffer', 'rendered_per_obj_in_scene'):
        if key in rets[0]:
            ret[key] = gather([entry[key] for entry in rets], target_device)
    
    #---- Miscs
    if 'rendered_class_names' in rets[0]:
        ret['rendered_class_names'] = list(set(chain.from_iterable(entry['rendered_class_names'] for entry in rets)))
    if 'rendered_model_ids' in rets[0]:
        ret['rendered_model_ids'] = list(set(chain.from_iterable(entry['rendered_model_ids'] for entry in rets)))
    if 'rendered_obj_ids' in rets[0]:
        ret['rendered_obj_ids'] = list(set(chain.from_iterable(entry['rendered_obj_ids'] for entry in rets)))
    return ret
    
def render_parallel(
    *ray_inputs, 
    renderer: nn.Module, scene: Scene, observer: SceneNode = None, 
    devices=..., output_device=None, detach=True, **kwargs):
    if len(devices) == 1:
        return renderer(*ray_inputs, scene=scene, observer=observer, **kwargs)
    output_device = devices[0] if output_device is None else output_device
    #------------------------------------------------------------------
    #------------------ Replicate scene (and its assetbank) and observer
    #------------------------------------------------------------------
    scene_replicas = replicate_scene(scene, devices=devices, detach=detach) # 83 ms
    if observer is not None:
        observer_replicas = [s.observers[observer.id] for s in scene_replicas]
        if isinstance(observer.id, (Tuple, list)):
            observer_replicas = [o[0].make_bundle(o) for o in observer_replicas]
    else:
        observer_replicas = [None] * len(devices)
    renderer_replicas = replicate(renderer, devices=devices, detach=detach) # NOTE: Do not assign asset_bank to be an attribute of renderer !!
    
    #------------------------------------------------------------------
    #------------------ Scatter inputs
    #------------------------------------------------------------------
    scattered_ray_inputs = tuple(zip(*(scatter(r, devices) for r in ray_inputs)))
    scattered_kwargs = tuple(scatter(kwargs, devices))
    for j, kw in enumerate(scattered_kwargs):
        kw.update(observer=observer_replicas[j], scene=scene_replicas[j])
    ray_chunk_sizes = [r[0].size(0) for r in scattered_ray_inputs]
    
    #------------------------------------------------------------------
    #------------------ Parallel Apply
    #------------------------------------------------------------------
    outputs = parallel_apply(renderer_replicas, scattered_ray_inputs, scattered_kwargs, devices)
    #---- DEBUG: non-parallel
    # outputs = [re(*ri, **kw) for re,ri,kw in zip(renderer_replicas, scattered_ray_inputs, scattered_kwargs)]

    #------------------------------------------------------------------
    #------------------ Gather outputs
    #------------------------------------------------------------------
    outputs = gather_render_ret(outputs, ray_chunk_sizes, output_device) # 6.4 ms   
    
    # NOTE: A temporary fix for conditional models: set_condition
    #       Needed by uniform samples and other model related functionalities
    if renderer.training:
        for _, obj_raw_ret in outputs['raw_per_obj_model'].items():
            if obj_raw_ret['volume_buffer']['type'] == 'empty':
                continue # Skip not rendered models
            model_id = obj_raw_ret['model_id']
            model = scene.asset_bank[model_id]
            if model.assigned_to in [AssetAssignment.MULTI_OBJ, AssetAssignment.MULTI_OBJ_ONE_SCENE]:
                batched_infos = {
                    'ins_id': [o.full_unique_id for o in scene.all_nodes[obj_raw_ret['obj_id']]], 
                    'scene_ts': scene.i if scene.i_is_timestamp else scene.frame_global_ts[scene.i]
                }
                model.set_condition(batched_infos) # 33 ms
    
    return outputs

def render_parallel_with_replicas(
    *ray_inputs, 
    renderer_replicas: List[nn.Module], scene_replicas: List[Scene], observer_replicas: List[SceneNode] = None, 
    devices=None, output_device=None, **kwargs):
    #------------------------------------------------------------------
    #------------------ Scatter inputs
    #------------------------------------------------------------------
    scattered_ray_inputs = tuple(zip(*(scatter(r, devices) for r in ray_inputs)))
    scattered_kwargs = tuple(scatter(kwargs, devices))
    for j, kw in enumerate(scattered_kwargs):
        kw.update(observer=observer_replicas[j], scene=scene_replicas[j])
    ray_chunk_sizes = [r[0].size(0) for r in scattered_ray_inputs]
    
    #------------------------------------------------------------------
    #------------------ Parallel Apply
    #------------------------------------------------------------------
    outputs = parallel_apply(renderer_replicas, scattered_ray_inputs, scattered_kwargs, devices)
    
    #------------------------------------------------------------------
    #------------------ Gather outputs
    #------------------------------------------------------------------
    outputs = gather_render_ret(outputs, ray_chunk_sizes, output_device)
    return outputs

class EvalParallelWrapper(object):
    def __init__(
        self, asset_bank: nn.Module, renderer: nn.Module, devices: list) -> None:
        self.devices = devices
        self.renderer_replicas = replicate(renderer, devices=devices, detach=True)
        self.asset_bank_replicas = replicate(asset_bank, devices=devices, detach=True)

    def replicate_scene_observer(self, scene, observer, detach=True):
        return replicate_scene_observer(
            scene, observer, 
            asset_bank_replicas=self.asset_bank_replicas, 
            devices=self.devices, detach=detach)

if __name__ == "__main__":
    def unit_test():
        from icecream import ic
        from torch.nn.parallel.scatter_gather import scatter_kwargs
        from torch.nn.parallel.replicate import replicate
        from torch.nn.parallel import comm
        from torch.nn.parallel._functions import Broadcast
        
        devices = [torch.device('cuda:0'), torch.device('cuda:1')]
        device0 = devices[0]
        
        t1 = torch.randn([7, 3], device=device0, dtype=torch.float)
        t2 = torch.randn([123,123,7], device=device0, dtype=torch.float)
        t3 = torch.randn([1234,], device=device0, dtype=torch.float)
        
        # NOTE: This will return a list, with length=len(devices)
        ret = comm.broadcast_coalesced([t1,t2,t3], devices)
        
        class DummyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.n = nn.Parameter(torch.randn([7,3], device=device0))
                self.is_dummy = False
            def extra_repr(self) -> str:
                return f"n={self.n.shape}, n.device={self.n.device}, dummy={self.is_dummy}"
        
        m = DummyModel()
        m.is_dummy = True
        # NOTE: This verifies that temporary attributes like `is_dummy` is also successfully replicated. 
        dummy = m._replicate_for_data_parallel()
        ic(hasattr(dummy, 'is_dummy'))
        # NOTE: This will return a list, with length=len(devices)
        ret = replicate(m, devices)

    def unit_test_attr_grad():
        """
        Unit test of replicate_scene, to see whether it correctly pass nodes' attributes' gradients
        """
        from icecream import ic
        from torch.utils.benchmark import Timer
        from nr3d_lib.models.attributes import TransformRT, Translation, AttrNested, Valid
        num_total_frames = 10
        
        devices = [torch.device('cuda:0'), torch.device('cuda:1')]
        device0 = devices[0]
        scene = Scene(device=device0)
        scene.n_frames = num_total_frames
        node = SceneNode('dummy')
        node.n_frames = node.n_global_frames = num_total_frames
        learnable_pose = TransformRT(
            rot=torch.eye(3, device=device0).tile(num_total_frames,1,1), 
            trans=Translation(torch.zeros([num_total_frames,3], device=device0), learnable=True))
        frame_data = AttrNested(
            allow_new_attr=True, 
            transform=learnable_pose
        )
        # NOTE: Must use object.__setattr__; 
        #       otherwise `frame_data` is considered to be a new attr to be registered since its type is AttrNested
        object.__setattr__(node, 'frame_data', frame_data)
        scene.add_node(node)
        
        # NOTE: 744 us each
        print(Timer(
            stmt="replicate_scene(scene, devices=devices, detach=False)", 
            globals={'replicate_scene':replicate_scene, 'scene':scene, 'devices':devices}
        ).blocked_autorange())
        
        #---- Test replicating of an unfrozen scene
        for n, p in learnable_pose.named_parameters():
            p.grad = None
        
        scene_replicas = replicate_scene(scene, devices=devices)
        losses = []
        for i, s in enumerate(scene_replicas):
            s.slice_at(4+i) # 4,5
            loss_i = (i+1) * s.all_nodes['dummy'].world_transform.translation().mean()
            losses.append(loss_i)
        losses = gather(losses, target_device=device0)
        losses.sum().backward()
        
        # [Checked] Should see different grad on [4],[5]
        ic(learnable_pose.subattr.trans.tensor.grad)
        
        
        #---- Test replicating of an frozen scene
        for n, p in learnable_pose.named_parameters():
            p.grad = None
        
        scene.slice_at(4)
        scene_replicas = replicate_scene(scene, devices=devices)
        losses = []
        for i, s in enumerate(scene_replicas):
            loss_i = s.all_nodes['dummy'].world_transform.translation()[i] * (i+1)
            losses.append(loss_i)
        losses = gather(losses, target_device=device0)
        losses.sum().backward()

        # [Checked] Should see different grad on [4, (0,1)]
        ic(learnable_pose.subattr.trans.tensor.grad)

        #---- Test additional grad on original scene
        for n, p in learnable_pose.named_parameters():
            p.grad = None
        
        scene.slice_at(4)
        scene_replicas = replicate_scene(scene, devices=devices)
        losses = [scene.all_nodes['dummy'].world_transform.translation()[2] * 10]
        for i, s in enumerate(scene_replicas):
            loss_i = s.all_nodes['dummy'].world_transform.translation()[i] * (i+1)
            losses.append(loss_i)
        losses = gather(losses, target_device=device0)
        losses.sum().backward()

        # [Checked] Should see different grad on [4, (0,1,2)]
        ic(learnable_pose.subattr.trans.tensor.grad)

    # unit_test()
    unit_test_attr_grad()
    