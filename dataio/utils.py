"""
@file   utils.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Common utilities for processing datasets
"""

import numpy as np

def clip_node_data(odict: dict, start, stop):
    """
    In-place clip odict['data'][...], odict['start_frame'], odict['n_frames']
    
    start/stop: [int] start/stop frame ind, or [float] start/stop timestamp
                both under start <= ... < stop convention, i.e. item \in [start,stop)
    """
    assert 'data' in odict, "Only works with node with data"
    if (start is None and stop is None) or ('global_frame_inds' not in odict['data'] and 'global_timestamps' not in odict['data']):
        t = None
    else:
        if 'global_frame_inds' in odict['data']:
            assert (start is None or isinstance(start, int)) and (stop is None or isinstance(stop, int)), f"Please use frame ind clipping."
            t = odict['data']['global_frame_inds']
        else:
            assert (start is None or isinstance(start, float)) and (stop is None or isinstance(stop, float)), f"Please use timestamp clipping."
            t = odict['data']['global_timestamps']
    
    if t is not None:
        if start is None:
            mask = t < stop
        elif stop is None:
            mask = t >= start
        else:
            mask = (t >= start) & (t < stop)
        
        if ~mask.any():
            # NOTE: DO NOT let any useful nodes to be not-used parent's children, 
            #       because all children and descendants of a not-used node will be ignored. 
            return
        
        # Modified keys: data{}, start_frame, n_frames
        for k, v in odict['data'].items():
            odict['data'][k] = v[mask]
        odict['n_frames'] = np.sum(mask)
        if 'global_frame_inds' in odict['data']:
            if start is not None:
                # TODO: This only supports `start`, `stop` being frame inds, not ts
                odict['data']['global_frame_inds'] -= start
            odict['start_frame'] = np.min(odict['data']['global_frame_inds'])
    return odict

def clip_node_segments(odict: dict, start, stop):
    """
    In-place clip odict['segments'] (seg['data'][...], seg['start_frame'], seg['n_frames'])
    """
    assert 'segments' in odict, "Only works with node with segments"
    old_segs = odict.pop('segments')
    new_segs = []
    for seg in old_segs:
        if (start is None and stop is None) or ('global_frame_inds' not in seg['data'] and 'global_timestamps' not in seg['data']):
            new_segs.append(seg)
            continue
        else:
            if 'global_frame_inds' in seg['data']:
                assert (start is None or isinstance(start, int)) and (stop is None or isinstance(stop, int)), f"Please use frame ind clipping."
                t = seg['data']['global_frame_inds']
            else:
                assert (start is None or isinstance(start, float)) and (stop is None or isinstance(stop, float)), f"Please use timestamp clipping."
                t = seg['data']['global_timestamps']
        
        if start is None:
            mask = t < stop
        elif stop is None:
            mask = t >= start
        else:
            mask = (t >= start) & (t < stop)
        
        if ~mask.any():
            continue
        
        # Modified keys: data{}, start_frame, n_frames
        for k, v in seg['data'].items():
            seg['data'][k] = v[mask]
        seg['n_frames'] = np.sum(mask)
        if 'global_frame_inds' in seg['data']:
            if start is not None:
                # TODO: This only supports `start`, `stop` being frame inds, not ts
                seg['data']['global_frame_inds'] -= start
            seg['start_frame'] = np.min(seg['data']['global_frame_inds'])
        new_segs.append(seg)
    
    odict['segments'] = new_segs
    return odict