"""
@file   filter_dynamic.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Filter dynamic objects
"""

import argparse
import numpy as np
from typing import List

def collect_loc_motion(scenario: dict, loc_eps=0.03):
    """
    return path: {id: path_xyz}
    path_xyz: path_x: [] (3*length)
    """
    categ_stats = {}
    for oid, odict in scenario['objects'].items():
        class_name = odict['class_name']
        # Location at world coordinate
        loc_diff_norms = []
        for seg in odict['segments']:
            locations = seg['data']['transform'][:, :3, 3]
            loc_diff = np.diff(locations, axis=0)
            loc_diff_norm = np.linalg.norm(loc_diff, axis=-1)
            loc_diff_norms.append(loc_diff_norm)
        categ_stats.setdefault(class_name, {})[oid] = loc_diff_norms
    return categ_stats

def stat_dynamic_objects(scenario: dict, loc_eps=0.03, speed_eps=None, all_class_name: List[str] = None):
    if all_class_name is None:
        all_class_name = [odict['class_name'] for oid, odict in scenario['objects'].items()]
        all_class_name = list(set(all_class_name))
    stats = {cls_name:{'n_dynamic': 0, 'is_dynamic':[], 'by_speed': [], 'by_loc': []} for cls_name in all_class_name}
    #------------------------------------------------
    # Filter according to center_x and center_y
    loc_motion_stats = collect_loc_motion(scenario)
    for cls_name, cls_dict in loc_motion_stats.items():
        by_loc = []
        for oid, loc_diff_norms in cls_dict.items():
            loc_diff_norms = np.concatenate(loc_diff_norms)
            if len(loc_diff_norms) == 0:
                continue
            # print(oid, loc_diff_norms.max())
            if loc_diff_norms.max() > loc_eps:
                by_loc.append(oid)
        stats[cls_name]['by_loc'] = by_loc
    #------------------------------------------------
    # Filter according to speed_x and speed_y
    pass
    #------------------------------------------------
    # Gather results from box_speed and loc_motion
    for cls_name, cls_dict in stats.items():
        # li_dyna = list(set(cls_dict['by_speed']) | set(cls_dict['by_loc']))
        li_dyna = cls_dict['by_loc']
        cls_dict['is_dynamic'] = li_dyna
        cls_dict['n_dynamic'] = len(li_dyna)
    
    return stats