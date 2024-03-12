"""
@file   filter_dynamic.py
@author Jianfei Guo, Shanghai AI Lab & Xiaohang Yang, Shanghai AI Lab
@brief  Get statistics on waymo's sequences' dynamic objects.
"""

import os
import json
import argparse
import functools
import numpy as np
from glob import glob
from tqdm import tqdm

def collect_loc_motion(dataset):
    """
    return path: {id: path_xyz}
    path_xyz: path_x: [] (3*length)
    """
    from .waymo_dataset import WAYMO_CLASSES
    categories = {}
    ego_path = np.empty((3, 0))
    for frame in dataset:
        v2w = np.array(frame.pose.transform, copy=True).reshape(4, 4)
        ego_path = np.concatenate((ego_path, v2w[:3, 3:]), axis=1)

        for label in frame.laser_labels:
            class_name = WAYMO_CLASSES[int(label.type)]
            if class_name not in categories:
                categories[class_name] = {}

            box = label.box
            # location at world coordinate
            # NOTE: sometime the z-value may be floating
            # loc = (v2w @ np.array([[box.center_x], [box.center_y], [box.center_z], [1.0]])).squeeze()[:3]
            loc = (v2w @ np.array([[box.center_x], [box.center_y], [box.center_z], [1.0]])).squeeze()[:2]
            if label.id not in categories[class_name]:
                categories[class_name][label.id] = dict(paths=[loc], motions=[0.])
            else:
                motion = np.linalg.norm(categories[class_name][label.id]['paths'][-1] - loc)
                categories[class_name][label.id]['paths'].append(loc)
                categories[class_name][label.id]['motions'].append(motion)

    return categories, ego_path

def collect_box_speed(dataset):
    """
    initial version: xiaohang
    filter the bbox based on speed
    input: one label
    output: True, moving; False, static
    """
    from .waymo_dataset import WAYMO_CLASSES
    categories = {}
    for frame in dataset:
        for label in frame.laser_labels:
            class_name = WAYMO_CLASSES[int(label.type)]
            if class_name not in categories:
                categories[class_name] = {}
            
            meta = label.metadata
            if label.id not in categories[class_name]:
                categories[class_name][label.id] = dict(motions=[])
            
            categories[class_name][label.id]['motions'].append(np.linalg.norm([meta.speed_x, meta.speed_y]))

    return categories

def count_all(dataset):
    """
    initial version: xiaohang
    filter the bbox based on speed
    input: one label
    output: True, moving; False, static
    """
    from .waymo_dataset import WAYMO_CLASSES
    categories = {}
    for frame in dataset:
        for label in frame.laser_labels:
            class_name = WAYMO_CLASSES[int(label.type)]
            if class_name not in categories:
                categories[class_name] = {}
            
            if label.id not in categories[class_name]:
                categories[class_name][label.id] = 0
            else:
                categories[class_name][label.id] += 1
    return categories

def stat_dynamic_objects(dataset, speed_eps=0.2, loc_eps=0.03):
    from .waymo_dataset import WAYMO_CLASSES
    stats = {cls_name:{'n_dynamic': 0, 'is_dynamic':[], 'by_speed': [], 'by_loc': []} for cls_name in WAYMO_CLASSES}
    #------------------------------------------------
    # Filter according to speed_x and speed_y
    speed_stats = collect_box_speed(dataset)
    for cls_name, cls_dict in speed_stats.items():
        by_speed = []
        for str_id, item_dict in cls_dict.items():
            if (np.array(item_dict['motions']).max() > speed_eps):
                by_speed.append(str_id)
        stats[cls_name]['by_speed'] = by_speed
    #------------------------------------------------
    # Filter according to center_x and center_y
    loc_motion_stats, _ = collect_loc_motion(dataset)
    for cls_name, cls_dict in loc_motion_stats.items():
        by_loc = []
        for str_id, item_dict in cls_dict.items():
            if (np.array(item_dict['motions']).max() > loc_eps):
                by_loc.append(str_id)
        stats[cls_name]['by_loc'] = by_loc
    #------------------------------------------------
    # Collect results from box_speed and loc_motion
    for cls_name, cls_dict in stats.items():
        li_dyna = list(set(cls_dict['by_speed']) | set(cls_dict['by_loc']))
        stats[cls_name]['is_dynamic'] = li_dyna
        stats[cls_name]['n_dynamic'] = len(li_dyna)

    return stats

def main_function(args):
    from dataio.autonomous_driving.waymo.waymo_filereader import WaymoDataFileReader
    from dataio.autonomous_driving.waymo.waymo_dataset import parse_seq_file_list, WAYMO_CLASSES, file_to_scene_id
    import concurrent.futures as futures

    def process_single_sequence(seq_fpath: str, out_root: str):
        scene_id = file_to_scene_id(seq_fpath)
        os.makedirs(os.path.join(out_root, scene_id), exist_ok=True)
        dataset = WaymoDataFileReader(str(seq_fpath))


        #------------------------------------------------
        #    dynamic object statistics
        #------------------------------------------------
        stats = stat_dynamic_objects(dataset)

        #------------------------------------------------
        #    Category stats
        #------------------------------------------------
        category_stats = count_all(dataset, WAYMO_CLASSES)
        for cls_name, cls_dict in stats.items():
            if cls_name in category_stats:
                stats[cls_name]['n_total'] = len(category_stats[cls_name])
                stats[cls_name]['freq'] = category_stats[cls_name]
            else:
                stats[cls_name]['n_total'] = 0
                stats[cls_name]['freq'] = {}

        meta_fpath = os.path.join(out_root, scene_id, "metadata.json")
        if os.path.exists(meta_fpath):
            with open(meta_fpath, 'r') as f:
                dic = json.load(f)
        else:
            dic = {}
        dic['dynamic_stats'] = stats
        with open(meta_fpath, 'w') as f:
            json.dump(dic, f)
        print(f"=> File saved to {meta_fpath}")

    os.makedirs(args.out_root, exist_ok=True)

    seq_fpath_list = sorted(parse_seq_file_list(args.root, args.seq_list))
    num_workers = min(args.j, len(seq_fpath_list))
    
    #-------------------------------------
    # filter all available sequences, generate "metas" in the processing directory
    if args.generate_meta:
        print("=> Generating metadata...")
        if num_workers > 1:
            with futures.ThreadPoolExecutor(num_workers) as executor:
                executor.map(functools.partial(process_single_sequence, out_root=args.out_root), seq_fpath_list)
        else:
            for seq_fpath in tqdm(seq_fpath_list):
                process_single_sequence(seq_fpath, out_root=args.out_root)
    
    print("=> Start filtering...")
    #-------------------------------------
    # read metas, generate a seq_list to process
    seq_list_morethan20 = []
    seq_list = []
    for meta_fpath in tqdm(sorted(glob(os.path.join(args.out_root, "*", 'metadata.json'))), desc="=> Filtering..."):
        with open(meta_fpath, 'r') as f:
            stats = json.load(f)['dynamic_stats']
            # n_dyna = stats['Vehicle']['n_dynamic']
            n_dyna = len(  list(set(stats['Vehicle']['by_speed']) & set(stats['Vehicle']['by_loc']))  )
            n_total = stats['Vehicle']['n_total']
            seq_list.append([n_dyna, n_total, file_to_scene_id(meta_fpath)])
            if n_dyna > 20:
                seq_list_morethan20.append(file_to_scene_id(meta_fpath))
    
    output_list = os.path.join(args.out_root, 'vehicle_debug.seq_list')
    seq_list = sorted(seq_list, key=(lambda it: it[0]), reverse=True)
    with open(output_list, 'w') as f:
        f.writelines('\n'.join([f"{it[0]:04d}, {it[1]:04d}, {it[2]}" for it in seq_list]))
    print(f"=> File saved to {output_list}")
    
    output_list = os.path.join(args.out_root, 'vehicle_dynamic>20.seq_list')
    with open(output_list, 'w') as f:
        f.writelines('\n'.join(seq_list_morethan20))
    print(f"=> File saved to {output_list}")
    print("=> Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=str, default="/media/guojianfei/DataBank0/dataset/waymo/training", required=True, 
        help="Root directory of raw .tfrecords")
    parser.add_argument(
        "--seq_list", type=str, default=None, 
        help="Optional specify subset of sequences. If None, will process all sequences contained in args.root")
    parser.add_argument(
        "--out_root", type=str, default="/data1/waymo/processed", required=True, 
        help="Output root directory")
    parser.add_argument('-j', type=int, default=4, help='max num workers')
    parser.add_argument('--generate_meta', action='store_true', help='whether to generate_meta meta info. ')
    args = parser.parse_args()
    main_function(args)