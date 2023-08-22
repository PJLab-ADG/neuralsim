"""
@file   filter_demo.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Try to filter some nice waymo sequence for making demos.
"""

import os
import json
import argparse
import functools
import numpy as np
from glob import glob
from tqdm import tqdm

def main_function(args):
    from dataio.autonomous_driving.waymo.waymo_filereader import WaymoDataFileReader
    from dataio.autonomous_driving.waymo.waymo_dataset import parse_seq_file_list, WAYMO_CLASSES, file_to_scene_id
    import concurrent.futures as futures

    def process_single_sequence(seq_fpath: str, out_root: str):
        scene_id = file_to_scene_id(seq_fpath)
        os.makedirs(os.path.join(out_root, scene_id), exist_ok=True)
        
        dataset = WaymoDataFileReader(str(seq_fpath))
        frame0 = next(iter(dataset))
        stats = frame0.context.stats
        
        meta_fpath = os.path.join(out_root, scene_id, "metadata.json")
        dic = {
            'stats': {
                'name': frame0.context.name,
                'time_of_day': stats.time_of_day,
                'weather': stats.weather,
                'location': stats.location
            }
        }
        if os.path.exists(meta_fpath):
            with open(meta_fpath, 'r') as f:
                dic2 = json.load(f)
                dic.update(dic2)
        with open(meta_fpath, 'w') as f:
            json.dump(dic, f)
        print(f"=> File saved to {meta_fpath}")

    os.makedirs(args.out_root, exist_ok=True)
    seq_fpath_list = sorted(parse_seq_file_list(args.root, args.seq_list))
    num_workers = min(args.j, len(seq_fpath_list))

    if args.generate_meta:
        print("=> Generating metadata...")
        if num_workers > 1:
            with futures.ThreadPoolExecutor(num_workers) as executor:
                executor.map(functools.partial(process_single_sequence, out_root=args.out_root), seq_fpath_list)
        else:
            for seq_fpath in tqdm(seq_fpath_list):
                process_single_sequence(seq_fpath, out_root=args.out_root)

    seq_list = []
    for meta_fpath in tqdm(sorted(glob(os.path.join(args.out_root, "*", 'metadata.json'))), desc="=> Filtering..."):
        with open(meta_fpath, 'r') as f:
            meta = json.load(f)

            dyna_stats = meta['dynamic_stats']
            # n_dyna = stats['Vehicle']['n_dynamic']
            n_dyna = len(  list(set(dyna_stats['Vehicle']['by_speed']) & set(dyna_stats['Vehicle']['by_loc']))  )
            n_total = dyna_stats['Vehicle']['n_total']

            stats = meta['stats']

            egomotion = meta['egomotion']
            linear = np.array(egomotion['linear']).max()
            angular = np.array(egomotion['angular']).max()

            if linear < 8. and angular < 15.:
                seq_list.append([n_dyna, n_total, linear, angular, stats, file_to_scene_id(meta_fpath)])

    output_list = os.path.join(args.out_root, 'vehicle_debug.seq_list')
    seq_list = sorted(seq_list, key=(lambda it: it[0]), reverse=True)
    with open(output_list, 'w') as f:
        f.writelines('\n'.join([f"{n_dyna:04d}, {n_total:04d}, {linear}, {angular}, {stats['time_of_day']:10s}, {stats['weather']:16s}, {stats['location']:16s}, {scene_id}" for (n_dyna, n_total, linear, angular, stats, scene_id) in seq_list]))
    print(f"=> File saved to {output_list}")

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