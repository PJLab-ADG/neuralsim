"""
Copy processed data of selected waymo sequences to a target parent directory (for data transfering between devices)
"""

import os
from tqdm import tqdm
from glob import glob
from pathlib import Path

import shutil
from os import PathLike
from typing import List, Union

def copy_files(
    out_root_dir: str, root_dir: Union[str, PathLike], select_scene_ids: List[str], 
    fields=['masks', 'normals', 'depths']):
    src_root = Path(root_dir).expanduser().resolve(strict=True)
    out_root = Path(out_root_dir).expanduser().resolve(strict=True)
    for field in ['lidars', 'images', 'masks', 'normals', 'depths']:
        if field in fields:
            for seq_i, seq in enumerate(tqdm(select_scene_ids, f'copying {field}...')):
                shutil.copytree(src_root.joinpath(field, seq), out_root.joinpath(field, seq))
    if 'scenarios' in fields:
        out_scenario_root = out_root.joinpath('scenarios')
        os.makedirs(out_scenario_root, exist_ok=True)
        for seq_i, seq in enumerate(tqdm(select_scene_ids, 'copying scenarios...')):
            sce_fname = f"{seq}.pt"
            sce_fpath = src_root.joinpath('scenarios', sce_fname)
            shutil.copyfile(sce_fpath, out_scenario_root.joinpath(sce_fname))

def copy_raw(
    out_root_dir: str, root_dir: str, select_scene_ids: List[str], 
):
    for seq_i, seq in enumerate(tqdm(select_scene_ids, f'copying raw...')):
        shutil.copyfile(os.path.join(root_dir, f"{seq}.tfrecord"), os.path.join(out_root_dir, f"{seq}.tfrecord"))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_list", type=str, help='specify --seq_list if you want to limit the list of seqs', default="dataio/autonomous_driving/waymo/waymo_static_32.lst")
    parser.add_argument("--data_root", type=str, default="/data1/waymo/processed")
    parser.add_argument("--out_root", help='output root directory')
    args = parser.parse_args()
    
    assert args.out_root is not None
    os.makedirs(args.out_root, exist_ok=True)
    
    with open(args.seq_list, 'r') as f:
        seq_list = f.read().splitlines()
    select_scene_ids = [s.split(',')[0].rstrip(".tfrecord") for s in seq_list]
    
    copy_files(args.out_root, args.data_root, select_scene_ids)
    # copy_raw(args.out_root, args.data_root, select_scene_ids)
    