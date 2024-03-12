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
    for i, scene_id in enumerate(tqdm(select_scene_ids, f'copying...')):
        for field in fields:
            if field == 'scenarios':
                shutil.copy(src_root.joinpath(scene_id, 'scenario.pt'), out_root.joinpath(scene_id, 'scenario.pt'))
            else:
                shutil.copytree(src_root.joinpath(scene_id, field), out_root.joinpath(scene_id, field))

def copy_raw(
    out_root_dir: str, root_dir: str, select_scene_ids: List[str], 
):
    for i, scene_id in enumerate(tqdm(select_scene_ids, f'copying raw...')):
        shutil.copyfile(os.path.join(root_dir, f"{scene_id}.tfrecord"), os.path.join(out_root_dir, f"{scene_id}.tfrecord"))

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
    # copy_files(args.out_root, args.data_root, select_scene_ids, ['masks_vit_adapter'])
    # copy_raw(args.out_root, args.data_root, select_scene_ids)
    