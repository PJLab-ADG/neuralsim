"""
Compress processed data of selected waymo sequences to a zip file (for data transfering between devices)
"""

from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from os import PathLike
from typing import List, Union

def make_zip(
    zip_name: str, root_dir: Union[str, PathLike], select_scene_ids: List[str], 
    fields=['images', 'lidars', 'masks', 'normals', 'depths', 'scenarios']):
    src_path = Path(root_dir).expanduser().resolve(strict=True)
    with ZipFile(zip_name, 'w', ZIP_DEFLATED) as zf:
        def recursive_append_to_zip(path: Path):
            # for file in path.rglob('*.npz'):
            for file in path.rglob('*'):
                zf.write(file, file.relative_to(src_path))
        for i, scene_id in enumerate(tqdm(select_scene_ids, f'compressing ...')):
            for field in fields:
                if field == 'scenarios':
                    scenario_file = src_path.joinpath(scene_id, f"scenario.pt")
                    zf.write(scenario_file, scenario_file.relative_to(src_path))
                else:
                    recursive_append_to_zip(src_path.joinpath(scene_id, field))

def make_multi_zip(
    out_root: Union[str, PathLike], root: Union[str, PathLike], select_scene_ids: List[str], 
    fields=['images', 'lidars', 'masks', 'normals', 'depths', 'scenarios']):
    src_path = Path(root).expanduser().resolve(strict=True)
    dst_path = Path(out_root).expanduser().resolve(strict=True)
    
    def make_single_zip(scene_id: str):
        zip_name = dst_path.joinpath(f"{scene_id}.zip")
        with ZipFile(zip_name, 'w', ZIP_DEFLATED) as zf:
            def recursive_append_to_zip(path: Path):
                for file in path.rglob('*'):
                    zf.write(file, file.relative_to(src_path))
            for field in fields:
                if field == 'scenarios':
                    scenario_file = src_path.joinpath(scene_id, f"scenario.pt")
                    zf.write(scenario_file, scenario_file.relative_to(src_path))
                else:
                    recursive_append_to_zip(src_path.joinpath(scene_id, field))
    
    thread_map(make_single_zip, select_scene_ids, desc='Compressing')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_list", type=str, help='specify --seq_list if you want to limit the list of seqs', default="dataio/autonomous_driving/waymo/seq_list_select_20221214.lst")
    parser.add_argument("--data_root", type=str, default="/data1/waymo/processed")
    parser.add_argument("--out", help='output zip file path', default=None)
    parser.add_argument("--out_root", help='output zip files root dir', default=None)
    args = parser.parse_args()
    
    with open(args.seq_list, 'r') as f:
        seq_list = f.read().splitlines()
    select_scene_ids = [s.split(',')[0].rstrip(".tfrecord") for s in seq_list]
    
    if args.out is not None:
        make_zip(args.out, args.data_root, select_scene_ids)
        # make_zip(args.out, args.data_root, select_scene_ids, ['masks_vit_adapter'])
    elif args.out_root is not None:
        make_multi_zip(args.out_root, args.data_root, select_scene_ids)
    