"""
Compress processed data of selected waymo sequences to a zip file (for data transfering between devices)
"""

from tqdm import tqdm
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
            for file in path.rglob('*'):
                zf.write(file, file.relative_to(src_path))
        for field in ['lidars', 'images', 'masks', 'normals', 'depths']:
            if field in fields:
                for seq_i, seq in enumerate(tqdm(select_scene_ids, f'compressing {field}...')):
                    recursive_append_to_zip(src_path.joinpath(field, seq))
        if 'scenarios' in fields:
            for seq_i, seq in enumerate(tqdm(select_scene_ids, 'compressing scenarios...')):
                scenario_file = src_path.joinpath('scenarios', f"{seq}.pt")
                zf.write(scenario_file, scenario_file.relative_to(src_path))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_list", type=str, help='specify --seq_list if you want to limit the list of seqs', default="dataio/autonomous_driving/waymo/seq_list_select_20221214.lst")
    parser.add_argument("--data_root", type=str, default="/data1/waymo/processed")
    parser.add_argument("--out", help='output zip file path')
    args = parser.parse_args()
    
    assert args.out is not None
    
    with open(args.seq_list, 'r') as f:
        seq_list = f.read().splitlines()
    select_scene_ids = [s.split(',')[0].rstrip(".tfrecord") for s in seq_list]
    
    make_zip(args.out, args.data_root, select_scene_ids)
    