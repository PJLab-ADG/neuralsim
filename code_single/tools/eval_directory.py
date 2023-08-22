"""
@file   eval_directory.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Appearance evaluation for all exps in a specified directory (--overall_dir)
"""
import os
import sys
def set_env(depth: int):
    # Add project root to sys.path
    current_file_path = os.path.abspath(__file__)
    project_root_path = os.path.dirname(current_file_path)
    for _ in range(depth):
        project_root_path = os.path.dirname(project_root_path)
    if project_root_path not in sys.path:
        sys.path.append(project_root_path)
        print(f"Added {project_root_path} to sys.path")
set_env(2)

import os
from tqdm import tqdm

from nr3d_lib.config import OmegaConf, ConfigDict

from code_single.tools.eval import main_function as eval_main, make_parser as eval_parser
from nr3d_lib.utils import is_file_being_written

def main_function(args: ConfigDict):
    sdir_list = list(sorted(os.listdir(args.overall_dir)))
    for sdirname in tqdm(sdir_list, 'evaluating...'):
        full_sdir = os.path.join(args.overall_dir, sdirname)
        if os.path.isdir(full_sdir):
            # NOTE: Only eval finished exps
            ckpt_dir = os.path.join(full_sdir, 'ckpts')
            finished = False
            if os.path.exists(ckpt_dir):
                ckpt_list = os.listdir(ckpt_dir)
                for ckpt in ckpt_list:
                    if 'final' in ckpt and not is_file_being_written(ckpt):
                        finished = True
                        break
            if not finished:
                continue
            
            eval_dir = os.path.join(full_sdir, args.dirname)
            if not args.no_ignore_existing and os.path.exists(eval_dir):
                continue
            
            sargs = OmegaConf.create(args.to_dict())
            sargs.exp_dir = full_sdir
            detail_cfg = OmegaConf.load(os.path.join(full_sdir, 'config.yaml'))
            
            sargs = OmegaConf.merge(detail_cfg, sargs)
            sargs = ConfigDict(OmegaConf.to_container(sargs, resolve=True))
            try:
                eval_main(sargs)
            except FileExistsError:
                pass

if __name__ == "__main__":
    bc = eval_parser()
    bc.parser.add_argument("--overall_dir", type=str, required=True, help="Specifies the overall directory.")
    bc.parser.add_argument("--no_ignore_existing", action='store_true', help="If set, existing evaluations will NOT be ignored.")
    args = bc.parse(stand_alone=False)
    main_function(args)