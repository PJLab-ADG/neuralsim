"""
@file   run.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Run [train/eval/render etc.] at once for a certain exp

NOTE: Example usage:

Example 1. Run only one type of task (e.g. train): 
    python code_multi/tools/run.py train --config test.yaml

Example 2. Run multiple tasks one by one (e.g. train,eval): 
    python code_multi/tools/run.py train,eval --config test.yaml --eval.downscale=2
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

import sys
import torch
import inspect
from icecream import ic
from datetime import datetime
from nr3d_lib.config import BaseConfig


from code_multi.tools.train import main_function as train_main, make_parser as make_train_parser
from code_multi.tools.render import main_function as render_main, make_parser as make_render_parser
from code_multi.tools.eval import main_function as eval_main, make_parser as make_eval_parser
# from code_multi.tools.eval_lidar import main_function as eval_lidar_main, make_parser as make_eval_lidar_parser
# from code_multi.tools.extract_mesh import main_function as extract_mesh_main, make_parser as make_extract_mesh_parser
# from code_multi.tools.extract_occgrid import main_function as extract_occgrid_main, make_parser as make_extact_occgrid_parser

if __name__ == "__main__":
    #---------------------------------
    #-------- Get the required tasks, e.g. train, eval, eval_lidar, extract_mesh, etc.
    sub_parsers = {
        'train': make_train_parser(), 
        'render': make_render_parser(), 
        'eval': make_eval_parser(), 
        # 'eval_lidar': make_eval_lidar_parser(), 
        # 'extract_mesh': make_extract_mesh_parser(), 
        # 'extract_occgrid': make_extact_occgrid_parser()
    }

    sub_main_fns = {
        'train': train_main, 
        'render': render_main, 
        'eval':eval_main, 
        # 'eval_lidar': eval_lidar_main, 
        # 'extract_mesh': extract_mesh_main, 
        # 'extract_occgrid': extract_occgrid_main
    }

    argv = sys.argv[1:] # Remove filename
    
    help_msg = f"Please specify the task(s) you want to run. Supported: {list(sub_parsers.keys())}"
    
    if len(argv) == 0:
        print(help_msg)
        sys.exit()
    
    tasks_str = argv[0]
    if tasks_str == '-h' or tasks_str == '--help':
        print(help_msg)
        sys.exit()
    else:
        tasks = tasks_str.split(',')
        if not all([t in sub_parsers.keys() for t in tasks]):
            raise RuntimeError(f"Got invalid tasks={tasks_str}. Should be one or a combination (comma seperated) of {list(sub_parsers.keys())}")
    
    #---------------------------------
    #-------- Assemble holistic run_parser
    bc = BaseConfig()
    # Commnon 
    bc.parser.add_argument("--outbase", type=str, default=datetime.now().strftime("%Y_%m_%d_%H_%M_%S"), help="Sets the basename of the output file (without extension).")
    # NOTE: Combine all subtasks parser to one parser
    #       e.g. python run.py train,eval,eval_lidar,extract_mesh --eval.downscale=2 --eval_lidar.lidar_id=lidar_TOP
    for prefix, sparser in sub_parsers.items():
        if prefix not in tasks:
            continue
        print(f"=> [Help] Specify {prefix} args with e.g. `--{prefix}.xxx=yyy`. This will be passed to {prefix}.py as `--xxx=yyy`")
        for action in sparser.parser._actions:
            # Merge spec configs and skip common configs from subparsers
            if action.dest not in  ['help', 'resume_dir', 'config', 'port', 'ddp', 'outbase', 'device_ids']:
                kwargs = {
                    'action': action.__class__,
                    'dest': f"{prefix}.{action.dest}",
                }
                init_signature = inspect.signature(action.__class__.__init__)
                for k in ['const', 'default', 'type', 'choices', 'required', 'help', 'metavar', 'nargs']:
                    # Only feed supported kwargs
                    if k in init_signature.parameters and getattr(action, k) is not None:
                        kwargs[k] = getattr(action, k)
                # NOTE: "--train.xxx", "--eval.yyy", etc.
                bc.parser.add_argument(f"--{prefix}.{action.dest}", **kwargs)
    
    args = bc.parse(argv[1:], print_config=False)
    task_specifics = {}
    for task in tasks:
        # Pop out "$task.xxx" cli configs
        sub_cli_config_keys = [k for k,v in args.items() if (k[:len(task)+1] == (task+"."))]
        sub_cli_configs = {k[len(task)+1:]: args.pop(k) for k in sub_cli_config_keys}
        # Merge them with {$task: xxx} i.e. parsed dot-list configs
        sub_configs = args.pop(task, {})
        sub_configs.update(sub_cli_configs)
        # Store the specific configs
        task_specifics[task] = sub_configs
    
    #---------------------------------
    #-------- Run required tasks
    for task, spec in task_specifics.items():
        main_fn = sub_main_fns[task]
        
        sargs = args.deepcopy()
        sargs.update(spec)
        
        print("".center(40, "="))
        print()
        print(f"=> Runing {task} with spec={spec}")
        print()
        print("".center(40, "="))
        torch.cuda.empty_cache()
        main_fn(sargs)
        torch.cuda.empty_cache()