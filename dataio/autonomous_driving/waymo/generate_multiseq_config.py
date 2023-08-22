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
set_env(3)

import os
from glob import glob
from nr3d_lib.config import ConfigDict, save_config

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, default="/data1/waymo/processed/")
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--stop", type=int, default=None)
parser.add_argument("--out", type=str, required=True, default="dataio/autonomous_driving/waymo/all.yaml")
args = parser.parse_args()

scenario_root = os.path.join(os.path.normpath(args.data_root), 'scenarios')
scenario_file_list = list(sorted(glob(os.path.join(scenario_root, "*.pt"))))
scenario_list = [os.path.splitext(os.path.basename(s))[0] for s in scenario_file_list]

if args.stop is None:
    args.stop = len(scenario_list)
scenario_list = scenario_list[args.start:args.stop]

config = ConfigDict(seq_list=scenario_list)

save_config(config, args.out, ignore_fields=[])
print(f"=> Seq config saved to {args.out}")
