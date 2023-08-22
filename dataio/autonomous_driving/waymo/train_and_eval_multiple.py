
import os
from tqdm import tqdm

from app.resources import parse_scene_bank_cfg
from code_single.tools.train import main_function as train_main
from code_single.tools.eval import main_function as eval_main, make_parser as eval_parser

if __name__ == "__main__":
    parser = eval_parser()
    args = parser.parse(stand_alone=False)
    scenario_cfg_list = args.scenebank_cfg.pop('scenarios')
    exp_parent_dir = args.exp_parent_dir
    for sce_cfg in tqdm(scenario_cfg_list):
        try:
            sce_id, _, _ = parse_scene_bank_cfg(sce_cfg)
            waymo_short_id = sce_id[8:14]
            
            # Make local config
            sce_args = args.deepcopy()
            sce_args.scenebank_cfg.scenarios = [sce_cfg]
            if 'test_scenebank_cfg' in sce_args:
                sce_args.test_scenebank_cfg.scenarios = [sce_cfg]
            
            exp_dir = sce_args.exp_dir = os.path.join(exp_parent_dir, "seg"+waymo_short_id)
            ckpt_dir = os.path.join(exp_dir, 'ckpts')
            eval_dir = os.path.join(exp_dir, args.dirname)
            if not os.path.exists(exp_dir):
                train_main(sce_args)
            if not os.path.exists(eval_dir):
                finished = False
                if os.path.exists(ckpt_dir):
                    ckpt_list = os.listdir(ckpt_dir)
                    for ckpt in ckpt_list:
                        if 'final' in ckpt:
                            finished = True
                            break
                # NOTE: Only eval finished exps
                if finished:
                    eval_main(sce_args)
        except FileExistsError:
            pass
