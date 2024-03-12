"""
@file   utils.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Utility functions for scene resources
"""

__all__ = ['load_scenes_and_assets']

import os
import torch
from typing import Tuple, Dict, Any

from nr3d_lib.fmt import log
from nr3d_lib.checkpoint import sorted_ckpts
from nr3d_lib.config import ConfigDict
from nr3d_lib.utils import IDListedDict

from app.resources.asset_bank import AssetBank
from app.resources.scenes import Scene
from app.resources.scene_bank import load_scene_bank


def load_scenes_and_assets(exp_dir: str, assetbank_cfg: ConfigDict, training: ConfigDict, *,
                           load_pt: str = None, device=None, **kwargs) -> Tuple[IDListedDict[Scene], AssetBank, Dict[str, Any]]:
    # ---------------------------------------------
    # -----------  Load Checkpoint   --------------
    # ---------------------------------------------
    # Automatically load 'final_xxx.pt' or 'latest.pt' if argument load_pt is not specified
    ckpt_file = load_pt or sorted_ckpts(os.path.join(exp_dir, 'ckpts'))[-1]
    log.info("=> Use ckpt:" + str(ckpt_file))
    state_dict = torch.load(ckpt_file, map_location=device)

    # ---------------------------------------------
    # -----------     Scene Bank     --------------
    # ---------------------------------------------
    scene_bank, _ = load_scene_bank(os.path.join(exp_dir, 'scenarios'), device=device)

    # ---------------------------------------------
    # -----------     Asset Bank     --------------
    # ---------------------------------------------
    asset_bank = AssetBank(assetbank_cfg)
    asset_bank.create_asset_bank(scene_bank, load_state_dict=state_dict['asset_bank'], device=device)
    asset_bank.training_before_per_step(training.num_iters)
    asset_bank.eval()

    # ---------------------------------------------
    # ----    Load assets to scene objects     ----
    # ---------------------------------------------
    for scene in scene_bank:
        scene.load_assets(asset_bank)

    return scene_bank, asset_bank, state_dict
