#!/bin/bash

exp_dir=$1
PY_ARGS=${@:2}

python code_multi/tools/replay.py --resume_dir $exp_dir --render_lidar --no_cam --downscale 4 --forward_inv_s 64000 --lidar_model=original_reren --lidar_id=lidar_TOP_reren --dirname demo_lidar_original_reren --lidar_vis_vmin=-2. --lidar_vis_vmax=9. assetbank_cfg.class_name_cfgs.Vehicle.model_params.framework.model_params.ray_query_cfg.forward_inv_s=6400 ${PY_ARGS}