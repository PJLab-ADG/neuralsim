#!/bin/bash

EXP_DIR=$1
PY_ARGS=${@:2}

LIDAR_MODELS=(original original_reren Risley_prism Surround Surround Surround Surround Surround Surround Solid_state)
LIDAR_NAMES=(lidar_TOP lidar_TOP horizon pandar_qt pandar128 pandar_xt hdl64 vlp16 bpearl rs_m1)
LIDAR_RADIUS=(2 2 1 4 2 2 2 2 4 4)

for ((i = 0; i < ${#LIDAR_MODELS[@]}; i++)); do
    python code_single/tools/render.py --resume_dir $EXP_DIR --render_lidar --fps=24 --no_cam --dirname demo_lidar_sim --lidar_forward_inv_s 64000 --lidar_id="${LIDAR_NAMES[i]}" --lidar_model="${LIDAR_MODELS[i]}" --lidar_vis_radius="${LIDAR_RADIUS[i]}" --lidar_vis_vmin=-2. --lidar_vis_vmax=9. --lidar_vis_rgb_choice=height ${PY_ARGS}
done