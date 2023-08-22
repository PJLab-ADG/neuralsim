#!/bin/bash

overall_dir=$1
lidar_model=$2 # original_reren
lidar_id=$3 # lidar_TOP
PY_ARGS=${@:4}

for expdir in $overall_dir/*
do
    expname=$(basename $expdir)
    echo $expdir "->" $expname
    python code_single/tools/replay.py --resume_dir $expdir --no_cam --render_lidar --forward_inv_s 64000 --lidar_model=${lidar_model} --lidar_id=${lidar_id} --dirname=render_${lidar_model}_${lidar_id} --lidar_vis_vmin=-2. --lidar_vis_vmax=9. ${PY_ARGS}
done

echo "Done render_lidar_directory.sh in dir ${overall_dir}"