#!/usr/bin/bash
EXP_DIR=/home/dengnianchen/Work/neuralsim/logs/seg745856_merged_exp220
ANIM=/home/dengnianchen/Work/neuralsim/logs/seg745856_merged_exp220/vehicle_trajectory1.json
DATASET_ROOT=/data/waymo
START_FRAME=52
STOP_FRAME=100
FPS=18
LIDAR_VIS_WIDTH=480

EXTRA_ARGS=(
    "--lidar_vis_view" "TBDnDBg3lh6aQuGsMgqpgQ@45,70"
    #"--draw_anno" "TBDnDBg3lh6aQuGsMgqpgQ" "--anno_color" "0"
)

if [ -n "$ANIM" ]
then
      EXTRA_ARGS+=("--anim" "$ANIM")
fi

LIDAR_MODELS=(original_reren Risley_prism Surround Surround Surround Surround Surround Surround Solid_state)
LIDAR_NAMES=(lidar_TOP horizon pandar_qt pandar128 pandar_xt hdl64 vlp16 bpearl rs_m1)

for ((i = 0; i < ${#LIDAR_MODELS[@]}; i++)); do
    python code_multi/tools/render_anim.py --resume_dir "$EXP_DIR" dataset_cfg.param.root="$DATASET_ROOT" \
        --start_frame $START_FRAME --stop_frame $STOP_FRAME --fps $FPS --no_gt --no_cam --render_lidar \
        --forward_inv_s 64000 assetbank_cfg.class_name_cfgs.Vehicle.model_params.framework.model_params.ray_query_cfg.forward_inv_s=6400 \
        --lidar_model="${LIDAR_MODELS[i]}" --lidar_id="${LIDAR_NAMES[i]}" \
        --lidar_vis_vmin=-2. --lidar_vis_vmax=9. --lidar_vis_width $LIDAR_VIS_WIDTH \
        ${EXTRA_ARGS[*]}
done
