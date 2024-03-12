# export EXP_DIR=/home/guojianfei/neuralsim_results/seg767010_exp108_splitblock_netv4_lv3_far=240.0_nr8192_cstep=0.6_step=0.05
export EXP_DIR=/home/guojianfei/neuralsim_results/seg767010_exp111_15_splitblock_netv4_lv3_vi=10.0_far=280.0_nr8192_cstep=0.

# original rerender
python code_multi/tools/replay.py --resume_dir $EXP_DIR --render_lidar --start_frame 41 --stop_frame 95 dataset_cfg.param.root=/data1/waymo/processed --no_cam --downscale 4 --forward_inv_s 64000 --lidar_model=original_reren --lidar_id=lidar_TOP --dirname demo_lidar_original_reren --lidar_vis_vmin=-2. --lidar_vis_vmax=9. assetbank_cfg.class_name_cfgs.Vehicle.model_params.framework.model_params.ray_query_cfg.forward_inv_s=6400

# horizon
python code_multi/tools/replay.py --resume_dir $EXP_DIR --render_lidar --start_frame 41 --stop_frame 95 dataset_cfg.param.root=/data1/waymo/processed --no_cam --downscale 4 --forward_inv_s 64000 --lidar_model=Risley_prism --lidar_id=horizon --dirname demo_lidar_horizon --lidar_vis_vmin=-2. --lidar_vis_vmax=9. assetbank_cfg.class_name_cfgs.Vehicle.model_params.framework.model_params.ray_query_cfg.forward_inv_s=6400

# panda qt
python code_multi/tools/replay.py --resume_dir $EXP_DIR --render_lidar --start_frame 41 --stop_frame 95 dataset_cfg.param.root=/data1/waymo/processed --no_cam --downscale 4 --forward_inv_s 64000 --lidar_model=Surround  --lidar_id=pandar_qt --dirname demo_lidar_pandar_qt --lidar_vis_vmin=-2. --lidar_vis_width=960 --lidar_vis_vmax=9. assetbank_cfg.class_name_cfgs.Vehicle.model_params.framework.model_params.ray_query_cfg.forward_inv_s=6400

# panda 128
python code_multi/tools/replay.py --resume_dir $EXP_DIR --render_lidar --start_frame 41 --stop_frame 95 dataset_cfg.param.root=/data1/waymo/processed --no_cam --downscale 4 --forward_inv_s 64000 --lidar_model=Surround  --lidar_id=pandar128 --dirname demo_lidar_pandar_128 --lidar_vis_vmin=-2. --lidar_vis_vmax=9. assetbank_cfg.class_name_cfgs.Vehicle.model_params.framework.model_params.ray_query_cfg.forward_inv_s=6400

# panda xt
python code_multi/tools/replay.py --resume_dir $EXP_DIR --render_lidar --start_frame 41 --stop_frame 95 dataset_cfg.param.root=/data1/waymo/processed --no_cam --downscale 4 --forward_inv_s 64000 --lidar_model=Surround  --lidar_id=pandar_xt --dirname demo_lidar_pandar_xt --lidar_vis_vmin=-2. --lidar_vis_vmax=9. assetbank_cfg.class_name_cfgs.Vehicle.model_params.framework.model_params.ray_query_cfg.forward_inv_s=6400

# hdl64
python code_multi/tools/replay.py --resume_dir $EXP_DIR --render_lidar --start_frame 41 --stop_frame 95 dataset_cfg.param.root=/data1/waymo/processed --no_cam --downscale 4 --forward_inv_s 64000 --lidar_model=Surround  --lidar_id=hdl64 --dirname demo_lidar_hdl64 --lidar_vis_vmin=-2. --lidar_vis_vmax=9. assetbank_cfg.class_name_cfgs.Vehicle.model_params.framework.model_params.ray_query_cfg.forward_inv_s=6400

# vlp16
python code_multi/tools/replay.py  --resume_dir $EXP_DIR --render_lidar --start_frame 41 --stop_frame 95 dataset_cfg.param.root=/data1/waymo/processed --no_cam --downscale 4 --forward_inv_s 64000 --lidar_model=Surround  --lidar_id=vlp16 --dirname demo_lidar_vlp16 --lidar_vis_vmin=-2. --lidar_vis_vmax=9. assetbank_cfg.class_name_cfgs.Vehicle.model_params.framework.model_params.ray_query_cfg.forward_inv_s=6400

# bpearl
python code_multi/tools/replay.py --resume_dir $EXP_DIR --render_lidar --start_frame 41 --stop_frame 95 dataset_cfg.param.root=/data1/waymo/processed --no_cam --downscale 4 --forward_inv_s 64000 --lidar_model=Surround  --lidar_id=bpearl --dirname demo_lidar_bpearl --lidar_vis_vmin=-2. --lidar_vis_vmax=9. --lidar_vis_width=960 assetbank_cfg.class_name_cfgs.Vehicle.model_params.framework.model_params.ray_query_cfg.forward_inv_s=6400

# rs_m1
python code_multi/tools/replay.py --resume_dir $EXP_DIR --render_lidar --start_frame 41 --stop_frame 95 dataset_cfg.param.root=/data1/waymo/processed --no_cam --downscale 4 --forward_inv_s 64000 --lidar_model=Solid_state  --lidar_id=rs_m1 --dirname demo_lidar_rs_m1 --lidar_vis_vmin=-2. --lidar_vis_vmax=9. --lidar_vis_width=960 assetbank_cfg.class_name_cfgs.Vehicle.model_params.framework.model_params.ray_query_cfg.forward_inv_s=6400