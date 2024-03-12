PY_ARGS=${@:1} # --start_frame 80 --downscale 4

# source /etc/profile.d/conda.sh
# conda activate /cpfs2/user/guojianfei/ml/
# cd /cpfs2/user/guojianfei/ai_ws/neuralsim/
export PYTHONPATH="${DIR}":$PYTHONPATH

python code_multi/tools/manipulate.py --only_all --mode random ${PY_ARGS} --start_frame 60 --num_frames 60 --fix_gt --render_lidar --only_all
python code_multi/tools/manipulate.py --only_all --mode replay_random ${PY_ARGS} --start_frame 0 --num_frames 120 --render_lidar --only_all
# python code_multi/tools/manipulate.py --only_all --mode replay_rotation n_rots=5 --stop_frame -1 ${PY_ARGS}
# python code_multi/tools/manipulate.py --only_all --mode replay_translation --stop_frame -1 ${PY_ARGS}
# python code_multi/tools/manipulate.py --only_all --mode replay_scale --stop_frame -1 ${PY_ARGS}
# python code_multi/tools/manipulate.py --only_all --mode rotation --num_frames 48 --fix_gt n_rots=1.0  ${PY_ARGS}
# python code_multi/tools/manipulate.py --only_all --mode translation --fix_gt ${PY_ARGS}
# python code_multi/tools/manipulate.py --only_all --mode scale --num_frames 48 --fix_gt ${PY_ARGS}
# python code_multi/tools/manipulate.py --only_all --mode thanos --stop_frame -1  ${PY_ARGS}

python code_multi/tools/manipulate.py --only_all --mode self_fly ${PY_ARGS} --num_frames 48 --only_all
python code_multi/tools/manipulate.py --only_all --mode self_zoom_out_fix_obj ${PY_ARGS} --fix_gt --num_frames 48 --only_all
python code_multi/tools/manipulate.py --only_all --mode self_rotate ${PY_ARGS} --num_frames 120 --render_lidar --only_all
# python code_multi/tools/manipulate.py --only_all --mode self_trans ${PY_ARGS}

# python code_multi/tools/manipulate.py --only_all --mode clone --stop_frame -1 ${PY_ARGS}
