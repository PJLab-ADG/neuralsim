#!/bin/bash

# Mesh extraction for all exps in a specified directory ($1)

# $1: overall_dir
# the rest: other params

overall_dir=$1
PY_ARGS=${@:2}

shopt -s nullglob

for expdir in $overall_dir/*
do
    expname=$(basename $expdir)
    # echo $expdir "->" $expname

    # files=(${expdir}/meshes/*0.2.ply)
    # if [ ${#files[@]} -gt 0 ]; then
    #     echo "exist: $expdir"
    # else
    #     echo "no: $expdir"
    #     python code_single/tools/extract_mesh.py --resume_dir $expdir ${PY_ARGS}
    # fi

    python code_single/tools/extract_mesh.py --resume_dir $expdir ${PY_ARGS}
done

echo "Done extract_mesh_directory.sh in dir ${overall_dir}"