#!/bin/bash

# Render all experiments in a specified directory ($1)

overall_dir=$1
PY_ARGS=${@:2}

for expdir in $overall_dir/*
do
    expname=$(basename $expdir)
    echo $expdir "->" $expname
    python code_single/tools/replay.py --resume_dir $expdir ${PY_ARGS}
done

echo "Done replay_directory.sh in dir ${overall_dir}"