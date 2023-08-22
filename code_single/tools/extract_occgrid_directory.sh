#!/bin/bash

overall_dir=$1
PY_ARGS=${@:2}

for expdir in $overall_dir/*
do
    expname=$(basename $expdir)
    echo $expdir "->" $expname
    python code_single/tools/extract_occgrid.py --resume_dir $expdir ${PY_ARGS}
done

echo "Done extract_occgrid_directory.sh in dir ${overall_dir}"