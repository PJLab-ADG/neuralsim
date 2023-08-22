#!/bin/bash

# NOTE: Before proceeding, you need to fill out the Waymo terms of use and complete `gcloud auth login`.

lst=$1 # dataio/autonomous_driving/waymo/waymo_static_32.lst
dest=$2 # /data1/waymo/training/
source=gs://waymo_open_dataset_v_1_4_2/individual_files/training

mkdir -p $dest

# Get the total number of filenames
total_files=$(wc -l < $lst)
counter=0

# Read filenames from the .lst file and process them one by one
while IFS= read -r filename; do
    counter=$((counter + 1))
    echo "[${counter}/${total_files}] Dowloading $filename ..."
    gsutil cp -n ${source}/${filename}.tfrecord ${dest}
done < $lst
