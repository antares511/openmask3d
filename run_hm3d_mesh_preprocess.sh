#!/bin/bash

DATASET="hm3d"
DATA_DIR="/home/kumaraditya/datasets/hm3d_compressed"

SCENES=("00829")

cd openmask3d

for SCENE in ${SCENES[@]}; do
    echo "Processing scene: ${SCENE}"
    python openlex_utils/hm3d_mesh_preprocess.py \
    data=${DATASET} \
    data.data_path=${DATA_DIR} \
    data.scene=${SCENE}
done