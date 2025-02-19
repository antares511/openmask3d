#!/bin/bash

DATASET="hm3d"
DATA_DIR="/home/kumaraditya/datasets/hm3d_compressed"

# SCENES=("00829"
#         "00824"
#         "00843"
#         "00847"
#         "00873"
#         "00877"
#         "00890")

SCENES=("00890")

cd openmask3d

for SCENE in ${SCENES[@]}; do
    echo "Processing scene: ${SCENE}"
    python openlex_utils/hm3d_mesh_preprocess.py \
    data=${DATASET} \
    data.data_path=${DATA_DIR} \
    data.scene=${SCENE}
done