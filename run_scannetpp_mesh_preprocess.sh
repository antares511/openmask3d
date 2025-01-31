#!/bin/bash

DATASET="scannetpp"
DATA_DIR="/home/kumaraditya/datasets/scannetpp_openlex_v2/data"
# DATA_DIR="/home/kumaraditya/datasets/Replica"

SCENES=("0a76e06478"
        "0a7cc12c0e"  
        "1f7cbbdde1"  
        "410c470782"  
        "49a82360aa"  
        "4c5c60fa76"  
        "8a35ef3cfe"  
        "c0f5742640"  
        "d918af9c5f"  
        "fd361ab85f")

# SCENES=("room0")

cd openmask3d

for SCENE in ${SCENES[@]}; do
    echo "Processing scene: ${SCENE}"
    python openlex_utils/scannet_mesh_preprocess.py \
    data=${DATASET} \
    data.data_path=${DATA_DIR} \
    data.scene=${SCENE}
done