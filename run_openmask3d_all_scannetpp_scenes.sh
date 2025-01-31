#!/bin/bash
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine
set -e

# RUN OPENMASK3D FOR A SINGLE SCENE
# This script performs the following:
# 1. Compute class agnostic masks and save them
# 2. Compute mask features for each mask and save them

# --------
# NOTE: SET THESE PARAMETERS BASED ON YOUR SCENE!
# data paths
DATASET="scannetpp"
DATA_DIR="/home/kumaraditya/datasets/scannetpp_openlex_v2"
# SCENES=("0a76e06478"
#         "0a7cc12c0e"  
#         "1f7cbbdde1"  
#         "410c470782"  
#         "49a82360aa"  
#         "4c5c60fa76"  
#         "8a35ef3cfe"  
#         "c0f5742640"  
#         "d918af9c5f"  
#         "fd361ab85f")

SCENES=("410c470782"
        "49a82360aa"  
        "4c5c60fa76"  
        "8a35ef3cfe"  
        "c0f5742640"  
        "d918af9c5f"  
        "fd361ab85f")

VOXEL_SIZE=0.01
STRIDE=5
EXPERIMENT_NAME="final-eval-${VOXEL_SIZE}-${STRIDE}"

# model ckpt paths
MASK_MODULE_CKPT_PATH="$(pwd)/resources/scannet200_model.ckpt"
SAM_CKPT_PATH="$(pwd)/resources/sam_vit_h_4b8939.pth"
# output directories to save masks and mask features
TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")
SAVE_VISUALIZATIONS=false #if set to true, saves pyviz3d visualizations
SAVE_CROPS=false 
# gpu optimization
OPTIMIZE_GPU_USAGE=false

cd openmask3d

for SCENE in ${SCENES[@]}; do

    SCENE_PLY_PATH="${DATA_DIR}/data/${SCENE}/scans/pcd_sampled_aligned_${VOXEL_SIZE}.ply"
    OUTPUT_DIRECTORY="$(pwd)/output/${DATASET}/${SCENE}"
    OUTPUT_FOLDER_DIRECTORY="${OUTPUT_DIRECTORY}/${TIMESTAMP}-${EXPERIMENT_NAME}"

    # 1. Compute class agnostic masks and save them
    echo "[INFO] Extracting class agnostic masks..."
    python class_agnostic_mask_computation/get_masks_single_scene.py \
    general.experiment_name=${EXPERIMENT_NAME} \
    general.checkpoint=${MASK_MODULE_CKPT_PATH} \
    general.train_mode=false \
    data.test_mode=test \
    model.num_queries=120 \
    general.use_dbscan=true \
    general.dbscan_eps=0.95 \
    general.save_visualizations=${SAVE_VISUALIZATIONS} \
    general.scene_path=${SCENE_PLY_PATH} \
    general.mask_save_dir="${OUTPUT_FOLDER_DIRECTORY}" \
    hydra.run.dir="${OUTPUT_FOLDER_DIRECTORY}/hydra_outputs/class_agnostic_mask_computation" 
    echo "[INFO] Mask computation done!"

    # get the path of the saved masks
    MASK_FILE_BASE=$(echo $SCENE_PLY_PATH | sed 's:.*/::')
    MASK_FILE_NAME=${MASK_FILE_BASE/.ply/_masks.pt}
    SCENE_MASK_PATH="${OUTPUT_FOLDER_DIRECTORY}/${MASK_FILE_NAME}"
    echo "[INFO] Masks saved to ${SCENE_MASK_PATH}."

    # 2. Compute mask features for each mask and save them
    echo "[INFO] Computing mask features..."

    python compute_features_single_scene.py \
    data=${DATASET} \
    data.data_path=${DATA_DIR} \
    data.scene=${SCENE} \
    data.masks.masks_path=${SCENE_MASK_PATH} \
    data.point_cloud_path=${SCENE_PLY_PATH} \
    data.stride=${STRIDE} \
    output.output_directory=${OUTPUT_FOLDER_DIRECTORY} \
    output.save_crops=${SAVE_CROPS} \
    hydra.run.dir="${OUTPUT_FOLDER_DIRECTORY}/hydra_outputs/mask_features_computation" \
    external.sam_checkpoint=${SAM_CKPT_PATH} \
    gpu.optimize_gpu_usage=${OPTIMIZE_GPU_USAGE}
    echo "[INFO] Feature computation done!"

    python openlex_utils/compute_mask_indices.py \
    data=${DATASET} \
    data.data_path=${DATA_DIR} \
    data.scene=${SCENE} \
    data.masks.masks_path=${SCENE_MASK_PATH} \
    data.point_cloud_path=${SCENE_PLY_PATH} \
    data.stride=${STRIDE} \
    hydra.run.dir="${OUTPUT_FOLDER_DIRECTORY}/hydra_outputs/mask_features_computation" \
    output.output_directory=${OUTPUT_FOLDER_DIRECTORY}
    echo "[INFO] Mask indices saved!"
done
