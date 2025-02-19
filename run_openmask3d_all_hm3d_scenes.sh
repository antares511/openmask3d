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
DATASET="hm3d"
DATA_DIR="/home/kumaraditya/datasets/hm3d_compressed"

# SCENES=("00824"
#         "00843"
#         "00847"
#         "00873"
#         "00877"
#         "00829"
#         "00890")

SCENES=("00890")

VOXEL_SIZE=0.02
STRIDE=10
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

    SCENE_PLY_PATH_ROTATED="${DATA_DIR}/${SCENE}/scene_rgb_downsampled_${VOXEL_SIZE}_rotated.ply"
    SCENE_PLY_PATH="${DATA_DIR}/${SCENE}/scene_rgb_downsampled_${VOXEL_SIZE}.ply"
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
    general.scene_path=${SCENE_PLY_PATH_ROTATED} \
    general.mask_save_dir="${OUTPUT_FOLDER_DIRECTORY}" \
    hydra.run.dir="${OUTPUT_FOLDER_DIRECTORY}/hydra_outputs/class_agnostic_mask_computation" 
    echo "[INFO] Mask computation done!"

    # get the path of the saved masks
    MASK_FILE_BASE=$(echo $SCENE_PLY_PATH_ROTATED | sed 's:.*/::')
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
