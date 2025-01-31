import numpy as np
import torch
import os
from pathlib import Path
from natsort import natsorted
import open3d as o3d

openmask3d_output_directory = Path(
    "/home/kumaraditya/openmask3d/openmask3d/output/replica"
)
final_output_directory = Path("/home/kumaraditya/openlex3d_results/Replica")
dataset_directory = Path("/home/kumaraditya/datasets/Replica")
scenes = [
    "room0",
    "room1",
    "room2",
    "office0",
    "office1",
    "office2",
    "office3",
    "office4",
]


def get_filtered_data(path_scene_pcd, path_pred_mask_indices):

    scene_pcd = o3d.io.read_point_cloud(path_scene_pcd)

    # -1 index indicates that the point does not belong to any mask
    pred_mask_indices = np.load(path_pred_mask_indices)  # (num_instances,)

    keep = np.where(pred_mask_indices > -1)[0]
    scene_pcd_f = scene_pcd.select_by_index(keep)
    indices_f = pred_mask_indices[keep]

    return scene_pcd_f, indices_f


def get_data(output_dir, scene):
    scene_output_directory = output_dir / scene
    scene_exp_folders = natsorted(
        [f for f in scene_output_directory.iterdir() if f.is_dir()]
    )
    latest_exp_folder = scene_exp_folders[-1]

    original_pcd_path = dataset_directory / f"{scene}_mesh.ply"
    openmask3d_mask_indices_path = (
        latest_exp_folder / "mesh_aligned_0.05_mask_indices.npy"
    )
    openmask3d_features_path = (
        latest_exp_folder / f"{scene}_mesh_openmask3d_features.npy"
    )

    filtered_pcd, filtered_indices = get_filtered_data(
        str(original_pcd_path), str(openmask3d_mask_indices_path)
    )

    openmask3d_features = np.load(openmask3d_features_path)

    return filtered_pcd, filtered_indices, openmask3d_features


def main():
    for scene in scenes:

        scene_pcd, indices, features = get_data(openmask3d_output_directory, scene)

        # create final output directory if it does not exist
        scene_output_directory = final_output_directory / scene
        scene_output_directory.mkdir(parents=True, exist_ok=True)

        pcd_save_path = scene_output_directory / "point_cloud.pcd"
        indices_save_path = scene_output_directory / "index.npy"
        features_save_path = scene_output_directory / "embeddings.npy"

        o3d.io.write_point_cloud(str(pcd_save_path), scene_pcd)
        np.save(indices_save_path, indices)
        np.save(features_save_path, features)

        print(f"Scene: {scene}", "-" * 50)
        print(f"Scene PCD points: {len(scene_pcd.points)}")
        print(f"Indices shape: {indices.shape}")

    return


if __name__ == "__main__":
    main()
