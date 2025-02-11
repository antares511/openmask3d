import numpy as np
import torch
import os
from pathlib import Path
from natsort import natsorted
import open3d as o3d

openmask3d_output_directory = Path(
    "/home/kumaraditya/openmask3d/openmask3d/output/replica"
)
final_output_directory = Path(
    "/home/kumaraditya/openlex3d_results/openmask3d_duplicated/replica"
)
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


def get_duplicated_pcd_and_mask_indices(openmask3d_pcd, openmask3d_masks):
    """
    openmask3d_pcd: An open3d.geometry.PointCloud object with N points (and colors).
    openmask3d_masks: A NumPy array of shape (M, N), where M is the number of
                      masks (instances), and N is the number of points.
                      Each row is a binary mask indicating which points belong
                      to that instance.

    Returns:
        duplicated_pcd: A new open3d.geometry.PointCloud object of size K points
                      The points in duplicated_pcd are grouped by mask.
        mask_indices: A 1D NumPy array of length K indicating the mask ID
                      (0 to M-1) to which each point in filtered_pcd belongs.
    """

    # Extract the original points and colors from the Open3D point cloud
    original_points = np.asarray(openmask3d_pcd.points)  # shape: (N, 3)
    original_colors = np.asarray(openmask3d_pcd.colors)  # shape: (N, 3)

    duplicated_points_list = []
    duplicated_colors_list = []
    mask_indices_list = []

    M, N = openmask3d_masks.shape
    for mask_id in range(M):
        # Identify points belonging to this mask
        mask = openmask3d_masks[mask_id]
        active_indices = np.where(mask == 1)[0]

        # Retrieve those points and colors
        pcd_chunk = original_points[active_indices]
        color_chunk = original_colors[active_indices]

        # Collect them
        duplicated_points_list.append(pcd_chunk)
        duplicated_colors_list.append(color_chunk)
        mask_indices_list.append(np.full(len(active_indices), mask_id, dtype=np.int32))

    # Concatenate all chunks
    duplicated_points = np.concatenate(duplicated_points_list, axis=0)
    duplicated_colors = np.concatenate(duplicated_colors_list, axis=0)
    mask_indices = np.concatenate(mask_indices_list, axis=0)

    # Create a new Open3D point cloud
    duplicated_pcd = o3d.geometry.PointCloud()
    duplicated_pcd.points = o3d.utility.Vector3dVector(duplicated_points)
    duplicated_pcd.colors = o3d.utility.Vector3dVector(duplicated_colors)

    return duplicated_pcd, mask_indices


def get_filtered_pcd_and_mask_indices(scene_pcd, pred_mask_indices):

    # -1 index in pred_mask_indicesindicates that the point does not belong to any mask
    keep = np.where(pred_mask_indices > -1)[0]
    scene_pcd_f = scene_pcd.select_by_index(keep)
    indices_f = pred_mask_indices[keep]

    return scene_pcd_f, indices_f


def get_openlex_save_data(output_dir, scene):
    scene_output_directory = output_dir / scene
    scene_exp_folders = natsorted(
        [f for f in scene_output_directory.iterdir() if f.is_dir()]
    )
    latest_exp_folder = scene_exp_folders[-1]

    original_pcd_path = dataset_directory / f"{scene}_mesh.ply"
    openmask3d_masks_path = latest_exp_folder / f"{scene}_mesh_masks.pt"
    openmask3d_mask_indices_path = (
        latest_exp_folder / "mesh_aligned_0.05_mask_indices.npy"
    )
    openmask3d_features_path = (
        latest_exp_folder / f"{scene}_mesh_openmask3d_features.npy"
    )

    openmask3d_pcd = o3d.io.read_point_cloud(str(original_pcd_path))  # (num_points, 3)
    openmask3d_masks = np.asarray(
        torch.load(openmask3d_masks_path)
    ).T  # (num_instances, num_points)
    openmask3d_features = np.load(
        openmask3d_features_path
    )  # (num_instances, feature_dim)
    openmask3d_indices = np.load(openmask3d_mask_indices_path)  # (num_points,)

    # filtered_pcd, filtered_indices = get_filtered_pcd_and_mask_indices(
    #     openmask3d_pcd, openmask3d_indices
    # )

    duplicated_pcd, duplicated_indices = get_duplicated_pcd_and_mask_indices(
        openmask3d_pcd, openmask3d_masks
    )

    return duplicated_pcd, duplicated_indices, openmask3d_features


def main():
    for scene in scenes:

        scene_pcd, indices, features = get_openlex_save_data(
            openmask3d_output_directory, scene
        )

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
