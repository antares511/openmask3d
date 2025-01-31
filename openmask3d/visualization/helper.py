import open3d as o3d
import numpy as np
import torch


def main():
    path_scene_pcd = "/path/to/mesh.ply"
    path_pred_masks = "/path/to/masks.pt"
    path_pred_mask_indices = "/path/to/mask_indices.npy"
    path_openmask3d_features = "/path/to/openmask3d_features.npy"

    scene_pcd = o3d.io.read_point_cloud(path_scene_pcd)
    pred_masks = np.asarray(
        torch.load(path_pred_masks)
    ).T  # (num_instances, num_points)
    pred_mask_indices = np.load(path_pred_mask_indices)  # (num_instances,)
    openmask3d_features = np.load(path_openmask3d_features)  # (num_instances, feat_dim)

    return


if __name__ == "__main__":
    main()
