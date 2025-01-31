import hydra
import open3d as o3d
import numpy as np
from omegaconf import DictConfig
from pathlib import Path


def sample_mesh_with_adaptive_sampling(mesh, initial_factor=1, voxel_size=0.02):
    num_vertices = np.asarray(mesh.vertices).shape[0]
    sample_factor = initial_factor
    final_points = 0

    print(f"Original mesh with {num_vertices} points")

    while True:
        # Sample points from the mesh
        pcd = mesh.sample_points_uniformly(int(sample_factor * num_vertices))
        pcd_downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)

        pcd_num_points = np.asarray(pcd.points).shape[0]
        pcd_downsampled_num_points = np.asarray(pcd_downsampled.points).shape[0]

        final_points = pcd_downsampled_num_points

        # If the number of points remains the same, increase the sample factor
        if pcd_downsampled_num_points == pcd_num_points:
            sample_factor *= 1.5  # Increase the sample count
        else:
            break

    print(f"Sampled mesh with {final_points} points, sample factor: {sample_factor}")

    return pcd_downsampled


@hydra.main(config_path="../configs", config_name="openmask3d_inference")
def main(ctx: DictConfig):
    data_path = ctx.data.data_path
    scene = ctx.data.scene
    voxel_size = 0.01

    # mesh_path = Path(data_path) / f"{scene}_mesh.ply"
    mesh_path = Path(data_path) / scene / "scans" / "mesh_aligned_0.05.ply"
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))

    pcd = sample_mesh_with_adaptive_sampling(mesh, voxel_size=voxel_size)

    # pcd_save_path = Path(data_path) / f"{scene}_pcd_sampled_0.02.ply"
    pcd_save_path = (
        Path(data_path) / scene / "scans" / f"pcd_sampled_aligned_{voxel_size}.ply"
    )
    o3d.io.write_point_cloud(str(pcd_save_path), pcd)


if __name__ == "__main__":
    main()
