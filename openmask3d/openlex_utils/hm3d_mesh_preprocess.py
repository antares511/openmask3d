import hydra
import open3d as o3d
import numpy as np
from omegaconf import DictConfig
from pathlib import Path


@hydra.main(config_path="../configs", config_name="openmask3d_inference")
def main(ctx: DictConfig):
    data_path = ctx.data.data_path
    scene = ctx.data.scene
    voxel_size = 0.02

    pcd_path = Path(data_path) / scene / "scene_rgb.ply"
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    pcd_rotated = o3d.io.read_point_cloud(str(pcd_path))

    pcd = pcd.voxel_down_sample(voxel_size)
    pcd_rotated = pcd_rotated.voxel_down_sample(voxel_size)

    # rotate pcd by 90 degrees about x axis
    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    pcd_rotated.rotate(rotation_matrix, center=(0, 0, 0))

    pcd_save_path = Path(data_path) / scene / f"scene_rgb_downsampled_{voxel_size}.ply"
    pcd_rotated_save_path = (
        Path(data_path) / scene / f"scene_rgb_downsampled_{voxel_size}_rotated.ply"
    )
    o3d.io.write_point_cloud(str(pcd_save_path), pcd)
    o3d.io.write_point_cloud(str(pcd_rotated_save_path), pcd_rotated)


if __name__ == "__main__":
    main()
