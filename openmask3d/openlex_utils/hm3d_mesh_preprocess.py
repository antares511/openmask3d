import hydra
import open3d as o3d
import numpy as np
from omegaconf import DictConfig
from pathlib import Path


@hydra.main(config_path="../configs", config_name="openmask3d_inference")
def main(ctx: DictConfig):
    data_path = ctx.data.data_path
    scene = ctx.data.scene
    voxel_size = 0.01

    pcd_path = Path(data_path) / scene / "scene_rgb.ply"
    pcd = o3d.io.read_point_cloud(str(pcd_path))

    pcd = pcd.voxel_down_sample(voxel_size)

    pcd_save_path = Path(data_path) / scene / "scene_rgb_downsampled.ply"
    o3d.io.write_point_cloud(str(pcd_save_path), pcd)


if __name__ == "__main__":
    main()
