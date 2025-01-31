import open3d as o3d

if __name__ == "__main__":
    path_scene_pcd = (
        "/home/kumaraditya/datasets/hm3d_compressed/00829/scene_panoptic.ply"
    )
    scene_pcd = o3d.io.read_point_cloud(path_scene_pcd)
    # scene_pcd = o3d.io.read_triangle_mesh(path_scene_pcd)
    o3d.visualization.draw_geometries([scene_pcd])
