import open3d as o3d

if __name__ == "__main__":
    path_scene_pcd = "/home/kumaraditya/datasets/scannetpp_openlex_v2/data/8a35ef3cfe/scans/pcd_sampled_aligned_0.01.ply"
    scene_pcd = o3d.io.read_point_cloud(path_scene_pcd)
    o3d.visualization.draw_geometries([scene_pcd])
