import open3d as o3d
import numpy as np
import torch

# Define file paths
path_scene_pcd = "/home/kumaraditya/datasets/Replica/room0_mesh.ply"
path_pred_masks = "/home/kumaraditya/openmask3d/output/replica/room0/2025-01-29-13-16-05-eval/room0_mesh_masks.pt"

# Load the scene point cloud
scene_pcd = o3d.io.read_point_cloud(path_scene_pcd)

# Load the predicted masks
pred_masks = np.asarray(torch.load(path_pred_masks)).T  # (num_instances, num_points)


class InteractiveMaskVisualizer:
    def __init__(self, scene_pcd, pred_masks):
        self.scene_pcd = scene_pcd
        self.pred_masks = pred_masks
        self.visualizer = o3d.visualization.VisualizerWithKeyCallback()
        self.view_control = None

    def filter_points_by_mask(self, mask_id):
        """Filter and display points based on the selected mask ID."""
        if mask_id < 0 or mask_id >= self.pred_masks.shape[0]:
            print(
                f"Invalid mask ID: {mask_id}. Must be in range 0-{self.pred_masks.shape[0] - 1}."
            )
            return None

        mask = self.pred_masks[mask_id]  # Get mask for the given ID
        filtered_indices = np.where(mask > 0)[0]  # Get indices of active points

        if len(filtered_indices) == 0:
            print(f"No points found for mask ID {mask_id}.")
            return None

        filtered_pcd = self.scene_pcd.select_by_index(filtered_indices)
        return filtered_pcd

    def key_callback_filter(self, vis):
        """Callback to input a mask ID and visualize its corresponding points."""
        mask_id = input("Enter the Mask ID to visualize: ")
        try:
            mask_id = int(mask_id)
            self.view_control = vis.get_view_control()
            filtered_pcd = self.filter_points_by_mask(mask_id)
            if filtered_pcd:
                vis.clear_geometries()
                vis.add_geometry(filtered_pcd)
                vis.update_renderer()
                print(f"Displaying Mask ID {mask_id}.")
        except ValueError:
            print("Invalid input. Please enter a valid integer Mask ID.")

    def key_callback_reset(self, vis):
        """Show the full point cloud while keeping the current view."""
        if self.view_control:
            vis.clear_geometries()
            vis.add_geometry(self.scene_pcd)
            vis.update_renderer()
            print("Full point cloud displayed while keeping the view.")

    def run_visualizer(self):
        self.visualizer.create_window("OpenMask3D Visualizer")
        self.visualizer.add_geometry(self.scene_pcd)
        self.view_control = self.visualizer.get_view_control()

        # Register key callbacks
        self.visualizer.register_key_callback(ord("F"), self.key_callback_filter)
        self.visualizer.register_key_callback(ord("R"), self.key_callback_reset)

        print("Press 'F' to filter by Mask ID.")
        print("Press 'R' to reset view while keeping the current perspective.")
        self.visualizer.run()
        self.visualizer.destroy_window()


# Run the interactive visualization
vis = InteractiveMaskVisualizer(scene_pcd, pred_masks)
vis.run_visualizer()
