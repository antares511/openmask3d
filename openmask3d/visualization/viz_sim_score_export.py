import numpy as np
import open3d as o3d
import torch
import clip
import pdb
import matplotlib
import matplotlib.pyplot as plt
from constants import *

import yaml
import os
from tqdm import tqdm
from mp3d_preds import MP3DEval, collate_fn
from torch.utils.data import DataLoader
from scipy.spatial import cKDTree

from matplotlib.colors import LinearSegmentedColormap


class QuerySimilarityComputation:
    def __init__(
        self,
    ):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.clip_model, _ = clip.load("ViT-L/14@336px", self.device)

    def get_query_embedding(self, text_query):
        text_input_processed = clip.tokenize(text_query).to(self.device)
        with torch.no_grad():
            sentence_embedding = self.clip_model.encode_text(text_input_processed)

        sentence_embedding_normalized = (
            (sentence_embedding / sentence_embedding.norm(dim=-1, keepdim=True))
            .float()
            .cpu()
        )
        return sentence_embedding_normalized.squeeze().numpy()

    def compute_similarity_scores(self, mask_features, text_query):
        text_emb = self.get_query_embedding(text_query)

        scores = np.zeros(len(mask_features))
        for mask_idx, mask_emb in enumerate(mask_features):
            mask_norm = np.linalg.norm(mask_emb)
            if mask_norm < 0.001:
                continue
            normalized_emb = mask_emb / mask_norm
            scores[mask_idx] = normalized_emb @ text_emb

        return scores

    def get_per_point_colors_for_similarity(
        self,
        per_mask_scores,
        masks,
        normalize_based_on_current_min_max=True,
        normalize_min_bound=0.16,  # only used for visualization if normalize_based_on_current_min_max is False
        normalize_max_bound=0.26,  # only used for visualization if normalize_based_on_current_min_max is False
        background_color=(0.77, 0.77, 0.77),
    ):
        # get colors based on the openmask3d per mask scores
        non_zero_points = per_mask_scores != 0
        openmask3d_per_mask_scores_rescaled = np.zeros_like(per_mask_scores)
        pms = per_mask_scores[non_zero_points]

        # in order to be able to visualize the score differences better, we can use a normalization scheme
        if (
            normalize_based_on_current_min_max
        ):  # if true, normalize the scores based on the min. and max. scores for this scene
            openmask3d_per_mask_scores_rescaled[non_zero_points] = (pms - pms.min()) / (
                pms.max() - pms.min()
            )
        else:  # if false, normalize the scores based on a pre-defined color scheme with min and max clipping bounds, normalize_min_bound and normalize_max_bound.
            new_scores = np.zeros_like(openmask3d_per_mask_scores_rescaled)
            new_indices = np.zeros_like(non_zero_points)
            new_indices[non_zero_points] += pms > normalize_min_bound
            new_scores[new_indices] = (
                pms[pms > normalize_min_bound] - normalize_min_bound
            ) / (normalize_max_bound - normalize_min_bound)
            openmask3d_per_mask_scores_rescaled = new_scores

        new_colors = np.ones((masks.shape[1], 3)) * 0 + background_color

        for mask_idx, mask in enumerate(masks[::-1, :]):
            # colors = [(0, 0, 1), (0, 1, 0), (1, 1, 0)]  # Blue, Green, Yellow
            # similarity_levels = [0, 0.5, 1]  # Corresponding similarity levels
            # cmap_name = "custom_similarity"
            # cmap = LinearSegmentedColormap.from_list(
            #     cmap_name, list(zip(similarity_levels, colors))
            # )
            # get color from matplotlib colormap
            new_colors[mask > 0.5, :] = plt.cm.viridis(
                openmask3d_per_mask_scores_rescaled[len(masks) - mask_idx - 1]
            )[:3]
            # new_colors[mask > 0.5, :] = cmap(
            #     openmask3d_per_mask_scores_rescaled[len(masks) - mask_idx - 1]
            # )[:3]

        return new_colors


def get_points_from_mask(mask):
    points = np.where(mask)
    points = np.stack((points[0], points[1]), axis=-1)
    return points


def img2worldtf(img_points: np.ndarray, pcd_tf_info: dict) -> dict:
    """
    Given a set of 2D points in the topdown/density map images, return the corresponding 3D points in the pointcloud.

    Parameters:
    img_points: np.ndarray of shape (N, 2)
    pcd_tf_info: dict with keys: "x_offset", "y_offset", "x_scale", "max_ext", and "tile_size"

    Returns:
    dict with keys
    world_points: np.ndarray of shape (N, 2)
    z_spread: np.ndarray of shape (2,) with min and max z values
    """

    # first multiply by the tile size
    world_points = img_points * pcd_tf_info["tile_size"]

    # then add the offset
    world_points[:, 0] += pcd_tf_info["x_offset"]
    world_points[:, 1] += pcd_tf_info["y_offset"]
    world_points = world_points * -1

    z_spread = np.array([pcd_tf_info["floor_height"], pcd_tf_info["ceiling_height"]])
    z_spread += pcd_tf_info["z_offset"]
    z_spread = z_spread * -1

    return {"world_points": world_points, "z_spread": z_spread}


def get_kd_tree(room_masks):
    # Create a KD-tree for each room
    kd_trees = {}
    for room_id, (x, y) in room_masks.items():
        points = np.stack((x, y), axis=-1)
        kd_trees[room_id] = cKDTree(points)

    return kd_trees


def save_openmask3d_room_mesh(
    path_scene_pcd,
    room_kd_trees,
    similarity_colors,
    room_masks_to_openmask3d_features,
    save_dir,
    query_text,
):
    mesh = o3d.io.read_triangle_mesh(path_scene_pcd)

    # If the mesh doesn't have vertex colors, initialize them to white for visualization purposes
    if not mesh.has_vertex_colors():
        mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.ones((np.asarray(mesh.vertices).shape[0], 3))
        )

    # Convert mesh vertices to numpy array for processing
    vertices = np.asarray(mesh.vertices)

    # This will hold the new colors for each point based on room similarity
    new_colors = np.zeros((vertices.shape[0], 3))

    for i, vertex in enumerate(tqdm(vertices)):
        # For each vertex, find the nearest room mask index using KD-trees
        min_distance = float("inf")
        closest_room_idx = None
        for room_idx, kd_tree in room_kd_trees.items():
            distance, _ = kd_tree.query(vertex[:2], k=1)  # k=1 for the closest point
            if distance < min_distance:
                min_distance = distance
                closest_room_idx = room_idx

        # Use the closest_room_idx to get the corresponding color
        if closest_room_idx is not None:
            color_idx = list(room_masks_to_openmask3d_features.keys()).index(
                closest_room_idx
            )
            new_colors[i] = similarity_colors[color_idx]

    # Update the mesh vertex colors
    mesh.vertex_colors = o3d.utility.Vector3dVector(new_colors)

    # Save or display the colored mesh
    o3d.io.write_triangle_mesh(
        os.path.join(save_dir, f"avg_embed_{query_text}.ply"), mesh
    )  # To save the mesh

    return


def main():
    # # Room Masks
    # with open("openmask3d/configs/config_mp3d_eval.yaml", "r") as f:
    #     config = yaml.safe_load(f)

    # val_dataset = MP3DEval(**config["eval_dataset"])

    # val_loader = DataLoader(
    #     val_dataset, collate_fn=collate_fn, **config["eval_dataloader"]
    # )
    # X_scene_name = None
    # X = None
    # y = None
    # for i, (X, y) in enumerate(val_loader):
    #     X = X[0]  # batch size 1
    #     y = y[0]

    #     X_scene_name = X.split("/")[-1]
    #     print(X_scene_name)
    #     if i == 0:
    #         break

    # room_masks = y["masks"].numpy()
    # room_tf_masks = {}
    # # z_spread = np.zeros((2,))

    # for idx, mask in enumerate(room_masks):
    #     points = get_points_from_mask(mask)
    #     # points = points[np.random.choice(points.shape[0], 1000, replace=False)]

    #     pcd_tf_info = y["pcd_tf_info"]
    #     world_data = img2worldtf(points, pcd_tf_info)

    #     world_points = world_data["world_points"]
    #     world_points = (world_points[:, 0], world_points[:, 1])

    #     # z_spread = world_data["z_spread"]

    #     room_tf_masks[idx] = world_points

    # room_kd_trees = get_kd_tree(room_tf_masks)

    # --------------------------------
    # Set the paths
    # --------------------------------
    path_dataset = "/scratch/kumaraditya_gupta/Datasets/arkitscenes/ChallengeDevelopmentSet/42445173"
    path_scene_pcd = os.path.join(path_dataset, "42445173_3dod_mesh.ply")
    path_pred_masks = os.path.join(
        path_dataset,
        "output/2024-06-08-18-36-51-experiment/42445173_3dod_mesh_masks.pt",
    )
    path_openmask3d_features = os.path.join(
        path_dataset,
        "output/2024-06-08-18-36-51-experiment/42445173_3dod_mesh_openmask3d_features.npy",
    )
    path_save_pcd = os.path.join(path_dataset, "output/2024-06-08-18-36-51-experiment/")
    QUERIES = ["cushion with an alpaca print"]

    # --------------------------------
    # Load data
    # --------------------------------
    # load the scene pcd
    scene_pcd = o3d.io.read_point_cloud(path_scene_pcd)
    scene_points = np.asarray(scene_pcd.points)

    # # load the predicted masks
    pred_masks = np.asarray(
        torch.load(path_pred_masks)
    ).T  # (num_instances, num_points)

    # # get the centroid of each mask
    # centroids = np.zeros((pred_masks.shape[0], 3))
    # for i, mask in enumerate(pred_masks):
    #     # pred_masks (num_masks, num_points); scene_points (num_points, 3)
    #     mask_points = scene_points[mask > 0.5]
    #     centroids[i] = np.mean(mask_points, axis=0)

    # room_masks_to_centroid = {}
    # for centroid_idx, centroid in enumerate(centroids):
    #     min_room_dist = float("inf")
    #     min_room_id = -1

    #     for room_id, tree in room_kd_trees.items():
    #         dist, idx = tree.query(centroid[:2])
    #         if dist < min_room_dist:
    #             min_room_dist = dist
    #             min_room_id = room_id

    #     if min_room_id not in room_masks_to_centroid:
    #         room_masks_to_centroid[min_room_id] = []
    #     room_masks_to_centroid[min_room_id].append(centroid_idx)

    # load the openmask3d features
    openmask3d_features = np.load(path_openmask3d_features)  # (num_instances, 768)

    # room_masks_to_openmask3d_features = {}
    # for room_id, mask_indices in room_masks_to_centroid.items():
    #     all_room_features = openmask3d_features[mask_indices]
    #     room_masks_to_openmask3d_features[room_id] = all_room_features.mean(axis=0)
    # room_masks_to_openmask3d_features = dict(
    #     sorted(room_masks_to_openmask3d_features.items())
    # )

    # room_masks_to_openmask3d_features_np = np.array(
    #     list(room_masks_to_openmask3d_features.values())
    # )

    # initialize the query similarity computer
    query_similarity_computer = QuerySimilarityComputation()

    # --------------------------------
    # Set the query text
    # --------------------------------
    for query_text in QUERIES:
        # --------------------------------
        # Get the similarity scores
        # --------------------------------
        # get the per mask similarity scores, i.e. the cosine similarity between the query embedding and each openmask3d mask-feature for each object instance
        per_mask_query_sim_scores = query_similarity_computer.compute_similarity_scores(
            openmask3d_features, query_text
        )

        # avg_embed_sim_scores = query_similarity_computer.compute_similarity_scores(
        #     room_masks_to_openmask3d_features_np, query_text
        # )
        # normalized_avg_embed_sim_scores = (
        #     avg_embed_sim_scores - avg_embed_sim_scores.min()
        # ) / (avg_embed_sim_scores.max() - avg_embed_sim_scores.min())
        # cmap = matplotlib.colormaps.get_cmap("viridis")
        # similarity_colors = cmap(normalized_avg_embed_sim_scores)[..., :3]

        # save_dir = "/scratch/kumaraditya_gupta/Datasets/openmask3d/2t7WUuJeko7/output/2024-03-05-16-18-47-experiment"
        # save_openmask3d_room_mesh(
        #     path_scene_pcd,
        #     room_kd_trees,
        #     similarity_colors,
        #     room_masks_to_openmask3d_features,
        #     save_dir,
        #     query_text,
        # )

        # --------------------------------
        # Visualize the similarity scores
        # --------------------------------
        # get the per-point heatmap colors for the similarity scores
        per_point_similarity_colors = query_similarity_computer.get_per_point_colors_for_similarity(
            per_mask_query_sim_scores, pred_masks
        )  # note: for normalizing the similarity heatmap colors for better clarity, you can check the arguments for the function get_per_point_colors_for_similarity

        # visualize the scene with the similarity heatmap
        scene_pcd_w_sim_colors = o3d.geometry.PointCloud()
        scene_pcd_w_sim_colors.points = scene_pcd.points
        scene_pcd_w_sim_colors.colors = o3d.utility.Vector3dVector(
            per_point_similarity_colors
        )
        scene_pcd_w_sim_colors.estimate_normals()
        # o3d.visualization.draw_geometries([scene_pcd_w_sim_colors])
        # alternatively, you can save the scene_pcd_w_sim_colors as a .ply file
        o3d.io.write_point_cloud(
            os.path.join(path_save_pcd, f"pcd_{'_'.join(query_text.split(' '))}.ply"),
            scene_pcd_w_sim_colors,
        )
        # o3d.io.write_point_cloud(
        #     "/scratch/kumaraditya_gupta/Datasets/openmask3d/RPmz2sHmrrY/output/2024-03-12-15-59-17-experiment/scene_pcd_w_sim_colors_{}.ply".format(
        #         "_".join(query_text.split(" "))
        #     ),
        #     scene_pcd_w_sim_colors,
        # )


if __name__ == "__main__":
    main()
