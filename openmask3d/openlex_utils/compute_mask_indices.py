import hydra
from omegaconf import DictConfig
import numpy as np
import torch
import os


@hydra.main(config_path="../configs", config_name="openmask3d_inference")
def main(ctx: DictConfig):
    path_pred_masks = ctx.data.masks.masks_path
    pred_masks = np.asarray(
        torch.load(path_pred_masks)
    ).T  # (num_instances, num_points)

    pred_masks = pred_masks.astype(int)

    # Find the index of the first mask each point appears in
    pred_mask_indices = np.argmax(pred_masks, axis=0)  # (num_points,)

    # Set the index to -1 if the point is not in any mask
    pred_mask_indices[~np.any(pred_masks, axis=0)] = -1

    np.save(
        ctx.output.output_directory + "/mesh_mask_indices.npy",
        pred_mask_indices,
    )


if __name__ == "__main__":
    main()
