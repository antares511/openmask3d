import numpy as np
import pickle
import os
import cv2

import yaml

from scipy.spatial import cKDTree
from tqdm import tqdm
import open3d as o3d

import pandas as pd
import random
import skimage
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json
from PIL import Image
from pathlib import Path
import torch


class MP3DEval(Dataset):
    def __init__(
        self,
        img_folder: Path,  # $DATASET_ROOT / "train"  / "dmaps" for example
        room_masks_folder: Path,  # $DATASET_ROOT / "train" /"room_mask_instances" for example
        room_id_json: Path,  # $DATASET_ROOT / "room_id.json" for example
    ):
        super(Dataset, self).__init__()
        self.img_folder = img_folder
        self.room_masks_folder = room_masks_folder

        with open(room_id_json) as f:
            self.image_regions_dict = json.load(f)

        self.ids = list(sorted(os.listdir(self.img_folder)))

        self.mask_transforms = transforms.Compose([transforms.ToTensor()])

    def get_data(self, img_id):
        img_path = img_id
        img_name = img_path.split(".")[0]
        input = os.path.join(self.img_folder, img_path)

        current_room_masks_folder = os.path.join(
            self.room_masks_folder, img_path.split(".")[-2]
        )
        room_mask_instances = os.listdir(current_room_masks_folder)
        room_mask_instances = [
            os.path.join(current_room_masks_folder, room_mask_instance)
            for room_mask_instance in room_mask_instances
        ]

        masks = torch.zeros((len(room_mask_instances), 512, 512))
        boxes = torch.zeros([len(room_mask_instances), 4], dtype=torch.float32)
        labels = torch.zeros(len(room_mask_instances), dtype=torch.int64)

        for i, room_mask_instance in enumerate(room_mask_instances):
            region_id = room_mask_instance.split("/")[-1].split(".")[0]

            mask = cv2.imread(room_mask_instance, cv2.IMREAD_GRAYSCALE)

            (x, y, w, h) = cv2.boundingRect(mask)

            masks[i] = self.mask_transforms(Image.fromarray(mask))
            boxes[i] = torch.tensor([x, y, x + w, y + h])
            # our room label id from FINAL_DICT
            labels[i] = self.image_regions_dict[img_name]["region_id"][region_id][
                "our_room_id"
            ]

        pcd_tf_info = self.image_regions_dict[img_name]["pcd_tf_info"]

        gt_room_labels = {
            "pcd_tf_info": pcd_tf_info,
            "masks": masks,
            "boxes": boxes,
            "labels": labels,
        }

        return input, gt_room_labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return self.get_data(self.ids[index])


def collate_fn(batch):
    image_names, targets = list(zip(*batch))
    image_names = list(image_names)
    return image_names, targets
