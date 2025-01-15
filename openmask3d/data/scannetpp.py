import numpy as np
from PIL import Image
import imageio
import math
import os
from path import Path
from scipy.spatial.transform import Rotation
from pathlib import Path


def get_number_of_images(poses_path):
    i = 0
    while os.path.isfile(os.path.join(poses_path, str(i) + ".txt")):
        i += 1
    return i


def invert_se3(se3_matrix: np.ndarray) -> np.ndarray:
    rotation_inv = se3_matrix[:3, :3].T
    translation_inv = -rotation_inv @ se3_matrix[:3, 3]

    inverted_se3_matrix = np.eye(4)
    inverted_se3_matrix[:3, :3] = rotation_inv
    inverted_se3_matrix[:3, 3] = translation_inv

    return inverted_se3_matrix


class COLMAP_image_txt:
    def __init__(self, path):
        self.file = open(str(path), "r")

    def __iter__(self):
        return self

    def __next__(self):
        line = next(self.file).split()
        while not len(line) or line[0] == "#":
            line = next(self.file).split()
        return line


class Camera:
    def __init__(
        self,
        data_path,
        scene,
        fx,
        fy,
        cx,
        cy,
        intrinsic_resolution,
        depth_scale,
        stride,
    ):
        self.intrinsic = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        self.intrinsic_original_resolution = intrinsic_resolution

        self.data_path = data_path
        self.scene = scene

        self.poses = self.get_se3_poses()
        self.indices = np.arange(0, len(self.poses), step=stride)  # Need to keep this
        self.poses = [self.poses[i] for i in self.indices]

        self.depth_paths = self.get_depth_paths()
        self.depth_scale = depth_scale
        self.depths = [self.load_depth(idx) for idx in self.indices]

        self.height = self.depths[0].shape[0]
        self.width = self.depths[0].shape[1]

    def get_depth_paths(self):
        depth_paths = []
        lines = COLMAP_image_txt(
            Path(self.data_path)
            / "data"
            / self.scene
            / "iphone"
            / "colmap"
            / "images.txt"
        )
        for line in lines:
            file_name = line[-1].replace("jpg", "png")
            depth_paths.append(
                Path(self.data_path)
                / "data"
                / self.scene
                / "iphone"
                / "depth"
                / file_name
            )

        return depth_paths

    def load_depth(self, idx):
        depth_path = str(self.depth_paths[idx])
        sensor_depth = imageio.v2.imread(depth_path) / self.depth_scale
        return sensor_depth

    def get_se3_poses(self):
        se3_poses = []
        lines = COLMAP_image_txt(
            Path(self.data_path)
            / "data"
            / self.scene
            / "iphone"
            / "colmap"
            / "images.txt"
        )
        for line in lines:
            quat = np.array(line[1:5]).astype(float)  # (qw, qx, qy, qz)
            quat = np.array([quat[1], quat[2], quat[3], quat[0]])  # (qx, qy, qz, qw)
            translation = np.array(line[5:8]).astype(float)
            se3 = np.eye(4)
            se3[:3, :3] = Rotation.from_quat(quat).as_matrix()
            se3[:3, 3] = translation
            # Append 3x4 pose matrix
            se3_poses.append(se3[:3, :])

        return se3_poses

    def get_adapted_intrinsic(self, desired_resolution):
        """Get adjusted camera intrinsics."""
        if self.intrinsic_original_resolution == desired_resolution:
            return self.intrinsic

        resize_width = int(
            math.floor(
                desired_resolution[1]
                * float(self.intrinsic_original_resolution[0])
                / float(self.intrinsic_original_resolution[1])
            )
        )

        adapted_intrinsic = self.intrinsic.copy()
        adapted_intrinsic[0, 0] *= float(resize_width) / float(
            self.intrinsic_original_resolution[0]
        )
        adapted_intrinsic[1, 1] *= float(desired_resolution[1]) / float(
            self.intrinsic_original_resolution[1]
        )
        adapted_intrinsic[0, 2] *= float(desired_resolution[0] - 1) / float(
            self.intrinsic_original_resolution[0] - 1
        )
        adapted_intrinsic[1, 2] *= float(desired_resolution[1] - 1) / float(
            self.intrinsic_original_resolution[1] - 1
        )
        return adapted_intrinsic


class Images:
    def __init__(self, data_path, scene, indices, height, width):
        self.data_path = data_path
        self.scene = scene
        self.indices = indices
        self.height = height
        self.width = width

        self.rgb_paths = self.get_rgb_paths()
        self.images = [self.load_images(idx) for idx in self.indices]

    def get_rgb_paths(self):
        rgb_paths = []
        lines = COLMAP_image_txt(
            Path(self.data_path)
            / "data"
            / self.scene
            / "iphone"
            / "colmap"
            / "images.txt"
        )
        for line in lines:
            rgb_paths.append(
                Path(self.data_path) / "data" / self.scene / "iphone" / "rgb" / line[-1]
            )

        return rgb_paths

    def load_images(self, idx):
        img_path = str(self.rgb_paths[idx])
        img = Image.open(img_path).convert("RGB")

        if img.size != (self.width, self.height):
            img = img.resize((self.width, self.height), Image.Resampling.BICUBIC)

        return img

    def get_as_np_list(self):
        images = []
        for i in range(len(self.images)):
            images.append(np.asarray(self.images[i]))
        return images
