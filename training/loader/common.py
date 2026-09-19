
import math
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Sequence, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


MODALITY_CHANNELS = {"rgb": 3, "depth": 1, "rgbd": 4}
VALID_TARGETS = {"skeleton", "width"}


def min_max_normalize(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    minimum = float(np.nanmin(array))
    maximum = float(np.nanmax(array))
    return np.nan_to_num((array - minimum) / (maximum - minimum + 1e-8))


def grasp_columns(columns: Sequence[str]) -> List[int]:
    pattern = re.compile(r"^CX(\d+)$")
    indices = []
    column_set = set(columns)
    for column in columns:
        match = pattern.match(column)
        if not match:
            continue
        index = int(match.group(1))
        if all(f"{prefix}{index}" in column_set for prefix in ("CX", "CY", "Theta", "W")):
            indices.append(index)
    return sorted(indices)


def generate_grasp_maps(
    row: pd.Series,
    indices: Sequence[int],
    shape: Tuple[int, int],
    grasp_height: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create the quality, cos(2 theta), sin(2 theta), and width maps."""
    quality_map = np.zeros(shape, dtype=np.float32)
    cosine_map = np.zeros(shape, dtype=np.float32)
    sine_map = np.zeros(shape, dtype=np.float32)
    width_map = np.zeros(shape, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(
        np.arange(shape[1], dtype=np.float32),
        np.arange(shape[0], dtype=np.float32),
    )

    for index in indices:
        cx = row[f"CX{index}"]
        if pd.isna(cx):
            break
        cy = float(row[f"CY{index}"])
        theta = float(row[f"Theta{index}"])
        width = float(row[f"W{index}"])
        cx = float(cx)
        if not all(math.isfinite(value) for value in (cx, cy, theta, width)) or width <= 0:
            continue

        theta_rad = math.radians(theta)
        centered_x = grid_x - cx
        centered_y = grid_y - cy
        rotated_x = centered_x * math.cos(theta_rad) + centered_y * math.sin(theta_rad) + cx
        rotated_y = -centered_x * math.sin(theta_rad) + centered_y * math.cos(theta_rad) + cy

        mask = (
            (rotated_x >= cx - width / 2.0)
            & (rotated_x <= cx + width / 2.0)
            & (rotated_y >= cy - grasp_height / 2.0)
            & (rotated_y <= cy + grasp_height / 2.0)
        )
        sigma = max(width / 2.0, 1e-8)
        gaussian = np.exp(-((np.abs(rotated_x - cx) ** 2) / (2.0 * sigma**2)))
        quality_map[mask] = gaussian[mask]

        doubled_angle = math.radians(2.0 * theta)
        cosine_map[mask] = math.cos(doubled_angle)
        sine_map[mask] = math.sin(doubled_angle)
        width_map[mask] = width / 100.0

    return quality_map, cosine_map, sine_map, width_map


class GraspDataset(Dataset, ABC):
    """Base loader shared by Jacquard and Dex-Net."""

    shift_probabilities = {"rgb": 0.9, "depth": 0.9, "rgbd": 0.9}
    rgb_shift_probability = 0.9

    def __init__(
        self,
        data_root: Path,
        records: pd.DataFrame,
        modality: str,
        target: str,
        augment: bool = False,
    ):
        if modality not in MODALITY_CHANNELS:
            raise ValueError(f"Unsupported modality: {modality}")
        if target not in VALID_TARGETS:
            raise ValueError(f"Unsupported target: {target}")

        self.data_root = Path(data_root)
        self.records = records.reset_index(drop=True)
        self.modality = modality
        self.target = target
        self.augment = augment
        self.grasp_indices = grasp_columns(self.records.columns)
        if not self.grasp_indices:
            raise ValueError("No CX/CY/Theta/W grasp columns were found in the CSV")

        additional_targets = {
            "image": "image",
            "qual": "image",
            "cos": "image",
            "sin": "image",
            self.target: "image",
        }
        if self.modality == "rgbd":
            additional_targets["depth"] = "image"

        self.shift_transform = A.Compose(
            [
                A.ShiftScaleRotate(
                    shift_limit=0.2,
                    scale_limit=0,
                    rotate_limit=0,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=self.shift_probabilities[self.modality],
                ),
                ToTensorV2(),
            ],
            additional_targets=additional_targets,
        )
        self.tensor_transform = A.Compose(
            [ToTensorV2()], additional_targets=additional_targets
        )

        if self.modality == "rgb":
            self.color_transform = A.Compose(
                [
                    A.RGBShift(
                        r_shift_limit=(0, 255),
                        g_shift_limit=(0, 255),
                        b_shift_limit=(0, 255),
                        p=self.rgb_shift_probability,
                    )
                ]
            )
        elif self.modality == "rgbd":
            self.color_transform = A.Compose(
                [A.RGBShift(p=1), A.Blur(p=1), A.GaussNoise(p=1)]
            )
        else:
            self.color_transform = None

    def __len__(self) -> int:
        return len(self.records)

    @abstractmethod
    def rgb_path(self, row: pd.Series) -> Path:
        raise NotImplementedError

    @abstractmethod
    def depth_path(self, row: pd.Series) -> Path:
        raise NotImplementedError

    def skeleton_path(self, row: pd.Series) -> Path:
        return self.data_root / "gradients" / str(row["gradient_name"])

    def paths_exist(self, row: pd.Series) -> bool:
        if self.modality in {"rgb", "rgbd"} and not self.rgb_path(row).is_file():
            return False
        if self.modality in {"depth", "rgbd"} and not self.depth_path(row).is_file():
            return False
        if self.target == "skeleton" and not self.skeleton_path(row).is_file():
            return False
        return True

    def _load_rgb(self, row: pd.Series) -> np.ndarray:
        image = cv2.imread(str(self.rgb_path(row)), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Could not read RGB image: {self.rgb_path(row)}")
        return image

    def _load_depth(self, row: pd.Series) -> np.ndarray:
        path = self.depth_path(row)
        if path.suffix.lower() == ".npy":
            image = np.load(path)
        else:
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise OSError(f"Could not read depth image: {path}")
        image = np.asarray(image).squeeze()
        if image.ndim != 2:
            raise ValueError(f"Expected a 2-D depth image at {path}, got shape {image.shape}")
        return min_max_normalize(image)

    def __getitem__(self, index: int):
        row = self.records.iloc[index]
        rgb = self._load_rgb(row) if self.modality in {"rgb", "rgbd"} else None
        depth = self._load_depth(row) if self.modality in {"depth", "rgbd"} else None
        reference = rgb if rgb is not None else depth
        if reference is None:
            raise RuntimeError("No input image was loaded")
        shape = reference.shape[:2]

        quality, cosine, sine, width = generate_grasp_maps(
            row, self.grasp_indices, shape
        )
        if self.target == "skeleton":
            selected_target = np.asarray(np.load(self.skeleton_path(row))).squeeze()
            if selected_target.shape != shape:
                raise ValueError(
                    f"Skeleton shape {selected_target.shape} does not match image shape {shape}"
                )
        else:
            selected_target = width

        if rgb is not None:
            if self.augment and self.color_transform is not None:
                rgb = self.color_transform(image=rgb)["image"]
            rgb = rgb.astype(float) / 255.0

        transform_data = {
            "image": depth if self.modality == "depth" else rgb,
            "qual": quality,
            "cos": cosine,
            "sin": sine,
            self.target: selected_target,
        }
        if self.modality == "rgbd":
            transform_data["depth"] = depth

        transform = self.shift_transform if self.augment else self.tensor_transform
        transformed = transform(**transform_data)

        if self.modality == "rgbd":
            input_tensor = torch.cat((transformed["image"], transformed["depth"]), dim=0)
        else:
            input_tensor = transformed["image"]

        target_tensors = (
            transformed["qual"],
            transformed["cos"],
            transformed["sin"],
            transformed[self.target],
        )
        return input_tensor, target_tensors
