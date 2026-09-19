
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Type

import numpy as np
import pandas as pd

from .common import GraspDataset, grasp_columns
from .dexnet import DexNetDataset
from .jacquard import JacquardDataset


@dataclass
class DatasetBundle:
    train: GraspDataset
    validation: GraspDataset
    csv_rows: int
    usable_rows: int


@dataclass
class TestDatasetBundle:
    dataset: GraspDataset
    csv_rows: int
    usable_rows: int


def _dataset_class(dataset: str) -> Type[GraspDataset]:
    if dataset == "jacquard":
        return JacquardDataset
    if dataset == "dexnet":
        return DexNetDataset
    raise ValueError(f"Unsupported dataset: {dataset}")


def _read_and_filter(
    dataset_class: Type[GraspDataset],
    csv_path: Path,
    data_root: Path,
    modality: str,
    target: str,
) -> tuple:
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")
    frame = pd.read_csv(csv_path)
    required = {"img_rgb", "img_depth", "gradient_name", "CX1", "CY1", "Theta1", "W1"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

    if not grasp_columns(frame.columns):
        raise ValueError("No CX/CY/Theta/W grasp columns were found in the CSV")

    data_root = Path(data_root)

    def relative_files(folder: str) -> set:
        base = data_root / folder
        if not base.is_dir():
            return set()
        return {
            path.relative_to(base).as_posix()
            for path in base.rglob("*")
            if path.is_file()
        }

    valid_mask = np.ones(len(frame), dtype=bool)
    if modality in {"rgb", "rgbd"}:
        rgb_names = relative_files("rgbs")
        valid_mask &= frame["img_rgb"].astype(str).isin(rgb_names).to_numpy()
    if modality in {"depth", "rgbd"}:
        depth_names = relative_files("depths")
        if dataset_class is JacquardDataset:
            depth_names.update(
                name.replace("_stereo_depth.tiff", "_depth.tiff") for name in tuple(depth_names)
            )
        valid_mask &= frame["img_depth"].astype(str).isin(depth_names).to_numpy()
    if target == "skeleton":
        gradient_names = relative_files("gradients")
        valid_mask &= frame["gradient_name"].astype(str).isin(gradient_names).to_numpy()
    return frame.loc[valid_mask].reset_index(drop=True), len(frame)


def build_datasets(
    dataset: str,
    data_root: Path,
    train_csv: Path,
    modality: str,
    target: str,
    val_csv: Optional[Path] = None,
    val_split: float = 0.3,
    seed: int = 42,
    augment: bool = True,
) -> DatasetBundle:
    dataset_class = _dataset_class(dataset)
    records, csv_rows = _read_and_filter(
        dataset_class, Path(train_csv), Path(data_root), modality, target
    )
    if records.empty:
        raise RuntimeError("No CSV rows have all files required by this configuration")
    train_usable_rows = len(records)

    if val_csv is not None:
        train_records = records
        validation_records, validation_csv_rows = _read_and_filter(
            dataset_class, Path(val_csv), Path(data_root), modality, target
        )
        csv_rows += validation_csv_rows
        usable_rows = train_usable_rows + len(validation_records)
    else:
        if not 0.0 < val_split < 1.0:
            raise ValueError("val_split must be between 0 and 1")
        if len(records) < 2:
            raise RuntimeError("At least two usable samples are required for a split")
        indices = np.random.default_rng(seed).permutation(len(records))
        validation_size = max(1, int(round(len(records) * val_split)))
        validation_indices = indices[:validation_size]
        train_indices = indices[validation_size:]
        if len(train_indices) == 0:
            raise RuntimeError("The validation split left no training samples")
        train_records = records.iloc[train_indices].reset_index(drop=True)
        validation_records = records.iloc[validation_indices].reset_index(drop=True)
        usable_rows = train_usable_rows

    train_dataset = dataset_class(
        data_root, train_records, modality, target, augment=augment
    )
    validation_dataset = dataset_class(
        data_root, validation_records, modality, target, augment=augment
    )
    return DatasetBundle(train_dataset, validation_dataset, csv_rows, usable_rows)


def build_test_dataset(
    dataset: str,
    data_root: Path,
    test_csv: Path,
    modality: str,
    target: str,
) -> TestDatasetBundle:
    dataset_class = _dataset_class(dataset)
    records, csv_rows = _read_and_filter(
        dataset_class, Path(test_csv), Path(data_root), modality, target
    )
    if records.empty:
        raise RuntimeError("No test CSV rows have all files required by this configuration")
    test_dataset = dataset_class(
        data_root, records, modality, target, augment=False
    )
    return TestDatasetBundle(test_dataset, csv_rows, len(records))
