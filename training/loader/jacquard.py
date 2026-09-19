
from pathlib import Path

import pandas as pd

from .common import GraspDataset


class JacquardDataset(GraspDataset):
    shift_probabilities = {"rgb": 0.9, "depth": 0.9, "rgbd": 0.9}
    rgb_shift_probability = 0.8

    def rgb_path(self, row: pd.Series) -> Path:
        return self.data_root / "rgbs" / str(row["img_rgb"])

    def depth_path(self, row: pd.Series) -> Path:
        name = str(row["img_depth"])
        direct_path = self.data_root / "depths" / name
        if direct_path.is_file():
            return direct_path
        stereo_name = name.replace("_depth.tiff", "_stereo_depth.tiff")
        return self.data_root / "depths" / stereo_name
