
from pathlib import Path

import pandas as pd

from .common import GraspDataset


class DexNetDataset(GraspDataset):
    shift_probabilities = {"rgb": 0.97, "depth": 0.9, "rgbd": 0.9}
    rgb_shift_probability = 0.9

    def rgb_path(self, row: pd.Series) -> Path:
        return self.data_root / "rgbs" / str(row["img_rgb"])

    def depth_path(self, row: pd.Series) -> Path:
        return self.data_root / "depths" / str(row["img_depth"])
