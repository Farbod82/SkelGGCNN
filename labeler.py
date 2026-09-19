
from __future__ import annotations

import argparse
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Collection, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from keypoint_generator import KeypointGeneration
from keypoint_utils import calculate_angle, distance


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEPTH_EXTENSIONS = {".npy", ".png", ".tif", ".tiff", ".exr"}


@dataclass(frozen=True)
class InputSample:
    """The three source files belonging to one sample."""

    mesh_name: str
    stem: str
    mask: Path
    rgb: Path
    depth: Path


def _index_files(folder: Path, extensions: Collection[str]) -> Dict[str, Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Required input folder does not exist: {folder}")

    indexed: Dict[str, Path] = {}
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        if path.stem in indexed:
            raise ValueError(
                f"Duplicate filename stem '{path.stem}' in {folder}: "
                f"{indexed[path.stem].name} and {path.name}"
            )
        indexed[path.stem] = path
    return indexed


def collect_input_samples(input_dir: Path) -> List[InputSample]:
    """Read collector mesh folders and pair files by identical stem."""

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")

    mesh_dirs = sorted(path for path in input_dir.iterdir() if path.is_dir())
    if not mesh_dirs:
        raise ValueError(f"No mesh folders found in {input_dir}")

    samples: List[InputSample] = []
    for mesh_dir in mesh_dirs:
        masks = _index_files(mesh_dir / "mask", IMAGE_EXTENSIONS)
        rgbs = _index_files(mesh_dir / "rgb", IMAGE_EXTENSIONS)
        depths = _index_files(mesh_dir / "depth", DEPTH_EXTENSIONS)

        if not masks:
            raise ValueError(f"No mask images found in {mesh_dir / 'mask'}")

        missing_rgb = sorted(masks.keys() - rgbs.keys())
        missing_depth = sorted(masks.keys() - depths.keys())
        if missing_rgb or missing_depth:
            messages = []
            if missing_rgb:
                messages.append("missing RGB files for: " + ", ".join(missing_rgb))
            if missing_depth:
                messages.append("missing depth files for: " + ", ".join(missing_depth))
            raise ValueError(
                f"Files in '{mesh_dir.name}' must share the same stem; "
                + "; ".join(messages)
            )

        samples.extend(
            InputSample(
                mesh_name=mesh_dir.name,
                stem=stem,
                mask=masks[stem],
                rgb=rgbs[stem],
                depth=depths[stem],
            )
            for stem in sorted(masks)
        )

    return samples


def _image_shape(path: Path, label: str) -> Tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read {label} file: {path}")
    return image.shape[:2]


def _depth_shape(path: Path) -> Tuple[int, int]:
    if path.suffix.lower() == ".npy":
        depth = np.load(path, mmap_mode="r")
        shape = depth.shape
    else:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"Could not read depth file: {path}")
        shape = depth.shape

    if len(shape) == 2:
        return int(shape[0]), int(shape[1])
    if len(shape) == 3 and shape[2] == 1:
        return int(shape[0]), int(shape[1])
    raise ValueError(f"Depth must have shape (H, W) or (H, W, 1), got {shape}: {path}")


def _validate_shapes(sample: InputSample) -> None:
    mask_shape = _image_shape(sample.mask, "mask")
    rgb_shape = _image_shape(sample.rgb, "RGB")
    depth_shape = _depth_shape(sample.depth)
    if mask_shape != rgb_shape or mask_shape != depth_shape:
        raise ValueError(
            f"Spatial shapes do not match for '{sample.stem}': "
            f"mask={mask_shape}, RGB={rgb_shape}, depth={depth_shape}"
        )


def _grasp_values(p1, p2) -> Tuple[float, float, float, float]:
    values = (
        (float(p1[0]) + float(p2[0])) / 2.0,
        (float(p1[1]) + float(p2[1])) / 2.0,
        float(calculate_angle(p1, p2)),
        float(distance(p1, p2)),
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Straight-skeleton algorithm returned a non-finite grasp: {values}")
    return values


class DatasetBuilder2D:
    """Run the existing 2D algorithm and write trainer-compatible files."""

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        output_csv: Path,
        generate_gradient: bool = False,
        overwrite: bool = False,
        mesh: str = None,
        sample: str = None,
        save_visualizations: bool = False,
    ) -> None:
        self.input_dir = input_dir.resolve()
        self.output_dir = output_dir.resolve()
        self.output_csv = output_csv.resolve()
        self.generate_gradient = generate_gradient
        self.overwrite = overwrite
        self.mesh = mesh
        self.sample = sample
        self.save_visualizations = save_visualizations

        if sample is not None and mesh is None:
            raise ValueError("--sample requires --mesh to identify the object")

        if self.input_dir == self.output_dir:
            raise ValueError("--input-dir and --output-dir must be different folders")

    def _output_paths(self, sample: InputSample) -> Dict[str, Path]:
        paths = {
            "mask": self.output_dir / "masks" / sample.mesh_name / sample.mask.name,
            "rgb": self.output_dir / "rgbs" / sample.mesh_name / sample.rgb.name,
            "depth": self.output_dir / "depths" / sample.mesh_name / sample.depth.name,
        }
        if self.generate_gradient:
            paths["gradient"] = (
                self.output_dir
                / "gradients"
                / sample.mesh_name
                / f"{sample.stem}_gradient.npy"
            )
        if self.save_visualizations:
            paths["visualization"] = (
                self.output_dir / "visualizations" / sample.mesh_name / f"{sample.stem}.png"
            )
        return paths

    def _prepare_output(self, samples: List[InputSample]) -> None:
        destinations = [self.output_csv]
        for sample in samples:
            destinations.extend(self._output_paths(sample).values())

        existing = [path for path in destinations if path.exists()]
        if existing and not self.overwrite:
            preview = ", ".join(str(path) for path in existing[:5])
            if len(existing) > 5:
                preview += f", and {len(existing) - 5} more"
            raise FileExistsError(
                f"Output already exists ({preview}). Pass --overwrite to replace generated files."
            )

        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        for sample in samples:
            for path in self._output_paths(sample).values():
                path.parent.mkdir(parents=True, exist_ok=True)

    def build(self) -> pd.DataFrame:
        samples = collect_input_samples(self.input_dir)
        if self.mesh is not None:
            samples = [sample for sample in samples if sample.mesh_name == self.mesh]
        if self.sample is not None:
            samples = [sample for sample in samples if sample.stem == self.sample]
        if not samples:
            raise ValueError("No input samples match --mesh/--sample")
        self._prepare_output(samples)
        generator = KeypointGeneration.for_2d()
        rows = []

        for sample in tqdm(samples, desc="Generating 2D grasp labels"):
            _validate_shapes(sample)
            results = generator.returnGraspPoses(
                str(sample.mask), generate_gradient=self.generate_gradient
            )
            if not isinstance(results, tuple) or len(results) != 2:
                raise RuntimeError(
                    f"Straight-skeleton labeling failed for mask '{sample.mask}'"
                )
            grasp_poses, gradient = results

            output_paths = self._output_paths(sample)
            shutil.copy2(sample.mask, output_paths["mask"])
            shutil.copy2(sample.rgb, output_paths["rgb"])
            shutil.copy2(sample.depth, output_paths["depth"])

            gradient_name = ""
            if self.generate_gradient:
                if gradient is None:
                    raise RuntimeError(
                        f"No gradient was returned for mask '{sample.mask}'"
                    )
                np.save(output_paths["gradient"], gradient)
                gradient_name = output_paths["gradient"].name

            grasps = [_grasp_values(p1, p2) for p1, p2 in grasp_poses]
            if self.save_visualizations:
                image = cv2.imread(str(sample.rgb), cv2.IMREAD_COLOR)
                for cx, cy, theta, width in grasps:
                    # Match the training loader's 10-pixel grasp-map height.
                    corners = np.rint(
                        cv2.boxPoints(((cx, cy), (width, 10.0), theta))
                    ).astype(np.int32)
                    for index in range(4):
                        color = (0, 0, 255) if index % 2 == 0 else (255, 0, 0)
                        cv2.line(
                            image, tuple(corners[index]), tuple(corners[(index + 1) % 4]),
                            color, 2, cv2.LINE_AA,
                        )
                cv2.putText(
                    image, f"Grasps: {len(grasps)}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
                )
                if not cv2.imwrite(str(output_paths["visualization"]), image):
                    raise OSError(f"Could not save visualization: {output_paths['visualization']}")
            rows.append(
                {
                    "img_rgb": (Path(sample.mesh_name) / output_paths["rgb"].name).as_posix(),
                    "img_depth": (Path(sample.mesh_name) / output_paths["depth"].name).as_posix(),
                    "gradient_name": (
                        (Path(sample.mesh_name) / gradient_name).as_posix()
                        if gradient_name
                        else ""
                    ),
                    "grasps": grasps,
                }
            )

        # Keep at least one grasp group so an empty-detection dataset still has
        # the same schema expected by the training loaders.
        max_grasps = max(1, max(len(row["grasps"]) for row in rows))
        csv_rows = []
        for row in rows:
            csv_row = {
                "img_rgb": row["img_rgb"],
                "img_depth": row["img_depth"],
                "gradient_name": row["gradient_name"],
            }
            for index in range(1, max_grasps + 1):
                if index <= len(row["grasps"]):
                    cx, cy, theta, width = row["grasps"][index - 1]
                else:
                    cx = cy = theta = width = np.nan
                csv_row[f"CX{index}"] = cx
                csv_row[f"CY{index}"] = cy
                csv_row[f"Theta{index}"] = theta
                csv_row[f"W{index}"] = width
            csv_rows.append(csv_row)

        frame = pd.DataFrame(csv_rows)
        frame.to_csv(self.output_csv, index=False)
        return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a 2D training dataset from matching masks, RGB images, and "
            "depth files using keypoint_generator.KeypointGeneration."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Collector output containing <mesh>/mask, <mesh>/rgb, and <mesh>/depth",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Folder where the processed dataset will be written",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Path for the generated trainer-compatible annotation CSV",
    )
    parser.add_argument(
        "--generate-gradient",
        action="store_true",
        help="Generate and save the straight-skeleton gradient map",
    )
    parser.add_argument(
        "--mesh",
        help="Process only this exact input mesh/object folder name",
    )
    parser.add_argument(
        "--sample",
        help="Process only this filename stem within --mesh, for example 1",
    )
    parser.add_argument(
        "--save-visualizations",
        action="store_true",
        help="Save RGB grasp-box overlays to <output-dir>/visualizations/<mesh>/<sample>.png",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow generated output files to be replaced",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    builder = DatasetBuilder2D(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        output_csv=args.output_csv,
        generate_gradient=args.generate_gradient,
        overwrite=args.overwrite,
        mesh=args.mesh,
        sample=args.sample,
        save_visualizations=args.save_visualizations,
    )
    frame = builder.build()
    print(f"Wrote {len(frame)} samples to {builder.output_csv}")


if __name__ == "__main__":
    main()
