
import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.feature import peak_local_max
from skimage.filters import gaussian
from tqdm import tqdm

from training.loader import build_test_dataset
from training.loader.common import MODALITY_CHANNELS
from training.model import SkeletonGG_CNN
from training.trainer import resolve_device


def get_line_pixels(x0, y0, angle, image_width, image_height):
    """Trace both directions of the angle line using the original decoder rule."""
    side1 = []
    side2 = []
    if angle == 0:
        dy = 0
        dx = 1
    elif angle == 90:
        dx = 0
        dy = 1
    elif angle < 0:
        dx = 1
        dy = math.tan(math.radians(angle))
    else:
        dx = 1
        dy = math.tan(math.radians(angle))

    x = x0
    y = y0
    while True:
        side1.append((round(x), round(y)))
        x += dx
        y += dy
        if x < 0 or x >= image_width or y < 0 or y >= image_height:
            break

    x = x0
    y = y0
    dx *= -1
    dy *= -1
    while True:
        side2.append((round(x), round(y)))
        x += dx
        y += dy
        if x < 0 or x >= image_width or y < 0 or y >= image_height:
            break
    return side1, side2


def find_final_width(x0, y0, theta, skeleton_map):
    """Measure the skeleton crossing using the original 256-pixel decoder rule."""
    if theta == 0 or theta == 180:
        dy = 0
        dx = 1 if theta == 0 else -1
    elif theta == 90:
        dx = 0
        dy = 1
    elif theta < 90:
        dx = 1
        dy = math.tan(math.radians(theta))
    else:
        dx = 1
        dy = math.tan(math.radians(theta - 180))

    d_max = max(abs(dx), abs(dy))
    dx /= d_max
    dy /= d_max
    x = x0
    y = y0
    while True:
        x += dx
        y += dy
        if x <= 0 or x >= 255 or y <= 0 or y >= 255 or skeleton_map[int(y)][int(x)] <= 0.1:
            x1, y1 = x, y
            break

    x = x0
    y = y0
    dx *= -1
    dy *= -1
    while True:
        x += dx
        y += dy
        if x <= 0 or x >= 255 or y <= 0 or y >= 255 or skeleton_map[int(y)][int(x)] <= 0.1:
            x2, y2 = x, y
            break
    return math.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))


@dataclass
class GraspBox:
    x: float
    y: float
    width: float
    angle_degrees: float
    quality: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("jacquard", "dexnet"), required=True)
    parser.add_argument("--modality", choices=("rgb", "depth", "rgbd"), required=True)
    parser.add_argument(
        "--target", choices=("skeleton", "width"), default="skeleton"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Absolute quality threshold passed to peak_local_max",
    )
    parser.add_argument(
        "--num-grasps",
        type=int,
        default=None,
        help="Maximum local maxima per image; omit to leave num_peaks unset",
    )
    parser.add_argument(
        "--random-samples",
        type=int,
        default=None,
        help="Randomly visualize this many test samples; omit to use the full test set",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for reproducible random sample selection",
    )
    parser.add_argument("--device", choices=("cpu", "auto", "cuda"), default="auto")
    return parser.parse_args()


def detect_grasps(
    quality_map: np.ndarray,
    threshold: float,
    num_grasps: Optional[int] = None,
) -> np.ndarray:
    """Select local quality maxima using the decoder settings from the project."""
    if num_grasps is None:
        return peak_local_max(
            quality_map,
            min_distance=10,
            threshold_abs=threshold,
        )
    return peak_local_max(
        quality_map,
        min_distance=10,
        threshold_abs=threshold,
        num_peaks=num_grasps,
    )


def _angle_at(cosine_map: np.ndarray, sine_map: np.ndarray, y: int, x: int) -> float:
    cosine = float(np.clip(cosine_map[y, x], -1.0, 1.0))
    sine = float(np.clip(sine_map[y, x], -1.0, 1.0))
    return math.degrees(math.atan2(sine, cosine) / 2.0)


def _skeleton_box(
    y: int,
    x: int,
    quality_map: np.ndarray,
    cosine_map: np.ndarray,
    sine_map: np.ndarray,
    skeleton_map: np.ndarray,
) -> GraspBox:
    angle = _angle_at(cosine_map, sine_map, y, x)
    height, width = quality_map.shape
    side1, side2 = get_line_pixels(x, y, angle, width, height)

    connected_points = []
    for side in (side1, side2):
        for point_x, point_y in side:
            if skeleton_map[point_y, point_x] < 0.01 or quality_map[point_y, point_x] < 0.01:
                break
            connected_points.append((point_x, point_y))

    center_x, center_y = x, y
    if connected_points:
        center_x, center_y = max(
            connected_points,
            key=lambda point: skeleton_map[point[1], point[0]],
        )
    grasp_width = find_final_width(center_x, center_y, angle, skeleton_map) + 15.0
    return GraspBox(
        float(center_x),
        float(center_y),
        float(grasp_width),
        float(angle),
        float(quality_map[y, x]),
    )


def decode_grasp_boxes(
    quality_map: np.ndarray,
    cosine_map: np.ndarray,
    sine_map: np.ndarray,
    target_map: np.ndarray,
    target: str,
    threshold: float,
    num_grasps: Optional[int],
) -> Tuple[List[GraspBox], np.ndarray]:
    peaks = detect_grasps(quality_map, threshold, num_grasps)
    boxes = []
    for y, x in peaks:
        if target == "skeleton":
            box = _skeleton_box(
                int(y),
                int(x),
                quality_map,
                cosine_map,
                sine_map,
                target_map,
            )
        else:
            box = GraspBox(
                float(x),
                float(y),
                float(target_map[y, x] * 100.0),
                _angle_at(cosine_map, sine_map, int(y), int(x)),
                float(quality_map[y, x]),
            )
        boxes.append(box)
    return boxes, peaks


def _background_image(input_tensor: torch.Tensor, modality: str) -> np.ndarray:
    array = input_tensor.detach().cpu().numpy()
    if modality in {"rgb", "rgbd"}:
        return np.clip(array[:3].transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
    depth = np.clip(array[0] * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(depth, cv2.COLORMAP_JET)


def draw_grasp_boxes(image: np.ndarray, boxes: Sequence[GraspBox]) -> np.ndarray:
    result = image.copy()
    for box in boxes:
        rectangle = (
            (float(box.x), float(box.y)),
            (float(box.width), 10.0),
            float(box.angle_degrees),
        )
        points = np.intp(cv2.boxPoints(rectangle))
        cv2.drawContours(result, [points], 0, (255, 255, 255), 2)
    return result


def _save_heatmap(
    path: Path,
    heatmap: np.ndarray,
    title: str,
    vmin: float,
    vmax: float,
    peaks: Optional[np.ndarray] = None,
) -> None:
    figure, axis = plt.subplots(figsize=(5, 5))
    image = axis.imshow(heatmap, cmap="jet", vmin=vmin, vmax=vmax)
    if peaks is not None and len(peaks):
        axis.scatter(peaks[:, 1], peaks[:, 0], c="black", s=12)
    axis.set_title(title)
    axis.axis("off")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_visualization(
    output_dir: Path,
    input_image: np.ndarray,
    boxed_image: np.ndarray,
    quality_map: np.ndarray,
    cosine_map: np.ndarray,
    sine_map: np.ndarray,
    target_map: np.ndarray,
    target: str,
    peaks: np.ndarray,
    boxes: Sequence[GraspBox],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "input.png"), input_image)
    cv2.imwrite(str(output_dir / "boxes.png"), boxed_image)
    _save_heatmap(output_dir / "quality.png", quality_map, "Quality", 0.0, 1.0, peaks)
    _save_heatmap(output_dir / "cosine.png", cosine_map, "Cosine", -1.0, 1.0)
    _save_heatmap(output_dir / "sine.png", sine_map, "Sine", -1.0, 1.0)
    _save_heatmap(
        output_dir / f"{target}.png",
        target_map,
        target.capitalize(),
        -1.0 if target == "skeleton" else 0.0,
        1.0,
    )
    (output_dir / "grasps.json").write_text(
        json.dumps([asdict(box) for box in boxes], indent=2), encoding="utf-8"
    )


def load_model(
    checkpoint_path: Path,
    modality: str,
    dataset: str,
    target: str,
    device: torch.device,
) -> SkeletonGG_CNN:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        saved_config = checkpoint.get("config", {})
        for key, selected in (
            ("dataset", dataset),
            ("modality", modality),
            ("target", target),
        ):
            saved = saved_config.get(key)
            if saved is not None and saved != selected:
                raise ValueError(
                    f"Checkpoint {key} is {saved!r}, but the selected value is {selected!r}"
                )
    else:
        state_dict = checkpoint

    model = SkeletonGG_CNN(input_channels=MODALITY_CHANNELS[modality])
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def select_sample_indices(
    dataset_size: int,
    random_samples: Optional[int],
    seed: Optional[int],
) -> np.ndarray:
    if random_samples is None:
        return np.arange(dataset_size)
    if random_samples < 1:
        raise ValueError("random_samples must be at least 1 when provided")
    if random_samples > dataset_size:
        raise ValueError(
            f"random_samples={random_samples} exceeds the usable test set size "
            f"of {dataset_size}"
        )
    return np.random.default_rng(seed).choice(
        dataset_size, size=random_samples, replace=False
    )


def main() -> None:
    args = parse_args()
    if args.num_grasps is not None and args.num_grasps < 1:
        raise ValueError("num_grasps must be at least 1 when provided")

    device = resolve_device(args.device)
    bundle = build_test_dataset(
        dataset=args.dataset,
        data_root=args.data_root.resolve(),
        test_csv=args.test_csv.resolve(),
        modality=args.modality,
        target=args.target,
    )
    model = load_model(
        args.checkpoint.resolve(),
        args.modality,
        args.dataset,
        args.target,
        device,
    )
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    sample_indices = select_sample_indices(
        len(bundle.dataset), args.random_samples, args.seed
    )

    ignored = bundle.csv_rows - bundle.usable_rows
    print(
        f"test={bundle.usable_rows} selected={len(sample_indices)} "
        f"ignored_missing_rows={ignored} "
        f"device={device} threshold={args.threshold} num_grasps={args.num_grasps}"
    )
    with torch.no_grad():
        for index in tqdm(sample_indices, desc="visualize"):
            index = int(index)
            input_tensor, _ = bundle.dataset[index]
            outputs = model(input_tensor.unsqueeze(0).to(device=device, dtype=torch.float32))
            quality_map = gaussian(
                outputs[0][0, 0].detach().cpu().numpy(), 2.0, preserve_range=True
            )
            cosine_map = gaussian(
                outputs[1][0, 0].detach().cpu().numpy(), 2.0, preserve_range=True
            )
            sine_map = gaussian(
                outputs[2][0, 0].detach().cpu().numpy(), 2.0, preserve_range=True
            )
            target_map = gaussian(
                outputs[3][0, 0].detach().cpu().numpy(), 1.0, preserve_range=True
            )
            boxes, peaks = decode_grasp_boxes(
                quality_map,
                cosine_map,
                sine_map,
                target_map,
                args.target,
                args.threshold,
                args.num_grasps,
            )
            input_image = _background_image(input_tensor, args.modality)
            boxed_image = draw_grasp_boxes(input_image, boxes)
            row = bundle.dataset.records.iloc[index]
            name_column = "img_rgb" if args.modality == "rgb" else "img_depth"
            sample_name = Path(str(row[name_column])).stem
            save_visualization(
                output_root / f"{index:06d}_{sample_name}",
                input_image,
                boxed_image,
                quality_map,
                cosine_map,
                sine_map,
                target_map,
                args.target,
                peaks,
                boxes,
            )

    print(f"Visualizations saved to: {output_root}")


if __name__ == "__main__":
    main()
