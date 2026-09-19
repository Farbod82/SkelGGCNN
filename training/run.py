
import argparse
from pathlib import Path
from typing import Optional

from torch.utils.data import DataLoader

from training.loader import build_datasets
from training.loader.common import MODALITY_CHANNELS
from training.model import SkeletonGG_CNN
from training.trainer import TrainerConfig, train


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("jacquard", "dexnet"), required=True)
    parser.add_argument("--modality", choices=("rgb", "depth", "rgbd"), required=True)
    parser.add_argument(
        "--target",
        choices=("skeleton", "width"),
        default="skeleton",
        help="Fourth prediction target (default: skeleton)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Dataset directory containing rgbs, depths, and gradients",
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        required=True,
        help="CSV containing filenames and grasp annotations",
    )
    parser.add_argument("--val-csv", type=Path)
    parser.add_argument("--val-split", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "auto", "cuda"), default="cpu")
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply notebook-equivalent RGB shifts and translations to training data",
    )
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    train_csv = args.train_csv.resolve()
    val_csv: Optional[Path] = args.val_csv.resolve() if args.val_csv else None

    bundle = build_datasets(
        dataset=args.dataset,
        data_root=data_root,
        train_csv=train_csv,
        val_csv=val_csv,
        modality=args.modality,
        target=args.target,
        val_split=args.val_split,
        seed=args.seed,
        augment=args.augment,
    )
    skipped = bundle.csv_rows - bundle.usable_rows
    print(
        f"dataset={args.dataset} modality={args.modality} target={args.target} | "
        f"train={len(bundle.train)} validation={len(bundle.validation)} | "
        f"ignored_missing_rows={skipped}"
    )

    train_sample = bundle.train[0]
    print(
        f"sample input={tuple(train_sample[0].shape)} "
        f"targets={[tuple(item.shape) for item in train_sample[1]]}"
    )
    if train_sample[0].shape[0] != MODALITY_CHANNELS[args.modality]:
        raise RuntimeError("Loaded input channel count does not match the selected modality")
    if train_sample[0].shape[-2] % 16 or train_sample[0].shape[-1] % 16:
        raise ValueError("Input height and width must be divisible by 16")

    train_loader = DataLoader(
        bundle.train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=args.device == "cuda",
    )
    validation_loader = DataLoader(
        bundle.validation,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=args.device == "cuda",
    )
    output_dir = args.output_dir or (
        PROJECT_ROOT / "runs" / f"{args.dataset}_{args.modality}_{args.target}"
    )
    config = TrainerConfig(
        dataset=args.dataset,
        modality=args.modality,
        target=args.target,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        workers=args.workers,
        device=args.device,
        seed=args.seed,
        output_dir=str(output_dir.resolve()),
        resume=str(args.resume.resolve()) if args.resume else None,
    )
    model = SkeletonGG_CNN(
        input_channels=MODALITY_CHANNELS[args.modality], dropout_prob=args.dropout
    )
    best_checkpoint = train(model, train_loader, validation_loader, config)
    print(f"Training complete. Best checkpoint: {best_checkpoint}")


if __name__ == "__main__":
    main()
