
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import SkeletonGG_CNN


@dataclass
class TrainerConfig:
    dataset: str
    modality: str
    target: str
    epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 1e-3
    workers: int = 0
    device: str = "cpu"
    seed: int = 42
    output_dir: str = "runs"
    resume: Optional[str] = None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this PyTorch installation has no CUDA support")
    return torch.device(requested)


def _run_epoch(
    model: SkeletonGG_CNN,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[Adam],
    target_name: str,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "position": 0.0, "cosine": 0.0, "sine": 0.0, target_name: 0.0}
    sample_count = 0

    progress = tqdm(loader, desc="train" if training else "validation", leave=False)
    for inputs, targets in progress:
        inputs = inputs.to(device=device, dtype=torch.float32, non_blocking=True)
        targets = tuple(
            item.to(device=device, dtype=torch.float32, non_blocking=True)
            for item in targets
        )
        batch_size = inputs.shape[0]

        if training:
            optimizer.zero_grad(set_to_none=True)
            result = model.compute_loss(inputs, targets)
            result["loss"].backward()
            optimizer.step()
        else:
            with torch.no_grad():
                result = model.compute_loss(inputs, targets)

        totals["loss"] += float(result["loss"].detach()) * batch_size
        totals["position"] += float(result["losses"]["position"].detach()) * batch_size
        totals["cosine"] += float(result["losses"]["cosine"].detach()) * batch_size
        totals["sine"] += float(result["losses"]["sine"].detach()) * batch_size
        totals[target_name] += float(result["losses"]["target"].detach()) * batch_size
        sample_count += batch_size
        progress.set_postfix(loss=f"{totals['loss'] / sample_count:.5f}")

    if sample_count == 0:
        raise RuntimeError("The data loader produced no batches")
    return {name: value / sample_count for name, value in totals.items()}


def _save_checkpoint(
    path: Path,
    model: SkeletonGG_CNN,
    optimizer: Adam,
    epoch: int,
    validation_loss: float,
    config: TrainerConfig,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "validation_loss": validation_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
        },
        path,
    )


def train(
    model: SkeletonGG_CNN,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: TrainerConfig,
) -> Path:
    if config.epochs < 1:
        raise ValueError("epochs must be at least 1")
    seed_everything(config.seed)
    device = resolve_device(config.device)
    model.to(device)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    start_epoch = 1
    best_validation_loss = float("inf")

    if config.resume:
        checkpoint = torch.load(config.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation_loss = float(checkpoint.get("validation_loss", float("inf")))

    history = []
    for epoch in range(start_epoch, config.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, device, optimizer, config.target)
        validation_metrics = _run_epoch(
            model, validation_loader, device, optimizer=None, target_name=config.target
        )
        scheduler.step(validation_metrics["loss"])

        entry = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(entry)
        print(
            f"Epoch {epoch:03d}/{config.epochs:03d} | "
            f"train={train_metrics['loss']:.6f} | "
            f"validation={validation_metrics['loss']:.6f}"
        )

        _save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            epoch,
            validation_metrics["loss"],
            config,
        )
        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]
            _save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                epoch,
                best_validation_loss,
                config,
            )

        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

    return output_dir / "best.pt"
