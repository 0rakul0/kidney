import argparse
import csv
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.core.checkpoint_metadata import save_checkpoint_metadata
from src.segmentation.experiments.train_inner_deeplab import (
    CLASS_NAMES,
    IntrarenalMulticlassDataset,
    metrics_from_logits,
    multiclass_dice_loss,
)
from src.segmentation.experiments.train_unet import UNet


DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT
    / "dataset_aumentado"
    / "dataset_intrarrenal"
    / "supervisionado"
    / "regions_multiclass_annotator_1"
)
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "intrarenal_unet_multiclass_annotator1.pth"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "intrarenal_model3"


@dataclass
class TrainConfig:
    dataset_root: str
    checkpoint_path: str
    experiment_name: str = "intrarenal_unet_multiclass_annotator1"
    img_size: int = 256
    epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    early_stopping: int = 10
    num_workers: int = 0
    seed: int = 42
    clahe: bool = True
    augment: bool = True
    base_channels: int = 64
    dice_weight: float = 1.0
    ce_weight: float = 1.0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(base_channels=64):
    return UNet(in_channels=3, out_channels=len(CLASS_NAMES), base_channels=base_channels)


def forward_logits(model, images):
    return model(images)


def evaluate(model, loader, device, criterion, config):
    model.eval()
    losses, logits_all, targets_all = [], [], []
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = forward_logits(model, images)
            ce = criterion(logits, targets)
            dice = multiclass_dice_loss(logits, targets, len(CLASS_NAMES))
            loss = config.ce_weight * ce + config.dice_weight * dice
            losses.append(float(loss.item()))
            logits_all.append(logits.cpu())
            targets_all.append(targets.cpu())
    metrics = metrics_from_logits(torch.cat(logits_all), torch.cat(targets_all), len(CLASS_NAMES))
    return float(np.mean(losses)), metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Treina U-Net multiclasse para cortex, medulla e CEC dentro da ROI renal."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--experiment-name", default="intrarenal_unet_multiclass_annotator1")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--early-stopping", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-clahe", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = TrainConfig(
        dataset_root=str(args.dataset_root),
        checkpoint_path=str(args.checkpoint),
        experiment_name=args.experiment_name,
        img_size=args.img_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        early_stopping=args.early_stopping,
        num_workers=args.num_workers,
        seed=args.seed,
        clahe=not args.no_clahe,
        augment=not args.no_augment,
        base_channels=args.base_channels,
    )
    set_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    datasets = {
        split: IntrarenalMulticlassDataset(
            args.dataset_root / split,
            config.img_size,
            split == "train" and config.augment,
            config.clahe,
        )
        for split in ("train", "val", "test")
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=split == "train",
            num_workers=config.num_workers,
            pin_memory=device == "cuda",
        )
        for split, dataset in datasets.items()
    }
    print({split: len(dataset) for split, dataset in datasets.items()}, "device=", device)

    model = build_model(base_channels=config.base_channels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=4,
    )

    best, without_improvement, history = None, 0, []
    started = time.time()
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses = []
        for images, targets in loaders["train"]:
            images, targets = images.to(device), targets.to(device)
            logits = forward_logits(model, images)
            ce = criterion(logits, targets)
            dice = multiclass_dice_loss(logits, targets, len(CLASS_NAMES))
            loss = config.ce_weight * ce + config.dice_weight * dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        val_loss, val = evaluate(model, loaders["val"], device, criterion, config)
        scheduler.step(val["score"])
        row = {
            "epoch": epoch,
            "train_loss": round(float(np.mean(train_losses)), 6),
            "val_loss": round(val_loss, 6),
            **{key: round(value, 6) for key, value in val.items()},
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d} train_loss={row['train_loss']:.4f} "
            f"val_mDice={val['mean_foreground_dice']:.4f} "
            f"val_mIoU={val['mean_foreground_iou']:.4f}"
        )
        if best is None or val["score"] > best["score"]:
            best = {**val, "epoch": epoch}
            without_improvement = 0
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.checkpoint)
        else:
            without_improvement += 1
            if without_improvement >= config.early_stopping:
                break

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    test_loss, test = evaluate(model, loaders["test"], device, criterion, config)
    results_dir = DEFAULT_RESULTS_ROOT / config.experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    summary = {
        "architecture": "U-Net",
        "task": "intrarenal_multiclass_roi",
        "classes": {index: name for index, name in enumerate(CLASS_NAMES)},
        "input_channels": ["grayscale_roi", "masked_kidney_roi", "kidney_mask"],
        "checkpoint": str(args.checkpoint),
        "device": device,
        "dataset_sizes": {split: len(dataset) for split, dataset in datasets.items()},
        "best_epoch": best["epoch"],
        "best_val": {key: round(value, 6) for key, value in best.items() if key != "epoch"},
        "test_loss": round(test_loss, 6),
        "test": {key: round(value, 6) for key, value in test.items()},
        "elapsed_seconds": round(time.time() - started, 2),
        "config": asdict(config),
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    save_checkpoint_metadata(args.checkpoint, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
