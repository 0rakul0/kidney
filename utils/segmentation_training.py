import argparse
import csv
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.checkpoint_metadata import save_checkpoint_metadata
from utils.dataset import KidneyDataset
from utils.losses import dice_loss
from utils.metrics import dice_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_THRESHOLD_CANDIDATES = tuple(round(x / 100, 2) for x in range(35, 70, 5))


@dataclass
class SegmentationTrainingConfig:
    model_name: str
    experiment_name: str
    checkpoint_name: str
    dataset_path: str
    model_kind: str
    img_size: int = 256
    batch_size: int = 8
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    optimizer_name: str = "adam"
    scheduler_name: str = "plateau"
    scheduler_patience: int = 4
    scheduler_factor: float = 0.5
    min_lr: float = 1e-6
    early_stopping_patience: int = 10
    bce_weight: float = 1.0
    dice_weight: float = 1.0
    threshold: float = 0.5
    threshold_search: bool = True
    threshold_candidates: tuple = DEFAULT_THRESHOLD_CANDIDATES
    pos_weight: float | None = None
    auto_pos_weight: bool = False
    augment: bool = False
    clahe: bool = False
    seed: int = 42
    num_workers: int = 0
    model_kwargs: dict | None = None

    @property
    def dataset_root(self):
        return Path(self.dataset_path).resolve()

    @property
    def checkpoint_path(self):
        return (PROJECT_ROOT / "models" / self.checkpoint_name).resolve()

    @property
    def history_path(self):
        return (
            PROJECT_ROOT
            / "results"
            / "segmentation_experiments"
            / f"{self.experiment_name}_history.csv"
        ).resolve()

    @property
    def summary_path(self):
        return (
            PROJECT_ROOT
            / "results"
            / "segmentation_experiments"
            / f"{self.experiment_name}_summary.json"
        ).resolve()


def add_training_args(parser, default_model_name, default_checkpoint_name, default_experiment_name):

    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    parser.add_argument("--scheduler", choices=["none", "plateau", "cosine"], default="plateau")
    parser.add_argument("--scheduler-patience", type=int, default=4)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--early-stopping", type=int, default=10)
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--threshold-candidates",
        type=str,
        default="0.35,0.40,0.45,0.50,0.55,0.60,0.65"
    )
    parser.add_argument(
        "--threshold-search",
        action=argparse.BooleanOptionalAction,
        default=True
    )
    parser.add_argument("--pos-weight", type=float, default=None)
    parser.add_argument(
        "--auto-pos-weight",
        action=argparse.BooleanOptionalAction,
        default=False
    )
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--clahe", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dataset-path", type=str, default=str((PROJECT_ROOT / "dataset").resolve()))
    parser.add_argument("--experiment-name", type=str, default=default_experiment_name)
    parser.add_argument("--checkpoint-name", type=str, default=default_checkpoint_name)
    parser.add_argument("--model-name", type=str, default=default_model_name)


def build_training_config(args, model_kind, model_kwargs=None):

    threshold_candidates = tuple(
        float(value.strip())
        for value in args.threshold_candidates.split(",")
        if value.strip()
    )

    return SegmentationTrainingConfig(
        model_name=args.model_name,
        experiment_name=args.experiment_name,
        checkpoint_name=args.checkpoint_name,
        dataset_path=args.dataset_path,
        model_kind=model_kind,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        optimizer_name=args.optimizer,
        scheduler_name=args.scheduler,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        min_lr=args.min_lr,
        early_stopping_patience=args.early_stopping,
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        threshold=args.threshold,
        threshold_search=args.threshold_search,
        threshold_candidates=threshold_candidates,
        pos_weight=args.pos_weight,
        auto_pos_weight=args.auto_pos_weight,
        augment=args.augment,
        clahe=args.clahe,
        seed=args.seed,
        num_workers=args.num_workers,
        model_kwargs=model_kwargs or {}
    )


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model, config):

    if config.optimizer_name == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

    return optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )


def build_scheduler(optimizer, config):

    if config.scheduler_name == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            min_lr=config.min_lr
        )

    if config.scheduler_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(config.epochs, 1),
            eta_min=config.min_lr
        )

    return None


def forward_logits(model, imgs, model_kind):

    if model_kind == "segformer":
        preds = model(pixel_values=imgs).logits
        return torch.nn.functional.interpolate(
            preds,
            size=imgs.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

    if model_kind == "deeplab":
        return model(imgs)["out"]

    return model(imgs)


def resolve_pos_weight(mask_dir, img_size):

    positive_pixels = 0.0
    total_pixels = 0.0

    for mask_name in os.listdir(mask_dir):
        mask_path = mask_dir / mask_name
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        positive_pixels += float((mask > 0).sum())
        total_pixels += float(mask.size)

    negative_pixels = max(total_pixels - positive_pixels, 1.0)

    return max(negative_pixels / max(positive_pixels, 1.0), 1.0)


def compute_loss(preds, masks, bce, config):

    preds = preds.squeeze(1)
    bce_term = bce(preds, masks)
    dice_term = dice_loss(preds, masks)

    return config.bce_weight * bce_term + config.dice_weight * dice_term


def evaluate_thresholds(logits_tensor, masks_tensor, candidates):

    best_threshold = candidates[0]
    best_dice = -1.0

    for threshold in candidates:
        dice_value = dice_score(
            logits_tensor,
            masks_tensor,
            threshold=threshold,
            from_logits=True
        )

        if dice_value > best_dice:
            best_dice = dice_value
            best_threshold = threshold

    return best_threshold, best_dice


def run_evaluation(model, loader, device, bce, config, threshold=None, search_threshold=False):

    model.eval()

    total_loss = 0.0
    logits_list = []
    masks_list = []

    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            preds = forward_logits(model, imgs, config.model_kind)
            loss = compute_loss(preds, masks, bce, config)

            total_loss += loss.item()
            logits_list.append(preds.squeeze(1).cpu())
            masks_list.append(masks.cpu())

    logits_tensor = torch.cat(logits_list, dim=0)
    masks_tensor = torch.cat(masks_list, dim=0)

    if search_threshold:
        threshold, dice_value = evaluate_thresholds(
            logits_tensor,
            masks_tensor,
            config.threshold_candidates
        )
    else:
        threshold = config.threshold if threshold is None else threshold
        dice_value = dice_score(
            logits_tensor,
            masks_tensor,
            threshold=threshold,
            from_logits=True
        )

    return total_loss / max(len(loader), 1), dice_value, threshold


def save_history(history_path, rows):

    history_path.parent.mkdir(parents=True, exist_ok=True)

    with history_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_summary(summary_path, summary):

    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def train_segmentation_model(build_model, config):

    set_seed(config.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = KidneyDataset(
        config.dataset_root / "train",
        img_size=config.img_size,
        augment=config.augment,
        clahe=config.clahe
    )
    val_dataset = KidneyDataset(
        config.dataset_root / "val",
        img_size=config.img_size,
        augment=False,
        clahe=config.clahe
    )
    test_dataset = KidneyDataset(
        config.dataset_root / "test",
        img_size=config.img_size,
        augment=False,
        clahe=config.clahe
    )

    pin_memory = device == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=pin_memory
    )

    print("Train:", len(train_dataset))
    print("Val:", len(val_dataset))
    print("Test:", len(test_dataset))

    resolved_pos_weight = config.pos_weight

    if config.auto_pos_weight and resolved_pos_weight is None:
        resolved_pos_weight = resolve_pos_weight(config.dataset_root / "train" / "mask", config.img_size)

    pos_weight_tensor = None
    if resolved_pos_weight is not None:
        pos_weight_tensor = torch.tensor([resolved_pos_weight], dtype=torch.float32, device=device)

    model = build_model(**(config.model_kwargs or {})).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    best_val_dice = -1.0
    best_threshold = config.threshold
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    training_start = time.time()

    for epoch in range(config.epochs):
        model.train()

        running_loss = 0.0
        running_dice = 0.0

        loop = tqdm(train_loader, desc=f"{config.model_name} epoch {epoch + 1}/{config.epochs}")

        for imgs, masks in loop:
            imgs = imgs.to(device)
            masks = masks.to(device)

            preds = forward_logits(model, imgs, config.model_kind)
            loss = compute_loss(preds, masks, bce, config)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_dice += dice_score(
                preds.squeeze(1).detach(),
                masks,
                threshold=config.threshold,
                from_logits=True
            )

            loop.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / max(len(train_loader), 1)
        train_dice = running_dice / max(len(train_loader), 1)

        val_loss, val_dice, selected_threshold = run_evaluation(
            model,
            val_loader,
            device,
            bce,
            config,
            threshold=best_threshold,
            search_threshold=config.threshold_search
        )

        if scheduler is not None:
            if config.scheduler_name == "plateau":
                scheduler.step(val_dice)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 6),
                "train_dice": round(train_dice, 6),
                "val_loss": round(val_loss, 6),
                "val_dice": round(val_dice, 6),
                "selected_threshold": round(float(selected_threshold), 4),
                "lr": round(float(current_lr), 10)
            }
        )

        print(
            f"Epoch {epoch + 1} | "
            f"Train Loss {train_loss:.4f} | "
            f"Train Dice {train_dice:.4f} | "
            f"Val Loss {val_loss:.4f} | "
            f"Val Dice {val_dice:.4f} | "
            f"Threshold {selected_threshold:.2f} | "
            f"LR {current_lr:.2e}"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_threshold = selected_threshold
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), config.checkpoint_path)

            metadata = {
                "model_name": config.model_name,
                "checkpoint_name": config.checkpoint_name,
                "experiment_name": config.experiment_name,
                "best_epoch": best_epoch,
                "best_val_dice": round(float(best_val_dice), 6),
                "best_threshold": round(float(best_threshold), 4),
                "img_size": config.img_size,
                "model_kwargs": config.model_kwargs or {},
                "hyperparameters": asdict(config),
                "resolved_pos_weight": None if resolved_pos_weight is None else round(float(resolved_pos_weight), 6)
            }

            save_checkpoint_metadata(config.checkpoint_path, metadata)
            print("Modelo salvo:", config.checkpoint_path)
        else:
            epochs_without_improvement += 1

        if (
            config.early_stopping_patience > 0
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            print("Early stopping acionado.")
            break

    best_model = build_model(**(config.model_kwargs or {})).to(device)
    best_model.load_state_dict(torch.load(config.checkpoint_path, map_location=device))

    test_loss, test_dice, _ = run_evaluation(
        best_model,
        test_loader,
        device,
        bce,
        config,
        threshold=best_threshold,
        search_threshold=False
    )

    elapsed_seconds = time.time() - training_start

    summary = {
        "model_name": config.model_name,
        "experiment_name": config.experiment_name,
        "checkpoint_path": str(config.checkpoint_path),
        "history_path": str(config.history_path),
        "best_epoch": int(best_epoch),
        "best_val_dice": round(float(best_val_dice), 6),
        "best_threshold": round(float(best_threshold), 4),
        "test_loss": round(float(test_loss), 6),
        "test_dice": round(float(test_dice), 6),
        "resolved_pos_weight": None if resolved_pos_weight is None else round(float(resolved_pos_weight), 6),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "elapsed_seconds": round(float(elapsed_seconds), 2),
        "model_kwargs": config.model_kwargs or {},
        "hyperparameters": asdict(config)
    }

    save_history(config.history_path, history)
    save_summary(config.summary_path, summary)

    print("History saved:", config.history_path)
    print("Summary saved:", config.summary_path)
    print("Test Dice:", f"{test_dice:.4f}")

    return summary
