import argparse
import csv
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.core.checkpoint_metadata import save_checkpoint_metadata


DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT
    / "dataset_aumentado"
    / "dataset_intrarrenal"
    / "supervisionado"
    / "regions_multiclass_annotator_1"
)
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "intrarenal_deeplab_resnet50_multiclass_annotator1.pth"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "intrarenal_model3"
CLASS_NAMES = ["background", "cortex", "medulla", "central_echo_complex"]


@dataclass
class TrainConfig:
    dataset_root: str
    checkpoint_path: str
    experiment_name: str = "intrarenal_deeplab_resnet50_multiclass_annotator1"
    backbone: str = "resnet50"
    img_size: int = 256
    epochs: int = 50
    batch_size: int = 6
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    early_stopping: int = 10
    num_workers: int = 0
    seed: int = 42
    clahe: bool = True
    augment: bool = True
    dice_weight: float = 1.0
    ce_weight: float = 1.0


class IntrarenalMulticlassDataset(Dataset):
    def __init__(self, root_dir, img_size=256, augment=False, clahe=True):
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / "image"
        self.mask_dir = self.root_dir / "mask"
        self.kidney_dir = self.root_dir / "kidney_mask"
        self.names = sorted(path.name for path in self.image_dir.glob("*.png"))
        self.img_size = img_size
        self.augment = augment
        self.clahe = clahe

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        image = cv2.imread(str(self.image_dir / name), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(self.mask_dir / name), cv2.IMREAD_GRAYSCALE)
        kidney = cv2.imread(str(self.kidney_dir / name), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None or kidney is None:
            raise FileNotFoundError(f"Exemplo multiclasse incompleto: {name}")

        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        kidney = cv2.resize(kidney, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        if self.clahe:
            image = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)
        image = image.astype(np.float32) / 255.0
        kidney = (kidney > 0).astype(np.float32)
        mask = mask.astype(np.int64)
        mask[kidney == 0] = 0

        if self.augment:
            image, mask, kidney = self._augment(image, mask, kidney)

        channels = np.stack([image, image * kidney, kidney], axis=0)
        return torch.tensor(channels, dtype=torch.float32), torch.tensor(mask, dtype=torch.long)

    def _augment(self, image, mask, kidney):
        if np.random.rand() < 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
            kidney = cv2.flip(kidney, 1)
        if np.random.rand() < 0.3:
            angle = np.random.uniform(-12.0, 12.0)
            matrix = cv2.getRotationMatrix2D((self.img_size / 2, self.img_size / 2), angle, 1.0)
            image = cv2.warpAffine(image, matrix, (self.img_size, self.img_size), borderMode=cv2.BORDER_REFLECT_101)
            mask = cv2.warpAffine(mask, matrix, (self.img_size, self.img_size), flags=cv2.INTER_NEAREST)
            kidney = cv2.warpAffine(kidney, matrix, (self.img_size, self.img_size), flags=cv2.INTER_NEAREST)
        if np.random.rand() < 0.3:
            image = np.clip(image * np.random.uniform(0.90, 1.12) + np.random.uniform(-0.04, 0.04), 0, 1)
        return image, mask, kidney


def build_model(backbone="resnet50", pretrained=True, num_classes=4):
    builders = {
        "resnet50": torchvision.models.segmentation.deeplabv3_resnet50,
        "resnet101": torchvision.models.segmentation.deeplabv3_resnet101,
    }
    if backbone not in builders:
        raise ValueError(f"Backbone nao suportado: {backbone}")
    weights = "DEFAULT" if pretrained else None
    model = builders[backbone](weights=weights, aux_loss=True)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    if model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model


def multiclass_dice_loss(logits, target, num_classes):
    probabilities = torch.softmax(logits, dim=1)
    one_hot = torch.nn.functional.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = (probabilities * one_hot).sum(dims)
    denominator = probabilities.sum(dims) + one_hot.sum(dims)
    dice = (2 * intersection + 1e-6) / (denominator + 1e-6)
    return 1 - dice[1:].mean()


def metrics_from_logits(logits, target, num_classes):
    prediction = logits.argmax(dim=1)
    result = {}
    foreground_dice = []
    foreground_iou = []
    for class_id, class_name in enumerate(CLASS_NAMES[:num_classes]):
        pred = prediction == class_id
        truth = target == class_id
        intersection = torch.logical_and(pred, truth).sum().item()
        union = torch.logical_or(pred, truth).sum().item()
        pred_sum = pred.sum().item()
        truth_sum = truth.sum().item()
        dice = 1.0 if pred_sum + truth_sum == 0 else (2 * intersection / (pred_sum + truth_sum))
        iou = 1.0 if union == 0 else (intersection / union)
        result[f"{class_name}_dice"] = float(dice)
        result[f"{class_name}_iou"] = float(iou)
        if class_id > 0:
            foreground_dice.append(dice)
            foreground_iou.append(iou)
    result["mean_foreground_dice"] = float(np.mean(foreground_dice))
    result["mean_foreground_iou"] = float(np.mean(foreground_iou))
    result["score"] = (result["mean_foreground_dice"] + result["mean_foreground_iou"]) / 2
    return result


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device, criterion, config):
    model.eval()
    losses, logits_all, targets_all = [], [], []
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)["out"]
            ce = criterion(logits, targets)
            dice = multiclass_dice_loss(logits, targets, len(CLASS_NAMES))
            loss = config.ce_weight * ce + config.dice_weight * dice
            losses.append(float(loss.item()))
            logits_all.append(logits.cpu())
            targets_all.append(targets.cpu())
    metrics = metrics_from_logits(torch.cat(logits_all), torch.cat(targets_all), len(CLASS_NAMES))
    return float(np.mean(losses)), metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Treina DeepLabV3 multiclasse para regioes intrarrenais dentro da ROI renal.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--experiment-name", default="intrarenal_deeplab_resnet50_multiclass_annotator1")
    parser.add_argument("--backbone", choices=["resnet50", "resnet101"], default="resnet50")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--early-stopping", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-clahe", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = TrainConfig(
        dataset_root=str(args.dataset_root),
        checkpoint_path=str(args.checkpoint),
        experiment_name=args.experiment_name,
        backbone=args.backbone,
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
    )
    set_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    datasets = {
        split: IntrarenalMulticlassDataset(args.dataset_root / split, config.img_size, split == "train" and config.augment, config.clahe)
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
    model = build_model(config.backbone, pretrained=not args.no_pretrained, num_classes=len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)
    best, without_improvement, history = None, 0, []
    started = time.time()

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses = []
        for images, targets in loaders["train"]:
            images, targets = images.to(device), targets.to(device)
            output = model(images)
            logits = output["out"]
            ce = criterion(logits, targets)
            dice = multiclass_dice_loss(logits, targets, len(CLASS_NAMES))
            loss = config.ce_weight * ce + config.dice_weight * dice
            if "aux" in output:
                loss = loss + 0.4 * criterion(output["aux"], targets)
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
            f"val_mDice={val['mean_foreground_dice']:.4f} val_mIoU={val['mean_foreground_iou']:.4f}"
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
        "architecture": "DeepLabV3",
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
