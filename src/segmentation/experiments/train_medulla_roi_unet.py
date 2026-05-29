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
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.core.checkpoint_metadata import save_checkpoint_metadata
from src.segmentation.core.losses import focal_tversky_loss
from src.segmentation.core.metrics import dice_score, iou_score


INTRARENAL_ROOT = PROJECT_ROOT / "dataset_aumentado" / "dataset_intrarrenal"
DEFAULT_DATASET_ROOT = INTRARENAL_ROOT / "supervisionado" / "medulla_annotator_1"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "medulla_roi_unet_annotator1.pth"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "intrarenal_model3"


@dataclass
class TrainConfig:
    dataset_root: str
    checkpoint_path: str
    experiment_name: str = "medulla_roi_unet_annotator1"
    img_size: int = 256
    base_channels: int = 32
    epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    early_stopping: int = 10
    num_workers: int = 0
    seed: int = 42
    clahe: bool = True
    augment: bool = True
    threshold_candidates: tuple = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65)
    tversky_alpha: float = 0.45
    tversky_beta: float = 0.55
    focal_tversky_gamma: float = 1.33


class MedullaROIDataset(Dataset):
    def __init__(self, root_dir, img_size=256, augment=False, clahe=True):
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / "image"
        self.target_dir = self.root_dir / "mask"
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
        target = cv2.imread(str(self.target_dir / name), cv2.IMREAD_GRAYSCALE)
        kidney = cv2.imread(str(self.kidney_dir / name), cv2.IMREAD_GRAYSCALE)
        if image is None or target is None or kidney is None:
            raise FileNotFoundError(f"Exemplo intrarrenal incompleto: {name}")
        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        target = cv2.resize(target, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        kidney = cv2.resize(kidney, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        if self.clahe:
            image = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)
        image = image.astype(np.float32) / 255.0
        target = (target > 0).astype(np.float32)
        kidney = (kidney > 0).astype(np.float32)
        target *= kidney
        if self.augment:
            image, target, kidney = self._augment(image, target, kidney)
        channels = np.stack([image, image * kidney, kidney], axis=0)
        return (
            torch.tensor(channels, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
            torch.tensor(kidney, dtype=torch.float32),
        )

    def _augment(self, image, target, kidney):
        if np.random.rand() < 0.5:
            image = cv2.flip(image, 1)
            target = cv2.flip(target, 1)
            kidney = cv2.flip(kidney, 1)
        if np.random.rand() < 0.3:
            angle = np.random.uniform(-12.0, 12.0)
            matrix = cv2.getRotationMatrix2D((self.img_size / 2, self.img_size / 2), angle, 1.0)
            image = cv2.warpAffine(image, matrix, (self.img_size, self.img_size), borderMode=cv2.BORDER_REFLECT_101)
            target = cv2.warpAffine(target, matrix, (self.img_size, self.img_size), flags=cv2.INTER_NEAREST)
            kidney = cv2.warpAffine(kidney, matrix, (self.img_size, self.img_size), flags=cv2.INTER_NEAREST)
        if np.random.rand() < 0.3:
            image = np.clip(image * np.random.uniform(0.90, 1.12) + np.random.uniform(-0.04, 0.04), 0, 1)
        return image, target, kidney


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class MedullaROIUNet(nn.Module):
    """Segmenta Medulla a partir da imagem, imagem mascarada e mascara renal."""

    def __init__(self, in_channels=3, out_channels=1, base_channels=32):
        super().__init__()
        c1, c2, c3, c4 = (base_channels, base_channels * 2, base_channels * 4, base_channels * 8)
        self.pool = nn.MaxPool2d(2)
        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.bottleneck = ConvBlock(c3, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = ConvBlock(c3 * 2, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = ConvBlock(c2 * 2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = ConvBlock(c1 * 2, c1)
        self.output = nn.Conv2d(c1, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        bottleneck = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(bottleneck), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.output(d1)


def build_model(base_channels=32):
    return MedullaROIUNet(base_channels=base_channels)


def constrain_logits(logits, kidney):
    return logits.squeeze(1) * kidney + (-20.0 * (1.0 - kidney))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device, config, search_threshold=False, threshold=0.5):
    model.eval()
    losses, logits_all, targets_all = [], [], []
    with torch.no_grad():
        for images, targets, kidney in loader:
            images, targets, kidney = images.to(device), targets.to(device), kidney.to(device)
            logits = constrain_logits(model(images), kidney)
            losses.append(float(focal_tversky_loss(
                logits, targets, alpha=config.tversky_alpha,
                beta=config.tversky_beta, gamma=config.focal_tversky_gamma
            ).item()))
            logits_all.append(logits.cpu())
            targets_all.append(targets.cpu())
    logits = torch.cat(logits_all)
    targets = torch.cat(targets_all)
    candidates = config.threshold_candidates if search_threshold else (threshold,)
    best = None
    for candidate in candidates:
        dice = dice_score(logits, targets, threshold=candidate, from_logits=True)
        iou = iou_score(logits, targets, threshold=candidate, from_logits=True)
        score = (dice + iou) / 2
        if best is None or score > best["score"]:
            best = {"threshold": candidate, "dice": dice, "iou": iou, "score": score}
    return float(np.mean(losses)), best


def parse_args():
    parser = argparse.ArgumentParser(description="Treina U-Net condicionada pela ROI renal para segmentar estrutura intrarrenal.")
    parser.add_argument("--target", choices=["medulla", "cortex"], default="medulla")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--early-stopping", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dataset_root is None:
        args.dataset_root = (
            DEFAULT_DATASET_ROOT
            if args.target == "medulla"
            else INTRARENAL_ROOT / "supervisionado" / "cortex_annotator_1"
        )
    if args.checkpoint is None:
        args.checkpoint = (
            DEFAULT_CHECKPOINT
            if args.target == "medulla"
            else PROJECT_ROOT / "models" / "cortex_roi_unet_annotator1.pth"
        )
    if args.experiment_name is None:
        args.experiment_name = f"{args.target}_roi_unet_annotator1"
    config = TrainConfig(
        dataset_root=str(args.dataset_root),
        checkpoint_path=str(args.checkpoint),
        experiment_name=args.experiment_name,
        img_size=args.img_size,
        base_channels=args.base_channels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        early_stopping=args.early_stopping,
        num_workers=args.num_workers,
    )
    set_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    datasets = {
        split: MedullaROIDataset(args.dataset_root / split, config.img_size, split == "train", config.clahe)
        for split in ("train", "val", "test")
    }
    loaders = {
        split: DataLoader(
            dataset, batch_size=config.batch_size, shuffle=split == "train",
            num_workers=config.num_workers, pin_memory=device == "cuda"
        )
        for split, dataset in datasets.items()
    }
    print({split: len(dataset) for split, dataset in datasets.items()}, "device=", device)
    model = build_model(config.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)
    best, without_improvement, history = None, 0, []
    started = time.time()
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses = []
        for images, targets, kidney in loaders["train"]:
            images, targets, kidney = images.to(device), targets.to(device), kidney.to(device)
            logits = constrain_logits(model(images), kidney)
            loss = focal_tversky_loss(
                logits, targets, alpha=config.tversky_alpha,
                beta=config.tversky_beta, gamma=config.focal_tversky_gamma
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        val_loss, val = evaluate(model, loaders["val"], device, config, search_threshold=True)
        scheduler.step(val["score"])
        history.append({
            "epoch": epoch, "train_loss": round(float(np.mean(train_losses)), 6),
            "val_loss": round(val_loss, 6), "val_dice": round(val["dice"], 6),
            "val_iou": round(val["iou"], 6), "threshold": val["threshold"],
        })
        print(
            f"Epoch {epoch:02d} train_loss={np.mean(train_losses):.4f} "
            f"val_dice={val['dice']:.4f} val_iou={val['iou']:.4f} threshold={val['threshold']:.2f}"
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
    test_loss, test = evaluate(model, loaders["test"], device, config, threshold=best["threshold"])
    results_dir = DEFAULT_RESULTS_ROOT / config.experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    summary = {
        "architecture": "IntrarenalROIUNet",
        "input_channels": ["grayscale_roi", "masked_kidney_roi", "kidney_mask"],
        "output": f"{args.target}_mask_constrained_to_kidney",
        "target": args.target,
        "checkpoint": str(args.checkpoint),
        "device": device,
        "dataset_sizes": {split: len(dataset) for split, dataset in datasets.items()},
        "best_epoch": best["epoch"],
        "best_threshold": best["threshold"],
        "best_val_dice": round(best["dice"], 6),
        "best_val_iou": round(best["iou"], 6),
        "test_loss": round(test_loss, 6),
        "test_dice": round(test["dice"], 6),
        "test_iou": round(test["iou"], 6),
        "elapsed_seconds": round(time.time() - started, 2),
        "config": asdict(config),
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    save_checkpoint_metadata(args.checkpoint, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
