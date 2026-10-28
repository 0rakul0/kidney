"""Treina uma U-Net estudante multiclasse para a etapa intrarrenal.

A estudante aprende com as máscaras manuais e com a média das probabilidades
produzidas pela U-Net e pela DeepLabV3 multiclasse. Os professores são usados
somente durante o treinamento; a inferência da aplicação carrega apenas a
estudante.
"""

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
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.core.checkpoint_metadata import load_checkpoint_metadata, save_checkpoint_metadata
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
DEFAULT_UNET_TEACHER = PROJECT_ROOT / "models" / "intrarenal_unet_multiclass_annotator1.pth"
DEFAULT_DEEPLAB_TEACHER = PROJECT_ROOT / "models" / "intrarenal_deeplab_resnet50_multiclass_annotator1.pth"
DEFAULT_STUDENT = PROJECT_ROOT / "models" / "intrarenal_student_unet_deeplab.pth"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "intrarenal_distillation"
DEFAULT_CACHE = DEFAULT_RESULTS_ROOT / "teacher_cache"


@dataclass
class TrainConfig:
    dataset_root: str
    student_checkpoint: str
    teacher_unet: str
    teacher_deeplab: str
    experiment_name: str = "intrarenal_student_unet_deeplab"
    img_size: int = 256
    epochs: int = 50
    batch_size: int = 8
    teacher_batch_size: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    early_stopping: int = 10
    num_workers: int = 0
    seed: int = 42
    clahe: bool = True
    augment: bool = True
    base_channels: int = 32
    temperature: float = 2.0
    supervised_weight: float = 1.0
    distillation_weight: float = 0.5


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_deeplab(num_classes=len(CLASS_NAMES)):
    model = torchvision.models.segmentation.deeplabv3_resnet50(weights=None, aux_loss=True)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    if model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model


def load_unet_teacher(checkpoint, device):
    metadata = load_checkpoint_metadata(checkpoint)
    base_channels = int(metadata.get("config", {}).get("base_channels", 64))
    model = UNet(in_channels=3, out_channels=len(CLASS_NAMES), base_channels=base_channels).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model.eval()


def load_deeplab_teacher(checkpoint, device):
    model = build_deeplab().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model.eval()


def teacher_logits(model, model_name, images):
    output = model(images)
    return output["out"] if model_name == "deeplab" else output


def normalize_teacher_probabilities(probabilities, kidney):
    """Força a região externa à ROI a permanecer como background."""
    probabilities = np.asarray(probabilities, dtype=np.float32)
    outside = kidney <= 0
    probabilities[:, outside] = 0.0
    probabilities[0, outside] = 1.0
    denominator = np.clip(probabilities.sum(axis=0, keepdims=True), 1e-6, None)
    return probabilities / denominator


def cache_teacher_predictions(config, device):
    cache_root = Path(config.teacher_cache)
    base_datasets = {
        split: IntrarenalMulticlassDataset(
            Path(config.dataset_root) / split,
            config.img_size,
            augment=False,
            clahe=config.clahe,
        )
        for split in ("train", "val", "test")
    }
    teacher_specs = (
        ("unet", Path(config.teacher_unet), load_unet_teacher),
        ("deeplab", Path(config.teacher_deeplab), load_deeplab_teacher),
    )
    for teacher_name, checkpoint, loader in teacher_specs:
        print(f"Carregando professor intrarrenal {teacher_name}: {checkpoint}")
        model = loader(checkpoint, device)
        for split, dataset in base_datasets.items():
            output_dir = cache_root / split / teacher_name
            output_dir.mkdir(parents=True, exist_ok=True)
            loader_data = DataLoader(
                dataset,
                batch_size=config.teacher_batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=device == "cuda",
            )
            with torch.no_grad():
                offset = 0
                for images, _ in loader_data:
                    images = images.to(device, non_blocking=True)
                    probabilities = torch.softmax(teacher_logits(model, teacher_name, images), dim=1)
                    probabilities = probabilities.cpu().numpy()
                    for batch_index in range(probabilities.shape[0]):
                        name = dataset.names[offset + batch_index]
                        kidney = images[batch_index, 2].detach().cpu().numpy()
                        prediction = normalize_teacher_probabilities(probabilities[batch_index], kidney)
                        np.save(output_dir / f"{Path(name).stem}.npy", prediction.astype(np.float32))
                    offset += probabilities.shape[0]
            print(f"  cache {split}: {len(dataset)} imagens")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()


class DistilledIntrarenalDataset(Dataset):
    def __init__(self, root_dir, cache_dir, img_size=256, augment=False, clahe=True):
        self.root_dir = Path(root_dir)
        self.cache_dir = Path(cache_dir)
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
        teacher_unet = np.load(self.cache_dir / "unet" / f"{Path(name).stem}.npy")
        teacher_deeplab = np.load(self.cache_dir / "deeplab" / f"{Path(name).stem}.npy")
        teacher = (teacher_unet + teacher_deeplab) / 2.0
        teacher = normalize_teacher_probabilities(teacher, kidney > 0)
        mask[kidney == 0] = 0

        if self.clahe:
            image = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)
        image = image.astype(np.float32) / 255.0
        kidney = (kidney > 0).astype(np.float32)
        mask = mask.astype(np.int64)

        if self.augment:
            image, mask, kidney, teacher = self._augment(image, mask, kidney, teacher)

        channels = np.stack([image, image * kidney, kidney], axis=0)
        return (
            torch.tensor(channels, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.long),
            torch.tensor(teacher, dtype=torch.float32),
        )

    def _augment(self, image, mask, kidney, teacher):
        if np.random.rand() < 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
            kidney = cv2.flip(kidney, 1)
            teacher = np.flip(teacher, axis=2).copy()
        if np.random.rand() < 0.3:
            angle = np.random.uniform(-12.0, 12.0)
            matrix = cv2.getRotationMatrix2D((self.img_size / 2, self.img_size / 2), angle, 1.0)
            image = cv2.warpAffine(image, matrix, (self.img_size, self.img_size), borderMode=cv2.BORDER_REFLECT_101)
            mask = cv2.warpAffine(mask, matrix, (self.img_size, self.img_size), flags=cv2.INTER_NEAREST)
            kidney = cv2.warpAffine(kidney, matrix, (self.img_size, self.img_size), flags=cv2.INTER_NEAREST)
            teacher = np.stack(
                [cv2.warpAffine(channel, matrix, (self.img_size, self.img_size), flags=cv2.INTER_LINEAR) for channel in teacher]
            )
        if np.random.rand() < 0.3:
            image = np.clip(image * np.random.uniform(0.90, 1.12) + np.random.uniform(-0.04, 0.04), 0, 1)
        return image, mask, kidney, teacher


def multiclass_distillation_loss(logits, teacher_probabilities, temperature):
    teacher_probabilities = teacher_probabilities.clamp_min(1e-5)
    teacher_probabilities = teacher_probabilities / teacher_probabilities.sum(dim=1, keepdim=True).clamp_min(1e-6)
    teacher_soft = F.softmax(torch.log(teacher_probabilities) / temperature, dim=1)
    student_log_soft = F.log_softmax(logits / temperature, dim=1)
    loss = F.kl_div(student_log_soft, teacher_soft, reduction="none").sum(dim=1).mean()
    return loss * (temperature ** 2)


def evaluate(model, loader, device, criterion, config):
    model.eval()
    losses, logits_all, targets_all = [], [], []
    with torch.no_grad():
        for images, targets, teacher in loader:
            images, targets, teacher = images.to(device), targets.to(device), teacher.to(device)
            logits = model(images)
            ce = criterion(logits, targets)
            dice = multiclass_dice_loss(logits, targets, len(CLASS_NAMES))
            kd = multiclass_distillation_loss(logits, teacher, config.temperature)
            loss = config.supervised_weight * (ce + dice) + config.distillation_weight * kd
            losses.append(float(loss.item()))
            logits_all.append(logits.cpu())
            targets_all.append(targets.cpu())
    metrics = metrics_from_logits(torch.cat(logits_all), torch.cat(targets_all), len(CLASS_NAMES))
    return float(np.mean(losses)), metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Treina U-Net estudante multiclasse por destilação intrarrenal.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--teacher-unet", type=Path, default=DEFAULT_UNET_TEACHER)
    parser.add_argument("--teacher-deeplab", type=Path, default=DEFAULT_DEEPLAB_TEACHER)
    parser.add_argument("--student-checkpoint", type=Path, default=DEFAULT_STUDENT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--teacher-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--experiment-name", default="intrarenal_student_unet_deeplab")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--teacher-batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--early-stopping", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--distillation-weight", type=float, default=0.5)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-clahe", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = TrainConfig(
        dataset_root=str(args.dataset_root.resolve()),
        student_checkpoint=str(args.student_checkpoint.resolve()),
        teacher_unet=str(args.teacher_unet.resolve()),
        teacher_deeplab=str(args.teacher_deeplab.resolve()),
        experiment_name=args.experiment_name,
        img_size=args.img_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        teacher_batch_size=args.teacher_batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        early_stopping=args.early_stopping,
        num_workers=args.num_workers,
        seed=args.seed,
        clahe=not args.no_clahe,
        augment=not args.no_augment,
        base_channels=args.base_channels,
        temperature=args.temperature,
        distillation_weight=args.distillation_weight,
    )
    config.teacher_cache = str(args.teacher_cache.resolve())
    set_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print({"device": device, "dataset": config.dataset_root, "student": config.student_checkpoint})

    cache_teacher_predictions(config, device)
    datasets = {
        split: DistilledIntrarenalDataset(
            Path(config.dataset_root) / split,
            Path(config.teacher_cache) / split,
            config.img_size,
            augment=split == "train" and config.augment,
            clahe=config.clahe,
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
    model = UNet(in_channels=3, out_channels=len(CLASS_NAMES), base_channels=config.base_channels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)
    best, without_improvement, history = None, 0, []
    started = time.time()

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses = []
        for images, targets, teacher in loaders["train"]:
            images, targets, teacher = images.to(device), targets.to(device), teacher.to(device)
            logits = model(images)
            ce = criterion(logits, targets)
            dice = multiclass_dice_loss(logits, targets, len(CLASS_NAMES))
            kd = multiclass_distillation_loss(logits, teacher, config.temperature)
            loss = config.supervised_weight * (ce + dice) + config.distillation_weight * kd
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
        print(f"Epoch {epoch:02d} train_loss={row['train_loss']:.4f} val_mDice={val['mean_foreground_dice']:.4f} val_mIoU={val['mean_foreground_iou']:.4f}")
        if best is None or val["score"] > best["score"]:
            best = {**val, "epoch": epoch}
            without_improvement = 0
            Path(config.student_checkpoint).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), config.student_checkpoint)
        else:
            without_improvement += 1
            if without_improvement >= config.early_stopping:
                break

    model.load_state_dict(torch.load(config.student_checkpoint, map_location=device))
    test_loss, test = evaluate(model, loaders["test"], device, criterion, config)
    results_dir = Path(args.results_root) / config.experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    summary = {
        "architecture": "Distilled U-Net",
        "task": "intrarenal_multiclass_roi",
        "classes": {index: name for index, name in enumerate(CLASS_NAMES)},
        "input_channels": ["grayscale_roi", "masked_kidney_roi", "kidney_mask"],
        "checkpoint": config.student_checkpoint,
        "device": device,
        "teachers": {"unet": config.teacher_unet, "deeplab": config.teacher_deeplab},
        "fusion": "arithmetic_mean_of_teacher_probabilities",
        "method": "knowledge_distillation",
        "best_epoch": best["epoch"],
        "best_val": {key: round(value, 6) for key, value in best.items() if key != "epoch"},
        "test_loss": round(test_loss, 6),
        "test": {key: round(value, 6) for key, value in test.items()},
        "elapsed_seconds": round(time.time() - started, 2),
        "model_kwargs": {"in_channels": 3, "out_channels": len(CLASS_NAMES), "base_channels": config.base_channels},
        "config": asdict(config),
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    save_checkpoint_metadata(config.student_checkpoint, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
