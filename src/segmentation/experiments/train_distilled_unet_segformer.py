"""Treina uma U-Net estudante por destilação de U-Net e SegFormer-B0.

O protocolo preserva o teste manual independente. Os dois professores geram
probabilidades para as imagens de treino, validação e teste; a estudante é
treinada somente com treino e validação, combinando supervisão manual com uma
perda de destilação. O teste é utilizado apenas para a avaliação final.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.build_dataset_geral import predict_probability, prepare_tensor
from src.segmentation.core.checkpoint_metadata import load_checkpoint_metadata, save_checkpoint_metadata
from src.segmentation.core.model_loader import load_model_bundle
from src.segmentation.core.metrics import dice_score, iou_score
from src.segmentation.experiments.train_unet import UNet


DEFAULT_DATASET = (
    PROJECT_ROOT
    / "dataset_aumentado"
    / "dataset_intrarrenal"
    / "supervisionado"
    / "capsule_annotator_1_deduplicated"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "segmentation_experiments" / "capsule_distilled_unet_segformer"
DEFAULT_CACHE = DEFAULT_OUTPUT / "teacher_cache"
DEFAULT_UNET_TEACHER = PROJECT_ROOT / "models" / "kidneyus_capsule_dedup_unet.pth"
DEFAULT_SEGFORMER_TEACHER = PROJECT_ROOT / "models" / "kidneyus_capsule_dedup_segformer.pth"
DEFAULT_STUDENT = PROJECT_ROOT / "models" / "kidneyus_capsule_distilled_unet_segformer.pth"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def binary_dice(first, second):
    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)
    denominator = int(first.sum() + second.sum())
    if denominator == 0:
        return 1.0
    return float(2.0 * np.logical_and(first, second).sum() / denominator)


def fuse_teacher_probabilities(unet_probability, segformer_probability):
    """Fusão simples e auditável: média das probabilidades calibradas."""
    first = np.asarray(unet_probability, dtype=np.float32)
    second = np.asarray(segformer_probability, dtype=np.float32)
    if first.shape != second.shape:
        raise ValueError("As probabilidades dos professores têm dimensões diferentes.")
    return np.clip((first + second) / 2.0, 0.0, 1.0)


def teacher_prediction(bundle, image, img_size, device, use_tta=True):
    original = prepare_tensor(image, img_size, device, clahe=True)
    with torch.no_grad():
        probability = predict_probability(bundle, original, img_size)
        if use_tta:
            flipped_image = cv2.flip(image, 1)
            flipped = prepare_tensor(flipped_image, img_size, device, clahe=True)
            flipped_probability = np.fliplr(predict_probability(bundle, flipped, img_size))
            probability = (probability + flipped_probability) / 2.0
    return np.clip(probability, 0.0, 1.0).astype(np.float32)


def split_images(dataset_root: Path, split: str):
    image_dir = dataset_root / split / "image"
    mask_dir = dataset_root / split / "mask"
    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(f"Divisão ausente: {image_dir} / {mask_dir}")
    records = []
    for image_path in sorted(image_dir.glob("*.png")):
        mask_path = mask_dir / image_path.name
        if not mask_path.exists():
            raise FileNotFoundError(f"Máscara manual ausente para {image_path.name}")
        records.append((image_path, mask_path))
    if not records:
        raise ValueError(f"Nenhuma imagem encontrada em {image_dir}")
    return records


def cache_path(cache_root: Path, split: str, image_path: Path):
    return cache_root / split / f"{image_path.stem}.npz"


def cache_teachers(args, records_by_split, device):
    """Gera cache em etapas para não manter U-Net e SegFormer na GPU juntos."""
    args.cache_root.mkdir(parents=True, exist_ok=True)
    all_records = [(split, image_path) for split, records in records_by_split.items() for image_path, _ in records]
    cache_ready = all(
        cache_path(args.cache_root, split, image_path).exists()
        for split, image_path in all_records
    )
    metadata_path = args.cache_root / "metadata.json"
    if cache_ready and metadata_path.exists() and not args.refresh_cache:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (metadata.get("teacher_unet") == str(args.teacher_unet)
            and metadata.get("teacher_segformer") == str(args.teacher_segformer)
            and metadata.get("img_size") == args.img_size
            and metadata.get("tta") == args.tta):
            return
        raise RuntimeError("Cache corresponde a outros professores/configurações. Use --refresh-cache ou outro --cache-root.")

    teacher_paths = {
        "unet": args.teacher_unet,
        "segformer": args.teacher_segformer,
    }
    teacher_probabilities = {key: {} for key in teacher_paths}
    thresholds = {}
    for teacher_name, checkpoint in teacher_paths.items():
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint do professor não encontrado: {checkpoint}")
        print(f"Carregando professor {teacher_name}: {checkpoint}")
        bundle = load_model_bundle(teacher_name, device=device, checkpoint_path=checkpoint)
        thresholds[teacher_name] = float(bundle.get("threshold", 0.5))
        for split, image_path in all_records:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(f"Não foi possível ler a imagem: {image_path}")
            teacher_probabilities[teacher_name][(split, image_path.name)] = teacher_prediction(
                bundle, image, args.img_size, device, args.tta
            )
        del bundle
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for split, records in records_by_split.items():
        destination = args.cache_root / split
        destination.mkdir(parents=True, exist_ok=True)
        for image_path, _ in records:
            unet_probability = teacher_probabilities["unet"][(split, image_path.name)]
            segformer_probability = teacher_probabilities["segformer"][(split, image_path.name)]
            unet_mask = unet_probability >= thresholds["unet"]
            segformer_mask = segformer_probability >= thresholds["segformer"]
            np.savez_compressed(
                cache_path(args.cache_root, split, image_path),
                unet_probability=unet_probability,
                segformer_probability=segformer_probability,
                teacher_agreement=np.float32(binary_dice(unet_mask, segformer_mask)),
            )

    (args.cache_root / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "teacher_unet": str(args.teacher_unet),
                "teacher_segformer": str(args.teacher_segformer),
                "img_size": args.img_size,
                "tta": args.tta,
                "thresholds": thresholds,
                "fusion": "arithmetic_mean_of_teacher_probabilities",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


@dataclass
class DistillationRecord:
    image_path: Path
    mask_path: Path
    cache_path: Path
    teacher_agreement: float


class DistillationDataset(Dataset):
    def __init__(self, records, img_size=256, clahe=True, augment=False):
        self.records = records
        self.img_size = img_size
        self.clahe = clahe
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = cv2.imread(str(record.image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(record.mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(f"Par inválido: {record.image_path}")
        with np.load(record.cache_path) as cache:
            teacher_probability = fuse_teacher_probabilities(
                cache["unet_probability"], cache["segformer_probability"]
            )
        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        teacher_probability = cv2.resize(
            teacher_probability, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR
        )
        if self.clahe:
            image = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)
        image = image.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)

        if self.augment:
            if np.random.rand() < 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()
                teacher_probability = np.fliplr(teacher_probability).copy()
            if np.random.rand() < 0.3:
                angle = float(np.random.uniform(-12.0, 12.0))
                center = (self.img_size / 2, self.img_size / 2)
                matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                image = cv2.warpAffine(image, matrix, (self.img_size, self.img_size), borderMode=cv2.BORDER_REFLECT_101)
                mask = cv2.warpAffine(mask, matrix, (self.img_size, self.img_size), flags=cv2.INTER_NEAREST, borderValue=0)
                teacher_probability = cv2.warpAffine(
                    teacher_probability, matrix, (self.img_size, self.img_size), flags=cv2.INTER_LINEAR, borderValue=0
                )

        image_tensor = torch.tensor(np.stack([image, image, image]), dtype=torch.float32)
        return (
            image_tensor,
            torch.tensor(mask, dtype=torch.float32),
            torch.tensor(np.clip(teacher_probability, 0.0, 1.0), dtype=torch.float32),
        )


def make_records(args, split, records):
    result = []
    for image_path, mask_path in records:
        path = cache_path(args.cache_root, split, image_path)
        if not path.exists():
            raise FileNotFoundError(f"Cache de professores ausente: {path}")
        with np.load(path) as cache:
            agreement = float(cache["teacher_agreement"])
        if args.consensus_only and agreement < args.min_teacher_agreement:
            continue
        result.append(DistillationRecord(image_path, mask_path, path, agreement))
    if not result:
        raise ValueError(f"Nenhuma amostra restou na divisão {split} após o filtro de consenso.")
    return result


def supervised_loss(logits, target):
    logits = logits.squeeze(1)
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probability = torch.sigmoid(logits)
    intersection = (probability * target).flatten(1).sum(1)
    denominator = probability.flatten(1).sum(1) + target.flatten(1).sum(1)
    dice = 1.0 - ((2.0 * intersection + 1e-6) / (denominator + 1e-6)).mean()
    return bce + dice


def distillation_loss(logits, teacher_probability, temperature):
    logits = logits.squeeze(1)
    teacher_probability = teacher_probability.clamp(1e-4, 1.0 - 1e-4)
    teacher_logit = torch.logit(teacher_probability)
    teacher_soft = torch.sigmoid(teacher_logit / temperature)
    return F.binary_cross_entropy_with_logits(logits / temperature, teacher_soft) * (temperature ** 2)


def evaluate(model, loader, device, threshold_candidates):
    model.eval()
    logits = []
    masks = []
    with torch.no_grad():
        for images, manual_masks, _ in loader:
            logits.append(model(images.to(device)).squeeze(1).cpu())
            masks.append(manual_masks)
    logits = torch.cat(logits)
    masks = torch.cat(masks)
    best = None
    for threshold in threshold_candidates:
        dice = dice_score(logits, masks, threshold=threshold, from_logits=True)
        iou = iou_score(logits, masks, threshold=threshold, from_logits=True)
        objective = (dice + iou) / 2.0
        candidate = {"threshold": threshold, "dice": dice, "iou": iou, "objective": objective}
        if best is None or candidate["objective"] > best["objective"]:
            best = candidate
    return best


def teacher_metrics(records, teacher_name, threshold=0.5):
    dices = []
    ious = []
    for record in records:
        image_mask = cv2.imread(str(record.mask_path), cv2.IMREAD_GRAYSCALE) > 0
        with np.load(record.cache_path) as cache:
            probability = cache[f"{teacher_name}_probability"]
        reference = cv2.resize(image_mask.astype(np.uint8), probability.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
        prediction = probability >= threshold
        intersection = np.logical_and(reference, prediction).sum()
        union = np.logical_or(reference, prediction).sum()
        dices.append(binary_dice(reference, prediction))
        ious.append(float(intersection / union) if union else 1.0)
    return {"dice": float(np.mean(dices)), "iou": float(np.mean(ious)), "samples": len(dices)}


def write_history(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Treina U-Net estudante por destilação de U-Net e SegFormer-B0.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--teacher-unet", type=Path, default=DEFAULT_UNET_TEACHER)
    parser.add_argument("--teacher-segformer", type=Path, default=DEFAULT_SEGFORMER_TEACHER)
    parser.add_argument("--student-checkpoint", type=Path, default=DEFAULT_STUDENT)
    parser.add_argument("--experiment-name", default="capsule_distilled_unet_segformer_v1")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--supervised-weight", type=float, default=1.0)
    parser.add_argument("--distillation-weight", type=float, default=0.5)
    parser.add_argument("--early-stopping", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--clahe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tta", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--consensus-only", action="store_true")
    parser.add_argument("--min-teacher-agreement", type=float, default=0.94)
    return parser.parse_args()


def main():
    args = parse_args()
    args.dataset_root = args.dataset_root.resolve()
    args.output_root = args.output_root.resolve()
    args.cache_root = args.cache_root.resolve()
    args.teacher_unet = args.teacher_unet.resolve()
    args.teacher_segformer = args.teacher_segformer.resolve()
    args.student_checkpoint = args.student_checkpoint.resolve()
    args.history_path = args.output_root / f"{args.experiment_name}_history.csv"
    args.summary_path = args.output_root / f"{args.experiment_name}_summary.json"

    if not 0.0 <= args.min_teacher_agreement <= 1.0:
        raise ValueError("--min-teacher-agreement deve estar entre 0 e 1.")
    if args.temperature <= 0:
        raise ValueError("--temperature deve ser positiva.")
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo da destilação: {device}")

    records_by_split = {split: split_images(args.dataset_root, split) for split in ("train", "val", "test")}
    cache_teachers(args, records_by_split, device)
    records = {split: make_records(args, split, rows) for split, rows in records_by_split.items()}
    print("Amostras após filtro:", {split: len(rows) for split, rows in records.items()})

    train_dataset = DistillationDataset(records["train"], args.img_size, args.clahe, args.augment)
    val_dataset = DistillationDataset(records["val"], args.img_size, args.clahe, False)
    test_dataset = DistillationDataset(records["test"], args.img_size, args.clahe, False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device == "cuda")
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=device == "cuda")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=device == "cuda")

    model = UNet(in_channels=3, out_channels=1, base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4, min_lr=1e-6)
    thresholds = tuple(round(value / 100, 2) for value in range(35, 70, 5))
    best_val = -1.0
    best_epoch = 0
    best_threshold = 0.5
    patience = 0
    history = []
    start = time.time()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"DistilledUNet {epoch + 1}/{args.epochs}")
        for images, manual_masks, teacher_probability in progress:
            images = images.to(device, non_blocking=True)
            manual_masks = manual_masks.to(device, non_blocking=True)
            teacher_probability = teacher_probability.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss_manual = supervised_loss(logits, manual_masks)
            loss_distill = distillation_loss(logits, teacher_probability, args.temperature)
            loss = args.supervised_weight * loss_manual + args.distillation_weight * loss_distill
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            progress.set_postfix(loss=f"{loss.item():.4f}")

        val = evaluate(model, val_loader, device, thresholds)
        scheduler.step(val["dice"])
        row = {
            "epoch": epoch + 1,
            "train_loss": total_loss / max(len(train_loader), 1),
            "val_dice": val["dice"],
            "val_iou": val["iou"],
            "val_threshold": val["threshold"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(row)
        if val["objective"] > best_val:
            best_val = val["objective"]
            best_epoch = epoch + 1
            best_threshold = val["threshold"]
            patience = 0
            args.student_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.student_checkpoint)
        else:
            patience += 1
            if patience >= args.early_stopping:
                print(f"Early stopping na época {epoch + 1}.")
                break

    write_history(args.history_path, history)
    student_meta = {
        "schema_version": 1,
        "model_name": "DistilledUNet",
        "model_kwargs": {"in_channels": 3, "out_channels": 1, "base_channels": args.base_channels},
        "best_threshold": best_threshold,
        "experiment": args.experiment_name,
        "method": "knowledge_distillation",
        "teachers": {"unet": str(args.teacher_unet), "segformer": str(args.teacher_segformer)},
        "fusion": "arithmetic_mean_of_teacher_probabilities",
        "temperature": args.temperature,
        "supervised_weight": args.supervised_weight,
        "distillation_weight": args.distillation_weight,
        "dataset_root": str(args.dataset_root),
        "consensus_only": args.consensus_only,
        "min_teacher_agreement": args.min_teacher_agreement,
        "best_epoch": best_epoch,
    }
    save_checkpoint_metadata(args.student_checkpoint, student_meta)

    checkpoint = torch.load(args.student_checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    val = evaluate(model, val_loader, device, thresholds)
    test = evaluate(model, test_loader, device, (best_threshold,))
    summary = {
        **student_meta,
        "device": device,
        "elapsed_seconds": time.time() - start,
        "samples": {split: len(rows) for split, rows in records.items()},
        "teacher_metrics_test": {
            "unet": teacher_metrics(
                records["test"],
                "unet",
                float(load_checkpoint_metadata(args.teacher_unet).get("best_threshold", 0.5)),
            ),
            "segformer": teacher_metrics(
                records["test"],
                "segformer",
                float(load_checkpoint_metadata(args.teacher_segformer).get("best_threshold", 0.5)),
            ),
        },
        "student_metrics": {"val": val, "test": test},
        "independent_test": True,
        "test_threshold_fixed_from_validation": best_threshold,
        "test_warning": "O teste manual foi utilizado apenas após a seleção do checkpoint.",
    }
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
