import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.core.model_loader import load_model_bundle
from src.segmentation.tools.compare.unet_preprocess import (
    apply_clahe,
    apply_superres,
    load_swinir_model,
    swinir_upscale,
)


DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT
    / "dataset_aumentado"
    / "dataset_intrarrenal"
    / "supervisionado"
    / "capsule_annotator_1"
    / "test"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "segmentation_experiments" / "capsule_preprocess_benchmark"
CHECKPOINTS = {
    "unet": PROJECT_ROOT / "models" / "kidneyus_capsule_unet.pth",
    "unetplusplus": PROJECT_ROOT / "models" / "kidneyus_capsule_unetplusplus.pth",
    "deeplab": PROJECT_ROOT / "models" / "kidneyus_capsule_deeplab.pth",
    "segformer": PROJECT_ROOT / "models" / "kidneyus_capsule_segformer.pth",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reavalia os quatro modelos da capsula com preprocessamentos alternativos."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument(
        "--preprocess",
        choices=["raw", "clahe", "lanczos_clahe", "swinir_clahe"],
        default="lanczos_clahe",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--swinir-model",
        type=Path,
        default=PROJECT_ROOT / "models" / "super_resolution" / "001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth",
    )
    return parser.parse_args()


def read_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Nao foi possivel ler imagem: {path}")
    return image


def prepare_image(image, preprocess, swinir_model, device):
    if preprocess == "raw":
        return image
    if preprocess == "clahe":
        return apply_clahe(image)
    if preprocess == "lanczos_clahe":
        return apply_clahe(apply_superres(image, 2))
    if preprocess == "swinir_clahe":
        return apply_clahe(swinir_upscale(image, swinir_model, device, scale=2))
    raise ValueError(f"Preprocessamento desconhecido: {preprocess}")


def prepare_tensor(image, img_size, device):
    resized = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype(np.float32) / 255.0
    stacked = np.stack([normalized, normalized, normalized], axis=0)
    return torch.tensor(stacked, dtype=torch.float32).unsqueeze(0).to(device)


def predict_probability(bundle, tensor, img_size):
    model = bundle["model"]
    display_name = bundle["display_name"]
    if display_name == "SegFormer":
        logits = model(pixel_values=tensor).logits
        logits = torch.nn.functional.interpolate(
            logits,
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        )
    elif display_name == "DeepLab":
        logits = model(tensor)["out"]
    else:
        logits = model(tensor)
    return torch.sigmoid(logits).squeeze().detach().cpu().numpy()


def keep_largest_component(mask):
    mask = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 2:
        return mask
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest_label).astype(np.uint8)


def predict_mask(bundle, image, original_shape, args, device):
    tensor = prepare_tensor(image, args.img_size, device)
    with torch.no_grad():
        probability = predict_probability(bundle, tensor, args.img_size)
    probability = cv2.resize(
        probability,
        (original_shape[1], original_shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    return keep_largest_component(probability >= float(bundle["threshold"]))


def metrics(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    pred_sum = pred.sum()
    target_sum = target.sum()
    dice = (2 * intersection) / max(pred_sum + target_sum, 1)
    iou = intersection / max(union, 1)
    precision = intersection / max(pred_sum, 1)
    recall = intersection / max(target_sum, 1)
    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
    }


def aggregate(rows, model_key):
    result = {"model_key": model_key}
    for field in ("dice", "iou", "precision", "recall"):
        values = [float(row[field]) for row in rows if row["model_key"] == model_key]
        result[field] = float(np.mean(values))
        result[f"{field}_std"] = float(np.std(values))
    return result


def main():
    args = parse_args()
    run_dir = args.output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    image_dir = args.dataset_root / "image"
    mask_dir = args.dataset_root / "mask"
    image_paths = sorted(image_dir.glob("*.png"))
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    swinir_model = None
    if args.preprocess == "swinir_clahe":
        swinir_model = load_swinir_model(args.swinir_model, device)

    bundles = {
        key: load_model_bundle(key, device=device, checkpoint_path=checkpoint)
        for key, checkpoint in CHECKPOINTS.items()
    }
    prepared_images = {}
    targets = {}
    for image_path in image_paths:
        image = read_gray(image_path)
        prepared_images[image_path.name] = (
            prepare_image(image, args.preprocess, swinir_model, device),
            image.shape,
        )
        targets[image_path.name] = (read_gray(mask_dir / image_path.name) > 0).astype(np.uint8)

    rows = []
    for model_key, bundle in bundles.items():
        for image_name, (model_image, original_shape) in prepared_images.items():
            pred = predict_mask(bundle, model_image, original_shape, args, device)
            row = {
                "image_name": image_name,
                "model_key": model_key,
                "model": bundle["display_name"],
                "threshold": float(bundle["threshold"]),
                "preprocess": args.preprocess,
            }
            row.update(metrics(pred, targets[image_name]))
            rows.append(row)

    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = run_dir / "metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = [aggregate(rows, model_key) for model_key in bundles]
    summary_csv = run_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    best = max(summary_rows, key=lambda row: row["dice"])
    summary = {
        "dataset_root": str(args.dataset_root),
        "preprocess": args.preprocess,
        "samples": len(image_paths),
        "device": device,
        "img_size": args.img_size,
        "summary_csv": str(summary_csv),
        "metrics_csv": str(metrics_csv),
        "best_model": best,
        "note": "A super-resolucao foi usada apenas como pre-processamento auxiliar para segmentacao.",
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
