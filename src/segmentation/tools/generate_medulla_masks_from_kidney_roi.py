import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.build_dataset_geral import predict_probability, prepare_tensor
from src.segmentation.core.checkpoint_metadata import load_checkpoint_metadata
from src.segmentation.core.model_loader import load_model_bundle
from src.segmentation.experiments.train_medulla_roi_unet import build_model as build_medulla_roi_unet


DEFAULT_DATASET_ROOT = PROJECT_ROOT / "dataset_aumentado" / "dataset_geral"
DEFAULT_DEEPLAB_CHECKPOINT = PROJECT_ROOT / "models" / "medulla_deeplab_resnet50_annotator1_baseline.pth"
DEFAULT_ROI_UNET_CHECKPOINT = PROJECT_ROOT / "models" / "medulla_roi_unet_annotator1.pth"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "intrarenal_model3" / "medulla_predictions_dataset_geral"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Gera pseudo-mascaras candidatas de estrutura intrarrenal dentro das mascaras "
            "renais ja existentes em dataset_geral."
        )
    )
    parser.add_argument("--target", choices=["medulla", "cortex"], default="medulla")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--architecture", choices=["deeplab", "roi_unet"], default="deeplab")
    parser.add_argument("--model", choices=["unet", "unetplusplus", "deeplab", "segformer"], default="deeplab")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--pad-ratio", type=float, default=0.12)
    parser.add_argument("--min-medulla-ratio", type=float, default=0.01)
    parser.add_argument("--max-medulla-ratio", type=float, default=0.65)
    parser.add_argument("--preview-count", type=int, default=24)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def read_manifest(path):
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Nao foi possivel ler imagem: {path}")
    return image


def load_mask(path, shape):
    mask = load_image(path)
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


def roi_bounds(mask, pad_ratio):
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    width = int(xs.max() - xs.min() + 1)
    height = int(ys.max() - ys.min() + 1)
    pad = max(4, int(round(max(width, height) * pad_ratio)))
    return (
        max(0, int(xs.min()) - pad),
        max(0, int(ys.min()) - pad),
        min(mask.shape[1], int(xs.max()) + pad + 1),
        min(mask.shape[0], int(ys.max()) + pad + 1),
    )


def crop(array, bounds):
    x1, y1, x2, y2 = bounds
    return array[y1:y2, x1:x2]


def save_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Nao foi possivel salvar imagem: {path}")


def create_preview(image, kidney_mask, medulla_mask, bounds, label):
    roi_image = crop(image, bounds)
    roi_kidney = crop(kidney_mask, bounds)
    roi_medulla = crop(medulla_mask, bounds)
    base = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2BGR)
    kidney_overlay = base.copy()
    kidney_overlay[roi_kidney > 0] = (0, 180, 255)
    kidney_overlay = cv2.addWeighted(base, 0.73, kidney_overlay, 0.27, 0)
    medulla_overlay = base.copy()
    medulla_overlay[roi_medulla > 0] = (0, 0, 255)
    medulla_overlay = cv2.addWeighted(base, 0.70, medulla_overlay, 0.30, 0)
    mask_rgb = cv2.cvtColor(medulla_mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    mask_rgb = crop(mask_rgb, bounds)
    tiles = [
        cv2.resize(tile, (280, 220), interpolation=cv2.INTER_AREA)
        for tile in (kidney_overlay, medulla_overlay, mask_rgb)
    ]
    panel = cv2.hconcat(tiles)
    cv2.putText(panel, label[:108], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (0, 255, 255), 2, cv2.LINE_AA)
    return panel


def load_medulla_bundle(args, device):
    if args.architecture == "roi_unet":
        metadata = load_checkpoint_metadata(args.checkpoint)
        base_channels = metadata.get("config", {}).get("base_channels", 32)
        model = build_medulla_roi_unet(base_channels=base_channels).to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        return {
            "model": model.eval(),
            "threshold": float(metadata.get("best_threshold", 0.5)),
            "architecture": "roi_unet",
        }
    bundle = load_model_bundle(args.model, device=device, checkpoint_path=args.checkpoint)
    bundle["architecture"] = "deeplab"
    return bundle


def predict_roi_probability(bundle, roi_image, roi_kidney, args, device):
    if bundle["architecture"] == "roi_unet":
        resized = cv2.resize(roi_image, (args.img_size, args.img_size), interpolation=cv2.INTER_LINEAR)
        kidney = cv2.resize(roi_kidney, (args.img_size, args.img_size), interpolation=cv2.INTER_NEAREST)
        image_float = resized.astype(np.float32) / 255.0
        kidney_float = (kidney > 0).astype(np.float32)
        channels = np.stack([image_float, image_float * kidney_float, kidney_float], axis=0)
        tensor = torch.tensor(channels, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            return torch.sigmoid(bundle["model"](tensor)).cpu().numpy()[0, 0]
    tensor = prepare_tensor(roi_image, args.img_size, device)
    with torch.no_grad():
        return predict_probability(bundle, tensor, args.img_size)


def generate_one(bundle, row, args, device):
    image = load_image(Path(row["dataset_image_path"]))
    kidney_mask = load_mask(Path(row["dataset_mask_path"]), image.shape)
    bounds = roi_bounds(kidney_mask, args.pad_ratio)
    if bounds is None:
        return None

    roi_image = crop(image, bounds)
    roi_kidney = crop(kidney_mask, bounds)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    model_image = clahe.apply(roi_image)
    probability = predict_roi_probability(bundle, model_image, roi_kidney, args, device)
    probability = cv2.resize(
        probability,
        (roi_image.shape[1], roi_image.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    threshold = float(bundle["threshold"])
    roi_medulla = ((probability >= threshold) & (roi_kidney > 0)).astype(np.uint8)
    medulla_mask = np.zeros_like(kidney_mask)
    x1, y1, x2, y2 = bounds
    medulla_mask[y1:y2, x1:x2] = roi_medulla

    pixels = int(medulla_mask.sum())
    kidney_pixels = int(kidney_mask.sum())
    medulla_ratio = float(pixels / max(kidney_pixels, 1))
    confidence = float(probability[roi_medulla > 0].mean()) if pixels else 0.0
    if pixels == 0:
        status = "rejected_empty_prediction"
    elif medulla_ratio < args.min_medulla_ratio:
        status = "rejected_ratio_low"
    elif medulla_ratio > args.max_medulla_ratio:
        status = "rejected_ratio_high"
    else:
        status = "candidate_for_review"
    return image, kidney_mask, medulla_mask, bounds, status, pixels, medulla_ratio, confidence


def main():
    args = parse_args()
    if args.checkpoint is None:
        args.checkpoint = (
            (
                DEFAULT_ROI_UNET_CHECKPOINT
                if args.target == "medulla"
                else PROJECT_ROOT / "models" / "cortex_roi_unet_annotator1.pth"
            )
            if args.architecture == "roi_unet"
            else DEFAULT_DEEPLAB_CHECKPOINT
        )
    if args.output_root == DEFAULT_OUTPUT_ROOT and args.target == "cortex":
        args.output_root = PROJECT_ROOT / "results" / "intrarenal_model3" / "cortex_roi_unet_predictions_dataset_geral"
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint nao encontrado: {args.checkpoint}")
    rows = read_manifest(args.dataset_root / "manifest.csv")
    targets = [
        row
        for row in rows
        if row.get("has_mask", "").lower() == "true"
        and row.get("dataset_mask_path")
        and Path(row["dataset_mask_path"]).exists()
    ]
    if args.limit is not None:
        targets = targets[: args.limit]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = load_medulla_bundle(args, device)
    mask_path_field = f"predicted_{args.target}_mask_path"
    pixels_field = f"{args.target}_pixels"
    ratio_field = f"{args.target}_to_kidney_ratio"
    output_rows = []
    preview_count = 0
    for index, row in enumerate(targets, 1):
        generated = generate_one(bundle, row, args, device)
        if generated is None:
            output_rows.append(
                {
                    "image_id": row["image_id"],
                    "source_name": row.get("source_name", ""),
                    "kidney_mask_status": row.get("mask_status", ""),
                    "dataset_image_path": row["dataset_image_path"],
                    "dataset_kidney_mask_path": row["dataset_mask_path"],
                    mask_path_field: "",
                    "prediction_status": "skipped_empty_kidney_mask",
                    pixels_field: 0,
                    ratio_field: "0.000000",
                    "mean_foreground_probability": "0.000000",
                }
            )
            continue
        image, kidney_mask, medulla_mask, bounds, status, pixels, ratio, confidence = generated
        if status == "candidate_for_review":
            if row.get("mask_status", "") == "existing":
                status = "candidate_existing_kidney_mask"
            else:
                status = "candidate_requires_kidney_roi_review"
        image_id = row["image_id"]
        mask_path = args.output_root / "masks" / f"{image_id}.png"
        save_image(mask_path, medulla_mask * 255)
        if preview_count < args.preview_count and status.startswith("candidate_"):
            preview_count += 1
            panel = create_preview(
                image,
                kidney_mask,
                medulla_mask,
                bounds,
                f"{image_id} | ratio={ratio:.3f} conf={confidence:.3f}",
            )
            save_image(args.output_root / "previews" / f"candidate_{preview_count:03d}_{image_id}.png", panel)
        output_rows.append(
            {
                "image_id": image_id,
                "source_name": row.get("source_name", ""),
                "kidney_mask_status": row.get("mask_status", ""),
                "dataset_image_path": row["dataset_image_path"],
                "dataset_kidney_mask_path": row["dataset_mask_path"],
                mask_path_field: str(mask_path),
                "prediction_status": status,
                pixels_field: pixels,
                ratio_field: f"{ratio:.6f}",
                "mean_foreground_probability": f"{confidence:.6f}",
            }
        )
        if index % 250 == 0:
            print(f"Processadas {index}/{len(targets)} ROIs renais")

    write_csv(args.output_root / "manifest.csv", output_rows)
    status_counts = {}
    for row in output_rows:
        status_counts[row["prediction_status"]] = status_counts.get(row["prediction_status"], 0) + 1
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset_root": str(args.dataset_root),
        "checkpoint": str(args.checkpoint),
        "architecture": args.architecture,
        "target": args.target,
        "threshold_from_checkpoint": float(bundle["threshold"]),
        "device": device,
        "kidney_rois_processed": len(output_rows),
        "status_counts": status_counts,
        "output_root": str(args.output_root),
        "note": (
            f"Estas mascaras sao pseudo-rotulos candidatos de {args.target}. "
            "Predicoes baseadas em mascaras renais geradas requerem revisao "
            "anatomica da ROI antes de qualquer uso em retreinamento."
        ),
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
