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

from src.segmentation.core.checkpoint_metadata import load_checkpoint_metadata
from src.segmentation.experiments.train_deeplab_intrarenal_multiclass import (
    CLASS_NAMES,
    build_model,
)


DEFAULT_DATASET_ROOT = PROJECT_ROOT / "dataset_aumentado" / "dataset_geral"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "intrarenal_deeplab_resnet50_multiclass_annotator1.pth"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "intrarenal_model3" / "intrarenal_multiclass_predictions_dataset_geral"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Aplica o DeepLabV3 multiclasse intrarrenal dentro das mascaras "
            "renais existentes em dataset_geral."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--pad-ratio", type=float, default=0.12)
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


def load_bundle(args, device):
    metadata = load_checkpoint_metadata(args.checkpoint)
    config = metadata.get("config", {})
    img_size = args.img_size or int(config.get("img_size", 256))
    backbone = config.get("backbone", "resnet50")
    model = build_model(backbone=backbone, pretrained=False, num_classes=len(CLASS_NAMES)).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    return {"model": model.eval(), "metadata": metadata, "img_size": img_size, "backbone": backbone}


def predict_roi(bundle, roi_image, roi_kidney, device):
    img_size = bundle["img_size"]
    resized = cv2.resize(roi_image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    kidney = cv2.resize(roi_kidney, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    image_float = resized.astype(np.float32) / 255.0
    kidney_float = (kidney > 0).astype(np.float32)
    channels = np.stack([image_float, image_float * kidney_float, kidney_float], axis=0)
    tensor = torch.tensor(channels, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = bundle["model"](tensor)["out"]
        prediction = logits.argmax(dim=1).cpu().numpy()[0].astype(np.uint8)
    prediction[kidney == 0] = 0
    return prediction


def generate_one(bundle, row, args, device):
    image = load_image(Path(row["dataset_image_path"]))
    kidney_mask = load_mask(Path(row["dataset_mask_path"]), image.shape)
    bounds = roi_bounds(kidney_mask, args.pad_ratio)
    if bounds is None:
        return None
    roi_image = crop(image, bounds)
    roi_kidney = crop(kidney_mask, bounds)
    model_image = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(roi_image)
    roi_prediction = predict_roi(bundle, model_image, roi_kidney, device)
    roi_prediction = cv2.resize(
        roi_prediction,
        (roi_image.shape[1], roi_image.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    roi_prediction[roi_kidney == 0] = 0
    prediction = np.zeros_like(kidney_mask, dtype=np.uint8)
    x1, y1, x2, y2 = bounds
    prediction[y1:y2, x1:x2] = roi_prediction
    return image, kidney_mask, prediction, bounds


def mask_stats(prediction, kidney_mask):
    kidney_pixels = int(kidney_mask.sum())
    stats = {"kidney_pixels": kidney_pixels}
    for class_id, class_name in enumerate(CLASS_NAMES[1:], start=1):
        pixels = int((prediction == class_id).sum())
        stats[f"{class_name}_pixels"] = pixels
        stats[f"{class_name}_to_kidney_ratio"] = float(pixels / max(kidney_pixels, 1))
    return stats


def create_preview(image, kidney_mask, prediction, bounds, label):
    roi_image = crop(image, bounds)
    roi_kidney = crop(kidney_mask, bounds)
    roi_prediction = crop(prediction, bounds)
    base = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2BGR)
    overlay = base.copy()
    overlay[roi_kidney > 0] = (80, 80, 80)
    overlay[roi_prediction == 1] = (255, 220, 0)
    overlay[roi_prediction == 2] = (0, 220, 255)
    overlay[roi_prediction == 3] = (0, 128, 255)
    overlay = cv2.addWeighted(base, 0.62, overlay, 0.38, 0)
    label_map = (roi_prediction * 70).astype(np.uint8)
    label_rgb = cv2.cvtColor(label_map, cv2.COLOR_GRAY2BGR)
    tiles = [
        cv2.resize(tile, (280, 220), interpolation=cv2.INTER_AREA)
        for tile in (base, overlay, label_rgb)
    ]
    panel = cv2.hconcat(tiles)
    cv2.putText(panel, label[:108], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (0, 255, 255), 2, cv2.LINE_AA)
    return panel


def main():
    args = parse_args()
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
    bundle = load_bundle(args, device)
    output_rows = []
    preview_count = 0
    for index, row in enumerate(targets, start=1):
        generated = generate_one(bundle, row, args, device)
        image_id = row["image_id"]
        if generated is None:
            output_rows.append(
                {
                    "image_id": image_id,
                    "source_name": row.get("source_name", ""),
                    "kidney_mask_status": row.get("mask_status", ""),
                    "dataset_image_path": row["dataset_image_path"],
                    "dataset_kidney_mask_path": row["dataset_mask_path"],
                    "predicted_label_mask_path": "",
                    "predicted_cortex_mask_path": "",
                    "predicted_medulla_mask_path": "",
                    "predicted_central_echo_complex_mask_path": "",
                    "prediction_status": "skipped_empty_kidney_mask",
                }
            )
            continue

        image, kidney_mask, prediction, bounds = generated
        stats = mask_stats(prediction, kidney_mask)
        label_path = args.output_root / "labels" / f"{image_id}.png"
        save_image(label_path, prediction)
        class_paths = {}
        for class_id, class_name in enumerate(CLASS_NAMES[1:], start=1):
            path = args.output_root / "masks" / class_name / f"{image_id}.png"
            save_image(path, ((prediction == class_id).astype(np.uint8) * 255))
            class_paths[class_name] = path

        status = (
            "candidate_existing_kidney_mask"
            if row.get("mask_status", "") == "existing"
            else "candidate_requires_kidney_roi_review"
        )
        if preview_count < args.preview_count:
            preview_count += 1
            panel = create_preview(
                image,
                kidney_mask,
                prediction,
                bounds,
                f"{image_id} | c={stats['cortex_to_kidney_ratio']:.3f} "
                f"m={stats['medulla_to_kidney_ratio']:.3f} "
                f"cec={stats['central_echo_complex_to_kidney_ratio']:.3f}",
            )
            save_image(args.output_root / "previews" / f"candidate_{preview_count:03d}_{image_id}.png", panel)

        output_rows.append(
            {
                "image_id": image_id,
                "source_name": row.get("source_name", ""),
                "kidney_mask_status": row.get("mask_status", ""),
                "dataset_image_path": row["dataset_image_path"],
                "dataset_kidney_mask_path": row["dataset_mask_path"],
                "predicted_label_mask_path": str(label_path),
                "predicted_cortex_mask_path": str(class_paths["cortex"]),
                "predicted_medulla_mask_path": str(class_paths["medulla"]),
                "predicted_central_echo_complex_mask_path": str(class_paths["central_echo_complex"]),
                "prediction_status": status,
                **{
                    key: (f"{value:.6f}" if isinstance(value, float) else value)
                    for key, value in stats.items()
                },
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
        "architecture": "deeplabv3_multiclass_roi",
        "classes": {index: name for index, name in enumerate(CLASS_NAMES)},
        "device": device,
        "kidney_rois_processed": len(output_rows),
        "status_counts": status_counts,
        "output_root": str(args.output_root),
        "model_summary": bundle["metadata"],
        "note": (
            "Estas mascaras sao pseudo-rotulos multiclasse candidatos para "
            "cortex, medulla e central echo complex. Elas dependem da mascara "
            "renal da etapa 1 e requerem revisao humana antes de retreinamento."
        ),
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
